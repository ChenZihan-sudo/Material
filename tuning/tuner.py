import torch
import matplotlib.pyplot as plt

import os.path as osp
from tqdm import tqdm

from training.prepare import *

from models import *
from utils import *
from process import make_dataset

import ray
from ray import tune, train
from ray.train import Checkpoint
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler

import models

from datetime import datetime

from functools import partial


def save_result_data(
    args,
    dataset_args,
    epoch,
    model,
    optimizer,
    scheduler,
    result_path,
    train_losses,
    val_losses,
    test_losses=None,
    eval_results=None,
    regression_title="Model Regression",
):
    print("-------------------------------------------")
    print(f"Epoch {epoch}: Saving data and model...")
    save_hyper_parameter(args, result_path)
    save_train_progress(epoch - 1, train_losses, val_losses, test_losses, result_path)
    _, out, y = eval_results

    # Reverse normalization of out and y
    min, max = get_data_scale(dataset_args["get_parameters_from"])
    y = reverse_min_max_scalar_1d(y, min, max)
    out = reverse_min_max_scalar_1d(out, min, max)
    # loss = (out.squeeze() - y).abs().mean()
    # print("MAE loss: ", loss.item())

    # save results
    plot_training_progress(len(train_losses), train_losses, val_losses, test_losses, res_path=result_path, threshold=0.2)
    save_regression_result(out, y, result_path)
    mae = plot_regression_result(regression_title, result_path, plotfilename="regression_figure.jpeg")

    # save model
    save_model(result_path, model, epoch, mae, optimizer, scheduler)
    print("-------------------------------------------")
    
    return mae


# def trainable_model(args, model_name=None, dataset_name=None):


def trainable_model(args, dataset=None, model_name=None, dataset_name=None):
    print(f"############ Start Hyperparameter Tuning on {model_name} with {dataset_name} ############")
    # print("Hyperparameters: ", args)

    tune_args = args["Tuning"]
    dataset_args = args["Dataset"][dataset_name]
    model_args = args["Models"][model_name]

    # make dataset and data loader
    # train_dataset, validation_dataset, test_dataset = make_dataset(dataset_name, args, **(tune_args["dataset"]))
    train_dataset, validation_dataset, test_dataset = dataset
    dataloader_args = tune_args["data_loader"]
    train_loader, val_loader, test_loader = make_data_loader(train_dataset, validation_dataset, test_dataset, **dataloader_args)

    # print(f"dataset num, train:{len(train_dataset)}, val:{len(validation_dataset)}, test:{len(test_dataset)}")

    # get device
    device = get_device(args=args)

    # get model in dimension
    in_dim = train_dataset[0].x.shape[-1]
    # print(f"model in_dim: {in_dim}")
    if model_name in ("PNA", "ChemGNN"):
        deg = generate_deg(train_dataset).float()
        deg = deg.to(device)
        model = getattr(models, model_name)(deg, in_dim, **model_args)
        model = model.to(device)
    else:
        model = getattr(models, model_name)(in_dim, **model_args)
        model = model.to(device)

    # set optimizer
    optimizer_args = model_args["optimizer"]
    optimizer_params = optimizer_args["params"] if optimizer_args["params"] is not None else {}
    optimizer = getattr(torch.optim, optimizer_args["name"])(model.parameters(), lr=model_args["learning_rate"], **optimizer_params)

    # set scheduler
    scheduler_args = model_args["scheduler"]
    scheduler_params = scheduler_args["params"] if scheduler_args["params"] is not None else {}
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_args["name"])(optimizer, **scheduler_params)

    # create folder for recording results
    result_path = create_result_folder(osp.join(tune_args["save_result_on"], model_name))

    # get best model based on best test loss
    save_best_model = tune_args["save_best_model"]
    # save results every save_step epochs
    save_step = tune_args["save_step"]

    # model summary
    # model_summary(model)
    with open(osp.join(result_path, "model_info.txt"), "w") as file:
        model_summary(model, file=file)
        file.close()

    train_losses = []
    val_losses = []
    test_losses = []

    best_loss = None
    best_loss_epoch = None
    keep_best_epochs = 0

    epochs = model_args["epochs"]
    pbar = tqdm(total=(epochs + 1))
    for epoch in range(1, epochs + 1):

        eval_results = None

        model, train_loss = train_step(model, train_loader, train_dataset, optimizer, device)
        val_eval_results = test_evaluations(model, val_loader, validation_dataset, device)
        eval_results = val_eval_results
        val_loss = val_eval_results[0]

        eval_loss = val_loss
        
        test_loss = 0.0
        if epoch == epochs - 1:
            test_eval_results = test_evaluations(model, test_loader, test_dataset, device)
            eval_results = test_eval_results
            test_loss = test_eval_results[0]
            eval_loss = test_loss

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        save_params = (
            [args, dataset_args, epoch, model]
            + [optimizer, scheduler, result_path]
            + [train_losses, val_losses, test_losses, eval_results,model_name]
        )

        # save best model
        checkpoint = None
        if best_loss is None or eval_loss < best_loss:
            best_loss = eval_loss
            best_loss_epoch = epoch
            if save_best_model:
                save_result_data(*save_params)
        
        # save results every save_step epochs
        if save_best_model is False and epoch % save_step == 0:
            save_result_data(*save_params)

        # report results to tay tune
        keep_best_epochs = epoch - best_loss_epoch
        # checkpoint = Checkpoint.from_directory(result_path)
        train.report({"mean_absolute_error": best_loss, "keep_best_epochs": keep_best_epochs,"storage_path":result_path}, checkpoint=checkpoint)
        
        # stop condition
        if tune_args["keep_best_epochs"] <= keep_best_epochs:
            break

        # show messages
        progress_msg = f"epoch:{str(epoch)} train:{str(round(train_loss,4))} valid:{str(round(val_loss, 4))} test:{"-" if test_loss==0.0 else str(round(test_loss, 4))} lr:{str(round(current_lr, 8))} eval_best:{str(round(best_loss, 4))}"
        pbar.set_description(progress_msg)
        pbar.update(1)
    pbar.close()


def start_tuning(model_name, dataset_name, args):
    tune_args = args["Tuning"]
    
    tune_resources = tune_args["resources"]
    ray.init(num_cpus=tune_resources["num_cpus"],num_gpus=tune_resources["num_gpus"])

    search_alg = HyperOptSearch(metric="mean_absolute_error", mode="min")
    # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)

    scheduler = ASHAScheduler(metric="mean_absolute_error", mode="min", max_t=10000)

    reporter = tune.CLIReporter(max_progress_rows=100, metric_columns=["mean_absolute_error"])

    trial_name = f"{model_name}_{dataset_name}_{get_current_time()}"
    storage_path = tune_args["storage_path"]
    log_to_file = osp.join(storage_path, trial_name, "output.log") if tune_args["log_to_file"] else False
    print(f"Log file about trainable object save on {log_to_file}.")

    # trainable = partial(trainable_model, model_name=model_name, dataset_name=dataset_name)

    # create dataset in here
    dataset = make_dataset(dataset_name, args, **(tune_args["dataset"]))
    trainable = tune.with_parameters(trainable_model, dataset=dataset, model_name=model_name, dataset_name=dataset_name)
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": tune_resources["trial_cpus"], "gpu": tune_resources["trial_gpus"]}),
        tune_config=tune.TuneConfig(
            num_samples=tune_args["trial_num_samples"],
            search_alg=search_alg,
            scheduler=scheduler,
            max_concurrent_trials=tune_args["max_concurrent_trials"],
            time_budget_s=tune_args["time_budget_s"]
        ),
        run_config=train.RunConfig(
            name=trial_name,
            storage_path=storage_path,
            progress_reporter=reporter,
            log_to_file=log_to_file,
            # checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
        ),
        param_space=args,
    )

    results = tuner.fit()
    best_result = results.get_best_result("mean_absolute_error", mode="min")
    print("best_result", best_result)
    print("hyperparameters: ", best_result.config)
