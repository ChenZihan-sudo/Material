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
    train_losses,
    val_losses,
    test_losses,
    test_eval_results,
    optimizer,
    scheduler,
    result_path,
    regression_title="Model Regression",
):
    print("-------------------------------------------")
    print(f"Epoch {epoch}: Saving data and model...")
    save_hyper_parameter(args, result_path)
    save_train_progress(epoch - 1, train_losses, val_losses, test_losses, result_path)
    test_loss, test_out, test_y = test_eval_results

    # Reverse normalization of test_out and y
    min, max = get_data_scale(dataset_args["get_parameters_from"])
    test_y = reverse_min_max_scalar_1d(test_y, min, max)
    test_out = reverse_min_max_scalar_1d(test_out, min, max)
    loss = (test_out.squeeze() - test_y).abs().mean()
    print("MAE loss: ", loss.item())

    # save results
    plot_training_progress(len(train_losses), train_losses, val_losses, test_losses, res_path=result_path, threshold=0.2)
    save_regression_result(test_out, test_y, result_path)
    plot_regression_result(regression_title, result_path, plotfilename="regression_figure.jpeg")

    # save model
    save_model(result_path, model, epoch, loss, optimizer, scheduler)
    print("-------------------------------------------")

    return loss


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

    test_best_loss = None

    # get best model based on best test loss
    save_best_model = tune_args["save_best_model"]
    # save results every save_step epochs
    save_step = tune_args["save_step"]

    # model summary
    model_summary(model)
    with open(osp.join(result_path, "model_info.txt"), "w") as file:
        model_summary(model, file=file)
        file.close()

    train_losses = []
    val_losses = []
    test_losses = []

    epochs = model_args["epochs"]
    pbar = tqdm(total=(epochs + 1))
    for epoch in range(1, epochs + 1):

        model, train_loss = train_step(model, train_loader, train_dataset, optimizer, device)
        val_loss, _, _ = test_evaluations(model, val_loader, validation_dataset, device)
        test_eval_results = test_evaluations(model, test_loader, test_dataset, device)
        test_loss = test_eval_results[0]

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        save_params = (
            [args, dataset_args, epoch, model]
            + [train_losses, val_losses, test_losses, test_eval_results]
            + [optimizer, scheduler, result_path, model_name]
        )

        # save best model
        if test_best_loss is None or test_loss < test_best_loss:
            test_best_loss = test_loss
            if save_best_model:
                save_result_data(*save_params)

        # save results every save_step epochs
        if save_best_model is False and epoch % save_step == 0:
            save_result_data(*save_params)

        # report results to tay tune
        checkpoint = Checkpoint.from_directory(result_path)
        train.report({"mean_absolute_error": val_loss}, checkpoint=checkpoint)

        # show messages
        progress_msg = f"epoch:{str(epoch)} train:{str(round(train_loss,4))} valid:{str(round(val_loss, 4))} test:{str(round(test_loss, 4))} lr:{str(round(current_lr, 8))} best_test:{str(round(test_best_loss, 4))}"
        pbar.set_description(progress_msg)
        pbar.update(1)
    pbar.close()


def start_tuning(model_name, dataset_name, args):
    tune_args = args["Tuning"]

    ray.init(num_gpus=1, num_cpus=30)
    # ray.init(**(tune_args["resources"]))

    # hyperparameters = []

    search_alg = HyperOptSearch(metric="mean_absolute_error", mode="min")
    # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)

    scheduler = ASHAScheduler(metric="mean_absolute_error", mode="min")

    # parameter_columns = [hyperparameter for hyperparameter in hyperparameters.keys()]
    # reporter = tune.CLIReporter(max_progress_rows=30, metric_columns=["mean_absolute_error"], parameter_columns=parameter_columns)
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
        tune.with_resources(trainable, resources={"cpu": 30, "gpu": 1}),
        tune_config=tune.TuneConfig(
            num_samples=tune_args["trial_num_samples"],
            search_alg=search_alg,
            scheduler=scheduler,
            max_concurrent_trials=tune_args["max_concurrent_trials"],
        ),
        run_config=train.RunConfig(
            name=trial_name,
            storage_path=storage_path,
            progress_reporter=reporter,
            log_to_file=log_to_file,
        ),
        param_space=args,
    )

    results = tuner.fit()
    best_result = results.get_best_result("mean_absolute_error", mode="min")
    print("best_result", best_result)
    print("hyperparameters: ", best_result.config)
