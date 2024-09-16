import torch
import traceback
import matplotlib.pyplot as plt

import os.path as osp
from tqdm import tqdm

from training.prepare import *

from models import *
from utils import *
from process import make_dataset, delete_processed_data, create_dataset_occupy_table

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
    mae, r2 = plot_regression_result(regression_title, result_path, plotfilename="regression_figure.jpeg")

    # save model
    save_model(result_path, model, epoch, mae, optimizer, scheduler)
    print("-------------------------------------------")

    return mae, r2


# def trainable_model(args, model_name=None, dataset_name=None):
def trainable_model(load_args, args=None, dataset=None, model_name=None, dataset_name=None):
    """
    Params:
        - load_args: original arguments when restore the trial
        - args: current arguments file
    """
    print(f"############ Start Hyperparameter Tuning on {model_name} with {dataset_name} ############")

    # Tuning will inherit from current config file here to avoid problem about the absolute work dir
    tune_batch_size = load_args["Tuning"]["data_loader"]["batch_size"]  # batch_size is a tune argument and will not be changed.
    load_args["Tuning"] = args["Tuning"]
    load_args["Tuning"]["data_loader"]["batch_size"] = tune_batch_size
    load_args["Dataset"] = args["Dataset"]

    # load_args["Process"] = args["Process"]

    load_path = None
    if "restore_experiment_from" in args["Tuning"]:
        if args["Tuning"]["restore_experiment_from"] is not None:
            from ray.train.context import TrainContext

            content = TrainContext().get_storage()
            trial_path = osp.join(args["Tuning"]["restore_experiment_from"], content.trial_dir_name)

            progress_path = osp.join(trial_path, "progress.csv")
            if osp.exists(progress_path):  # restore training data
                # progress.csv file exist with model data path
                with open(progress_path, mode="r") as file:
                    reader = csv.DictReader(file)
                    for row in reader:
                        load_path = row["storage_path"]
                        break
                print("restore: get load model data path ", load_path)
                print(f"############ Resume Tuning on {model_name} with {dataset_name} ############")

    args = load_args

    tune_args = args["Tuning"]
    dataset_args = args["Dataset"][dataset_name]
    model_args = args["Models"][model_name]

    # tune_args["load_dataset_by_trainable"] = False  # debugging: remove this line
    # make dataset and data loader
    if tune_args["load_dataset_by_trainable"]:
        train_dataset, validation_dataset, test_dataset, data_processed_path = make_dataset(dataset_name, args, **(tune_args["dataset"]))
    else:  # use the dataset passed from start_tuning method
        train_dataset, validation_dataset, test_dataset, data_processed_path = dataset

    # sync the processed data path
    if args["Process"]["auto_processed_name"] is True:
        args["Dataset"][dataset_name]["processed_dir"] = data_processed_path
        dataset_args["processed_dir"] = data_processed_path
        dataset_args["get_parameters_from"] = dataset_args["processed_dir"]

    # patch: add identifier to delete datasets at a proper time
    create_dataset_occupy_table(osp.join(args["Default"]["absolute_work_dir"], "tuning/dataset_occupy_table.pt"), data_processed_path, 1)

    dataloader_args = tune_args["data_loader"]
    train_loader, val_loader, test_loader = make_data_loader(train_dataset, validation_dataset, test_dataset, **dataloader_args)

    # print("hyperparameters: ", args)
    # print(f"dataset num, train:{len(train_dataset)}, val:{len(validation_dataset)}, test:{len(test_dataset)}")

    # get device
    device = get_device(args=args)

    if load_path is not None:
        checkpoint = torch.load(osp.join(load_path, "checkpoint.pt"))
        model = checkpoint["model"]
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
    else:
        # get model in dimension
        in_dim = train_dataset[0].x.shape[-1]
        # print(f"model in_dim: {in_dim}")
        if model_name in ("PNA", "ChemGNN"):
            model_args["conv_params"]["edge_dim"] = args["Process"]["edge"]["edge_feature"]  # import edge_dim from Process.edge.edge_feature
            deg = generate_deg(train_dataset).float()
            deg = deg.to(device)
            model = getattr(models, model_name)(deg, in_dim, **model_args)
            model = model.to(device)
        else:
            model = getattr(models, model_name)(in_dim, **model_args)
            model = model.to(device)

    # set optimizer
    if load_path is not None:
        optimizer = checkpoint["optimizer"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        optimizer_args = model_args["optimizer"]
        optimizer_params = optimizer_args["params"] if optimizer_args["params"] is not None else {}
        optimizer = getattr(torch.optim, optimizer_args["name"])(model.parameters(), lr=model_args["learning_rate"], **optimizer_params)

    # set scheduler
    if load_path is not None:
        scheduler = checkpoint["scheduler"]
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    else:
        scheduler_args = model_args["scheduler"]
        scheduler_params = scheduler_args["params"] if scheduler_args["params"] is not None else {}
        scheduler = getattr(torch.optim.lr_scheduler, scheduler_args["name"])(optimizer, **scheduler_params)

    # create folder for recording results
    if load_path is not None:
        result_path = load_path
    else:
        result_path = create_result_folder(osp.join(tune_args["save_result_on"], model_name))

    best_loss = None
    best_loss_epoch = None
    keep_best_epochs = 0

    epoch = 0 if load_path is None else checkpoint["epoch"]
    epochs = tune_args["max_epochs"]
    pbar = tqdm(total=(epochs + 1), mininterval=10)
    pbar.update(epoch)

    train_losses = []
    val_losses = []
    test_losses = []

    if load_path is not None:
        with open(osp.join(load_path, "train_progress.csv"), mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                train_losses.append(float(row["train_losses"]))
                val_losses.append(float(row["val_losses"]))
                test_losses.append(float(row["test_losses"]))

    mae, r2 = 0.0, 0.0
    for epoch in range(epoch, epochs + 1):
        try:
            eval_results = None

            model, train_loss = train_step(model, train_loader, train_dataset, optimizer, device)
            val_eval_results = test_evaluations(model, val_loader, validation_dataset, device)
            eval_results = val_eval_results
            val_loss = val_eval_results[0]

            eval_loss = val_loss

            test_loss = 0.0
            if epoch == epochs - 1 or tune_args["keep_best_epochs"] <= keep_best_epochs:
                # get test loss
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
                + [train_losses, val_losses, test_losses, eval_results, model_name]
            )

            # save best model
            if best_loss is None or eval_loss < best_loss:
                best_loss = eval_loss
                best_loss_epoch = epoch
                # save condition
                if tune_args["save_loss_limit"] >= best_loss:
                    print("save result data")
                    mae, r2 = save_result_data(*save_params)

            # report results to tay tune
            keep_best_epochs = epoch - best_loss_epoch
            train.report(
                {"mean_absolute_error": best_loss, "keep_best_epochs": keep_best_epochs, "storage_path": result_path, "mae": mae, "r2": r2},
                checkpoint=None,
            )

            # show messages
            progress_msg = f"epoch:{str(epoch)} train:{str(round(train_loss,4))} valid:{str(round(val_loss, 4))} test:{'-' if test_loss==0.0 else str(round(test_loss, 4))} lr:{str(round(current_lr, 8))} eval_best:{str(round(best_loss, 4))}"
            pbar.set_description(progress_msg)
            pbar.update(1)

            # stop conditions
            # 1. reach last epoch
            if epoch == epochs - 1:
                print("training complete.")
            # 2. reach keep_best_epochs
            if tune_args["keep_best_epochs"] <= keep_best_epochs:
                print("training complete.")
                break
        except Exception as e:
            # delete all processed data
            occupied = create_dataset_occupy_table(
                osp.join(args["Default"]["absolute_work_dir"], "tuning/dataset_occupy_table.pt"), data_processed_path, -1
            )
            if occupied == 0:
                delete_processed_data(data_processed_path)
            traceback.print_exc()
            raise RuntimeError(f"error during training, error msg: {e}")
    pbar.close()

    # delete all processed data
    occupied = create_dataset_occupy_table(osp.join(args["Default"]["absolute_work_dir"], "tuning/dataset_occupy_table.pt"), data_processed_path, 1)
    if occupied == 0:
        delete_processed_data(data_processed_path)


def start_tuning(model_name, dataset_name, args):
    tune_args = args["Tuning"]

    tune_resources = tune_args["resources"]
    ray.init(num_cpus=tune_resources["num_cpus"], num_gpus=tune_resources["num_gpus"])

    search_alg = HyperOptSearch(metric="mean_absolute_error", mode="min")
    # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)

    scheduler = None
    # scheduler = ASHAScheduler(metric="mean_absolute_error", mode="min", max_t=10000)

    reporter = tune.CLIReporter(max_progress_rows=100, metric_columns=["mean_absolute_error"])

    trial_name = f"{model_name}_{dataset_name}_{get_current_time()}"
    storage_path = tune_args["storage_path"]
    log_to_file = osp.join(storage_path, trial_name, "output.log") if tune_args["log_to_file"] else False
    print(f"Log file about trainable object save on {log_to_file}.")

    # trainable = partial(trainable_model, model_name=model_name, dataset_name=dataset_name)

    # patch: add identifier to delete datasets at a proper time
    create_dataset_occupy_table(osp.join(args["Default"]["absolute_work_dir"], "tuning/dataset_occupy_table.pt"))

    # choose load dataset in this method or in the trainable method
    dataset = make_dataset(dataset_name, args, **(tune_args["dataset"])) if tune_args["load_dataset_by_trainable"] is False else None
    trainable = tune.with_parameters(trainable_model, args=args, dataset=dataset, model_name=model_name, dataset_name=dataset_name)

    # new tuner
    if tune_args["restore_experiment_from"] is None:
        tuner = tune.Tuner(
            tune.with_resources(trainable, resources={"cpu": tune_resources["trial_cpus"], "gpu": tune_resources["trial_gpus"]}),
            tune_config=tune.TuneConfig(
                num_samples=tune_args["trial_num_samples"],
                search_alg=search_alg,
                scheduler=scheduler,
                max_concurrent_trials=tune_args["max_concurrent_trials"],
                time_budget_s=tune_args["time_budget_s"],
            ),
            run_config=train.RunConfig(
                name=trial_name,
                storage_path=storage_path,
                progress_reporter=reporter,
                log_to_file=log_to_file,
                failure_config=train.FailureConfig(max_failures=3),
                # checkpoint_config=train.CheckpointConfig(checkpoint_at_end=True),
            ),
            param_space=args,
        )
    # restore tuner
    else:
        tuner = tune.Tuner.restore(
            tune_args["restore_experiment_from"],
            tune.with_resources(trainable, resources={"cpu": tune_resources["trial_cpus"], "gpu": tune_resources["trial_gpus"]}),
            resume_unfinished=True,
            resume_errored=False,
            restart_errored=False,
        )
        # tuner._local_tuner._tune_config.max_concurrent_trials = tune_args["max_concurrent_trials"]

    results = tuner.fit()
    best_result = results.get_best_result("mean_absolute_error", mode="min")
    print("best_result", best_result)
    print("hyperparameters: ", best_result.config)
