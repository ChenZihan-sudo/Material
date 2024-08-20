import torch
import matplotlib.pyplot as plt

import os.path as osp

from training.prepare import *

from models import *
from utils import *
from process import make_dataset

import ray
from ray import tune, train
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune.schedulers import ASHAScheduler

import models

from datetime import datetime

from functools import partial


def trainable_model(args, model_name=None, dataset_name=None):

    print(f"############ Start Hyperparameter Tuning on {model_name} with {dataset_name} ############")
    print("Hyperparameters: ", args)

    tune_args = args["Tuning"]
    model_args = args["Models"][model_name]

    # make dataset and data loader
    train_dataset, validation_dataset, test_dataset = make_dataset(dataset_name, args, **(tune_args["dataset"]))
    dataloader_args = tune_args["data_loader"]
    train_loader, val_loader, test_loader = make_data_loader(train_dataset, validation_dataset, test_dataset, **dataloader_args)

    # get device
    device = get_device(args=args)

    # get model in dimension
    in_dim = train_dataset[0].x.shape[-1]

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

    epochs = model_args["epochs"]
    for epoch in range(1, epochs + 1):

        model, train_loss = train_step(model, train_loader, train_dataset, optimizer, device)
        val_loss, _, _ = test_evaluations(model, val_loader, validation_dataset, device)
        # test_eval_results = test_evaluations(model, test_loader, test_dataset, device)
        # test_loss = test_eval_results[0]

        scheduler.step(val_loss)
        # current_lr = optimizer.param_groups[0]["lr"]
        # progress_msg = f"epoch:{str(epoch)} train:{str(round(train_loss,4))} valid:{str(round(val_loss, 4))} test:{str(round(test_loss, 4))} lr:{str(round(current_lr, 8))}"
        train.report({"mean_absolute_error": val_loss})


def start_tuning(model_name, dataset_name, args):
    ray.init(num_gpus=1, num_cpus=1)

    tune_args = args["Tuning"]
    print(tune_args)

    # hyperparameters = []

    search_alg = HyperOptSearch(metric="mean_absolute_error", mode="min")
    # search_alg = ConcurrencyLimiter(search_alg, max_concurrent=2)

    scheduler = ASHAScheduler(metric="mean_absolute_error", mode="min")

    # parameter_columns = [hyperparameter for hyperparameter in hyperparameters.keys()]
    # reporter = tune.CLIReporter(max_progress_rows=30, metric_columns=["mean_absolute_error"], parameter_columns=parameter_columns)
    reporter = tune.CLIReporter(max_progress_rows=30, metric_columns=["mean_absolute_error"])

    trial_name = f"{model_name}_{dataset_name}_{get_current_time()}"
    storage_path = tune_args["storage_path"]
    log_to_file = osp.join(storage_path, trial_name, "output.log") if tune_args["log_to_file"] else False
    print(f"Log file about trainable object save on {log_to_file}.")

    trainable = partial(trainable_model, model_name=model_name, dataset_name=dataset_name)
    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"gpu": 1}),
        tune_config=tune.TuneConfig(
            num_samples=tune_args["trial_num_samples"],
            search_alg=search_alg,
            scheduler=scheduler,
            max_concurrent_trials=tune_args["max_concurrent_trials"],
        ),
        run_config=train.RunConfig(name=trial_name, storage_path=storage_path, progress_reporter=reporter, log_to_file=log_to_file),
        param_space=args,
    )

    results = tuner.fit()
    best_result = results.get_best_result("mean_absolute_error", mode="min")
    print("best_result", best_result)
    print("hyperparameters: ", best_result.config)
