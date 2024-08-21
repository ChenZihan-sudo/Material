import torch
import matplotlib.pyplot as plt

import time

import os.path as osp
import numpy as np

from tqdm import tqdm

from .prepare import *

from models import *
from utils import *
from process import make_dataset

import models


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


# def record_model_object(model_object: dict, model, epoch, optimizer, scheduler):
#     from copy import deepcopy

#     for data in model_object.values():
#         del data

#     model_object["epoch"] = deepcopy(epoch)
#     model_object["model"] = deepcopy(model)
#     model_object["model_state_dict"] = deepcopy(model.state_dict())
#     model_object["optimizer"] = deepcopy(optimizer)
#     model_object["optimizer_state_dict"] = deepcopy(optimizer.state_dict())
#     model_object["scheduler"] = deepcopy(scheduler)
#     model_object["scheduler"] = deepcopy(scheduler.state_dict())
#     return model_object


def start_training(model_name, dataset_name, args):

    print(f"############ Start Training on {model_name} with {dataset_name} ############")
    train_args = args["Training"]
    dataset_args = args["Dataset"][dataset_name]
    model_args = args["Models"][model_name]

    # make dataset and data loader
    train_dataset, validation_dataset, test_dataset = make_dataset(dataset_name, args, **(train_args["dataset"]))
    dataloader_args = train_args["data_loader"]
    train_loader, val_loader, test_loader = make_data_loader(train_dataset, validation_dataset, test_dataset, **dataloader_args)

    print(f"dataset num, train:{len(train_dataset)}, val:{len(validation_dataset)}, test:{len(test_dataset)}")

    # get device
    device = get_device(args=args)

    # get model in dimension
    in_dim = train_dataset[0].x.shape[-1]
    print(f"model in_dim: {in_dim}")

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
    result_path = create_result_folder(osp.join(train_args["save_result_on"], model_name))

    test_best_loss = None
    # epoch = None
    # model_object = {
    #     "epoch": None,
    #     "loss": None,
    #     "model": None,
    #     "model_state_dict": None,
    #     "optimizer": None,
    #     "optimizer_state_dict": None,
    #     "scheduler": None,
    #     "scheduler_state_dict": None,
    # }

    # get best model based on best test loss
    save_best_model = train_args["save_best_model"]
    # save results every save_step epochs
    save_step = train_args["save_step"]

    model_summary(model)
    with open(osp.join(result_path, "model_info.txt"), "w") as file:
        model_summary(model, file=file)
        # print(model, file=file)
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
        # torch.cuda.empty_cache()

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

        progress_msg = f"epoch:{str(epoch)} train:{str(round(train_loss,4))} valid:{str(round(val_loss, 4))} test:{str(round(test_loss, 4))} lr:{str(round(current_lr, 8))} best_test:{str(round(test_best_loss, 4))}"
        pbar.set_description(progress_msg)
        pbar.update(1)

    pbar.close()
