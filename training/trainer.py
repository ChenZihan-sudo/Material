import json
import csv

import torch
import matplotlib.pyplot as plt

import time

import os.path as osp
import numpy as np

from tqdm import tqdm

from .prepare import *

from models import *
from utils import *
from process import make_dataset, process_dataset

import models
import process


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
    hypo_losses=None,
    eval_results=None,
    regression_title="Model Regression",
):
    print("-------------------------------------------")
    print(f"Epoch {epoch}: Saving data and model...")
    save_hyper_parameter(args, result_path)
    save_train_progress(epoch - 1, train_losses, val_losses, test_losses, result_path)
    _, out, y = eval_results

    # Reverse normalization of out and y
    if args["Process"]["target_normalization"]:
        data_path = process.get_parameter_file_path(dataset_args["get_parameters_from"], args)
        min, max = get_data_scale(data_path)
        y = reverse_min_max_scalar_1d(y, min, max)
        out = reverse_min_max_scalar_1d(out, min, max)

    # save results
    if args["Training"]["ignore_test_evaluation"] is True:
        test_losses = None
    # *- extra options: show hypo result
    if args["Training"]["show_hypo_result"] is False:
        hypo_losses = None
    plot_training_progress(len(train_losses), train_losses, val_losses, test_losses, hypo_losses, res_path=result_path, threshold=0.2)

    postfix_epoch = f"_{epoch}" if args["Training"]["save_all_epoch_result"] is True and args["Training"]["save_best_model_only"] is False else ""
    # save regression result
    save_regression_result(out, y, result_path, filename=f"regression_result{postfix_epoch}.txt")
    mae, r2 = plot_regression_result(
        regression_title, result_path, filename=f"regression_result{postfix_epoch}.txt", plotfilename=f"regression_figure{postfix_epoch}.jpeg"
    )
    print(f"MAE={mae}")
    print(f"R^2={r2}")
    # save model
    save_model(result_path, model, epoch, mae, optimizer, scheduler, model_filename=f"checkpoint{postfix_epoch}.pt")
    # *- extra options: show hypo result
    if args["Training"]["show_hypo_result"] is False:
        print("-------------------------------------------\n")

    return mae, r2


def start_training(model_name, dataset_name, args):
    print(f"############ Start Training on {model_name} with {dataset_name} ############")
    train_args = args["Training"]
    dataset_args = args["Dataset"][dataset_name]
    model_args = args["Models"][model_name]

    # set up for loading the model
    load_path = train_args["load_model_from"]
    if load_path is not None:
        # replace the parameters
        with open(osp.join(load_path, "hyperparameters.json"), "r") as file:
            load_args = json.load(file)

        # Default, Training, Models optimier and scheduler
        # will inherit from current config file
        load_args["Default"] = args["Default"]
        load_args["Training"] = args["Training"]
        # load optimizer and scheduler params from current configs
        load_args["Models"][model_name]["optimizer"] = args["Models"][model_name]["optimizer"]
        load_args["Models"][model_name]["scheduler"] = args["Models"][model_name]["scheduler"]

        args = load_args

        dataset_args = args["Dataset"][dataset_name]
        model_args = args["Models"][model_name]
        train_args = args["Training"]

        print(f"############ Resume Training on {model_name} with {dataset_name} ############")

    # make dataset and data loader
    train_dataset, validation_dataset, test_dataset, data_processed_path = make_dataset(dataset_name, args, **(train_args["dataset"]))
    dataloader_args = train_args["data_loader"]
    train_loader, val_loader, test_loader = make_data_loader(train_dataset, validation_dataset, test_dataset, **dataloader_args)

    # *- extra options: show hypo result
    if train_args["show_hypo_result"] == True:
        from torch_geometric.loader import DataLoader

        hypo_dataset, _ = process_dataset("OptimizedHypoDataset", args)
        hypo_loader = DataLoader(hypo_dataset, shuffle=False, batch_size=dataloader_args["batch_size"], num_workers=dataloader_args["num_workers"])
        print(f"sample_dataset: {hypo_dataset}")

    # sync the processed data path if auto processed the dataset name
    # if args["Process"]["auto_processed_name"] is True:
    # args["Dataset"][dataset_name]["processed_dir"] = data_processed_path
    # dataset_args["processed_dir"] = data_processed_path
    # dataset_args["get_parameters_from"] = dataset_args["processed_dir"]

    print(f"dataset num, train:{len(train_dataset)}, val:{len(validation_dataset)}, test:{len(test_dataset)}")

    # get device
    device = get_device(args=args)

    if load_path:
        checkpoint = torch.load(osp.join(load_path, "checkpoint.pt"))
        model = checkpoint["model"]
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
    else:
        # get model in dimension
        in_dim = train_dataset[0].x.shape[-1]
        print(f"model in_dim: {in_dim}")

        if model_name in ("PNA", "ChemGNN"):
            model_args["conv_params"]["edge_dim"] = args["Process"]["edge"]["edge_feature"]  # import edge_dim from Process.edge.edge_feature
            deg = generate_deg(train_dataset).float()
            deg = deg.to(device)
            model = getattr(models, model_name)(deg, in_dim, **model_args)
            model = model.to(device)
        elif model_name in "CGCNN":
            model_args["conv_params"]["dim"] = args["Process"]["edge"]["edge_feature"]
            model = getattr(models, model_name)(in_dim, **model_args)
            model = model.to(device)
        else:
            model = getattr(models, model_name)(in_dim, **model_args)
            model = model.to(device)

    # set optimizer
    optimizer_args = model_args["optimizer"]
    optimizer_params = optimizer_args["params"] if optimizer_args["params"] is not None else {}
    optimizer = getattr(torch.optim, optimizer_args["name"])(model.parameters(), lr=model_args["learning_rate"], **optimizer_params)
    if load_path and train_args["resume_training"] is True:
        # optimizer = checkpoint["optimizer"]
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # set scheduler
    scheduler_args = model_args["scheduler"]
    scheduler_params = scheduler_args["params"] if scheduler_args["params"] is not None else {}
    scheduler = getattr(torch.optim.lr_scheduler, scheduler_args["name"])(optimizer, **scheduler_params)
    if load_path and train_args["resume_training"] is True:
        # scheduler = checkpoint["scheduler"]
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # create folder for recording results
    if load_path and train_args["resume_training"] is True:
        result_path = load_path
    else:
        result_path = create_result_folder(osp.join(train_args["save_result_on"], model_name))

    model_summary(model)
    with open(osp.join(result_path, "model_info.txt"), "w") as file:
        model_summary(model, file=file)
        file.close()

    best_loss = None

    epoch = 0 if load_path is None else checkpoint["epoch"]
    epochs = train_args["epochs"]
    pbar = tqdm(total=(epochs + 1))
    pbar.update(epoch)

    train_losses = []
    val_losses = []
    test_losses = []
    hypo_losses = []

    if load_path:
        with open(osp.join(load_path, "train_progress.csv"), mode="r") as file:
            reader = csv.DictReader(file)
            for row in reader:
                train_losses.append(float(row["train_losses"]))
                val_losses.append(float(row["val_losses"]))
                test_losses.append(float(row["test_losses"]))

    for epoch in range(epoch, epochs + 1):
        eval_results = None

        model, train_loss = train_step(model, train_loader, train_dataset, optimizer, device)
        val_loss, val_out, val_y = test_evaluations(model, val_loader, validation_dataset, device)

        if train_args["ignore_test_evaluation"] is True:
            test_loss = 0.0
            eval_results = val_loss, val_out, val_y
            eval_loss = val_loss
        else:
            test_loss, test_out, test_y = test_evaluations(model, test_loader, test_dataset, device)
            eval_results = test_loss, test_out, test_y
            eval_loss = test_loss

        if epoch == epochs - 1 and train_args["ignore_test_evaluation"] is True:
            print("Last one.")
            test_loss, test_out, test_y = test_evaluations(model, test_loader, test_dataset, device)
            print(f"Final test loss is {test_loss}")
            # eval_loss = test_loss

        # *- extra options: show hypo result
        if train_args["show_hypo_result"] == True:
            hypo_loss, hypo_out, hypo_y = test_evaluations(model, hypo_loader, hypo_dataset, device)
            hypo_losses.append(hypo_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

        save_params = (
            [args, dataset_args, epoch, model]
            + [optimizer, scheduler, result_path]
            + [train_losses, val_losses, test_losses, hypo_losses, eval_results, model_name]
        )

        # get best loss
        get_best_loss = False
        if best_loss is None or eval_loss < best_loss:
            get_best_loss = True
            best_loss = eval_loss
            # print(f"Get best loss:{best_loss}")

        # save result data
        if train_args["save_best_model_only"] is True and get_best_loss:
            save_result_data(*save_params)

        # save result data
        if train_args["save_best_model_only"] is False:
            save_result_data(*save_params)
            
        # *- extra options: show hypo result
        if train_args["show_hypo_result"] == True:
            print(f"    [sample_opt_hypo]---------------------")
            # Reverse normalization of out and y
            if args["Process"]["target_normalization"]:
                data_path = process.get_parameter_file_path(args["Dataset"]["OptimizedHypoDataset"]["get_parameters_from"], args)
                hypo_min, hypo_max = get_data_scale(data_path)
                hypo_y = reverse_min_max_scalar_1d(hypo_y, hypo_min, hypo_max)
                hypo_out = reverse_min_max_scalar_1d(hypo_out, hypo_min, hypo_max)

            # save result
            postfix_epoch = (
                f"_{epoch}" if args["Training"]["save_all_epoch_result"] is True and args["Training"]["save_best_model_only"] is False else ""
            )
            save_regression_result(hypo_out, hypo_y, result_path, f"sample_opt_hypo_regression_result{postfix_epoch}.txt")
            hypo_mae, hypo_r2 = plot_regression_result(
                "Hypothetic Compounds Regression Result",
                result_path,
                filename=f"sample_opt_hypo_regression_result{postfix_epoch}.txt",
                plotfilename=f"sample_opt_hypo_regression_figure{postfix_epoch}.jpeg",
                scope=[-2, 4, -2, 4],
            )
            print(f"    MAE={hypo_mae}")
            print(f"    R^2={hypo_r2}")
            print(f"    [sample_opt_hypo]---------------------")
            print("-------------------------------------------\n")

        # show progress message
        progress_msg = f"epoch:{str(epoch)} train:{str(round(train_loss,4))} valid:{str(round(val_loss, 4))} test:{'-' if test_loss==0.0 else str(round(test_loss, 4))} lr:{str(round(current_lr, 8))} eval_best:{str(round(best_loss, 4))}"
        pbar.set_description(progress_msg)
        pbar.update(1)

    pbar.close()
