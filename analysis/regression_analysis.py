# generate regression graph based on the dataset target result (x axis)
# and the model prediction result (y axis)
from .model_prediction import model_prediction
from .dataset_target_distribution import dataset_target_distribution

def regression_analysis(config, model_path, batch_size, dataset_name, generation, postfix_epoch, **kwargs):
    import os
    import torch
    import process
    from os import path as osp
    from utils import get_device, save_regression_result, plot_regression_result, reverse_min_max_scalar_1d, get_data_scale

    device = get_device(device_name="cpu")

    # model prediction result
    model_pred_path = osp.join(model_path, f"{generation}_{dataset_name}_predict{"" if postfix_epoch == "" else f'_{postfix_epoch}'}.pt")
    if not osp.exists(model_pred_path):
        print(f"File {model_pred_path} not exists. Try to generate the model prediction result...")
        model_prediction(config, model_path, batch_size, dataset_name, generation, postfix_epoch)

    pred_data, pred_idx = torch.load(model_pred_path, map_location=device)
    # pred_data.to(device)

    # dataset target result
    dataset, _ = process.process_dataset(dataset_name, config)
    dataset_args = config["Dataset"][dataset_name]
    processed_path = dataset_args["processed_dir"]
    dataset_target_path = osp.join(processed_path, f"{dataset_name}_target.pt")
    if not osp.exists(dataset_target_path):
        print(f"File {dataset_target_path} not exists. Try to generate the dataset target result result...")
        dataset_target_distribution(config, model_path, batch_size, dataset_name, generation, postfix_epoch)
    
    target_data = torch.load(dataset_target_path, map_location=device)
    # target_data.to(device)

    # get regression graph
    if len(target_data) != len(pred_data):
        print(f"target data size ({len(target_data)}) unmatched the model predict data size ({len(pred_data)})")

    print(f"{len(target_data)} {len(pred_data)}")

    regression_result_name = f"{generation}_{dataset_name}_regression{"" if postfix_epoch == "" else f'_{postfix_epoch}'}.txt"
    regression_name = f"{generation}_{dataset_name}_regression{"" if postfix_epoch == "" else f'_{postfix_epoch}'}.png"
    regression_title = f"{generation}_{dataset_name}_regression"
    save_regression_result(pred_data, target_data, model_path, filename=regression_result_name)
    plot_regression_result(regression_title, model_path, filename=regression_result_name, plotfilename=regression_name, scope=[-2, 4, -2, 4])
