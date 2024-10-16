import torch
import math
import gc
from tqdm import tqdm
import yaml
import os
from os import path as osp


@torch.no_grad()
def predict_step(model, data, data_length, device):
    model.eval()

    pbar = tqdm(total=data_length)
    pbar.set_description("predicting")

    with torch.no_grad():
        res_out = torch.zeros(0).to(device)
        idx_out = torch.zeros(0).to(device)
        for i, d in enumerate(data):
            d = d.to(device)
            out = model(d, False)
            res_out = torch.cat((res_out, out), 0)
            idx_out = torch.cat((idx_out, d.id))
            pbar.update(1)
        pbar.close()

    return res_out, idx_out


def model_prediction(config, model_path, batch_size, dataset_name, generation, **kwargs):
    import process
    from torch_geometric.loader import DataLoader
    from models.utils import load_model
    from utils import get_device, get_data_scale, reverse_min_max_scalar_1d

    dataset = getattr(process, dataset_name)(config)
    dataset_args = config["Dataset"][dataset_name]

    device = get_device(args=config)
    model, _ = load_model(model_path, file_name="checkpoint.pt", load_dict=True, map_location=device)

    full_out = torch.zeros(0).to(device)
    idx_out = torch.zeros(0).to(device)

    print("prepare dataset...")
    if dataset_name == "HypoDataset":
        data_block = None
        for i in range(len(dataset)):
            print(f"total progress: {i+1}/{len(dataset)}")

            data_block = dataset[i]
            data_length = len(data_block)
            predict_epochs = math.ceil(data_length / batch_size)
            dataloader = DataLoader(data_block, batch_size=batch_size, shuffle=False, num_workers=0)

            out, idx = predict_step(model, dataloader, predict_epochs, device)
            full_out = torch.cat((full_out, out), 0)
            idx_out = torch.cat((idx_out, idx), 0)

            torch.cuda.empty_cache()
            del out, idx, data_block, dataloader
            gc.collect()

    if dataset_name == "OptimizedHypoDataset" or dataset_name == "UnoptimizedHypoDataset":
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        data_length = math.ceil(len(dataset) / batch_size)
        out, idx = predict_step(model, dataloader, data_length, device)
        full_out = torch.cat((full_out, out), 0)
        idx_out = torch.cat((idx_out, idx), 0)

    print(dataset_args)

    # reverse data scale
    if config["Process"]["target_normalization"]:
        min, max = get_data_scale(dataset_args["get_parameters_from"])
        print(f"data scale: {min}, {max}")
        get_out = reverse_min_max_scalar_1d(full_out, min, max)
    else:
        get_out = full_out
    print(f"data nums: {len(get_out)}")

    predicts_path = osp.join(model_path, f"{generation}_{dataset_name}_predict.pt")
    torch.save((get_out, idx_out), predicts_path)
    print("data saved on", predicts_path)


def analyse_model_prediction(config, model_path, batch_size, dataset_name, generation, **kwargs):
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    model_pred_path = osp.join(model_path, f"{generation}_{dataset_name}_predict.pt")
    if not osp.exists(model_pred_path):
        print(f"File {model_pred_path} not exists. Try to generate the model prediction result...")
        model_prediction(config, model_path, batch_size, dataset_name, generation)

    get_out, idx_out = torch.load(osp.join(model_path, f"{generation}_{dataset_name}_predict.pt"))

    # show results
    counts, bins = np.histogram(get_out.to("cpu"), bins=50, range=None)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = 0.8 * (bins[1] - bins[0])
    # print(bin_centers, counts)
    plt.figure(figsize=(7, 5))
    plt.title(f"Formation Energy Distribution on {generation} Model with {dataset_name}")
    plt.xlabel("Formation energy (ev/atom)")
    plt.ylabel("Number of count")
    plt.bar(bin_centers, height=counts, width=width)

    get_out_np = get_out.to("cpu").numpy()
    info = (
        f"total: {len(get_out_np)}\nEf<0: {len(get_out_np[get_out_np < 0.0])}\nEf<-0.2: {len(get_out_np[get_out_np < -0.2])}\nEf>0: {len(get_out_np[get_out_np > 0])}\n"
        + f"ratio of Ef<0: {round(len(get_out_np[get_out_np < 0.0]) / len(get_out_np),6)}\n"
        + f"ratio of Ef<-0.2: {round(len(get_out_np[get_out_np < -0.2]) / len(get_out_np),6)}"
    )
    print(info)

    # print("total:", len(get_out_np))
    # print("Ef<0:", len(get_out_np[get_out_np < 0.0]))
    # print("Ef<-0.2:", len(get_out_np[get_out_np < -0.2]))
    # print("Ef>0:", len(get_out_np[get_out_np > 0.0]))
    # print("ratio of Ef<0 ", len(get_out_np[get_out_np < 0.0]) / len(get_out_np))
    # print("ratio of Ef<-0.2 ", len(get_out_np[get_out_np < -0.2]) / len(get_out_np))

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_offset = (xlim[1] - xlim[0]) * 0.05
    y_offset = (ylim[1] - ylim[0]) * 0.05

    plt.text(xlim[1] - x_offset, ylim[1] - y_offset, info, fontsize=12, color="black", horizontalalignment="right", verticalalignment="top")
    file_path = osp.join(model_path, f"{generation}_{dataset_name}_predict_distribution.png")
    plt.savefig(fname=file_path)

    print(f"{generation} predict distribution saved on ", file_path)
    return get_out_np
