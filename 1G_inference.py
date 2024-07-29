import argparse
import torch
import math
import gc
from args import *
from train import *
from utils import *

parser = argparse.ArgumentParser(description="Evaluate the model")
parser.add_argument("-M", "--modelPath", required=True, type=str)
parser.add_argument("--batchsize", required=False, default=500, type=str)
parser.add_argument("--gen1GPred", help="Generate 1G predicts and save them to model path", required=False, action="store_true")
parser.add_argument("--analyse1G", help="Analyse the 1G predicts", required=False, action="store_true")
parser.add_argument("--sampleData", help="Generate sample data from generated data", required=False, action="store_true")
parser.add_argument(
    "--sampleData-fromModelPath", help="Get sample data indexs from a model instead of generate", required=False, default="", type=str
)
parser.add_argument("--sampleData-seed", required=False, default=114514, type=int)
parser.add_argument("--sampleData-ratio", required=False, default=0.01, type=float)

cmd_args = parser.parse_args()
batch_size = cmd_args.batchsize

model_path = osp.join(args["result_path"], cmd_args.modelPath)
print("Model in ", model_path)

# get hypothesis dataset
hypo_dataset = HypoDataset(args)
####################################################################################################


@torch.no_grad()
def predict_step(model, data, predict_epochs, device):
    model.eval()

    pbar = tqdm(total=predict_epochs)
    pbar.set_description("Predicting")

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


def gen_1G_predictions():
    device = get_device()

    # load model from file
    model, _ = load_model(model_path, file_name="checkpoint.pt", load_dict=True)

    batch_size = cmd_args.batchsize

    full_out = torch.zeros(0).to(device)
    idx_out = torch.zeros(0).to(device)
    for i, dataset in enumerate(hypo_dataset):
        data_length = len(dataset)
        predict_epochs = math.ceil(data_length / batch_size)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        out, idx = predict_step(model, dataloader, predict_epochs, device)
        full_out = torch.cat((full_out, out), 0)
        idx_out = torch.cat((idx_out, idx), 0)

        del out, idx, dataset, dataloader
        torch.cuda.empty_cache()
        gc.collect()

    # reverse data scale
    min, max = get_data_scale(args)
    print(min, max)
    get_out = reverse_min_max_scalar_1d(full_out, min, max)
    print(len(get_out))

    predicts_path = osp.join(model_path, "1G_predict.pt")
    torch.save((get_out, idx_out), predicts_path)
    print("data saved on", predicts_path)


if cmd_args.gen1GPred:
    gen_1G_predictions()


####################################################################################################


def analyse_1G():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    get_out, idx_out = torch.load(osp.join(model_path, "1G_predict.pt"))

    # show results
    # plt.hist(get_out.to("cpu"), range=(-0.5, 1.0), bins=50)
    plt.hist(get_out.to("cpu"), bins=50)
    file_path = osp.join(model_path, "1G_predict_distribution.png")
    plt.savefig(fname=file_path)

    get_out_np = get_out.to("cpu").numpy()
    print("total:", len(get_out_np))
    print("Ef<0:", len(get_out_np[get_out_np < 0.0]))
    print("Ef>0:", len(get_out_np[get_out_np > 0.0]))
    print("ratio of Ef<0 ", len(get_out_np[get_out_np < 0.0]) / len(get_out_np))
    print("1G predict distribution saved on ", file_path)
    return get_out_np


if cmd_args.analyse1G:
    analyse_1G()
####################################################################################################


def sample_data_from_model(model_path):
    sample_data = torch.load(osp.join(args["result_path"], model_path, "sample_data.pt"))
    random_idx = np.array(sample_data["idx_list"])
    print("random_idx:", random_idx)
    print("random_idx length:", len(random_idx))
    return random_idx


def sample_data_index():
    from_model_path = cmd_args.sampleData_fromModelPath

    get_out_np = analyse_1G()
    if from_model_path != "":
        return sample_data_from_model(from_model_path), get_out_np

    ratio = cmd_args.sampleData_ratio
    random_seed = cmd_args.sampleData_seed

    extract_out = get_out_np < 0.0
    extract_out_idx = np.where(extract_out)[0]

    np.random.seed(random_seed)
    extract_out_random_idx = np.random.choice(extract_out_idx, round(len(extract_out_idx) * ratio), replace=False)
    extract_out_random_idx.sort()
    random_idx = extract_out_random_idx

    print("random_idx:", random_idx)
    print("random_idx length:", len(random_idx))
    return random_idx, get_out_np


def sample_data():
    random_idx, get_out_np = sample_data_index()

    # get ase data block
    hypo_dataset = HypoDataset(args)

    last_idx = 0
    current_idx = 0

    idx_list = []
    data_list = []
    vasp_list = []
    pred_list = []

    for i in range(len(hypo_dataset)):
        data_block = hypo_dataset.get(i, False)
        # if len(data_block) != len(vasp_block):
        #     print("something wrong")

        last_idx = current_idx
        current_idx += len(data_block)
        print(last_idx, current_idx)

        # random_idx + 1 is ID.
        item = random_idx[np.where((random_idx >= last_idx) & (random_idx < current_idx))[0]]

        for d in item:
            data_list.append(data_block[d - last_idx])
        del data_block

        vasp_block = hypo_dataset.get(i, True)
        for d in item:
            vasp_list.append(vasp_block[d - last_idx])
        del vasp_block

        for d in item:
            idx_list.append(d)
            pred_list.append(get_out_np[d])

    print(len(idx_list))
    print(len(data_list))
    print(len(vasp_list))

    for i, d in zip(idx_list, vasp_list):
        if not i == d[0] - 1:
            print("something wrong")

    data = {"data_list": data_list, "vasp_list": vasp_list, "idx_list": idx_list, "pred_list": pred_list, "random_seed": cmd_args.sampleData_seed}
    torch.save(data, osp.join(model_path, "sample_data.pt"))


if cmd_args.sampleData:
    # sample_data_index()
    sample_data()
####################################################################################################
