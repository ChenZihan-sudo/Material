import argparse
import torch
import math
import gc
import tqdm
import yaml
import os

# from models.utils import load_model
# from utils import *
# import process

parser = argparse.ArgumentParser(description="sample and analysis by inference a model")

parser.add_argument("--config_path", default="config.yml", type=str, help="config file path (default: config.yml)")
parser.add_argument("--model_path", required=True, type=str, help="model path for inference (<model path>/checkpoint.pt)")
parser.add_argument("--dataset", required=True, type=str, help="dataset type (HypoDataset or OptimizedHypoDataset)")
parser.add_argument("--batch_size", required=False, default=500, type=str)
parser.add_argument("--generation", default="1G", type=str, help="generation tag of the model (default:1G)")

# task configuration
parser.add_argument(
    "--gen-pred", help="Generate model prediction results from a dataset and save them to model path", required=False, action="store_true"
)
parser.add_argument("--analyse_pred", help="analyse the model results from a dataset", required=False, action="store_true")
parser.add_argument("--sample_data", help="Sample the hypothesis compound data from the hypothesis dataset", required=False, action="store_true")
parser.add_argument(
    "--sample_data_from_model_path", help="Get sample data indexs from a model instead of generate", required=False, default="", type=str
)
parser.add_argument("--sample_data_seed", required=False, default=114514, type=int)
parser.add_argument("--sample_data_ratio", required=False, default=0.01, type=float)

parser.add_argument("--gen-regression-result", required=False, action="store_true")

cmd_args = parser.parse_args()
print(cmd_args.__dict__)


def convert_str_to_number(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, str):
                if v.isdecimal():
                    d[k] = int(v)
                elif v.isdigit():
                    d[k] = float(v)
            else:
                convert_str_to_number(v)
    return d


def yaml_from_template(config_path, recursive_render_time=5):
    from jinja2 import Template

    assert os.path.exists(config_path), "Config file not found in " + config_path
    with open(config_path, "r") as file:
        content = file.read()
    with open(config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    for i in range(recursive_render_time):
        content = Template(content)
        content = content.render(config)
        config = yaml.load(content, Loader=yaml.FullLoader)
    return config


config = yaml_from_template(cmd_args.config_path, recursive_render_time=10)
config = convert_str_to_number(config)

dataset = getattr(process, cmd_args.dataset)(config)
dataset_args = config["Dataset"][cmd_args.dataset]
batch_size = cmd_args.batch_size
model_path = cmd_args.model_path
print(f"model path: {model_path}")
####################################################################################################


@torch.no_grad()
def predict_step(model, data, data_length, device):
    model.eval()

    pbar = tqdm(total=data_length)
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


def gen_model_predictions():
    from torch_geometric.loader import DataLoader

    device = get_device()

    # load model from file
    model, _ = load_model(model_path, file_name="checkpoint.pt", load_dict=True)

    batch_size = cmd_args.batch_size

    full_out = torch.zeros(0).to(device)
    idx_out = torch.zeros(0).to(device)

    if cmd_args.dataset == "HypoDataset":
        for i, data_block in enumerate(dataset):
            data_length = len(data_block)
            predict_epochs = math.ceil(data_length / batch_size)
            dataloader = DataLoader(data_block, batch_size=batch_size, shuffle=False, num_workers=0)

            out, idx = predict_step(model, dataloader, predict_epochs, device)
            full_out = torch.cat((full_out, out), 0)
            idx_out = torch.cat((idx_out, idx), 0)

            del out, idx, data_block, dataloader
            torch.cuda.empty_cache()
            gc.collect()

    if cmd_args.dataset == "OptimizedHypoDataset":
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
        data_length = math.ceil(len(dataset) / batch_size)
        out, idx = predict_step(model, dataloader, data_length, device)
        full_out = torch.cat((full_out, out), 0)
        idx_out = torch.cat((idx_out, idx), 0)

    # reverse data scale
    min, max = get_data_scale(dataset_args["get_parameters_from"])
    print(f"data scale: {min}, {max}")
    get_out = reverse_min_max_scalar_1d(full_out, min, max)
    print(f"data nums: {len(get_out)}")

    predicts_path = osp.join(model_path, f"{cmd_args.generation}_{cmd_args.dataset}_predict.pt")
    torch.save((get_out, idx_out), predicts_path)
    print("data saved on", predicts_path)


if cmd_args.gen_pred:
    gen_model_predictions()


####################################################################################################


def analyse_model_predictions():
    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    get_out, idx_out = torch.load(osp.join(model_path, f"{cmd_args.generation}_{cmd_args.dataset}_predict.pt"))

    # show results
    # plt.hist(get_out.to("cpu"), range=(-0.5, 1.0), bins=50)
    plt.hist(get_out.to("cpu"), bins=50)
    file_path = osp.join(model_path, f"{cmd_args.generation}_{cmd_args.dataset}_predict_distribution.png")
    plt.savefig(fname=file_path)

    get_out_np = get_out.to("cpu").numpy()
    print("total:", len(get_out_np))
    print("Ef<0:", len(get_out_np[get_out_np < 0.0]))
    print("Ef>0:", len(get_out_np[get_out_np > 0.0]))
    print("ratio of Ef<0 ", len(get_out_np[get_out_np < 0.0]) / len(get_out_np))
    print(f"{cmd_args.generation} predict distribution saved on ", file_path)
    return get_out_np


if cmd_args.analyse_pred:
    analyse_model_predictions()
####################################################################################################


def sample_data_from_model(model_path):
    sample_data = torch.load(osp.join(model_path, "sample_data.pt"))
    random_idx = np.array(sample_data["idx_list"])
    print("random idx:", random_idx)
    print("random idx length:", len(random_idx))
    return random_idx


def sample_data_index():
    from_model_path = cmd_args.sampleData_fromModelPath

    get_out_np = analyse_model_predictions()
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

    print("random idx:", random_idx)
    print("random idx length:", len(random_idx))
    return random_idx, get_out_np


def sample_data():
    random_idx, get_out_np = sample_data_index()

    last_idx = 0
    current_idx = 0

    idx_list = []
    data_list = []
    vasp_list = []
    pred_list = []

    for i in range(len(dataset)):
        data_block = dataset.get(i, False)
        # if len(data_block) != len(vasp_block):
        #     print("something wrong")

        last_idx = current_idx
        current_idx += len(data_block)
        print(last_idx, current_idx)

        # random_idx + 1 is the ID
        item = random_idx[np.where((random_idx >= last_idx) & (random_idx < current_idx))[0]]

        for d in item:
            data_list.append(data_block[d - last_idx])
        del data_block

        vasp_block = dataset.get(i, True)
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
    torch.save(data, osp.join(model_path, f"{cmd_args.generation}_{cmd_args.dataset}_sample_data.pt"))


if cmd_args.sampleData:
    if cmd_args.dataset != "HypoDataset":
        print("Hypothesis dataset is needed to sample data")
    else:
        sample_data()

####################################################################################################


# if cmd_args.gen_regression_result:
#     if cmd_args.dataset != "OptimizedHypoDataset":
#         print("Optimized hypothesis dataset is needed.")

#     from prceopt_hypo_dataset import *


# def gen_regression_result():
#     device = get_device("cpu")

#     predict_data_path = osp.join(model_path, "1G_opt_hypo_predict.pt")
#     pred_data_list, pred_data_id_list = torch.load(predict_data_path)
#     pred_data_list = pred_data_list.to(device)
#     dft_data_list_ = torch.zeros(0).to(device)

#     dataset = Optimized_hypo_dataset(args)

#     for pred_data_id, data in zip(pred_data_id_list, dataset):
#         if int(pred_data_id.item()) != int(data.id.item()):
#             print("ID unmatched: ", pred_data_id.item())
#         y = data.y.to(device)
#         dft_data_list_ = torch.cat((dft_data_list_, y), 0)

#     min, max = get_data_scale(args, data_path=DATASET_OPT_HYPO_RAW_DIR)
#     print(min, max)
#     dft_data_list = reverse_min_max_scalar_1d(dft_data_list_, min, max)

#     total = len(dft_data_list)

#     # print("MAE: ")
#     print("negative formation energy ratio:", len(dft_data_list[dft_data_list < 0.0]) / total)
#     save_regression_result(pred_data_list, dft_data_list, "model_path", filename="1G_opt_hypo_regression_result.txt")
#     plot_regression_result(
#         "1G Optimized Hypothesis Regression",
#         "1G_opt_hypo_regression_result.txt",
#         plotfilename="1G_opt_hypo_regression_figure.jpeg",
#         scope=[-5, 2, -5, 2],
#     )
