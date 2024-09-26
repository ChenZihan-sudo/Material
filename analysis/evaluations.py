import argparse
import torch
import tqdm
import math
import numpy as np

from args import *
from dataset import *
from utils import *
from model import CEALNetwork, GCNNetwork, load_model

import os.path as osp
from train import make_data_loader, train_step, test_evaluations

from tool.madgap import *

parser = argparse.ArgumentParser(description="Evaluate the model")
parser.add_argument("-M", "--modelPath", required=False, type=str)
parser.add_argument("--evalBatchsize", required=False, default=500, type=str)
parser.add_argument("--MAD", help="Calculate the MAD", required=False, action="store_true")
parser.add_argument("--node", help="Calculate the reachable nodes and the shared reachable nodes", required=False, action="store_true")
parser.add_argument("--node-to-numlayer", required=False, default=1, type=int)
parser.add_argument("--predError", help="Find features of high prediction error compounds", required=False, action="store_true")
parser.add_argument("--predError-threshold", required=False, default=0.5, type=float)

cmd_args = parser.parse_args()

train_dataset, validation_dataset, test_dataset = make_dataset()
batch_size = cmd_args.evalBatchsize
train_loader, val_loader, test_loader = make_data_loader(train_dataset, validation_dataset, test_dataset, batch_size=batch_size)

# print(cmd_args)

if cmd_args.modelPath is not None:
    model_path = osp.join(args["result_path"], cmd_args.modelPath)
    print("Model in ", model_path)

####################################################################################################


def calculate_MAD(dataset, model_path, device=get_device()):
    mad_total = 0.0
    total_data_size = len(dataset)

    predict_epochs = total_data_size
    pbar = tqdm(total=predict_epochs)
    pbar.set_description("Progress")

    model, _ = load_model(model_path)
    model.eval()

    with torch.no_grad():
        for i, data in enumerate(dataset):
            data.to(device)
            node_embeddings = model(data, node_embedding=True)

            in_arr = node_embeddings.cpu().detach().numpy()

            num_nodes = data.x.shape[0]
            adj = torch.zeros((num_nodes, num_nodes))
            adj[data.edge_index[0], data.edge_index[1]] = 1
            mask_arr = adj.numpy()

            mad_single = mad_value(in_arr, mask_arr)
            mad_total += mad_single
            torch.cuda.empty_cache()

            pbar.update(1)
        pbar.close()

    return mad_total / total_data_size


if cmd_args.MAD is True:
    print("MAD:", calculate_MAD(test_dataset, model_path))

####################################################################################################


def reachable_nodes_mat(num_nodes, edge_index, num_layers=1, device=get_device()):
    adj = torch.tensor(np.eye(num_nodes), dtype=float).to(device)
    adj[edge_index[0], edge_index[1]] = 1
    if num_layers == 1:
        adj = adj.fill_diagonal_(0.0)
        return adj

    res_adj = adj.clone()
    for i in range(0, num_layers - 1):
        res_adj = res_adj @ adj

    res_adj = res_adj.fill_diagonal_(0.0)

    res_adj_idx = torch.where(res_adj >= 1.0)
    res_adj[res_adj_idx] = 1.0

    return res_adj


def get_avg_reachable_nodes(dataset, num_layers, device=get_device()):
    total_num_nodes = 0
    total_reachable_nodes = 0

    total_data_size = len(dataset)
    total_shared_nodes = 0

    predict_epochs = total_data_size
    pbar = tqdm(total=predict_epochs)
    pbar.set_description("Progress")

    for i, data in enumerate(dataset):
        num_nodes = data.num_nodes
        edge_index = data.edge_index.to(device)

        # reachable matrix
        reachable_mat = reachable_nodes_mat(num_nodes, edge_index, num_layers, device=device)
        reachable_nodes = torch.sum(reachable_mat)
        total_reachable_nodes += reachable_nodes

        # shared reachable matrix
        shared_mat = reachable_mat @ reachable_mat
        shared_mat = shared_mat.fill_diagonal_(0)
        shared_nodes = torch.sum(shared_mat)
        # solve avg. shared reachable nodes per node for this batch
        shared_nodes = shared_nodes / (num_nodes * (num_nodes + 1))
        total_shared_nodes += shared_nodes

        total_num_nodes += num_nodes

        torch.cuda.empty_cache()
        pbar.update(1)
    pbar.close()

    avg_reachable_nodes = (total_reachable_nodes / total_num_nodes).item()
    avg_shared_reachable_nodes = (total_shared_nodes / total_data_size).item()
    avg_nodes_on_graph = total_num_nodes / total_data_size
    print(f"Layer {num_layers}")
    print("Average reachable nodes:", round(avg_reachable_nodes, 4))
    print("Average shared reachable nodes:", round(avg_shared_reachable_nodes, 4))
    print("Average nodes on a graph:", round(avg_nodes_on_graph, 4))
    print(f"=================================")

    return avg_reachable_nodes, avg_shared_reachable_nodes, avg_nodes_on_graph


if cmd_args.node is True:
    print("Reachable nodes:")
    for i in range(1, cmd_args.node_to_numlayer + 1):
        get_avg_reachable_nodes(train_dataset, i)

####################################################################################################

from scipy.spatial.distance import cdist


def data_stat(data_list):
    avg_num_edges = 0
    avg_num_nodes = 0
    for i, d in enumerate(data_list):
        avg_num_edges += d.edge_index.shape[-1]
        avg_num_nodes += d.x.shape[0]
    total_graphs = i + 1
    return avg_num_nodes / total_graphs, avg_num_edges / total_graphs


def ase_data_stat(ase_data_list):
    avg_volume = 0.0
    avg_radius = 0.0

    for i, d in enumerate(ase_data_list):
        avg_volume += d.get_volume()

        positions = d.get_positions()
        dist_matrix = cdist(positions, positions)
        max_distances = dist_matrix.max(axis=1)
        radius = max_distances.max() / 2.0
        avg_radius += radius

    total = i + 1
    return avg_volume / total, avg_radius / total


def get_high_pred_error_stats(dataloader, model_path, threshold=0.5):

    high_pred_compounds = []

    model, _ = load_model(model_path)
    model.eval()

    with torch.no_grad():
        for i, batch_data in enumerate(dataloader):
            batch_data.to(get_device())
            out = model(batch_data, node_embedding=False)

            # reverse data scale
            min, max = get_data_scale(args)
            res_out = reverse_min_max_scalar_1d(out, min, max)
            res_y = reverse_min_max_scalar_1d(batch_data.y, min, max)

            # get high prediction error compounds
            error = (res_out.squeeze() - res_y).abs()
            index = torch.where(error > threshold)[0]
            compounds = [{"mid": data.mid, "idx": data.idx} for _, data in enumerate(batch_data[index])]
            high_pred_compounds.extend(compounds)

            torch.cuda.empty_cache()

    high_pred_compounds.sort(key=lambda x: int(x["idx"]))

    all_dataset = MPDataset(args)
    dataset_stat = data_stat(all_dataset)
    print(f"dataset.             avg_num_nodes:{round(dataset_stat[0],4)}, avg_num_edges:{round(dataset_stat[1],4)}")
    # read all ase file from the whole dataset
    whole_pred_ase = []
    for i, d in enumerate(all_dataset):
        path = osp.join(mp_args["raw_dir"], f"CONFIG_{int(d.idx)}.poscar")
        compound = ase_read(path, format="vasp")
        whole_pred_ase.append(compound)
    dataset_ase_stat = ase_data_stat(whole_pred_ase)
    print(f"dataset.               avg_volume:{round(dataset_ase_stat[0],4)}, avg_radius:{round(dataset_ase_stat[1],4)}")

    high_pred_data = [all_dataset[int(d["idx"]) - 1] for i, d in enumerate(high_pred_compounds)]
    high_stat = data_stat(high_pred_data)
    print(f"high_pred_compounds. avg_num_nodes:{round(high_stat[0],4)}, avg_num_edges:{round(high_stat[1],4)}")
    # read all ase file from high pred error compounds
    high_pred_ase = []
    for i, d in enumerate(high_pred_data):
        path = osp.join(mp_args["raw_dir"], f"CONFIG_{int(d.idx)}.poscar")
        compound = ase_read(path, format="vasp")
        high_pred_ase.append(compound)
    high_ase_stat = ase_data_stat(high_pred_ase)
    print(f"high_pred_compounds.  avg_volume:{round(high_ase_stat[0],4)}, avg_radius:{round(high_ase_stat[1],4)}")


if cmd_args.predError is True:
    print("Statistics of high prediction error compounds:")
    get_high_pred_error_stats(test_loader, model_path, threshold=cmd_args.predError_threshold)
    
