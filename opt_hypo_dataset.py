import argparse
import io
import random
import json
import csv
import os.path as osp
import ase
from ase.io import read as ase_read
import ase.io
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data, Dataset, InMemoryDataset
from mp_api.client import MPRester
from torch_geometric.utils import dense_to_sparse, add_self_loops
from torch.utils.data import random_split, Subset
from typing import Sequence
from itertools import permutations

from model import load_model
import copy

from args import *
from utils import *


parser = argparse.ArgumentParser(description="Optimized hypothesis dataset with default args in args.py")
parser.add_argument("-R", "--run", required=False, action="store_true", help="Run dataset processing progs.")
cmd_args, unknown = parser.parse_known_args()


# Process one compound data
def read_one_compound_info(id, y, file_path, max_cutoff_distance=args["max_cutoff_distance"]) -> Data:
    data = Data()

    data.id = id
    compound = ase_read(file_path, format="vasp")

    # get distance matrix
    distance_matrix = compound.get_all_distances(mic=True)

    # get mask by max cutoff distance
    cutoff_mask = distance_matrix > max_cutoff_distance
    # suppress invalid values using max cutoff distance
    distance_matrix = np.ma.array(distance_matrix, mask=cutoff_mask)
    # let '--' in the masked array to 0
    distance_matrix = np.nan_to_num(np.where(cutoff_mask, np.isnan(distance_matrix), distance_matrix))

    # make it as a tensor
    distance_matrix = torch.Tensor(distance_matrix)

    # dense transform to sparse to get edge_index and edge_weight
    sparse_distance_matrix = dense_to_sparse(distance_matrix)
    data.edge_index = sparse_distance_matrix[0]
    data.edge_weight = torch.Tensor(np.array(sparse_distance_matrix[1], dtype=np.float32)).t().contiguous()

    # add self loop
    data.edge_index, data.edge_weight = add_self_loops(data.edge_index, data.edge_weight, num_nodes=len(compound), fill_value=0)

    data.atomic_numbers = compound.get_atomic_numbers()
    data.y = torch.Tensor(np.array([y], dtype=np.float32))

    data.edge_attr = edge_weight_to_edge_attr(data.edge_weight)

    return data


# Process raw data and store them as data.pt in {DATASET_OPT_HYPO_PROCESSED_DIR}
def raw_data_process(opt_hypo_args, onehot_gen=True, onehot_range: list = None) -> list:
    """
    Args:
    - onehot_gen: set `True` will generate atomic number onehot from all compounds. Otherwise, using `onehot_range`.
    - onehot_range: if `onehot_gen` is not `True`, onehot of atomic number will use `range(onehot_range[0],onehot_range[-1])` instead.
    """
    print("Raw data processing...")

    total_num = opt_hypo_args["dataset_total_num"]
    opt_hypo_args["raw_dir"]
    # process progress bar
    pbar = tqdm(total=total_num)
    pbar.set_description("dataset processing")

    # Process single graph
    data_list = []
    atomic_number_set = set()

    total_valid = 0
    for id in range(1, total_num + 1):
        pbar.update(1)

        formation_energy_data_path = osp.join(opt_hypo_args["raw_dir"], opt_hypo_args["formation_energy_filename"] + str(id))
        compound_data_path = osp.join(opt_hypo_args["raw_dir"], opt_hypo_args["compound_filename"] + str(id))

        exist = False
        if not osp.exists(formation_energy_data_path):
            continue

        with open(formation_energy_data_path, "r") as file:
            res = file.readline()
            try:
                res = torch.Tensor([float(res)])
                exist = True
            except:
                print("Failed to load ", formation_energy_data_path)
        print(res)
        if not exist or res > 5.0 or not osp.exists(compound_data_path):
            continue

        total_valid += 1
        data = read_one_compound_info(id, res, compound_data_path)
        if onehot_gen is True:
            for a in data.atomic_numbers:
                atomic_number_set.add(a)
        data_list.append(data)
    pbar.close()

    print("Total valid compounds: ", total_valid)

    # Create one hot for data.x
    if onehot_gen is not None:
        atomic_number_set = set(list(range(onehot_range[0], onehot_range[-1])))
    onehot_dict = make_onehot_dict(atomic_number_set, data_path=opt_hypo_args["raw_dir"])
    for i, d in enumerate(data_list):
        d.x = torch.tensor(np.vstack([onehot_dict[i] for i in d.atomic_numbers]).astype(np.float32))
        delattr(d, "atomic_numbers")

    # Target normalization
    y_list = torch.tensor([data_list[i].y for i in range(len(data_list))])
    y_list, data_min, data_max = tensor_min_max_scalar_1d(y_list)
    for i, d in enumerate(data_list):
        d.y = torch.Tensor(np.array([y_list[i]], dtype=np.float32))

    # write parameters into {opt_hypo_args["raw_dir"]}/PARAMETERS
    atomic_numbers = list(atomic_number_set)
    atomic_numbers.sort()
    atomic_numbers = [int(a) for a in atomic_numbers]
    parameter = {"data_min": data_min, "data_max": data_max, "onehot_set": atomic_numbers}
    filename = osp.join(opt_hypo_args["raw_dir"], "PARAMETERS")
    with open(filename, "w") as f:
        json.dump(parameter, f)

    # print(data_list[0].x.tolist())
    return data_list


# The Optimized Hypothesis Dataset
class Optimized_hypo_dataset(InMemoryDataset):
    def __init__(self, args, transform=None, pre_transform=None, pre_filter=None):
        self.args = args
        self.opt_hypo_args = self.args["optimized_hypothesis_dataset"]
        super().__init__(args["dataset_dir"], transform, pre_transform, pre_filter)

        path = osp.join(self.processed_dir, "data.pt")
        self.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def raw_dir(self) -> str:
        return self.opt_hypo_args["raw_dir"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    @property
    def processed_dir(self) -> str:
        return self.opt_hypo_args["processed_dir"]

    def download(self):
        pass

    def process(self):
        data_list = raw_data_process(
            opt_hypo_args=self.opt_hypo_args, onehot_gen=self.opt_hypo_args["onehot_gen"], onehot_range=self.opt_hypo_args["onehot_range"]
        )
        self.data = data_list

        path = osp.join(self.processed_dir, "data.pt")
        self.save(data_list, path)


def random_split_dataset(dataset, lengths: Sequence[int | float] = None, seed=None) -> list[Subset]:
    g = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, lengths, generator=g)

    return train_dataset, validation_dataset, test_dataset


def make_dataset():
    dataset = Optimized_hypo_dataset(args)
    lengths = [opt_hypo_args["trainset_ratio"], opt_hypo_args["testset_ratio"], opt_hypo_args["valset_ratio"]]
    train_dataset, validation_dataset, test_dataset = random_split_dataset(dataset, lengths=lengths, seed=opt_hypo_args["split_dataset_seed"])
    return train_dataset, validation_dataset, test_dataset


if cmd_args.run:
    dataset = Optimized_hypo_dataset(args)
    print(dataset)
    print("First item: ", dataset[0])
    
