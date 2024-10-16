import argparse
import io
import random
import json
import csv
import os.path as osp
import ase
from ase.io import read as ase_read
from ase.io import write as ase_write
import ase.io
import numpy as np
import torch
from tqdm import tqdm
from torch_geometric.data import Data, Dataset, InMemoryDataset
from torch_geometric.utils import dense_to_sparse, add_self_loops
from torch.utils.data import random_split, Subset
from typing import Sequence

from utils import *

module_filename = __name__.split(".")[-1]


# Process one compound data
def read_one_compound_info(id, y, file_path, max_cutoff_distance, max_cutoff_neighbors=None) -> Data:
    data = Data()

    data.id = id
    compound = ase_read(file_path, format="vasp")

    # # output ase atom compounds to sample optimized data
    # ase_write(f"sample_optimized_data/CONFIG_{id}.poscar", compound, format="vasp")

    # get distance matrix
    distance_matrix = compound.get_all_distances(mic=True)

    # cutoff distance matrix based on max distance and max neigbors
    distance_matrix = distance_matrix_cutoff(distance_matrix, max_cutoff_distance, max_cutoff_neighbors)

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
def raw_data_process(args, onehot_range: list = None) -> list:
    print("Raw data processing...")
    d_args = args["Dataset"][module_filename]

    total_num = d_args["total_num"]
    raw_dir = d_args["raw_dir"]

    # process progress bar
    pbar = tqdm(total=total_num)
    pbar.set_description("dataset processing")

    # Process single graph
    data_list = []
    atomic_number_set = set()

    p_args = args["Process"]
    max_cutoff_distance = p_args["max_cutoff_distance"]
    max_cutoff_neighbors = p_args["max_cutoff_neighbors"]

    total_valid = 0
    for id in range(1, total_num + 1):
        pbar.update(1)

        formation_energy_data_path = osp.join(d_args["raw_dir"], d_args["formation_energy_filename"] + str(id))
        compound_data_path = osp.join(d_args["raw_dir"], d_args["compound_filename"] + str(id))

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
        # print(res)
        if not exist or res > 5.0 or not osp.exists(compound_data_path):
            continue

        total_valid += 1
        data = read_one_compound_info(id, res, compound_data_path, args["Process"]["max_cutoff_distance"])
        data_list.append(data)
    pbar.close()

    print("total valid compounds: ", total_valid)

    # Create one hot for data.x
    print("process one hot for nodes...")
    atomic_number_set = set(list(range(onehot_range[0], onehot_range[-1])))
    onehot_dict = make_onehot_dict(atomic_number_set, data_path=d_args["raw_dir"])
    for i, d in enumerate(data_list):
        d.x = torch.tensor(np.vstack([onehot_dict[i] for i in d.atomic_numbers]).astype(np.float32))
        delattr(d, "atomic_numbers")

    # edge process
    print("process edge features...")
    edge_args = args["Process"]["edge"]
    # edge normalization
    if edge_args["normalization"] is True:
        for i, data in enumerate(data_list):
            data.edge_weight, _, _ = normalization_1d(data.edge_weight, 0.0, max_cutoff_distance, 0.0, 1.0)
    # edge gaussian smearing
    if edge_args["gaussian_smearing"]["enable"] is True:
        edge_args["gaussian_smearing"]["resolution"] = edge_args["edge_feature"]
        for i, data in enumerate(data_list):
            data.edge_attr = gaussian_smearing(0.0, 1.0, data.edge_weight, **edge_args["gaussian_smearing"])
    else:
        for i, data in enumerate(data_list):
            data.edge_attr = edge_weight_to_edge_attr(data.edge_weight)

    # target normalization
    # * inherit paramters from the MPDataset
    # * inherit formation energy scale from the inherit_parameters_from
    y_list = torch.tensor([data_list[i].y for i in range(len(data_list))])
    data_min, data_max = 0.0, 1.0
    if p_args["target_normalization"]:
        data_min, data_max = get_data_scale(d_args["inherit_parameters_from"])
        print(f"get data scale from the inherit paramters, min:{data_min} max:{data_max}")
        y_list, data_min, data_max = tensor_min_max_scalar_1d(y_list, data_min=data_min, data_max=data_max)
    for i, d in enumerate(data_list):
        d.y = torch.Tensor(np.array([y_list[i]], dtype=np.float32))

    # write parameters into {d_args["raw_dir"]}/PARAMETERS
    if d_args["inherit_parameters_from"] is not None:
        atomic_numbers = list(atomic_number_set)
        atomic_numbers.sort()
        atomic_numbers = [int(a) for a in atomic_numbers]
        parameter = {"data_min": data_min, "data_max": data_max, "onehot_set": atomic_numbers}
        filename = osp.join(d_args["raw_dir"], "PARAMETERS")
        with open(filename, "w") as f:
            json.dump(parameter, f)

    return data_list


# The Optimized Hypothesis Dataset
class OptimizedHypoDataset(InMemoryDataset):
    def __init__(self, args, transform=None, pre_transform=None, pre_filter=None):
        self.args = args
        self.d_args = self.args["Dataset"][module_filename]
        super().__init__(args["Default"]["dataset_dir"], transform, pre_transform, pre_filter)

        path = osp.join(self.processed_dir, "data.pt")
        self.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def raw_dir(self) -> str:
        return self.d_args["raw_dir"]

    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    @property
    def processed_dir(self) -> str:
        return self.d_args["processed_dir"]

    def download(self):
        pass

    def process(self):
        data_list = raw_data_process(self.args, onehot_range=self.d_args["onehot_range"])
        self.data = data_list

        path = osp.join(self.processed_dir, "data.pt")
        self.save(data_list, path)
