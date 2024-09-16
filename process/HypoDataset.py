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
from torch_geometric.utils import dense_to_sparse, add_self_loops
from itertools import permutations
import copy

from utils import *

module_filename = __name__.split(".")[-1]


def get_replace_atomic_numbers(compound, target_atomic_numbers):
    """
    get atomic numbers replaced list from origin one to `target_atomic_numbers` with full permutations.

    (e.g, origin: [1,1,1,1,2,2], target_atomic_numbers:[3,4], result:[[3,3,3,3,4,4],[4,4,4,4,3,3]])

    (e.g, origin: [1,2,2,2,2,3,3], target_atomic_numbers:[4,5,6], result:[[4,5,5,5,5,6,6],[4,6,6,6,6,5,5],[5,4,4,4,4,6,6],[5,6,6,6,6,4,4],[6,5,5,5,5,4,4],[6,4,4,4,4,5,5]])
    """
    target_atomic_numbers = list(set(target_atomic_numbers))

    # full permutations of target_atomic_numbers
    targets = [list(i) for i in permutations(target_atomic_numbers, len(target_atomic_numbers))]

    np_atomic_numbers = compound.get_atomic_numbers()
    set_atomic_numbers = [i for i in set(np_atomic_numbers)]

    replace_atomic_numbers = []
    for j, target in enumerate(targets):
        array = np.array(np_atomic_numbers)
        final_array = np.zeros_like(array)
        for i, d in enumerate(set_atomic_numbers):
            final_array[array == d] = target[i]
        replace_atomic_numbers.append(final_array)
        # # TODO: remove this
        # replace_atomic_numbers.append(array)
    return replace_atomic_numbers


def scale_compound_volume(compound, scaling_factor):
    """
    scale the compound by volume
    """
    scaling_factor = scaling_factor ** (1 / 3)
    compound.set_cell(compound.cell * scaling_factor, scale_atoms=True)


def get_ase_hypothesis_compounds(scales, hypo_atomic_numbers, original_compound):
    """
    get ase format hypothesis compounds from one original compound

    total hypothesis compounds num is `len(scales) * len(hypo_atomic_numbers)`

    return `hypothesis_compounds` with ase compounds
    """
    # get replaced atomic numbers
    hypo_atomic_numbers = get_replace_atomic_numbers(original_compound, hypo_atomic_numbers)

    # get scaled compounds
    scaled_compounds = []
    for i, d in enumerate(scales):
        sacled_compound = copy.deepcopy(original_compound)
        scale_compound_volume(sacled_compound, d)
        scaled_compounds.append(sacled_compound)
    # print("num hypothesis scaled compounds:", len(scaled_compounds))

    # get hypothesis compounds
    hypothesis_compounds = []
    for i, sacled_compound in enumerate(scaled_compounds):
        for j, hypo_atomic_number in enumerate(hypo_atomic_numbers):
            compound = copy.deepcopy(sacled_compound)
            compound.set_atomic_numbers(hypo_atomic_number)
            hypothesis_compounds.append(compound)

    # print("total hypothesis for single origin compound: ", len(hypothesis_compounds))
    # [print("volume:", round(i.get_volume(), 2), i) for i in hypothesis_compounds]

    return hypothesis_compounds


def get_one_hypothesis_compound(compound, onehot_dict, max_cutoff_distance):
    """
    get Data() from one hypothesis compound
    """
    data = Data()

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

    data.edge_attr = edge_weight_to_edge_attr(data.edge_weight)
    delattr(data, "edge_weight")

    # get onehot x
    atomic_numbers = compound.get_atomic_numbers()
    data.x = torch.tensor(np.vstack([onehot_dict[str(i)] for i in atomic_numbers]).astype(np.float32))
    # print("================")
    # print(compound)
    # print(data.x.tolist())

    return data


def make_hypothesis_compounds_dataset(args, split_num=10):
    """
    Make hypothesis compounds

    Args:
        `args`: parameters
        `split_num`: specify how many splited blocks
    """
    d_args = args["Dataset"][module_filename]
    mp_args = args["Dataset"]["MPDataset"]

    indices_filename = osp.join("{}".format(mp_args["raw_dir"]), "INDICES")
    assert osp.exists(indices_filename), "INDICES file not exist in " + indices_filename
    with open(indices_filename) as f:
        reader = csv.reader(f)
        indices = [row for row in reader][1:]  # ignore first line of header

    data_dir = d_args["processed_dir"]

    scales = d_args["scales"]
    hypo_atomic_numbers = d_args["atomic_numbers"]

    # fully permutations of atomic numbers
    num_atomic_numbers_perms = len([i for i in permutations(hypo_atomic_numbers, len(hypo_atomic_numbers))])
    # hypothesis compounds for single original compound
    single_processed = len(scales) * num_atomic_numbers_perms
    # num all hypothesis compounds
    total_processed = len(indices) * single_processed

    # process progress bar
    pbar = tqdm(total=total_processed)
    pbar.set_description("hypothesis compounds dataset processing")

    # read one hot dict from {DATASET_MP_RAW_DIR}/onehot_dict.json
    onehot_dict = read_onehot_dict(mp_args["raw_dir"], "onehot_dict.json")

    # split data if need
    step = int(len(indices) / split_num)
    save_point = [i + step - 1 for i in list(range(0, len(indices)))[::step]]
    if save_point[-1] > len(indices):
        save_point = save_point[:-1]
        save_point[-1] = len(indices) - 1
    save_track = 0
    # save save_point
    torch.save(save_point, osp.join(data_dir, "save_point.pt"))
    print("save points: ", save_point)

    # Process single graph data
    hypo_data_track = 1
    hypo_indices = []
    data_list = []
    poscar_data_list = []
    for i, d in enumerate(indices):
        # i = 355
        # d = indices[i]

        # d: one item in indices (e.g. i=0, d=['1', 'mp-861724', '-0.41328523750000556'])
        # read original compound data
        idx, mid, y = d[0], d[1], d[2]
        filename = osp.join("{}".format(mp_args["raw_dir"]), "CONFIG_" + idx + ".poscar")
        original_compound = ase_read(filename, format="vasp")

        # get hypothesis ase compounds
        hypo_compounds = get_ase_hypothesis_compounds(scales, hypo_atomic_numbers, original_compound)

        # get data list of hypothesis compounds
        hypo_data_list = [
            get_one_hypothesis_compound(hypo_compound, onehot_dict, args["Process"]["max_cutoff_distance"]) for hypo_compound in hypo_compounds
        ]

        # append all of hypo data into data_list
        for j, hypo_data in enumerate(hypo_data_list):
            hypo_data.id = j + hypo_data_track
            data_list.append(hypo_data)

            # transform ase to poscar format string
            output = io.StringIO()
            hypo_compounds[j].write(output, format="vasp")
            poscar = (hypo_data.id, output)
            poscar_data_list.append(poscar)

        pbar.update(single_processed)
        hypo_data_track += single_processed

        # # save single data to data_dir
        # indices_range = [k for k in range(hypo_data_track - single_processed, hypo_data_track)]
        # for hypo_idx, data in zip(indices_range, hypo_data_list):
        #     file_path = osp.join(data_dir, "data_" + str(hypo_idx) + ".pt")
        #     torch.save(data, file_path)

        # save data if need
        hypo_indices.append(
            {
                "hypo_range": [hypo_data_track - single_processed, hypo_data_track - 1],
                "origin_idx": idx,
                "origin_mid": mid,
            }
        )
        if save_point[save_track] == i:
            save_track += 1
            file_path = osp.join(data_dir, f"{d_args['processed_filename']}_{str(save_track)}.pt")
            print("Data block ", save_track, " saved on ", file_path)
            torch.save(data_list, file_path)
            ase_file_path = osp.join(data_dir, f"{d_args['processed_ase_filename']}_{str(save_track)}.pt")
            torch.save(poscar_data_list, ase_file_path)
            print("Saved data length:", len(data_list))
            data_list = []
            poscar_data_list = []

        # if i == 1:
        # break

    # Path where the indices csv file is created.
    indices_filename = osp.join(data_dir, "indices.json")
    with open(indices_filename, "w") as f:
        json.dump(hypo_indices, f)

    pbar.close()


class HypoDataset(Dataset):
    def __init__(self, args, transform=None, pre_transform=None, pre_filter=None):
        self.args = args
        self.mp_args = args["Dataset"]["MPDataset"]
        self.d_args = args["Dataset"][module_filename]
        super().__init__(args["Default"]["dataset_dir"], transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["INDICES"]

    @property
    def raw_dir(self) -> str:
        return self.mp_args["raw_dir"]

    @property
    def processed_file_names(self):
        file_names = [self.d_args["processed_filename"] + "_" + str(i) + ".pt" for i in range(1, self.d_args["split_num"] + 1)]
        return file_names

    @property
    def processed_dir(self) -> str:
        return self.d_args["processed_dir"]

    def download(self):
        pass

    def process(self):
        make_hypothesis_compounds_dataset(self.args, self.d_args["split_num"])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx, get_ase_data=False):
        data = torch.load(osp.join(self.processed_dir, f"{self.d_args['processed_filename']}_{idx+1}.pt"))
        if get_ase_data is True:
            data = torch.load(osp.join(self.processed_dir, f"{self.d_args['processed_ase_filename']}_{idx+1}.pt"))
        return data
