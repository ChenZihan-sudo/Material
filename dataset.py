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


##################################################################
# For Material Project Dataset
##################################################################


# Download compounds raw data from the Material Project
def download_raw_data(exclude_elements=["O"], num_elements=(3, 3), keep_data_from=None):
    """
    Args:
    - keep_data_from: filter compounds data and sort id from the file `INDICE`
    """
    mpr = MPRester(MP_API_KEY)

    material_ids = None
    if keep_data_from is not None:
        with open(keep_data_from) as f:
            reader = csv.reader(f)
            indices = [row[1] for row in reader][1:]  # ignore first line of header
            material_ids = dict(zip(indices, range(1, len(indices) + 1)))

    raw_datasets = mpr.materials.summary.search(
        fields=["material_id", "formation_energy_per_atom", "structure"],
        exclude_elements=exclude_elements,
        num_elements=num_elements,
        chunk_size=args["chunk_size"],
        num_chunks=args["num_chunks"],
    )

    if keep_data_from is not None:
        raw_datasets = [i for i in filter(lambda doc: doc.material_id in material_ids, raw_datasets)]
        print(len(raw_datasets))
        raw_datasets.sort(key=lambda doc: material_ids[doc.material_id])

    return raw_datasets


# Preprocess the raw data and store them in {DATASET_RAW_DIR}
def raw_data_preprocess(dest_dir, raw_datasets):
    indices = []
    pbar = tqdm(total=len(raw_datasets))
    pbar.set_description("to conventional")
    for i, d in enumerate(raw_datasets):
        # Path where the poscar file is created. (e.g. {DATASET_RAW_DIR}/CONFIG_1.poscar)
        filename = osp.join(dest_dir, "CONFIG_" + str(i + 1) + ".poscar")
        # filename = osp.join(dest_dir, "CONFIG_" + str(i + 1) + ".cif")
        d.structure = d.structure.to_conventional()
        d.structure.to_file(filename=filename, fmt="poscar")
        # d.structure.to_file(filename=filename, fmt="cif")

        indices.append(
            {
                "idx": i + 1,
                "mid": str(d.material_id),
                "formation_energy_per_atom": d.formation_energy_per_atom,
            }
        )
        pbar.update(1)
    pbar.close()

    # Path where the indices csv file is created. (i.e. {DATASET_RAW_DIR}/INDICE)
    indices_filename = osp.join(dest_dir, "INDICES")
    with open(indices_filename, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=["idx", "mid", "formation_energy_per_atom"])
        cw.writeheader()
        cw.writerows(indices)


# Process one compound data
def read_one_compound_info(
    compound_data, max_cutoff_distance=args["max_cutoff_distance"], optimize=False, onehot_dict=None, model=None, device=get_device()
) -> Data:
    data = Data()

    idx, mid, y = compound_data[0], compound_data[1], compound_data[2]
    data.mid = mid
    data.idx = idx

    filename = osp.join("{}".format(DATASET_RAW_DIR), "CONFIG_" + idx + ".poscar")
    compound = ase_read(filename, format="vasp")

    threshold = 1000
    # optimze big volume compounds if need
    if optimize is True and compound.get_volume() > threshold:
        max_cutoff_distance += 2.0

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

    # if optimize is True:
    #     with torch.no_grad():
    #         min, max = get_data_scale(args)
    #         # print("before", data.y)
    #         # data.y = tensor_min_max_scalar_1d(data.y, min, max)
    #         # print("after", data.y)
    #         data.x = torch.tensor(np.vstack([onehot_dict[str(i)] for i in data.atomic_numbers]).astype(np.float32))
    #         data = data.to(device)
    #         out = model(data, node_embedding=False)

    #         # reverse data scale
    #         res_out = reverse_min_max_scalar_1d(out, min, max)
    #         # res_y = reverse_min_max_scalar_1d(data.y, min, max)
    #         res_y = data.y
    #         # print(res_out, res_y)

    #         # get high prediction error compounds
    #         threshold = 0.5
    #         error = (res_out.squeeze() - res_y).abs()
    #         max_cutoff_distance += 3.0
    #         # print("optimized", error, error > threshold)
    #         if error > threshold:
    #             print("changed", error, max_cutoff_distance)
    #             data = read_one_compound_info(compound_data, max_cutoff_distance)
    #         else:
    #             data = data.to(torch.device("cpu"))
    return data


# Process raw data and store them as data.pt in {DATASET_PROCESSED_DIR}
def raw_data_process(onehot_gen=True, onehot_range: list = None, optimize=args["data_optimize"], model_path=args["data_opt_model_path"]) -> list:
    """
    Args:
    - onehot_gen: set `True` will generate atomic number onehot from all compoundd. Otherwise, use `onehot_range`.
    - onehot_range: if `onehot_gen` is not `True`, onehot of atomic number will use `range(onehot_range[0],onehot_range[-1])` instead.
    """
    print("Raw data processing...")

    indices_filename = osp.join("{}".format(DATASET_RAW_DIR), "INDICES")
    assert osp.exists(indices_filename), "INDICES file not exist in " + indices_filename
    with open(indices_filename) as f:
        reader = csv.reader(f)
        indices = [row for row in reader][1:]  # ignore first line of header

    # process progress bar
    pbar = tqdm(total=len(indices))
    pbar.set_description("dataset processing")

    # optimze from compounds if need
    # if optimize is True:
    #     model, _ = load_model(model_path)
    #     model.eval()
    #     onehot_dict_optimize = read_onehot_dict(DATASET_RAW_DIR, "onehot_dict.json")

    # Process single graph
    data_list = []
    atomic_number_set = set()

    for i, d in enumerate(indices):
        # d: one item in indices (e.g. i=0, d=['1', 'mp-861724', '-0.41328523750000556'])
        data = read_one_compound_info(d, optimize=optimize)
        # data = read_one_compound_info(d, optimize=optimize, model=model, onehot_dict=onehot_dict_optimize)

        if onehot_gen is True:
            for a in data.atomic_numbers:
                atomic_number_set.add(a)
        data_list.append(data)
        pbar.update(1)
        # print(data)
        # if i == 0:
        #     break
    pbar.close()

    # Create one hot for data.x
    if onehot_gen is not None:
        atomic_number_set = set(list(range(onehot_range[0], onehot_range[-1])))
    onehot_dict = make_onehot_dict(atomic_number_set)
    for i, d in enumerate(data_list):
        d.x = torch.tensor(np.vstack([onehot_dict[i] for i in d.atomic_numbers]).astype(np.float32))
        delattr(d, "atomic_numbers")

    # Target normalization
    y_list = torch.tensor([data_list[i].y for i in range(len(data_list))])
    y_list, data_min, data_max = tensor_min_max_scalar_1d(y_list)
    for i, d in enumerate(data_list):
        d.y = torch.Tensor(np.array([y_list[i]], dtype=np.float32))

    # write parameters into {DATASET_RAW_DIR}/PARAMETERS
    atomic_numbers = list(atomic_number_set)
    atomic_numbers.sort()
    atomic_numbers = [int(a) for a in atomic_numbers]
    parameter = {"data_min": data_min, "data_max": data_max, "onehot_set": atomic_numbers}
    # Path where the indices csv file is created. (i.e. {DATASET_RAW_DIR}/PARAMETERS)
    filename = osp.join(DATASET_RAW_DIR, "PARAMETERS")
    with open(filename, "w") as f:
        json.dump(parameter, f)

    # print(data_list[0].x.tolist())
    return data_list


# The Material Project Dataset
class MPDataset(InMemoryDataset):
    def __init__(self, args, transform=None, pre_transform=None, pre_filter=None):
        self.args = args
        super().__init__(args["dataset_dir"], transform, pre_transform, pre_filter)

        path = osp.join(self.processed_dir, "data.pt")
        self.load(path)

    # Skip process if file exist
    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    # Get all filenames from {DATASET_RAW_DIR}/INDICES, skip download if those files exist
    @property
    def raw_file_names(self) -> list[str]:
        filenames = ["INDICES"]
        # indices_filename = osp.join("{}".format(DATASET_RAW_DIR), "INDICES")
        # if not osp.exists(indices_filename):
        #     return []

        # with open(indices_filename) as f:
        #     reader = csv.reader(f)
        #     indices = [row for row in reader][1:]
        # filenames = ["CONFIG_" + d[0] + ".vasp" for _, d in enumerate(indices)]
        return filenames

    @property
    def raw_dir(self) -> str:
        return self.args["dataset_raw_dir"]

    @property
    def processed_dir(self) -> str:
        return self.args["dataset_processed_dir"]

    def download(self):
        print("Downloading raw dataset...")
        raw_data = download_raw_data(keep_data_from=self.args["keep_data_from"])
        raw_data_preprocess(self.raw_dir, raw_data)

    def process(self):
        data_list = raw_data_process(onehot_gen=args["onehot_gen"], onehot_range=args["onehot_range"])
        self.data = data_list

        path = osp.join(self.processed_dir, "data.pt")
        self.save(data_list, path)


def random_split_dataset(dataset, lengths: Sequence[int | float] = None, seed=None) -> list[Subset]:
    g = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, lengths, generator=g)

    return train_dataset, validation_dataset, test_dataset


def make_dataset():
    dataset = MPDataset(args)
    lengths = [args["trainset_ratio"], args["testset_ratio"], args["valset_ratio"]]
    train_dataset, validation_dataset, test_dataset = random_split_dataset(dataset, lengths=lengths, seed=args["split_dataset_seed"])
    return train_dataset, validation_dataset, test_dataset


##################################################################
##################################################################

##################################################################
# For hypothesis dataset
##################################################################


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


def get_one_hypothesis_compound(compound, onehot_dict):
    """
    get Data() from one hypothesis compound
    """
    data = Data()

    # get distance matrix
    distance_matrix = compound.get_all_distances(mic=True)

    # get mask by max cutoff distance
    cutoff_mask = distance_matrix > args["max_cutoff_distance"]
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
    hypo_args = args["hypothesis_dataset"]

    indices_filename = osp.join("{}".format(DATASET_RAW_DIR), "INDICES")
    assert osp.exists(indices_filename), "INDICES file not exist in " + indices_filename
    with open(indices_filename) as f:
        reader = csv.reader(f)
        indices = [row for row in reader][1:]  # ignore first line of header

    data_dir = hypo_args["data_dir"]

    scales = hypo_args["scales"]
    hypo_atomic_numbers = hypo_args["atomic_numbers"]

    # fully permutations of atomic numbers
    num_atomic_numbers_perms = len([i for i in permutations(hypo_atomic_numbers, len(hypo_atomic_numbers))])
    # hypothesis compounds for single original compound
    single_processed = len(scales) * num_atomic_numbers_perms
    # num all hypothesis compounds
    total_processed = len(indices) * single_processed

    # process progress bar
    pbar = tqdm(total=total_processed)
    pbar.set_description("hypothesis compounds dataset processing")

    # read one hot dict from {DATASET_RAW_DIR}/onehot_dict.json
    onehot_dict = read_onehot_dict(DATASET_RAW_DIR, "onehot_dict.json")

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
        filename = osp.join("{}".format(DATASET_RAW_DIR), "CONFIG_" + idx + ".poscar")
        original_compound = ase_read(filename, format="vasp")

        # get hypothesis ase compounds
        hypo_compounds = get_ase_hypothesis_compounds(scales, hypo_atomic_numbers, original_compound)

        # get data list of hypothesis compounds
        hypo_data_list = [get_one_hypothesis_compound(hypo_compound, onehot_dict) for hypo_compound in hypo_compounds]

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
            file_path = osp.join(data_dir, hypo_args["data_filename"] + "_" + str(save_track) + ".pt")
            print("Data block ", save_track, " saved on ", file_path)
            torch.save(data_list, file_path)
            ase_file_path = osp.join(data_dir, f"ase_{str(save_track)}.pt")
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
        self.hypo_args = self.args["hypothesis_dataset"]
        super().__init__(args["dataset_dir"], transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return ["INDICES"]

    @property
    def raw_dir(self) -> str:
        return self.args["dataset_raw_dir"]

    @property
    def processed_file_names(self):
        file_names = [self.hypo_args["data_filename"] + "_" + str(i) + ".pt" for i in range(1, hypo_args["split_num"] + 1)]
        return file_names

    @property
    def processed_dir(self) -> str:
        return self.hypo_args["data_dir"]

    def download(self):
        pass

    def process(self):
        make_hypothesis_compounds_dataset(self.args, self.hypo_args["split_num"])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx, get_ase=False):
        data = torch.load(osp.join(self.processed_dir, f"{self.hypo_args['data_filename']}_{idx+1}.pt"))
        if get_ase is True:
            data = torch.load(osp.join(self.processed_dir, f"ase_{idx+1}.pt"))
        return data


##################################################################
##################################################################

if __name__ == "__main__":
    make_dataset()
