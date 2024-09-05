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
import copy

from utils import *

module_filename = __name__.split(".")[-1]


# Download compounds raw data from the Material Project
def download_raw_data(
    d_args,
    exclude_elements=["O"],
    num_elements=(3, 3),
    keep_data_from=None,
    # include_elements=["Ce", "Co", "Cu"],
):
    """
    Args:
    - keep_data_from: filter compounds data and sort id from the file `INDICE`
    """
    mpr = MPRester(d_args["api_key"])

    material_ids = None
    if keep_data_from is not None:
        with open(keep_data_from) as f:
            reader = csv.reader(f)
            indices = [row[1] for row in reader][1:]  # ignore first line of header
            material_ids = dict(zip(indices, range(1, len(indices) + 1)))

    raw_datasets = mpr.materials.summary.search(
        fields=["material_id", "formation_energy_per_atom", "structure"],
        exclude_elements=exclude_elements,
        # include_elements=include_elements,
        num_elements=num_elements,
        chunk_size=d_args["chunk_size"],
        num_chunks=d_args["num_chunks"],
    )

    if keep_data_from is not None:
        raw_datasets = [i for i in filter(lambda doc: doc.material_id in material_ids, raw_datasets)]
        print(len(raw_datasets))
        raw_datasets.sort(key=lambda doc: material_ids[doc.material_id])

    return raw_datasets


# Preprocess the raw data and store them in {DATASET_MP_RAW_DIR}
def raw_data_preprocess(dest_dir, raw_datasets):
    indices = []
    pbar = tqdm(total=len(raw_datasets))
    pbar.set_description("to conventional")
    for i, d in enumerate(raw_datasets):
        # Path where the poscar file is created. (e.g. {DATASET_MP_RAW_DIR}/CONFIG_1.poscar)
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

    # Path where the indices csv file is created. (i.e. {DATASET_MP_RAW_DIR}/INDICE)
    indices_filename = osp.join(dest_dir, "INDICES")
    with open(indices_filename, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=["idx", "mid", "formation_energy_per_atom"])
        cw.writeheader()
        cw.writerows(indices)


# Process one compound data
def read_one_compound_info(args, compound_data, max_cutoff_distance, max_cutoff_neighbors=None) -> Data:
    d_args = args["Dataset"][module_filename]

    data = Data()

    idx, mid, y = compound_data[0], compound_data[1], compound_data[2]
    data.mid = mid
    data.idx = idx

    filename = osp.join("{}".format(d_args["raw_dir"]), "CONFIG_" + idx + ".poscar")
    compound = ase_read(filename, format="vasp")

    # get distance matrix
    distance_matrix = compound.get_all_distances(mic=True)

    # # get mask by max cutoff distance
    # cutoff_mask = distance_matrix > max_cutoff_distance
    # # suppress invalid values using max cutoff distance
    # distance_matrix = np.ma.array(distance_matrix, mask=cutoff_mask)
    # # let '--' in the masked array to 0
    # distance_matrix = np.nan_to_num(np.where(cutoff_mask, np.isnan(distance_matrix), distance_matrix))

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

    return data


# Process raw data and store them as data.pt in {DATASET_PROCESSED_DIR}
def raw_data_process(args, onehot_gen=False, onehot_range: list = None) -> list:
    """
    Args:
    - onehot_gen: set `True` will generate atomic number onehot from all compounds. Otherwise, using `onehot_range`.
    - onehot_range: if `onehot_gen` is not `True`, onehot of atomic number will use `range(onehot_range[0],onehot_range[-1])` instead.
    """
    print("Raw data processing...")

    d_args = args["Dataset"][module_filename]

    indices_filename = osp.join("{}".format(d_args["raw_dir"]), "INDICES")
    assert osp.exists(indices_filename), "INDICES file not exist in " + indices_filename
    with open(indices_filename) as f:
        reader = csv.reader(f)
        indices = [row for row in reader][1:]  # ignore first line of header

    # process progress bar
    pbar = tqdm(total=len(indices))
    pbar.set_description("dataset processing")

    # Process single graph
    data_list = []
    atomic_number_set = set()

    p_args = args["Process"]
    max_cutoff_distance = p_args["max_cutoff_distance"]
    max_cutoff_neighbors = p_args["max_cutoff_neighbors"]

    for i, d in enumerate(indices):
        # d: one item in indices (e.g. i=0, d=['1', 'mp-861724', '-0.41328523750000556'])
        data = read_one_compound_info(args, d, max_cutoff_distance, max_cutoff_neighbors)

        if onehot_gen is True:
            for a in data.atomic_numbers:
                atomic_number_set.add(a)
        data_list.append(data)
        pbar.update(1)
    pbar.close()

    # Create one hot for data.x
    print("process one hot for nodes...")
    if onehot_gen is False:
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
    print("target normalization...")
    y_list = torch.tensor([data_list[i].y for i in range(len(data_list))])
    y_list, data_min, data_max = tensor_min_max_scalar_1d(y_list)
    for i, d in enumerate(data_list):
        d.y = torch.Tensor(np.array([y_list[i]], dtype=np.float32))

    # write parameters into {DATASET_MP_RAW_DIR}/PARAMETERS
    atomic_numbers = list(atomic_number_set)
    atomic_numbers.sort()
    atomic_numbers = [int(a) for a in atomic_numbers]
    parameter = {"data_min": data_min, "data_max": data_max, "onehot_set": atomic_numbers}
    # Path where the indices csv file is created. (i.e. {DATASET_MP_RAW_DIR}/PARAMETERS)
    filename = osp.join(d_args["processed_dir"], "PARAMETERS")
    with open(filename, "w") as f:
        json.dump(parameter, f)

    # print(data_list[0].x.tolist())
    return data_list


# The Material Project Dataset
class MPDataset(InMemoryDataset):
    def __init__(self, args, transform=None, pre_transform=None, pre_filter=None):
        self.args = args
        self.d_args = args["Dataset"][module_filename]
        super().__init__(args["Default"]["dataset_dir"], transform, pre_transform, pre_filter)

        path = osp.join(self.processed_dir, "data.pt")
        self.load(path)

    # Skip process if file exist
    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    # Get all filenames from {DATASET_MP_RAW_DIR}/INDICES, skip download if those files exist
    @property
    def raw_file_names(self) -> list[str]:
        filenames = ["INDICES"]
        # indices_filename = osp.join("{}".format(DATASET_MP_RAW_DIR), "INDICES")
        # if not osp.exists(indices_filename):
        #     return []

        # with open(indices_filename) as f:
        #     reader = csv.reader(f)
        #     indices = [row for row in reader][1:]
        # filenames = ["CONFIG_" + d[0] + ".vasp" for _, d in enumerate(indices)]
        return filenames

    @property
    def raw_dir(self) -> str:
        return self.d_args["raw_dir"]

    @property
    def processed_dir(self) -> str:
        return self.d_args["processed_dir"]

    def download(self):
        print("Downloading raw dataset...")
        raw_data = download_raw_data(self.d_args, keep_data_from=self.d_args["keep_data_from"])
        raw_data_preprocess(self.raw_dir, raw_data)

    def process(self):
        data_list = raw_data_process(self.args, onehot_gen=self.d_args["onehot_gen"], onehot_range=self.d_args["onehot_range"])
        self.data = data_list

        path = osp.join(self.processed_dir, "data.pt")
        self.save(data_list, path)
