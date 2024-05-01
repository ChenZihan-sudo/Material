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
from torch_geometric.data import Data, Dataset
from mp_api.client import MPRester
from torch_geometric.utils import dense_to_sparse
from torch.utils.data import random_split, Subset
from typing import Sequence

from args import *
from utils import *


# Download compounds raw data from the Material Project
def download_raw_data(exclude_elements=["O"], num_elements=(3, 3)):

    mpr = MPRester(MP_API_KEY)

    raw_datasets = mpr.materials.summary.search(
        fields=["material_id", "formation_energy_per_atom", "structure"],
        exclude_elements=exclude_elements,
        num_elements=num_elements,
        chunk_size=args["chunk_size"],
        num_chunks=args["num_chunks"],
    )
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
def read_one_compound_info(compound_data) -> Data:
    data = Data()

    idx, mid, y = compound_data[0], compound_data[1], compound_data[2]
    data.mid = mid
    data.idx = idx

    filename = osp.join("{}".format(DATASET_RAW_DIR), "CONFIG_" + idx + ".poscar")
    # filename = osp.join("{}".format(DATASET_RAW_DIR), "CONFIG_" + idx + ".cif")
    compound = ase_read(filename, format="vasp")
    # compound = ase_read(filename, format="cif")

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

    # dense transform to sparse to get edge_index and edge_attr
    sparse_distance_matrix = dense_to_sparse(distance_matrix)
    data.edge_index = sparse_distance_matrix[0]
    data.edge_attr = torch.Tensor(np.array([sparse_distance_matrix[1]], dtype=np.float32)).t().contiguous()

    data.atomic_numbers = compound.get_atomic_numbers()
    # data.x = torch.Tensor(np.array([*atomic_numbers], dtype=np.float32)).t().contiguous()
    data.y = torch.Tensor(np.array([y], dtype=np.float32))

    return data


# Process raw data and store them as data.pt in {DATASET_PROCESSED_DIR}
def raw_data_process(raw_data=None) -> list:
    print("Raw data processing...")

    indices_filename = osp.join("{}".format(DATASET_RAW_DIR), "INDICES")
    assert osp.exists(
        indices_filename), "INDICES file not exist in " + indices_filename
    with open(indices_filename) as f:
        reader = csv.reader(f)
        indices = [row for row in reader][1:]  # ignore first line of header

    # process progress bar
    pbar = tqdm(total=len(indices))
    description = "Processing dataset"
    pbar.set_description(description)

    # Process single graph
    data_list = []
    atomic_number_set = set()
    for i, d in enumerate(indices):
        # d: one item in indices (e.g. i=0, d=['1', 'mp-861724', '-0.41328523750000556'])
        # TODO: 如果raw_data存在则直接使用，而不是从文件中读出
        data = read_one_compound_info(d)
        for a in data.atomic_numbers:
            atomic_number_set.add(a)
        data_list.append(data)
        pbar.update(1)
        print(data)
        if i == 10:
            break
    pbar.close()

    # Create one hot for data.x
    onehotDict = genOneHotDict(atomic_number_set)
    print(onehotDict)
    print(onehotDict[77])

    # Target normalization
    y_list = torch.tensor([data_list[i].y for i in range(len(data_list))])
    y_list, data_min, data_max = tensor_min_max_scalar_1d(y_list)
    for i, d in enumerate(data_list):
        d.y = y_list[i]

    # write parameters into {DATASET_RAW_DIR}/PARAMETERS
    atomic_numbers = list(atomic_number_set)
    atomic_numbers.sort()
    atomic_numbers = [int(a) for a in atomic_numbers]
    parameter = {
        "data_min": data_min,
        "data_max": data_max,
        "onehot_set": atomic_numbers
    }
    # Path where the indices csv file is created. (i.e. {DATASET_RAW_DIR}/PARAMETERS)
    filename = osp.join(DATASET_RAW_DIR, "PARAMETERS")
    with open(filename, "w") as f:
        json.dump(parameter, f)

    torch.save(data_list,
               osp.join("{}".format(DATASET_PROCESSED_DIR), "data.pt"))

    return data_list


# The Material Project Dataset
class MPDataset(Dataset):

    def __init__(self, root, args, transform=None, pre_transform=None, pre_filter=None):
        self.args = args
        super().__init__(root, transform, pre_transform, pre_filter)
        # self.load(self.processed_paths[0])

        path = osp.join(self.processed_dir, "data.pt")
        self.data = torch.load(path)

    # Skip process if file exist
    @property
    def processed_file_names(self) -> list[str]:
        return ["data.pt"]

    # Get all filenames from {DATASET_RAW_DIR}/INDICES, skip download if those files exist
    @property
    def raw_file_names(self) -> list[str]:
        filenames = ["INDICES", "PARAMETERS"]
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
        return self.args["raw_dir"]

    @property
    def processed_dir(self) -> str:
        return self.args["processed_dir"]

    def download(self):
        print("Downloading raw dataset...")
        self.raw_data = download_raw_data()
        raw_data_preprocess(self.raw_dir, self.raw_data)

    def process(self):
        data_list = raw_data_process(self.raw_data)
        self.data = data_list

    def len(self) -> int:
        return len(self.data)

    def get(self, idx) -> any:
        return self.data[idx]


def random_split_dataset(dataset, lengths: Sequence[int | float] = None, seed=None) -> list[Subset]:

    g = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, lengths, generator=g)

    return train_dataset, validation_dataset, test_dataset


def make_dataset():
    dataset = MPDataset(DATASET_DIR, args)
    lengths = [args["trainset_ratio"], args["testset_ratio"], args["valset_ratio"]]
    train_dataset, validation_dataset, test_dataset = random_split_dataset(dataset, lengths=lengths, seed=args["split_dataset_seed"])
    return train_dataset, validation_dataset, test_dataset


# make_dataset()
raw_data_process()
# download_raw_data()

# def random_split_dataset2(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, lengths=None, shuffle=True, seed=None) -> list[Dataset]:
#     """
#     Split the dataset into train, valid and test datasets by ratio or length. \n
#     Provide lengths (e.g. lengths=[6000,2000,2000]) will split by length provided. \n
#     Otherwise split by ratio.
#     """
#     dataset_len = dataset.len()

#     lengths_mode = True if (isinstance(lengths, list) and len(lengths) == 3) else False

#     if lengths_mode is False and train_ratio + val_ratio + test_ratio != 1.0:
#         raise ValueError("The total ratio of split dataset is not 1.0.")

#     train_len = int(lengths[0] if lengths_mode else dataset_len * train_ratio)
#     val_len = int(lengths[1] if lengths_mode else dataset_len * val_ratio)
#     test_len = int(lengths[2] if lengths_mode else dataset_len * test_ratio)

#     idx = list(range(dataset_len))

#     if shuffle is True:
#         random.seed(None)
#         random.shuffle(idx)

#     train_dataset = dataset.index_select(idx[:train_len])
#     validation_dataset = dataset.index_select(idx[train_len : train_len + val_len])
#     test_dataset = dataset.index_select(idx[train_len + val_len : train_len + val_len + test_len])

#     return train_dataset, validation_dataset, test_dataset
