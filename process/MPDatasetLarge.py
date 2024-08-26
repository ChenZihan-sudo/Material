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
from .MPDataset import *

module_filename = __name__.split(".")[-1]


# The Material Project Dataset
class MPDataset(Dataset):
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
