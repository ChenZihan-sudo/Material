from .MPDataset import MPDataset
from .HypoDataset import HypoDataset
from .OptimizedHypoDataset import OptimizedHypoDataset
from .MPDatasetLarge import MPDatasetLarge
from .MPDatasetCeCoCuBased import MPDatasetCeCoCuBased

__all__ = [
    "MPDataset",
    "MPDatasetLarge",
    "MPDatasetCeCoCuBased",
    "HypoDataset",
    "OptimizedHypoDataset",
    "make_dataset",
    "create_dataset_occupy_table",
]


def random_split_dataset(dataset, lengths=None, seed=None):
    import torch
    from torch.utils.data import random_split

    g = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, lengths, generator=g)

    return train_dataset, validation_dataset, test_dataset


# make processed file name based on the config parameters
def make_processed_filename(dataset_name: str, args: dict):
    p = args["Process"]
    name_prefix = args["Dataset"][dataset_name]["processed_dir"]
    name_postfix = (
        "cut_" + str(p["max_cutoff_distance"]) + "_efeat_" + str(p["edge"]["edge_feature"]) + "_gwid_" + str(p["edge"]["gaussian_smearing"]["width"])
    )
    args["Dataset"][dataset_name]["processed_dir"] = name_prefix + "_" + name_postfix
    print("Processed path: ", args["Dataset"][dataset_name]["processed_dir"])
    return args


def make_dataset(
    dataset_name: str,
    args: dict,
    trainset_ratio=None,
    valset_ratio=None,
    testset_ratio=None,
    seed=None,
    **kwargs,
):
    # assert len(lengths) > 0 or (args is not None and seed is not None), "Provide split ratio(lengths) or config dict(args)"
    import sys

    if args["Process"]["auto_processed_name"] is True:
        args = make_processed_filename(dataset_name, args)

    processed_data_path = args["Dataset"][dataset_name]["processed_dir"]

    dataset = getattr(sys.modules[__name__], dataset_name)(args)
    lengths = [trainset_ratio, valset_ratio, testset_ratio]
    train_dataset, validation_dataset, test_dataset = random_split_dataset(dataset, lengths=lengths, seed=seed)
    return train_dataset, validation_dataset, test_dataset, processed_data_path


def delete_processed_data(processed_data_path: str):
    import os

    cmd_str = f"rm -rf {processed_data_path}"
    print(f"delete command: {cmd_str}")
    os.system(cmd_str)


# The dataset occupy table is designed to avoid deleting a dataset,
# where two more workers are paralleled but one is finished and one is still working.
# We get the processed data folder name as the key.
# When a worker starts to use it, the value + 1. When it is finished, the value - 1.
# Every worker when they finished will check the value, if it's 0 the dataset will be deleted.
def create_dataset_occupy_table(path="dataset_occupy_table.pt", data_processed_path: str = None, increment: int = 0):
    import torch

    # if increment is 0 create a occupy table to the path
    if increment == 0:
        data = {}
    else:  # load from path
        data = torch.load(path)

    dataset_occupy_table = data

    # add processed name to the occupy table
    if increment != 0 and data_processed_path is not None:
        dataset_postfix_name = data_processed_path.split("/")[-1]
        dataset_occupy_table[dataset_postfix_name] = (
            increment if dataset_postfix_name not in dataset_occupy_table else dataset_occupy_table[dataset_postfix_name] + increment
        )
        if dataset_occupy_table[dataset_postfix_name] < 0:
            dataset_occupy_table[dataset_postfix_name] = 0

    torch.save(dataset_occupy_table, path)
    print("Dataset Occupy Table: ", dataset_occupy_table)

    if increment != 0 and data_processed_path is not None:
        return dataset_occupy_table[dataset_postfix_name]
