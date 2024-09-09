from .MPDataset import MPDataset
from .HypoDataset import HypoDataset
from .OptimizedHypoDataset import OptimizedHypoDataset
from .MPDatasetLarge import MPDatasetLarge

__all__ = [
    "MPDataset",
    "MPDatasetLarge",
    "HypoDataset",
    "OptimizedHypoDataset",
    "make_dataset",
]


def random_split_dataset(dataset, lengths=None, seed=None):
    import torch
    from torch.utils.data import random_split

    g = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, lengths, generator=g)

    return train_dataset, validation_dataset, test_dataset


# make processed file name
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
