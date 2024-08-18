from .MPDataset import MPDataset
from .HypoDataset import HypoDataset
from .OptimizedHypoDataset import OptimizedHypoDataset

__all__ = [
    "MPDataset",
    "HypoDataset",
    "OptimizedHypoDataset",
    "make_dataset",
    "normalization_1d",
    "reverse_normalization_1d",
]


def random_split_dataset(dataset, lengths=None, seed=None):
    import torch
    from torch.utils.data import random_split

    g = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, lengths, generator=g)

    return train_dataset, validation_dataset, test_dataset


def make_dataset(dataset_name, args):
    # assert len(lengths) > 0 or (args is not None and seed is not None), "Provide split ratio(lengths) or config dict(args)"
    import sys

    dataset = getattr(sys.modules[__name__], dataset_name)(args)
    td_args = args["Training"]["dataset"]
    lengths = [td_args["trainset_ratio"], td_args["valset_ratio"], td_args["testset_ratio"]]
    train_dataset, validation_dataset, test_dataset = random_split_dataset(dataset, lengths=lengths, seed=td_args["seed"])
    return train_dataset, validation_dataset, test_dataset
