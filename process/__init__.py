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
    "normalization_1d",
    "reverse_normalization_1d",
]


def random_split_dataset(dataset, lengths=None, seed=None):
    import torch
    from torch.utils.data import random_split

    g = torch.Generator().manual_seed(seed)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, lengths, generator=g)

    return train_dataset, validation_dataset, test_dataset


def make_dataset(dataset_name, args, trainset_ratio=None, valset_ratio=None, testset_ratio=None, seed=None, **kwargs):
    # assert len(lengths) > 0 or (args is not None and seed is not None), "Provide split ratio(lengths) or config dict(args)"
    import sys

    dataset = getattr(sys.modules[__name__], dataset_name)(args)
    lengths = [trainset_ratio, valset_ratio, testset_ratio]
    train_dataset, validation_dataset, test_dataset = random_split_dataset(dataset, lengths=lengths, seed=seed)
    return train_dataset, validation_dataset, test_dataset
