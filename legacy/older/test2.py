import torch
import matplotlib.pyplot as plt

import os.path as osp
import numpy as np

from tqdm import tqdm
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, ModuleList
from torch_geometric.nn import (
    GCNConv,
    PNAConv,
    BatchNorm,
    global_add_pool,
    global_mean_pool,
)

from dataset import make_dataset
from train import make_data_loader
from utils import get_device

# from model import GCNNetwork
from args import *

train_dataset, validation_dataset, test_dataset = make_dataset(args)
train_loader, val_loader, test_loader = make_data_loader(train_dataset, validation_dataset, test_dataset)
print("make loader success.")


class GCNNetwork(torch.nn.Module):
    def __init__(self, in_channels, out_dim, numLayers=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.numLayers = numLayers
        self.conv1 = GCNConv(self.in_channels, out_dim)
        self.batch_norm1 = BatchNorm(self.out_dim)

        # This container holds 2nd and more GCN layers
        self.gcnConvs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(1, self.numLayers):
            gcnconv = GCNConv(self.out_dim, out_dim)
            batch_norm = BatchNorm(self.out_dim)
            self.gcnConvs.append(gcnconv)
            self.batch_norms.append(batch_norm)

        # self.pre_mlp = Sequential(Linear(self.in_channels, out_dim//2), ReLU(), Linear(self.out_dim//2, self.out_dim))
        self.post_mlp = Sequential(Linear(out_dim, 100), ReLU(), Linear(100, 1))

    def forward(self, batch_data):
        # Note: batch_data is provided by Dataloader
        x, edge_index = batch_data.x, batch_data.edge_index
        edge_weight = batch_data.edge_attr.squeeze()

        # x = self.pre_mlp(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        for gcnconv, batch_norm in zip(self.gcnConvs, self.batch_norms):
            x = batch_norm(gcnconv(x, edge_index, edge_weight))
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)

        # x = global_add_pool(x, batch_data.batch)
        x = global_mean_pool(x, batch_data.batch)
        out = self.post_mlp(x)
        return out


if __name__ == "__main__":

    print("load model success.")

    device = get_device()

    in_channels = train_dataset[0].x.shape[-1]
    out_channels = args["out_channels_to_mlp"]
    # print(in_channels, out_channels)

    model = GCNNetwork(in_channels, out_channels)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.8, patience=50, min_lr=1e-8)

    model.train()
    for i, data in enumerate(train_loader):
        data = data.to(device)
        print(data)
        out = model(data)
        print(out)
        if i == 0:
            break
