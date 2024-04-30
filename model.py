from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, ModuleList
from torch_geometric.nn import (
    GCNConv,
    PNAConv,
    BatchNorm,
    global_add_pool,
    global_mean_pool,
)
from args import *


# GCNConv Network
class GCNNetwork(torch.nn.Module):

    def __init__(self, in_channels, out_dim, numLayers=args["num_layers"]):
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

        # self.pre_mlp = Sequential(Linear(self.in_channels, out_dim // 2), ReLU(), Linear(self.out_dim // 2, self.out_dim))
        self.post_mlp = Sequential(Linear(out_dim, 100), ReLU(), Linear(100, 1))

    def forward(self, batch_data):
        # Note: batch_data is provided by Dataloader
        x, edge_index = batch_data.x, batch_data.edge_index
        edge_weight = batch_data.edge_attr.squeeze() if args["edge_weight"] is True else None

        if args["edge_weight"] is True:
            print("edge weight is added into the model.")

        # x = self.pre_mlp(x)
        x = self.conv1(x, edge_index, edge_weight)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=args["dropout_p"], training=self.training)

        for gcnconv, batch_norm in zip(self.gcnConvs, self.batch_norms):
            x = batch_norm(gcnconv(x, edge_index, edge_weight))
            x = F.relu(x)
            x = F.dropout(x, p=args["dropout_p"], training=self.training)

        # x = global_add_pool(x, batch_data.batch)
        x = global_mean_pool(x, batch_data.batch)
        out = self.post_mlp(x)
        return out
