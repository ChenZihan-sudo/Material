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


class GCNNetwork(torch.nn.Module):

    def __init__(
        self,
        in_dim,
        conv_out_dim=args["conv_out_dim"],
        num_pre_fc=args["num_pre_fc"],
        pre_fc_dim=args["pre_fc_dim"],
        num_post_fc=args["num_post_fc"],
        post_fc_dim=args["post_fc_dim"],
        num_layers=args["num_layers"],
        drop_rate=args["dropout_rate"],
        **kwargs,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.conv_out_dim = conv_out_dim
        self.num_layers = num_layers

        self.num_pre_fc = num_pre_fc
        self.pre_fc_dim = pre_fc_dim

        self.num_post_fc = num_post_fc
        self.post_fc_dim = post_fc_dim

        self.pre_fc = torch.nn.ModuleList(
            ([torch.nn.Linear(in_dim, pre_fc_dim)] if num_pre_fc > 0 else [])
            + [torch.nn.Linear(pre_fc_dim, pre_fc_dim) for i in range(num_pre_fc - 1)]
        )
        self.conv_in_dim = pre_fc_dim if num_pre_fc > 0 else in_dim
        self.convs = torch.nn.ModuleList(
            [GCNConv(self.conv_in_dim, conv_out_dim, improved=True)]
            + [GCNConv(conv_out_dim, conv_out_dim, improved=True) for i in range(num_layers - 1)]
        )
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(conv_out_dim) for i in range(num_layers)])
        self.drop_rate = drop_rate

        self.pool = global_mean_pool

        self.post_fc = torch.nn.ModuleList(
            ([torch.nn.Linear(conv_out_dim, post_fc_dim)] if num_post_fc > 0 else [])
            + [torch.nn.Linear(post_fc_dim, post_fc_dim) for i in range(num_post_fc - 1)]
        )

        self.out_dim = post_fc_dim if num_post_fc > 0 else conv_out_dim
        self.out_lin = torch.nn.Linear(self.out_dim, 1)

    def forward(self, batch_data):
        x, edge_index, edge_weight = batch_data.x, batch_data.edge_index, batch_data.edge_weight
        batch = batch_data.batch

        out = None

        # pre full connect
        for i, lin in enumerate(self.pre_fc):
            out = lin(x) if i == 0 else lin(out)
            out = F.relu(out)

        # gcn conv layers
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out, edge_index, edge_weight)
            out = bn(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        # global pooling
        out = self.pool(out, batch)

        # post full connect
        for i, lin in enumerate(self.post_fc):
            out = lin(out)
            out = F.relu(out)

        out = self.out_lin(out)

        return out


# # GCNConv Network
# class GCNNetwork(torch.nn.Module):

#     def __init__(self, in_channels, out_dim, numLayers=args["num_layers"]):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_dim = out_dim
#         self.numLayers = numLayers
#         self.conv1 = GCNConv(self.in_channels, out_dim)
#         self.batch_norm1 = BatchNorm(self.out_dim)

#         # This container holds 2nd and more GCN layers
#         self.gcnConvs = ModuleList()
#         self.batch_norms = ModuleList()
#         for _ in range(1, self.numLayers):
#             gcnconv = GCNConv(self.out_dim, out_dim)
#             batch_norm = BatchNorm(self.out_dim)
#             self.gcnConvs.append(gcnconv)
#             self.batch_norms.append(batch_norm)

#         self.pre_mlp = Sequential(Linear(self.in_channels, out_dim // 2), ReLU(), Linear(self.out_dim // 2, self.out_dim))
#         self.post_mlp = Sequential(Linear(out_dim, 100), ReLU(), Linear(100, 1))

#     def forward(self, batch_data):
#         # Note: batch_data is provided by Dataloader
#         x, edge_index = batch_data.x, batch_data.edge_index
#         edge_weight = batch_data.edge_attr.squeeze() if args["edge_weight"] is True else None

#         if args["edge_weight"] is True:
#             print("edge weight is added into the model.")

#         # x = self.pre_mlp(x)
#         x = self.conv1(x, edge_index, edge_weight)
#         x = self.batch_norm1(x)
#         x = F.relu(x)
#         x = F.dropout(x, p=args["dropout_p"], training=self.training)

#         for gcnconv, batch_norm in zip(self.gcnConvs, self.batch_norms):
#             x = batch_norm(gcnconv(x, edge_index, edge_weight))
#             x = F.relu(x)
#             x = F.dropout(x, p=args["dropout_p"], training=self.training)

#         # x = global_add_pool(x, batch_data.batch)
#         x = global_mean_pool(x, batch_data.batch)
#         out = self.post_mlp(x)
#         return out
