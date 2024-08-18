from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, ModuleList, BatchNorm1d
import torch_geometric.nn

from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        conv_out_dim=None,
        pre_fc_dim=None,
        post_fc_dim=None,
        num_layers=None,
        dropout_rate=None,
        pool=None,
        conv_params=None,
        **kwargs,
    ):
        super().__init__()

        assert num_layers > 0, "num layer should >0"

        self.in_dim = in_dim
        self.conv_out_dim = conv_out_dim
        self.num_layers = num_layers

        self.pre_fc_dim = pre_fc_dim
        self.post_fc_dim = post_fc_dim

        # pre fc
        self.num_pre_fc = len(self.pre_fc_dim)
        self.conv_in_dim = pre_fc_dim[-1] if self.num_pre_fc > 0 else in_dim
        self.pre_fc = torch.nn.ModuleList()
        self.pre_fc_bns = torch.nn.ModuleList()
        last_dim = in_dim
        for i in range(self.num_pre_fc):
            self.pre_fc.append(torch.nn.Linear(last_dim, self.pre_fc_dim[i]))
            self.pre_fc_bns.append(BatchNorm1d(self.pre_fc_dim[i]))
            last_dim = self.pre_fc_dim[i]

        # convs initialize
        conv_params = conv_params if conv_params is not None else {}
        self.convs = torch.nn.ModuleList(
            [GCNConv(self.conv_in_dim, conv_out_dim, conv_params)] + [GCNConv(conv_out_dim, conv_out_dim, conv_params) for i in range(num_layers - 1)]
        )
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(conv_out_dim) for i in range(num_layers)])
        self.dropout_rate = dropout_rate

        # pooling
        self.pool = pool

        # post fc
        self.num_post_fc = len(self.post_fc_dim)
        self.out_dim = post_fc_dim[-1] if self.num_post_fc > 0 else conv_out_dim
        self.post_fc = torch.nn.ModuleList()
        self.post_fc_bns = torch.nn.ModuleList()
        last_dim = conv_out_dim
        for i in range(self.num_post_fc):
            self.post_fc.append(torch.nn.Linear(last_dim, self.post_fc_dim[i]))
            self.post_fc_bns.append(BatchNorm1d(self.post_fc_dim[i]))
            last_dim = self.post_fc_dim[i]

        self.out_lin = torch.nn.Linear(self.out_dim, 1)

    def forward(self, batch_data, node_embedding=False):
        x, edge_index, edge_weight = batch_data.x, batch_data.edge_index, batch_data.edge_weight
        batch = batch_data.batch

        out = x

        # pre full connect
        for i, lin in enumerate(self.pre_fc):
            out = lin(x) if i == 0 else lin(out)
            out = self.pre_fc_bns[i](out)
            out = F.relu(out)

        # gcn conv layers
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out, edge_index, edge_weight)
            out = bn(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.dropout_rate, training=self.training)

        # get node embedding instead of predicting the result
        if node_embedding is True:
            return out

        # global pooling
        out = getattr(torch_geometric.nn, self.pool)(out, batch)

        # post full connect
        for i, lin in enumerate(self.post_fc):
            out = lin(out)
            out = self.post_fc_bns[i](out)
            out = F.relu(out)

        out = self.out_lin(out)

        return out
