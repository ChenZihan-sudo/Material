from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, ModuleList, BatchNorm1d
import torch_geometric.nn
from torch_geometric.nn import Set2Set

from torch_geometric.nn import CGConv
from .utils import convert_fc_dim


# (in_dim)pre_fc(pre_fc_im) => convs(conv_out_dim) => post_fc(post_fc_dim)
class CGCNN(torch.nn.Module):
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
        pre_fc_dim = convert_fc_dim(pre_fc_dim)
        post_fc_dim = convert_fc_dim(post_fc_dim)

        self.in_dim = in_dim
        # * Ignore conv_out_dim in CGCNN
        # self.conv_out_dim = conv_out_dim
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

        # cgcnn convs initialize
        # * CGCNN can not change the input channel and output channel separately.
        # * Use conv_in_dim as the paramter `channel` in CGConv
        conv_params = conv_params if conv_params is not None else {}
        default_conv = CGConv(self.conv_in_dim, **conv_params)
        # Except for the first conv, in_channels for last convs are out_dim.
        # That is (out_dim)conv(out_dim)
        last_convs = [CGConv(self.conv_in_dim, **conv_params) for i in range(num_layers - 1)]
        self.convs = ModuleList(([default_conv if num_layers > 0 else []]) + (last_convs))
        self.batch_norms = ModuleList([BatchNorm1d(self.conv_in_dim) for i in range(num_layers)])
        self.dropout_rate = dropout_rate

        # pooling
        self.pool = pool

        # post fc
        self.num_post_fc = len(self.post_fc_dim)
        self.post_fc = torch.nn.ModuleList()
        self.post_fc_bns = torch.nn.ModuleList()
        last_dim = self.conv_in_dim
        for i in range(self.num_post_fc):
            self.post_fc.append(torch.nn.Linear(last_dim, self.post_fc_dim[i]))
            self.post_fc_bns.append(BatchNorm1d(self.post_fc_dim[i]))
            last_dim = self.post_fc_dim[i]

        self.out_dim = post_fc_dim[-1] if self.num_post_fc > 0 else self.conv_in_dim
        self.out_lin = torch.nn.Linear(self.out_dim, 1)

    def forward(self, batch_data, node_embedding=False):
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
        batch = batch_data.batch

        out = x

        # pre full connect
        for i, lin in enumerate(self.pre_fc):
            out = lin(x) if i == 0 else lin(out)
            out = self.pre_fc_bns[i](out)
            out = F.relu(out)

        # conv layers
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            out = conv(out, edge_index, edge_attr)
            out = batch_norm(out)
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
