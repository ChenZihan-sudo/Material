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

import copy
from ceal import CEALConv

from utils import get_device

ceal_args = args["CEAL"]
gcn_args = args["GCN"]


class CEALNetwork(torch.nn.Module):
    def __init__(
        self,
        deg,
        in_dim,
        conv_out_dim=ceal_args["conv_out_dim"],
        num_pre_fc=ceal_args["num_pre_fc"],
        pre_fc_dim=ceal_args["pre_fc_dim"],
        num_post_fc=ceal_args["num_post_fc"],
        post_fc_dim=ceal_args["post_fc_dim"],
        num_layers=ceal_args["num_layers"],
        drop_rate=ceal_args["dropout_rate"],
        **kwargs,
    ):
        super().__init__()

        self.deg = deg

        self.in_dim = in_dim
        self.conv_out_dim = conv_out_dim
        self.num_layers = num_layers

        self.num_pre_fc = num_pre_fc
        self.pre_fc_dim = pre_fc_dim

        self.num_post_fc = num_post_fc
        self.post_fc_dim = post_fc_dim

        # pre fc
        self.pre_fc = torch.nn.ModuleList(
            ([torch.nn.Linear(in_dim, pre_fc_dim)] if num_pre_fc > 0 else [])
            + [torch.nn.Linear(pre_fc_dim, pre_fc_dim) for i in range(num_pre_fc - 1)]
        )
        self.pre_fc_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(pre_fc_dim) for i in range(num_pre_fc)])

        # ceal convs
        # (in_dim)conv(out_dim)->bn->relu->dropout->(out_dim)conv(out_dim)->bn->relu->dropout...
        self.conv_in_dim = pre_fc_dim if num_pre_fc > 0 else in_dim
        default_conv = CEALConv(
            self.conv_in_dim,
            conv_out_dim,
            aggregators=ceal_args["aggregators"],
            scalers=ceal_args["scalers"],
            deg=deg,
            edge_dim=ceal_args["edge_dim"],
            towers=ceal_args["towers"],
            pre_layers=ceal_args["pre_layers"],
            post_layers=ceal_args["post_layers"],
            divide_input=ceal_args["divide_input"],
            aggMLP=ceal_args["aggMLP"],
        )
        # Except for the first conv, in_channels for last convs are out_dim. (out_dim)conv(out_dim)
        last_convs = [
            CEALConv(
                conv_out_dim,
                conv_out_dim,
                aggregators=ceal_args["aggregators"],
                scalers=ceal_args["scalers"],
                deg=deg,
                edge_dim=ceal_args["edge_dim"],
                towers=ceal_args["towers"],
                pre_layers=ceal_args["pre_layers"],
                post_layers=ceal_args["post_layers"],
                divide_input=ceal_args["divide_input"],
                aggMLP=ceal_args["aggMLP"],
            )
            for i in range(num_layers - 1)
        ]
        self.convs = torch.nn.ModuleList(([default_conv] if num_layers > 0 else []) + (last_convs))
        # print(self.convs)
        self.bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(conv_out_dim) for i in range(num_layers)])
        self.drop_rate = drop_rate

        self.pool = global_mean_pool

        # post fc
        self.post_fc = torch.nn.ModuleList(
            ([torch.nn.Linear(conv_out_dim, post_fc_dim)] if num_post_fc > 0 else [])
            + [torch.nn.Linear(post_fc_dim, post_fc_dim) for i in range(num_post_fc - 1)]
        )
        self.post_fc_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(post_fc_dim) for i in range(num_post_fc)])

        self.out_dim = post_fc_dim if num_post_fc > 0 else conv_out_dim
        self.out_lin = torch.nn.Linear(self.out_dim, 1)

    def forward(self, batch_data, node_embedding=False):
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
        batch = batch_data.batch

        out = None

        # pre full connect
        for i, lin in enumerate(self.pre_fc):
            out = lin(x) if i == 0 else lin(out)
            out = self.pre_fc_bns[i](out)
            out = F.relu(out)

        # gcn conv layers
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out, edge_index, batch, edge_attr)
            out = bn(out)
            out = F.relu(out)
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        # get node embedding instead of predicting the result
        if node_embedding is True:
            return out

        # global pooling
        out = self.pool(out, batch)

        # post full connect
        for i, lin in enumerate(self.post_fc):
            out = lin(out)
            out = self.post_fc_bns[i](out)
            out = F.relu(out)

        out = self.out_lin(out)

        return out


class GCNNetwork(torch.nn.Module):
    def __init__(
        self,
        in_dim,
        conv_out_dim=gcn_args["conv_out_dim"],
        num_pre_fc=gcn_args["num_pre_fc"],
        pre_fc_dim=gcn_args["pre_fc_dim"],
        num_post_fc=gcn_args["num_post_fc"],
        post_fc_dim=gcn_args["post_fc_dim"],
        num_layers=gcn_args["num_layers"],
        drop_rate=gcn_args["dropout_rate"],
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
        self.pre_fc_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(pre_fc_dim) for i in range(num_pre_fc)])

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
        self.post_fc_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(post_fc_dim) for i in range(num_post_fc)])

        self.out_dim = post_fc_dim if num_post_fc > 0 else conv_out_dim
        self.out_lin = torch.nn.Linear(self.out_dim, 1)

    def forward(self, batch_data, node_embedding=True):
        x, edge_index, edge_weight = batch_data.x, batch_data.edge_index, batch_data.edge_weight
        batch = batch_data.batch

        out = None

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
            out = F.dropout(out, p=self.drop_rate, training=self.training)

        # get node embedding instead of predicting the result
        if node_embedding is True:
            return out

        # global pooling
        out = self.pool(out, batch)

        # post full connect
        for i, lin in enumerate(self.post_fc):
            out = lin(out)
            out = self.post_fc_bns[i](out)
            out = F.relu(out)

        out = self.out_lin(out)

        return out


def save_model(res_path, model, epoch=None, loss=None, optimizer=None, scheduler=None, model_filename="checkpoint.pt"):
    model_filename = osp.join(res_path, model_filename)
    torch.save(
        {
            "epoch": epoch,
            "loss": loss,
            "model": model,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        model_filename,
    )


def load_model(model_path, file_name="checkpoint.pt", load_dict=False, map_location=get_device()):
    model_path = osp.join(model_path, file_name)
    data = torch.load(model_path, map_location=map_location)
    model = data["model"]
    if load_dict is True:
        model.load_state_dict(data["model_state_dict"])
    model = model.to(map_location)
    return model, data
