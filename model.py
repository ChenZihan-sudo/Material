from tqdm import tqdm
import os.path as osp
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential, ModuleList, BatchNorm1d
from torch_geometric.nn import (
    GCNConv,
    PNAConv,
    BatchNorm,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
    global_sort_pool,
)

from args import *
from ceal import CEALConv
from utils import get_device

ceal_args = args["CEAL"]
gcn_args = args["GCN"]
pna_args = args["PNA"]


class OriginCEALNetwork(torch.nn.Module):

    def __init__(
        self,
        deg,
        in_channels,
        conv_out_dim=ceal_args["conv_out_dim"],
        aggregators=ceal_args["aggregators"],
        scalers=ceal_args["scalers"],
        numLayers=1,
        edge_dim=ceal_args["edge_dim"],
        towers=ceal_args["towers"],
        pre_layers=ceal_args["pre_layers"],
        post_layers=ceal_args["post_layers"],
        divide_input=ceal_args["divide_input"],
        aggMLP=ceal_args["aggMLP"],
        **kwargs,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.out_channels = conv_out_dim
        self.aggregators = aggregators
        self.scalers = scalers
        self.deg = deg
        self.numLayers = numLayers
        self.edge_dim = edge_dim
        self.towers = towers
        self.pre_layers = pre_layers
        self.post_layers = post_layers
        self.divide_input = divide_input
        self.aggMLP = aggMLP

        # The first CEAL layer
        self.ceal_1 = CEALConv(
            in_channels=conv_out_dim,
            out_channels=conv_out_dim,
            aggregators=aggregators,
            scalers=scalers,
            deg=deg,
            edge_dim=edge_dim,
            towers=1,
            pre_layers=1,
            post_layers=1,
            divide_input=False,
            aggMLP=aggMLP,
        )
        self.batch_norm_1 = BatchNorm(self.out_channels)

        # This container holds 2nd and more ceal layers
        self.cealConvs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(1, self.numLayers):
            cealconv = CEALConv(
                in_channels=conv_out_dim,
                out_channels=conv_out_dim,
                aggregators=aggregators,
                scalers=scalers,
                deg=deg,
                edge_dim=edge_dim,
                towers=1,
                pre_layers=1,
                post_layers=1,
                divide_input=False,
                aggMLP=aggMLP,
            )
            batch_norm = BatchNorm(self.out_channels)
            self.cealConvs.append(cealconv)
            self.batch_norms.append(batch_norm)

        self.pre_mlp = Sequential(Linear(in_channels, conv_out_dim // 2), ReLU(), Linear(conv_out_dim // 2, conv_out_dim))
        self.post_mlp = Sequential(Linear(self.out_channels, 100), ReLU(), Linear(100, 1))

    def forward(self, batch_data):
        # Note: batch_data is provided by Dataloader
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr

        x = self.pre_mlp(x)
        x = self.ceal_1(x, edge_index, batch_data.batch, edge_attr)
        x = self.batch_norm_1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        for cealconv, batch_norm in zip(self.cealConvs, self.batch_norms):
            x = batch_norm(cealconv(x, edge_index, batch_data.batch, edge_attr))
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        x = global_add_pool(x, batch_data.batch)
        out = self.post_mlp(x)
        return out


class CEALNetwork(torch.nn.Module):
    def __init__(
        self,
        deg,
        in_dim,
        conv_out_dim=ceal_args["conv_out_dim"],
        num_pre_fc=ceal_args["num_pre_fc"],
        pre_fc_dim=ceal_args["pre_fc_dim"],
        pre_fc_dim_factor=ceal_args["pre_fc_dim_factor"],
        num_post_fc=ceal_args["num_post_fc"],
        post_fc_dim=ceal_args["post_fc_dim"],
        post_fc_dim_factor=ceal_args["post_fc_dim_factor"],
        num_layers=ceal_args["num_layers"],
        drop_rate=ceal_args["dropout_rate"],
        **kwargs,
    ):
        """
        Args:
        - pre_fc_dim_factor: not None will ignore `pre_fc_dim` and follow
        the `in_dim` incresing or decresing to the `conv_in_dim` with `num_pre_fc` layers and `pre_fc_dim_factor`
        - post_fc_dim_factor: not None will ignore `post_fc_dim` and follow
        the `conv_out_dim` incresing or decresing to 1 dim with `num_post_fc` layers and `post_fc_dim_factor`
        - num_layers: conv model num of layers
        """
        super().__init__()

        self.deg = deg

        self.in_dim = in_dim
        self.conv_out_dim = conv_out_dim
        self.num_layers = num_layers

        self.num_pre_fc = num_pre_fc
        self.pre_fc_dim = pre_fc_dim
        self.pre_fc_dim_factor = pre_fc_dim_factor

        self.num_post_fc = num_post_fc
        self.post_fc_dim = post_fc_dim
        self.post_fc_dim_factor = post_fc_dim_factor

        # pre fc
        # pre_fc_dim_factor is deprecated
        if self.pre_fc_dim_factor is not None:
            dim = self.in_dim
            dims = []
            for i in range(self.num_pre_fc):
                dim = round(dim * self.pre_fc_dim_factor)
                dims.append(dim)
            # print(dims)
            self.pre_fc = torch.nn.ModuleList()
            self.pre_fc.append(torch.nn.Linear(self.in_dim, dims[0]))
            for i in range(len(dims) - 1):
                self.pre_fc.append(torch.nn.Linear(dims[i], dims[i + 1]))

            self.pre_fc_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(dims[i]) for i in range(len(dims))])
            self.conv_in_dim = dims[-1] if self.num_pre_fc > 0 else in_dim
            # print(self.pre_fc, self.pre_fc_bns, self.conv_in_dim, dims)
        else:
            if type(pre_fc_dim) is int:
                self.conv_in_dim = pre_fc_dim if self.num_pre_fc > 0 else in_dim
                self.pre_fc = torch.nn.ModuleList(
                    ([torch.nn.Linear(in_dim, pre_fc_dim)] if self.num_pre_fc > 0 else [])
                    + [torch.nn.Linear(pre_fc_dim, pre_fc_dim) for i in range(self.num_pre_fc - 1)]
                )
                self.pre_fc_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(pre_fc_dim) for i in range(self.num_pre_fc)])
            elif type(pre_fc_dim) is list:
                self.num_pre_fc = len(self.pre_fc_dim)
                self.conv_in_dim = pre_fc_dim[-1] if self.num_pre_fc > 0 else in_dim
                self.pre_fc = torch.nn.ModuleList()
                self.pre_fc_bns = torch.nn.ModuleList()
                last_dim = in_dim
                for i in range(self.num_pre_fc):
                    self.pre_fc.append(torch.nn.Linear(last_dim, self.pre_fc_dim[i]))
                    self.pre_fc_bns.append(torch.nn.BatchNorm1d(self.pre_fc_dim[i]))
                    last_dim = self.pre_fc_dim[i]
                # print(self.pre_fc)
                # print(self.pre_fc_bns)

        # ceal convs
        # (in_dim)conv(out_dim)->bn->relu->dropout->(out_dim)conv(out_dim)->bn->relu->dropout...
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
            aggMLP_factor=ceal_args["aggMLP_factor"],
        )
        # Except for the first conv, in_channels for last convs are out_dim.
        # That is (out_dim)conv(out_dim)
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

        # self.pool = global_mean_pool
        self.pool = global_add_pool

        # post fc
        if self.post_fc_dim_factor is not None:
            dim = self.conv_out_dim
            dims = []
            for i in range(self.num_post_fc):
                dim = round(dim * self.post_fc_dim_factor)
                dims.append(dim)

            self.post_fc = torch.nn.ModuleList()
            self.post_fc.append(torch.nn.Linear(self.conv_out_dim, dims[0]))
            for i in range(len(dims) - 1):
                self.post_fc.append(torch.nn.Linear(dims[i], dims[i + 1]))

            self.post_fc_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(dims[i]) for i in range(len(dims))])
            self.out_dim = dims[-1] if self.num_post_fc > 0 else conv_out_dim
            # print(self.post_fc, self.post_fc_bns, self.conv_out_dim, dims)
        else:
            if type(post_fc_dim) is int:
                self.post_fc = torch.nn.ModuleList(
                    ([torch.nn.Linear(conv_out_dim, post_fc_dim)] if self.num_post_fc > 0 else [])
                    + [torch.nn.Linear(post_fc_dim, post_fc_dim) for i in range(self.num_post_fc - 1)]
                )
                self.post_fc_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(post_fc_dim) for i in range(self.num_post_fc)])
                self.out_dim = post_fc_dim if self.num_post_fc > 0 else conv_out_dim
            elif type(post_fc_dim) is list:
                self.num_post_fc = len(self.post_fc_dim)
                self.out_dim = post_fc_dim[-1] if self.num_post_fc > 0 else conv_out_dim
                self.post_fc = torch.nn.ModuleList()
                self.post_fc_bns = torch.nn.ModuleList()
                last_dim = conv_out_dim
                for i in range(self.num_post_fc):
                    self.post_fc.append(torch.nn.Linear(last_dim, self.post_fc_dim[i]))
                    self.post_fc_bns.append(torch.nn.BatchNorm1d(self.post_fc_dim[i]))
                    last_dim = self.post_fc_dim[i]
                # print(self.post_fc)
                # print(self.post_fc_bns)

        self.out_lin = torch.nn.Linear(self.out_dim, 1)

    def forward(self, batch_data, node_embedding=False):
        x, edge_index, edge_attr = batch_data.x, batch_data.edge_index, batch_data.edge_attr
        batch = batch_data.batch

        out = x

        # pre full connect
        for i, lin in enumerate(self.pre_fc):
            out = lin(x) if i == 0 else lin(out)
            out = self.pre_fc_bns[i](out)
            out = F.leaky_relu(out)

        # ceal conv layers
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out, edge_index, batch, edge_attr)
            out = bn(out)
            out = F.leaky_relu(out)
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
            out = F.leaky_relu(out)

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
        # print("batch_data", batch_data)
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


# (in_dim)pre_fc(pre_fc_dim) => convs(conv_out_dim) => post_fc(post_fc_dim)
class PNANetwork(torch.nn.Module):
    def __init__(
        self,
        deg,
        in_dim,
        num_pre_fc=pna_args["num_pre_fc"],
        pre_fc_dim=pna_args["pre_fc_dim"],
        num_layers=pna_args["num_layers"],
        conv_out_dim=pna_args["conv_out_dim"],
        num_post_fc=pna_args["num_post_fc"],
        post_fc_dim=pna_args["post_fc_dim"],
        drop_rate=pna_args["drop_rate"],
        **kwargs,
    ):
        """
        Args:
        - num_layers: conv model num of layers
        """
        super().__init__()

        self.deg = deg

        self.in_dim = in_dim
        self.conv_out_dim = conv_out_dim
        self.num_layers = num_layers

        self.num_pre_fc = num_pre_fc
        self.pre_fc_dim = pre_fc_dim
        # self.pre_fc_dim_factor = pre_fc_dim_factor

        self.num_post_fc = num_post_fc
        self.post_fc_dim = post_fc_dim
        # self.post_fc_dim_factor = post_fc_dim_factor

        # pre fc initialize
        self.pre_fc = torch.nn.ModuleList(
            ([torch.nn.Linear(in_dim, pre_fc_dim)] if num_pre_fc > 0 else [])
            + [torch.nn.Linear(pre_fc_dim, pre_fc_dim) for i in range(num_pre_fc - 1)]
        )
        self.pre_fc_bns = torch.nn.ModuleList([torch.nn.BatchNorm1d(pre_fc_dim) for i in range(num_pre_fc)])
        self.conv_in_dim = pre_fc_dim if num_pre_fc > 0 else in_dim

        # pna convs initialize
        default_conv = PNAConv(
            self.conv_in_dim,
            conv_out_dim,
            aggregators=pna_args["aggregators"],
            scalers=pna_args["scalers"],
            deg=deg,
            edge_dim=pna_args["edge_dim"],
            towers=pna_args["towers"],
            pre_layers=pna_args["pre_layers"],
            post_layers=pna_args["post_layers"],
            divide_input=pna_args["divide_input"],
        )
        # Except for the first conv, in_channels for last convs are out_dim.
        # That is (out_dim)conv(out_dim)
        last_convs = [
            PNAConv(
                conv_out_dim,
                conv_out_dim,
                aggregators=pna_args["aggregators"],
                scalers=pna_args["scalers"],
                deg=deg,
                edge_dim=pna_args["edge_dim"],
                towers=pna_args["towers"],
                pre_layers=pna_args["pre_layers"],
                post_layers=pna_args["post_layers"],
                divide_input=pna_args["divide_input"],
            )
            for i in range(num_layers - 1)
        ]
        self.convs = ModuleList(([default_conv if num_layers > 0 else []]) + (last_convs))
        self.batch_norms = ModuleList([BatchNorm1d(conv_out_dim) for i in range(num_layers)])
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

        out = x

        # pre full connect
        for i, lin in enumerate(self.pre_fc):
            out = lin(x) if i == 0 else lin(out)
            out = self.pre_fc_bns[i](out)
            out = F.relu(out)

        # pna conv layers
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            out = conv(out, edge_index, edge_attr)
            out = batch_norm(out)
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
