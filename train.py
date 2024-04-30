from torch_geometric.loader import DataLoader
import torch
from dataset import *
from args import *
from utils import *


def make_data_loader(train_set, val_set, test_set) -> list[Dataset]:

    g = torch.Generator().manual_seed(args["data_loader_seed"])
    
    train_data_loader = DataLoader(
        train_set,
        batch_size=args["batch_size"],
        shuffle=args["data_loader_shuffle"],
        num_workers=args["num_workers"],
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    val_data_loader = DataLoader(val_set, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"])
    test_data_loader = DataLoader(test_set, batch_size=args["batch_size"], shuffle=False, num_workers=args["num_workers"])

    return train_data_loader, val_data_loader, test_data_loader


def train_step(model, train_data_loader, train_dataset, optimizer, device):
    model.train()
    total_loss = 0.0
    for i, data in enumerate(train_data_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)  # [batch_size, formation_energy_per_atom]
        # use Mean Absolute Error to calculate loss
        loss = (out.squeeze() - data.y).abs().mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs

    loss = total_loss / len(train_dataset)
    return model, loss


@torch.no_grad()
def test_evaluations(model, data_loader, dataset, device, ret_data=False):
    model.eval()
    total_loss = 0.0

    res_out = torch.zeros(0).to(device) if ret_data is True else None
    res_y = torch.zeros(0).to(device) if ret_data is True else None

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            out = model(data)
            total_loss += (out.squeeze() - data.y).abs().sum().item()

            if ret_data is True:
                res_out = torch.cat((res_out, out), 0)
                res_y = torch.cat((res_y, data.y), 0)

    loss = total_loss / len(dataset)
    return loss, res_out, res_y
