import torch

__all__ = ["make_data_loader", "train_step", "test_evaluations"]


def worker_init_fn(worker_id):
    import numpy as np

    np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.manual_seed(np.random.get_state()[1][0] + worker_id)


def make_data_loader(train_set, val_set, test_set, batch_size=None, seed=None, num_workers=None) -> list:
    assert batch_size is not None or seed is not None or num_workers is not None, "parameters need to make data loader"
    from torch_geometric.loader import DataLoader

    g = torch.Generator().manual_seed(seed)

    train_data_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        generator=g,
        worker_init_fn=worker_init_fn,
    )
    val_data_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_data_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

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
def test_evaluations(model, data_loader, dataset, device):
    model.eval()
    total_loss = 0.0

    res_out = torch.zeros(0).to(device)
    res_y = torch.zeros(0).to(device)

    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            out = model(data)
            total_loss += (out.squeeze() - data.y).abs().sum().item()
            
            res_out = torch.cat((res_out, out), 0)
            res_y = torch.cat((res_y, data.y), 0)

    loss = total_loss / len(dataset)
    return loss, res_out, res_y
