import torch
from args import *
from train import *


@torch.no_grad()
def predict_step(model, all_data, predict_epochs, device):
    model.eval()

    pbar = tqdm(total=predict_epochs)
    pbar.set_description("Predicting")

    with torch.no_grad():
        res_out = torch.zeros(0).to(device)
        for i, data in enumerate(all_data):
            data = data.to(device)
            out = model(data)
            res_out = torch.cat((res_out, out), 0)
            pbar.update(1)
        pbar.close()

    return res_out


def train_step():
    pass


if __name__ == "__main__":
    pass
