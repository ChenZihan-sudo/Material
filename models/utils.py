def save_model(res_path, model=None, epoch=None, loss=None, optimizer=None, scheduler=None, model_filename="checkpoint.pt", model_object=None):
    import torch
    from os import path as osp

    save_path = osp.join(res_path, model_filename)
    
    object = (
        {
            "epoch": epoch,
            "loss": loss,
            "model": model,
            "model_state_dict": model.state_dict(),
            "optimizer": optimizer,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler": scheduler,
            "scheduler_state_dict": scheduler.state_dict(),
        }
        if model_object is None
        else model_object
    )
    torch.save(
        object,
        save_path,
    )


def load_model(model_path, file_name="checkpoint.pt", load_dict=False, map_location=None):
    import torch
    from os import path as osp

    # if map_location is None:
    #     from utils import get_device

    #     map_location = get_device()

    model_path = osp.join(model_path, file_name)
    data = torch.load(model_path, map_location=map_location)
    model = data["model"]
    if load_dict is True:
        model.load_state_dict(data["model_state_dict"])
    model = model.to(map_location)
    return model, data


# for PNA and chemgnn
def generate_deg(dataset):
    import torch
    from torch_geometric.utils import degree

    max_degree = -1
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        # find the max degree in the whole dateset
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


# print model summary
def model_summary(model, file=None):
    import torch
    from sys import stdout

    file = stdout if file is None else file
    model_params_list = list(model.named_parameters())
    print("--------------------------------------------------------------------------", file=file)
    line_new = "{:>30}  {:>20} {:>20}".format("Layer.Parameter", "Param Tensor Shape", "Param #")
    print(line_new, file=file)
    print("--------------------------------------------------------------------------", file=file)
    for elem in model_params_list:
        p_name = elem[0]
        p_shape = list(elem[1].size())
        p_count = torch.tensor(elem[1].size()).prod().item()
        line_new = "{:>30}  {:>20} {:>20}".format(p_name, str(p_shape), str(p_count))
        print(line_new, file=file)
    print("--------------------------------------------------------------------------", file=file)
    total_params = sum([param.nelement() for param in model.parameters()])
    print("Total params:", total_params, file=file)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable params:", num_trainable_params, file=file)
    print("Non-trainable params:", total_params - num_trainable_params, file=file)


def convert_fc_dim(fc_dim):
    if not isinstance(fc_dim, dict):
        return fc_dim
    num_layer = fc_dim["num_layer"]
    dim = fc_dim["dim"]
    fc_dim = [dim for i in range(num_layer)]
    return fc_dim
