import torch
import numpy as np

def sample_data_from_model(model_path):
    sample_data = torch.load(osp.join(model_path, "sample_data.pt"))
    random_idx = np.array(sample_data["idx_list"])
    print("random idx:", random_idx)
    print("random idx length:", len(random_idx))
    return random_idx


def sample_data_index():
    from_model_path = cmd_args.sampleData_fromModelPath

    get_out_np = analyse_model_predictions()
    if from_model_path != "":
        return sample_data_from_model(from_model_path), get_out_np

    ratio = cmd_args.sampleData_ratio
    random_seed = cmd_args.sampleData_seed

    extract_out = get_out_np < 0.0
    extract_out_idx = np.where(extract_out)[0]

    np.random.seed(random_seed)
    extract_out_random_idx = np.random.choice(extract_out_idx, round(len(extract_out_idx) * ratio), replace=False)
    extract_out_random_idx.sort()
    random_idx = extract_out_random_idx

    print("random idx:", random_idx)
    print("random idx length:", len(random_idx))
    return random_idx, get_out_np


def sample_data():
    random_idx, get_out_np = sample_data_index()

    last_idx = 0
    current_idx = 0

    idx_list = []
    data_list = []
    vasp_list = []
    pred_list = []

    for i in range(len(dataset)):
        data_block = dataset.get(i, False)
        # if len(data_block) != len(vasp_block):
        #     print("something wrong")

        last_idx = current_idx
        current_idx += len(data_block)
        print(last_idx, current_idx)

        # random_idx + 1 is the ID
        item = random_idx[np.where((random_idx >= last_idx) & (random_idx < current_idx))[0]]

        for d in item:
            data_list.append(data_block[d - last_idx])
        del data_block

        vasp_block = dataset.get(i, True)
        for d in item:
            vasp_list.append(vasp_block[d - last_idx])
        del vasp_block

        for d in item:
            idx_list.append(d)
            pred_list.append(get_out_np[d])

    print(len(idx_list))
    print(len(data_list))
    print(len(vasp_list))

    for i, d in zip(idx_list, vasp_list):
        if not i == d[0] - 1:
            print("something wrong")

    data = {"data_list": data_list, "vasp_list": vasp_list, "idx_list": idx_list, "pred_list": pred_list, "random_seed": cmd_args.sampleData_seed}
    torch.save(data, osp.join(model_path, f"{cmd_args.generation}_{cmd_args.dataset}_sample_data.pt"))
