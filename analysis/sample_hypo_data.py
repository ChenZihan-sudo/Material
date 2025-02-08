# sample data from the hypothetic compounds dataset
import torch
import process
from os import path as osp
import numpy as np

from .model_prediction import get_model_prediction

def hypo_data_sample_index(
    config,
    model_path,
    batch_size,
    dataset_name,
    generation,
    postfix_epoch,
    sample_cutoff_value,
    sample_ratio,
    sample_seed,
    **kwargs,
):
    pred_out, idx_out = get_model_prediction(config, model_path, batch_size, dataset_name, generation, postfix_epoch)

    # model_path, f"{cmd_args.generation}_{cmd_args.dataset}_sample_data.pt"

    # get compounds { pred value < cutoff value }
    pred_out_np = pred_out.to("cpu").numpy()
    extract_out = pred_out_np < sample_cutoff_value
    extract_out_idx = np.where(extract_out)[0]

    # sample some compounds from the compounds that { pred value < cutoff value }
    np.random.seed(sample_seed)
    extract_out_random_idx = np.random.choice(extract_out_idx, round(len(extract_out_idx) * sample_ratio), replace=False)
    extract_out_random_idx.sort()
    
    # random_idx begins from 0
    random_idx = extract_out_random_idx
    print("random index:", random_idx)
    print("random index length:", len(random_idx))

    return pred_out_np, random_idx


def sample_hypo_data(
    config,
    model_path,
    batch_size,
    dataset_name,
    generation,
    postfix_epoch,
    sample_cutoff_value,
    sample_ratio,
    sample_seed,
    **kwargs,
):
    pred_out_np, random_idx = hypo_data_sample_index(
        config, model_path, batch_size, dataset_name, generation, postfix_epoch, sample_cutoff_value, sample_ratio, sample_seed
    )
    dataset, _ = process.process_dataset("HypoDataset", config)
    
    last_idx = 0
    current_idx = 0

    idx_list = []
    data_list = []
    vasp_list = []
    pred_list = []

    for i in range(len(dataset)):
        print(f"processing: {i}/{len(dataset)}")
        data_block = dataset.get(i, get_vasp_data=False)

        last_idx = current_idx
        current_idx += len(data_block)
        print(f"last index: {last_idx}, current index: {current_idx}")

        # random_idx begins from 0, random_idx + 1 is the ID
        # select range { last_idx <= random_idx < current_idx } in random_idx
        items_idx = random_idx[np.where((random_idx >= last_idx) & (random_idx < current_idx))[0]]

        for item_idx in items_idx:
            data_list.append(data_block[item_idx - last_idx])
        del data_block

        vasp_block = dataset.get(i, get_vasp_data=True)
        for item_idx in items_idx:
            vasp_list.append(vasp_block[item_idx - last_idx])
        del vasp_block

        for item_idx in items_idx:
            idx_list.append(item_idx)
            pred_list.append(pred_out_np[item_idx])

    print(len(idx_list))
    print(len(data_list))
    print(len(vasp_list))

    for i, d in zip(idx_list, vasp_list):
        if not i == d[0] - 1:
            print("something wrong")

    data = {"data_list": data_list, "vasp_list": vasp_list, "idx_list": idx_list, "pred_list": pred_list, "sample_seed": sample_seed}
    save_path = osp.join(model_path, model_path, f"{generation}_{dataset_name}_sample_data{"" if postfix_epoch == "" else f'_{postfix_epoch}'}.pt")
    torch.save(data, save_path)
    print(f"Sample data saved on {save_path}")
    
