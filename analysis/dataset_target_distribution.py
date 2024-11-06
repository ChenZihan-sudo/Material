# get dataset target (y, formation energy) distribution,
# show it by a histogram


def dataset_target_distribution(config, model_path, batch_size, dataset_name, generation, postfix_epoch, **kwargs):
    import torch
    import process
    import numpy as np
    from os import path as osp
    import matplotlib.pyplot as plt
    from utils import reverse_min_max_scalar_1d, get_data_scale

    dataset, _ = process.process_dataset(dataset_name, config)
    dataset_args = config["Dataset"][dataset_name]
    processed_path = dataset_args["processed_dir"]
    print(processed_path)
    
    if "y" not in dataset[0]:
        print(f"{dataset_name} don't have a target (y, formation energy)")
        return

    results = torch.zeros(0)
    for i, d in enumerate(dataset):
        results = torch.cat((results, d.y), 0)

    # reverse data target presentation
    if config["Process"]["target_normalization"]:
        data_path = process.get_parameter_file_path(dataset_args["get_parameters_from"], config)
        data_min, data_max = get_data_scale(data_path)
        results = reverse_min_max_scalar_1d(results, data_min, data_max)

    # save dataset target data
    torch.save(results, osp.join(processed_path, f"{dataset_name}_target.pt"))

    # show results
    get_out = results
    counts, bins = np.histogram(get_out.to("cpu"), bins=100)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = 0.8 * (bins[1] - bins[0])

    plt.figure(figsize=(7, 5))
    plt.title(f"Formation Energy Distribution on Dataset {dataset_name}")
    plt.xlabel("Formation energy (ev/atom)")
    plt.ylabel("Number of count")
    plt.bar(bin_centers, height=counts, width=width)

    get_out_np = get_out.to("cpu").numpy()
    info = (
        f"total: {len(get_out_np)}\nEf<0: {len(get_out_np[get_out_np < 0.0])}\nEf<-0.2: {len(get_out_np[get_out_np < -0.2])}\nEf>0: {len(get_out_np[get_out_np > 0])}\n"
        + f"ratio of Ef<0: {round(len(get_out_np[get_out_np < 0.0]) / len(get_out_np),6)}\n"
        + f"ratio of Ef<-0.2: {round(len(get_out_np[get_out_np < -0.2]) / len(get_out_np),6)}"
    )
    print(info)

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_offset = (xlim[1] - xlim[0]) * 0.05
    y_offset = (ylim[1] - ylim[0]) * 0.05

    plt.text(xlim[1] - x_offset, ylim[1] - y_offset, info, fontsize=12, color="black", horizontalalignment="right", verticalalignment="top")
    file_path = osp.join(processed_path, f"{dataset_name}_target_distribution.png")
    plt.savefig(fname=file_path)

    print(f"{dataset_name} target distribution saved on ", file_path)
    return get_out_np
