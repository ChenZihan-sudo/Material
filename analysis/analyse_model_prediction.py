from .model_prediction import get_model_prediction


def analyse_model_prediction(config, model_path, batch_size, dataset_name, generation, postfix_epoch, **kwargs):
    from os import path as osp
    import matplotlib.pyplot as plt
    import numpy as np

    pred_out, idx_out = get_model_prediction(config, model_path, batch_size, dataset_name, generation, postfix_epoch, **kwargs)
    
    # show results
    counts, bins = np.histogram(pred_out.to("cpu"), bins=50, range=[-1, 5])
    bin_centers = (bins[:-1] + bins[1:]) / 2
    width = 0.8 * (bins[1] - bins[0])
    # print(bin_centers, counts)
    plt.figure(figsize=(7, 5))
    plt.title(f"Formation Energy Distribution on {generation} Model with {dataset_name}")
    plt.xlabel("Formation energy (ev/atom)")
    plt.ylabel("Number of count")
    plt.bar(bin_centers, height=counts, width=width)

    pred_out_np = pred_out.to("cpu").numpy()
    info = (
        f"total: {len(pred_out_np)}\nEf<0: {len(pred_out_np[pred_out_np < 0.0])}\nEf<-0.2: {len(pred_out_np[pred_out_np < -0.2])}\nEf>0: {len(pred_out_np[pred_out_np > 0])}\n"
        + f"ratio of Ef<0: {round(len(pred_out_np[pred_out_np < 0.0]) / len(pred_out_np),6)}\n"
        + f"ratio of Ef<-0.2: {round(len(pred_out_np[pred_out_np < -0.2]) / len(pred_out_np),6)}"
    )
    print(info)

    # print("total:", len(pred_out_np))
    # print("Ef<0:", len(pred_out_np[pred_out_np < 0.0]))
    # print("Ef<-0.2:", len(pred_out_np[pred_out_np < -0.2]))
    # print("Ef>0:", len(pred_out_np[pred_out_np > 0.0]))
    # print("ratio of Ef<0 ", len(pred_out_np[pred_out_np < 0.0]) / len(pred_out_np))
    # print("ratio of Ef<-0.2 ", len(pred_out_np[pred_out_np < -0.2]) / len(pred_out_np))

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    x_offset = (xlim[1] - xlim[0]) * 0.05
    y_offset = (ylim[1] - ylim[0]) * 0.05

    plt.text(xlim[1] - x_offset, ylim[1] - y_offset, info, fontsize=12, color="black", horizontalalignment="right", verticalalignment="top")
    file_path = osp.join(model_path, f"{generation}_{dataset_name}_predict_distribution{"" if postfix_epoch == "" else f'_{postfix_epoch}'}.png")
    plt.savefig(fname=file_path)

    print(f"{generation} predict distribution saved on ", file_path)
