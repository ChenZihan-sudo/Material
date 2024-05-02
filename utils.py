import torch
import json
import csv
from args import *
import os
from os import path as osp
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


def get_device() -> str:
    return torch.device("cpu")
    # return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


# For dataloader
def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)
    torch.manual_seed(np.random.get_state()[1][0] + worker_id)


# Plot the training progress
def plot_training_progress(epoch, train_losses, val_losses, test_losses, title):
    if epoch % 10 != 0:
        return

    lw = 0.7  # linewidth
    ms = 0.7  # markersize

    threshold = 1
    for i in range(len(train_losses)):
        if train_losses[i] > threshold:
            train_losses[i] = threshold
        if val_losses[i] > threshold:
            val_losses[i] = threshold
        if test_losses[i] > threshold:
            test_losses[i] = threshold

    plt.plot(range(1, epoch + 1, 1), train_losses, marker="o", linestyle="-", color="b", lw=lw, ms=ms)
    plt.plot(range(1, epoch + 1, 1), val_losses, marker="s", linestyle="-", color="r", lw=lw, ms=ms)
    plt.plot(range(1, epoch + 1, 1), test_losses, marker="*", linestyle="-", color="g", lw=lw, ms=ms)

    legend_entries = [
        Line2D([0], [0], color="blue", label="train_loss (Blue)"),
        Line2D([0], [0], color="red", label="validate_loss (Red)"),
        Line2D([0], [0], color="green", label="test_loss (Green)"),
    ]
    plt.title(title)

    plt.legend(handles=legend_entries, loc="upper right")
    plt.pause(0.001)


# Save hyper parameters
def save_hyper_parameter(hyper_dict, result_path, hyper_para_filename="hyperparameters.json"):
    # The hyper parameters are saved in a .json file

    hyper_para_filename = osp.join(result_path, hyper_para_filename)
    with open(hyper_para_filename, "w") as file:
        # Write the variable names and their values to the .json file
        json.dump(hyper_dict, file)


# Create result folder with ns timestamp (e.g. ./results/1712654332689967)
def createResultFolder() -> str:
    results_path = "./results"
    isExist = osp.exists(results_path)
    if not isExist:
        os.makedirs(results_path)

    timestr = str(int(time.time() * 1000000))
    results_data_path = osp.join(results_path, timestr)
    isExist = osp.exists(results_data_path)
    if not isExist:
        os.makedirs(results_data_path)
    else:
        raise ResourceWarning(results_data_path + " already exists.")

    return results_data_path


# Save the training progress info to a csv file (e.g. ./results/1712654332689967/train_progress.csv)
def save_train_progress(epochs, train_losses, val_losses, test_losses, result_path, filename="train_progress.csv"):
    train_progress_path = osp.join(result_path, filename)
    train_progress_dict = [
        {
            "epochs": i,
            "train_losses": train_losses[i - 1],
            "val_losses": val_losses[i - 1],
            "test_losses": test_losses[i - 1],
        }
        for i in range(1, epochs + 1)
    ]
    with open(train_progress_path, "w", newline="") as f:
        cw = csv.DictWriter(f, fieldnames=["epochs", "train_losses", "val_losses", "test_losses"])
        cw.writeheader()
        cw.writerows(train_progress_dict)


# Save regression result to a file
def save_regression_result(test_out, test_y, result_path, filename="regression_result.txt"):
    """
    test_out: predicted values
    test_y: truth values
    """
    result_file = osp.join(result_path, filename)
    test_y = torch.unsqueeze(test_y, dim=1)
    result = torch.cat((test_out, test_y), dim=1)
    result = result.cpu().numpy()
    with open(result_file, "w") as file:
        for row in result:
            # Convert each row of the NumPy array to a space-separated string
            row_str = " ".join(str(element) for element in row)
            file.write(row_str + "\n")
        file.close()


# Save trained GCN model for later inference
def save_model_GCN(model, folder, model_filename, model_para_filename):
    # The state_dict of the model is saved in model_filename
    # The parameters to restore the saved model are saved in model_para_filename
    # The parameters to restore the saved model are saved in a .json file

    model_filename = osp.join(folder, model_filename)
    torch.save(model, model_filename)

    in_channels = model.in_channels
    out_dim = model.out_dim
    numLayers = model.numLayers

    data = {"in_channels": in_channels, "out_dim": out_dim, "numLayers": numLayers}
    model_para_filename = osp.join(folder, model_para_filename)
    with open(model_para_filename, "w") as file:
        # Write the variable names and their values to the .json file
        json.dump(data, file)


def tensor_min_max_scalar_1d(data, new_min=0.0, new_max=1.0) -> tuple[list, torch.Tensor]:
    assert isinstance(data, torch.Tensor)
    data_min = torch.min(data).item()
    data_max = torch.max(data).item()

    if data_max > data_min:
        core = (data - data_min) / (data_max - data_min)
        data_new = core * (new_max - new_min) + new_min
        return data_new, data_min, data_max
    else:
        return data, data_min, data_max


# Get data scale from {DATASET_RAW_DIR}/DATA_SCALE and write into args
def get_data_scale(args):
    filename = osp.join("{}".format(DATASET_RAW_DIR), "DATA_SCALE")
    if not osp.exists(filename):
        raise FileNotFoundError(filename, " not found.")
    with open(filename) as f:
        reader = csv.reader(f)
        data = [row for row in reader][1:]
    return float(data[0][0]), float(data[0][1])


def reverse_min_max_scalar_1d(
    data_normalized,
    data_min=None,
    data_max=None,
    new_min=0.0,
    new_max=1.0,
):
    core = (data_normalized - new_min) / (new_max - new_min)
    data_original = core * (data_max - data_min) + data_min
    return data_original


# Load regression results
def load_regression_results(folder, filename):

    filename = osp.join(folder, filename)

    with open(filename, "r") as f:
        data = f.readlines()
        f.close()
    data = [line.split() for line in data]
    data = np.asarray(data, dtype=float)
    y = data[:, 0]  # predicted
    x = data[:, 1]
    return y, x


# Plot the regression result
def plot_regression_result(title, res_path, filename="regression_result.txt", plotfilename=None, disp=False):
    """
    folder: the folder of the result file
    filename: the result file name
    """
    # data_min, data_max = get_data_scale(args)

    # print(args["data_min"], args["data_max"])

    y, x = load_regression_results(res_path, filename)
    # y = reverse_min_max_scalar_1d(y, data_min, data_max)
    # x = reverse_min_max_scalar_1d(x, data_min, data_max)
    x_min = np.min(x) - 0.001
    x_max = np.max(x) + 0.001
    y_min = np.min(y) - 0.001
    y_max = np.max(y) + 0.001

    # Calculate MAE
    mae = np.mean(np.abs(x - y))
    print("MAE= ", mae)

    # Calculate R^2
    r2 = calculate_r_squared(y, x)

    density = get_density(x, y)
    max_d = round(np.max(density))
    min_d = round(np.min(density))

    sc = plt.scatter(x, y, c=density, cmap="bwr", s=8, marker="o")
    plt.plot([min(x), max(x)], [min(y), max(y)], color="grey", linestyle="--")

    text_x = x_min + (x_max - x_min) * 0.7  # Adjust 0.7 to position the text along the line
    text_y = y_min + (y_max - y_min) * 0.1  # Adjust 0.7 to position the text along the line

    # Add MAE and R2 to the plotting
    mae_msg = f"MAE = {mae:.6f}"
    plt.text(text_x, text_y, mae_msg, fontsize=10)
    r2_msg = f"R^2 = {r2:.6f}"
    text_y = y_min + (y_max - y_min) * 0.05
    plt.text(text_x, text_y, r2_msg, fontsize=10)

    # Create a color bar
    cbar = plt.colorbar(sc)
    cbar.set_label("Color Mapping")

    # Add labels and a title
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(title)
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Save the plotting
    if plotfilename is not None:
        plotfilename = osp.join(res_path, plotfilename)
        plt.savefig(plotfilename)

    # Show the plot
    # plt.legend()
    plt.grid(True)
    if disp:
        plt.show()


# R^2 of predicted vs true values
def calculate_r_squared(y_predicted, y_actual):
    # y_actual: true values, numpy array
    # y_predicted: predicted values, numpy array

    ssr = np.sum((y_actual - y_predicted) ** 2)
    sst = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r_squared = 1 - (ssr / sst)

    return r_squared


# Calculate the point density
def get_density(x, y):
    """Get kernal density estimate for each (x, y) point."""
    values = np.vstack([x, y])
    kernel = gaussian_kde(values)
    density = kernel(values)
    return density


# generate one hot dict
def genOneHotDict(sets):
    sets = list(sets)
    sets.sort()
    from sklearn import preprocessing

    feature = [[d] for _, d in enumerate(sets)]
    coder = preprocessing.OneHotEncoder()
    coder.fit(feature)
    onehots = coder.transform(feature).toarray()
    dict = {}
    for f, o in zip(feature, onehots):
        dict[int(f[0])] = o.tolist()

    filename = osp.join("{}".format(DATASET_RAW_DIR), "onehot_dict.json")
    with open(filename, "w") as file:
        json.dump(dict, file)

    return dict
