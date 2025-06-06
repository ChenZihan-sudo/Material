import argparse
import os
import time
import csv
import sys
import json
import random
import numpy as np
import pprint
import yaml

parser = argparse.ArgumentParser(description="")

parser.add_argument(
    "-C",
    "--config_path",
    default="config.yml",
    type=str,
    help="Location of config file (default: config.yml)",
)

parser.add_argument(
    "--tune_config_path",
    default="tuning/tune_config.yml",
    type=str,
    help="Location of tune config file (default: tuning/tune_config.yml)",
)
parser.add_argument(
    "--tune_exp_path",
    default=None,
    type=str,
    help="Location of tune experiment folder. Path should be the absolute path. Default format: <Project path>/Material/tune_results/tune/<Model>_<Dataset>_<Year>-<Month>-<Day>_<Hour>-<Minute>-<Second>",
    required=False,
)

parser.add_argument("-M", "--model", default="ChemGNN", type=str, help="model name (ChemGNN, PNA, GCN, etc.)", required=False)
parser.add_argument(
    "-D",
    "--dataset",
    default="MPDataset",
    type=str,
    help="dataset name (MPDataset, MPDatasetLarge, HypoDataset, OptimizedHypoDataset, etc.)",
    required=False,
)

parser.add_argument("--use_config", action="store_true", help="", required=False)

# Process, Training, Tuning, Analysis
parser.add_argument("-T", "--task", default="Training", type=str, help="task name (Process, Training, Tuning and TuningAnalysis)", required=True)

# Get arguments from the command line
cmd_args = sys.argv[1:]
tag, b, e = False, 0, len(cmd_args) - 1
for i, d in enumerate(cmd_args):
    if d == "{":
        b = i
        tag = True
    if e == "{":
        e = i
extra_cmd_args = cmd_args[b + 1 : e]
print("cmd_args", cmd_args[: b if tag else len(cmd_args)])
# print("extra_cmd_args", extra_cmd_args)
cmd_args = parser.parse_args(cmd_args[: b if tag else len(cmd_args)])


def convert_str_to_number(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, str):
                if v.isdecimal():
                    d[k] = int(v)
                elif v.isdigit():
                    d[k] = float(v)
            else:
                convert_str_to_number(v)
    return d


def yaml_from_template(config_path, recursive_render_time=5):
    from jinja2 import Template

    assert os.path.exists(config_path), "Config file not found in " + config_path
    with open(config_path, "r") as file:
        content = file.read()
    with open(config_path, "r") as ymlfile:
        config = yaml.load(ymlfile, Loader=yaml.FullLoader)

    for i in range(recursive_render_time):
        content = Template(content)
        content = content.render(config)
        config = yaml.load(content, Loader=yaml.FullLoader)
    return config


config = yaml_from_template(cmd_args.config_path, recursive_render_time=10)
config = convert_str_to_number(config)

# add cmd paramters
model = cmd_args.model
dataset = cmd_args.dataset
config["Default"]["model_name"] = model
config["Default"]["dataset_name"] = dataset

# training
if cmd_args.task == "Training":
    import training

    training.start_training(model, dataset, config)

# process dataset manually
if cmd_args.task == "Process":
    import process

    dataset, _ = process.process_dataset(dataset, config)
    print(f"dataset total length: {len(dataset)}")

# analysis
if cmd_args.task == "Analysis":
    import analysis

    if cmd_args.use_config is True:
        extra_cmd_args = config["Analysis"]
        print("extra_cmd_args", extra_cmd_args)
        analysis.manager(config, extra_cmd_args, use_config=True)
    
    if cmd_args.use_config is False:
        print("extra_cmd_args", extra_cmd_args)
        analysis.manager(config, extra_cmd_args)


# tuning
if cmd_args.task == "Tuning":
    import tuning
    from tuning.prepare import make_tune_config

    tune_config = yaml_from_template(cmd_args.tune_config_path, recursive_render_time=0)
    config = make_tune_config(config, tune_config)
    tuning.start_tuning(model, dataset, config)

if cmd_args.task == "TuningAnalysis":
    from tuning import tuner
    from ray import tune

    if cmd_args.tune_exp_path is not None:
        experiment_path = cmd_args.tune_exp_path
        print(f"Loading results from {experiment_path}...")
        restored_tuner = tune.Tuner.restore(experiment_path, trainable=tuner.trainable_model)
        result_grid = restored_tuner.get_results()

        # add error trails
        error_trails = []
        for result in result_grid:
            if result.error is not None:
                error_trails.append(result)

        print(f"===============================================")
        print(f"Total trials: {len(result_grid)}")
        print(f"Error trials: {len(error_trails)}")
        print(f"==================BEST RESULT==================")
        # best_result = result_grid.get_best_result("mean_absolute_error", mode="min")
        best_result = result_grid.get_best_result("r2", mode="max")
        storage_path = best_result.metrics["storage_path"]
        mean_absolute_error = best_result.metrics["mean_absolute_error"]
        mae = best_result.metrics["mae"]
        r2 = best_result.metrics["r2"]
        print(f"Best result data path: {storage_path}")
        print(f"Mean absolute error (not scaled): {mean_absolute_error}")
        print(f"Mean absolute error (scaled): {mae}")
        print(f"R^2: {r2}")
        print(f"===============================================")
        tune_analysis_data = {"result_grid": result_grid}
        import torch

        path = "./tune_analysis_data.pt"
        torch.save(tune_analysis_data, path)
        print(f"Tuning analysis data saved on {path}")
    else:
        print("Parameter --tune_exp_path=<Your tune experiment path> needed to analyze tuning results.")
