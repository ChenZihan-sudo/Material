import os
import argparse
import time
import csv
import sys
import json
import random
import numpy as np
import pprint
import yaml

# import models
import process
import process.MPDatasetLarge

parser = argparse.ArgumentParser(description="")
parser.add_argument(
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

parser.add_argument("-M", "--model", default="ChemGNN", type=str, help="", required=False)
parser.add_argument("-D", "--dataset", default="MPDataset", type=str, help="", required=False)

# Process, Training, Tuning, Analysis
parser.add_argument("-T", "--task", default="Training", type=str, help="", required=True)

# Get arguments from command line
cmd_args = parser.parse_args(sys.argv[1:])


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

# training
if cmd_args.task == "Training":
    import training

    training.start_training("ChemGNN", "MPDataset", config)

# process dataset manually
if cmd_args.task == "Process":
    import process

    # process.MPDataset(config)

    process.make_dataset("MPDataset", config, 0.1, 0.1, 0.8, 1)
    # process.MPDatasetLarge(config)

# tuning
if cmd_args.task == "Tuning":
    import tuning
    from tuning.prepare import make_tune_config

    tune_config = yaml_from_template(cmd_args.tune_config_path, recursive_render_time=0)
    config = make_tune_config(config, tune_config)
    # print(config)
    tuning.start_tuning("ChemGNN", "MPDataset", config)
