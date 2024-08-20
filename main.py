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

# Get arguments from command line
cmd_args = parser.parse_args(sys.argv[1:])


def convert_str_to_number(d):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, str):
                if v.isdecimal():  # 判断是否为整数
                    d[k] = int(v)
                elif v.isdigit():  # 判断是否为浮点数
                    d[k] = float(v)
            else:
                convert_str_to_number(v)  # 递归处理嵌套结构
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


# import ray

# from ray.tune.search.sample import Float
# def a(default: Float):
#     print(default.domain_str)

# a(config["Models"]["ChemGNN"]["learning_rate"])


# print(config)
# def a(onehot_gen=None, **kwargs):
#     print(onehot_gen, kwargs)


# a(**config["Dataset"]["MPDataset"])
# process.HypoDataset(config)

# a, b, c = process.make_dataset("MPDataset", args=config)
# print(a[0])

# dataset = OptimizedHypoDataset.OptimizedHypoDataset(config)

# print(config)


# import training

# training.start_training("ChemGNN", "MPDataset", config)

# import process
# process.MPDataset(config)
# process.make_dataset("MPDataset", config, **(config["Training"]["dataset"]))



import tuning
from tuning.prepare import make_tune_config

tune_config = yaml_from_template(cmd_args.tune_config_path, recursive_render_time=0)
config = make_tune_config(config, tune_config)
tuning.start_tuning("ChemGNN", "MPDataset", config)


