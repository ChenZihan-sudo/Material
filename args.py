import os.path as osp

MP_API_KEY = "j61NN3yuDh8tQWf0OrkachbbUoJ8npVP"

WORK_DIR = "."
DATASET_DIR = osp.join("{}".format(WORK_DIR), "dataset")
DATASET_RAW_DIR = osp.join("{}".format(DATASET_DIR), "raw")
DATASET_PROCESSED_DIR = osp.join("{}".format(DATASET_DIR), "processed")

args = {}

args["raw_dir"] = DATASET_RAW_DIR
args["processed_dir"] = DATASET_PROCESSED_DIR

args["max_cutoff_distance"] = 5.0

# For dataset of Material Project
args["chunk_size"] = 1000
args["num_chunks"] = None

# random split dataset
args["trainset_ratio"] = 0.7
args["testset_ratio"] = 0.15
args["valset_ratio"] = 0.15
args["split_dataset_seed"] = 1024

# data loader
args["batch_size"] = 1024
args["data_loader_shuffle"] = True
args["data_loader_seed"] = 1024
args["num_workers"] = 32

# model
args["conv_out_dim"] = 100

args["num_layers"] = 1

args["num_pre_fc"] = 2
args["pre_fc_dim"] = 200

args["num_post_fc"] = 2
args["post_fc_dim"] = 200

args["dropout_rate"] = 0.6

# train
args["epochs"] = 1000
args["learning_rate"] = 0.01

# for ReduceLROnPlateau scheduler
args["sche_mode"] = "min"
args["sche_factor"] = 0.8
args["sche_patience"] = 20
args["sche_min_lr"] = 1e-8
