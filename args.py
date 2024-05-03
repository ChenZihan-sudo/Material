import os.path as osp

MP_API_KEY = "j61NN3yuDh8tQWf0OrkachbbUoJ8npVP"

CONVENTIONAL_UNIT_CELL = False

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
args["trainset_ratio"] = 0.6
args["testset_ratio"] = 0.2
args["valset_ratio"] = 0.2
args["split_dataset_seed"] = 1024

# data loader
args["batch_size"] = 256
args["data_loader_shuffle"] = True
args["data_loader_seed"] = 1024
args["num_workers"] = 0

# model
args["conv_out_dim"] = 150

args["num_layers"] = 3

args["num_pre_fc"] = 4
args["pre_fc_dim"] = 120

args["num_post_fc"] = 3
args["post_fc_dim"] = 120

args["dropout_rate"] = 0.3

# train
args["epochs"] = 1000
args["learning_rate"] = 0.01


# for reverse min max scalar
args["data_min"] = None
args["data_max"] = None

# for ReduceLROnPlateau scheduler
