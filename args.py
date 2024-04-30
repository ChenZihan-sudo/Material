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
# TODO: to get all datasets set 1000 if not debugging
args["chunk_size"] = 1000
# TODO: to get all datasets set None if not debugging
args["num_chunks"] = 1

# split dataset
args["trainset_ratio"] = 0.8
args["testset_ratio"] = 0.1
args["valset_ratio"] = 0.1
args["split_dataset_shuffle"] = True
args["split_dataset_seed"] = 1234

# data loader
args["batch_size"] = 128
args["data_loader_shuffle"] = True
args["data_loader_seed"] = 9876
args["num_workers"] = 0

# model
args["num_layers"] = 1

# train
args["epochs"] = 1000
args["learning_rate"] = 0.0005
args["out_channels_to_mlp"] = 100
args["dropout_p"] = 0.5
args["edge_weight"] = True

# for reverse min max scalar
args["data_min"] = None
args["data_max"] = None

# for ReduceLROnPlateau scheduler
