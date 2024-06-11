import os.path as osp

MP_API_KEY = "j61NN3yuDh8tQWf0OrkachbbUoJ8npVP"

WORK_DIR = "."

DATASET_ORIGIN_DIR = osp.join("{}".format(WORK_DIR), "dataset")
DATASET_RAW_DIR = osp.join("{}".format(DATASET_ORIGIN_DIR), "raw")

DATASET_DIR = osp.join("{}".format(WORK_DIR), "dataset.max_cutoff==3.5")
DATASET_PROCESSED_DIR = osp.join("{}".format(DATASET_DIR), "processed")

args = {}

args["dataset_dir"] = DATASET_DIR
args["dataset_raw_dir"] = DATASET_RAW_DIR
args["dataset_processed_dir"] = DATASET_PROCESSED_DIR

args["max_cutoff_distance"] = 3.5


# * result path
args["result_path"] = "./results"

# * For dataset of Material Project
args["chunk_size"] = 1000
args["num_chunks"] = None
args["keep_data_from"] = "./dataset/raw/INDICES"
args["onehot_gen"] = False
args["onehot_range"] = [1, 101]

args["data_optimize"] = False
args["data_opt_model_path"] = osp.join(args["result_path"], "CEAL/1717234278983621")

# * device
args["device"] = "cuda"

# * For hypothesis dataset
args["hypothesis_dataset"] = {}
hypo_args = args["hypothesis_dataset"]
hypo_args["scales"] = [0.96, 0.98, 1.00, 1.02, 1.04]
hypo_args["atomic_numbers"] = [58, 27, 29]  # Ce,Co,Cu
# meta data filename
hypo_args["data_filename"] = "hypo_data"
# split dataset to multiple data block
hypo_args["split_num"] = 10
# for large dataset
hypo_args["data_dir"] = osp.join("{}".format(DATASET_PROCESSED_DIR), "hypo_data")

# * random split dataset
args["trainset_ratio"] = 0.45
args["valset_ratio"] = 0.275
args["testset_ratio"] = 0.275
args["split_dataset_seed"] = 765

# * data loader
# 1000 700 500
args["batch_size"] = 800
args["data_loader_shuffle"] = True
args["data_loader_seed"] = 888
args["num_workers"] = 8

# * GCN model
args["GCN"] = {}
gcn_args = args["GCN"]
gcn_args["conv_out_dim"] = 100
gcn_args["num_layers"] = 2
gcn_args["num_pre_fc"] = 2
gcn_args["pre_fc_dim"] = 200
gcn_args["num_post_fc"] = 2
gcn_args["post_fc_dim"] = 200
gcn_args["dropout_rate"] = 0.6
# train
gcn_args["epochs"] = 1000
gcn_args["learning_rate"] = 0.01
# for ReduceLROnPlateau scheduler
gcn_args["sche_mode"] = "min"
gcn_args["sche_factor"] = 0.8
gcn_args["sche_patience"] = 20
gcn_args["sche_min_lr"] = 1e-8

# * CEAL model
args["CEAL"] = {}
ceal_args = args["CEAL"]
# ceal conv parameters
ceal_args["aggregators"] = ["sum", "mean", "min", "max", "std"]
ceal_args["scalers"] = ["identity", "amplification", "attenuation"]
ceal_args["edge_dim"] = 1
ceal_args["towers"] = 1
ceal_args["pre_layers"] = 1
ceal_args["post_layers"] = 1
ceal_args["divide_input"] = False
ceal_args["aggMLP"] = False
ceal_args["aggMLP_factor"] = 1.5

# model parameters
ceal_args["num_pre_fc"] = 1
ceal_args["pre_fc_dim"] = 100
ceal_args["pre_fc_dim_factor"] = None

ceal_args["conv_out_dim"] = 200
ceal_args["num_layers"] = 1

ceal_args["num_post_fc"] = 2
ceal_args["post_fc_dim"] = 150
ceal_args["post_fc_dim_factor"] = 0.4

ceal_args["dropout_rate"] = 0.3
# train
ceal_args["epochs"] = 10000
ceal_args["learning_rate"] = 0.01
# for ReduceLROnPlateau scheduler
ceal_args["sche_mode"] = "min"
ceal_args["sche_factor"] = 0.85
ceal_args["sche_patience"] = 25
ceal_args["sche_min_lr"] = 1e-8
