import os.path as osp

MP_API_KEY = "j61NN3yuDh8tQWf0OrkachbbUoJ8npVP"

WORK_DIR = "."

DATASET_ORIGIN_DIR = osp.join("{}".format(WORK_DIR), "dataset")
DATASET_MP_RAW_DIR = osp.join("{}".format(DATASET_ORIGIN_DIR), "raw_mp")
DATASET_OPT_HYPO_RAW_DIR = osp.join("{}".format(DATASET_ORIGIN_DIR), "raw_opt_hypo")

DATASET_DIR = osp.join("{}".format(WORK_DIR), "dataset.max_cutoff==3.5")
DATASET_PROCESSED_DIR = osp.join("{}".format(DATASET_DIR), "processed")
DATASET_OPT_HYPO_PROCESSED_DIR = osp.join("{}".format(DATASET_PROCESSED_DIR), "opt_hypo_data")

args = {}

args["dataset_dir"] = DATASET_DIR
args["dataset_processed_dir"] = DATASET_PROCESSED_DIR

# * Ignore the interactions beyond a certain distance.
args["max_cutoff_distance"] = 3.5

# * result path
args["result_path"] = "./results"

# * device
args["device"] = "cuda"


# * For dataset from Material Project
args["mp_dataset"] = {}
mp_args = args["mp_dataset"]
mp_args["raw_dir"] = DATASET_MP_RAW_DIR
mp_args["chunk_size"] = 1000
mp_args["num_chunks"] = None
# keep the data sorting from INDICES file
mp_args["keep_data_from"] = "./dataset/raw/INDICES"
mp_args["onehot_gen"] = False
mp_args["onehot_range"] = [1, 101]
# random split dataset
args["trainset_ratio"] = 0.70
args["valset_ratio"] = 0.15
args["testset_ratio"] = 0.15
args["split_dataset_seed"] = 777

args["data_optimize"] = False
args["data_opt_model_path"] = osp.join(args["result_path"], "CEAL/1717234278983621")


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


# * For optimized hypothesis dataset
args["optimized_hypothesis_dataset"] = {}
opt_hypo_args = args["optimized_hypothesis_dataset"]
opt_hypo_args["raw_dir"] = DATASET_OPT_HYPO_RAW_DIR
opt_hypo_args["processed_dir"] = DATASET_OPT_HYPO_PROCESSED_DIR
opt_hypo_args["onehot_gen"] = False
opt_hypo_args["onehot_range"] = [1, 101]
opt_hypo_args["dataset_total_num"] = 4542
opt_hypo_args["formation_energy_filename"] = "FORMATION_ENERGY_"
opt_hypo_args["compound_filename"] = "POSCAR_"
# random split dataset
opt_hypo_args["trainset_ratio"] = 0.70
opt_hypo_args["valset_ratio"] = 0.15
opt_hypo_args["testset_ratio"] = 0.15
opt_hypo_args["split_dataset_seed"] = 777


# * data loader
args["batch_size"] = 800
args["data_loader_shuffle"] = True
args["data_loader_seed"] = 3407
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
ceal_args["scalers"] = ["identity"]
ceal_args["edge_dim"] = 1
ceal_args["towers"] = 1
ceal_args["pre_layers"] = 1
ceal_args["post_layers"] = 1
ceal_args["divide_input"] = False
ceal_args["aggMLP"] = False
ceal_args["aggMLP_factor"] = 0.5

# model parameters
ceal_args["pre_fc_dim"] = [100]  # last one is conv_in_dim
# deprecated
ceal_args["num_pre_fc"] = None
ceal_args["pre_fc_dim_factor"] = None

ceal_args["num_layers"] = 1
ceal_args["conv_out_dim"] = 100

ceal_args["post_fc_dim"] = [100]
# deprecated
ceal_args["num_post_fc"] = None
ceal_args["post_fc_dim_factor"] = None

ceal_args["dropout_rate"] = 0.0

# train parameters
ceal_args["epochs"] = 10000
ceal_args["learning_rate"] = 0.01
# ReduceLROnPlateau scheduler
ceal_args["sche_mode"] = "min"
ceal_args["sche_factor"] = 0.85
ceal_args["sche_patience"] = 30
ceal_args["sche_min_lr"] = 1e-8


# * PNA model
args["PNA"] = {}
pna_args = args["PNA"]

# PNAConv parameters
pna_args["aggregators"] = ["sum", "mean", "min", "max", "std"]
pna_args["scalers"] = ["identity", "amplification", "attenuation"]
pna_args["edge_dim"] = 1
pna_args["towers"] = 1
pna_args["divide_input"] = False
pna_args["pre_layers"] = 1
pna_args["post_layers"] = 1

# model parameters
pna_args["num_pre_fc"] = 1
pna_args["pre_fc_dim"] = 100

pna_args["num_layers"] = 1
pna_args["conv_out_dim"] = 200

pna_args["num_post_fc"] = 2
pna_args["post_fc_dim"] = 150

pna_args["drop_rate"] = 0.3

# train parameters
pna_args["epochs"] = 10000
pna_args["learning_rate"] = 0.01
# ReduceLROnPlateau scheduler
pna_args["sche_mode"] = "min"
pna_args["sche_factor"] = 0.85
pna_args["sche_patience"] = 25
pna_args["sche_min_lr"] = 1e-8
