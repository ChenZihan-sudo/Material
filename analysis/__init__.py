from .model_prediction import model_prediction
from .analyse_model_prediction import analyse_model_prediction
from .dataset_target_distribution import dataset_target_distribution
from .regression_analysis import regression_analysis
from .sample_hypo_data import sample_hypo_data

# TODO: debugging, remove this when you see
from .model_prediction import get_model_prediction

# create a mapping of task names to functions
task_functions = {
    "model_prediction": model_prediction,
    "analyse_model_prediction": analyse_model_prediction,
    "dataset_target_distribution": dataset_target_distribution,
    "regression_analysis": regression_analysis,
    "sample_hypo_data": sample_hypo_data,
}


def manager(config, args, use_config=False):
    """
    Args:
        - config: the whole config dictionary comes from the main.py
        - args: extra arguments used for parser in this file
    """
    import argparse

    parser = argparse.ArgumentParser(description="sample and analysis by inference a model")

    parser.add_argument("-C", "--config_path", default="config.yml", type=str, help="config file path (default: config.yml)")
    parser.add_argument("-M", "--model_path", required=False, type=str, default="", help="model path for inference (<model path>/checkpoint.pt)")
    parser.add_argument("-D", "--dataset_name", required=True, type=str, help="dataset type (HypoDataset, OptimizedHypoDataset, etc.)")
    parser.add_argument("-B", "--batch_size", required=False, default=100, type=int)
    parser.add_argument("-G", "--generation", default="1G", type=str, help="the generation name tag of the model (default:1G)")
    parser.add_argument(
        "-E",
        "--postfix_epoch",
        required=False,
        default="",
        type=str,
        help="string, optional, set this if the model has the postfix epoch index (default: '')",
    )
    # parser.add_argument("-R", "--figure_range", default=[""], type=list, help="view range in the graph")

    # * task explainations *
    # model_prediction:             generate model prediction results from a dataset and save the results to the model path
    # analyse_model_prediction:     analyse the model results, generate a distribution histogram
    # dataset_target_distribution:  generate dataset target(formation energy) results and save the results to the dataset path. Analyse the target distribution, show it by a histogram.
    # regression_analysis:          generate regression graph based on the dataset target result (x axis) and the model prediction result (y axis). Store the graph to the model path.
    # sample_hypo_data:             sample the hypothesis compound data from the hypothesis dataset
    parser.add_argument(
        "-V",
        "--sample_cutoff_value",
        required=False,
        default=-0.2,
        type=float,
        help="a sample of some compounds which prediction value less than the cutoff value",
    )
    parser.add_argument(
        "-R",
        "--sample_ratio",
        required=False,
        default=1.0,
        type=float,
        help="a random sample of some compounds in a ratio from the compounds that already cutoff",
    )
    parser.add_argument("-S", "--sample_seed", required=False, default=114514, type=int, help="random sample seed for sample ratio")
    # * task explainations *
    parser.add_argument(
        "-T", "--task", default="", type=str, help="task name (model_prediction, analyse_model_prediction, sample_data)", required=True
    )

    cmd_args_dict = parser.parse_args(args).__dict__ if use_config is False else args
    print(f"parsed arguments: {cmd_args_dict}")

    task_function = task_functions.get(cmd_args_dict["task"])
    task_function(config, **(cmd_args_dict))


__all__ = ["manager"]


# parser.add_argument(
#     "--sample_data_from_model_path", help="Get sample data indexs from a model instead of generate", required=False, default="", type=str
# )
# parser.add_argument("--sample_data_seed", required=False, default=114514, type=int)
# parser.add_argument("--sample_data_ratio", required=False, default=0.01, type=float)

# parser.add_argument("--gen-regression-result", required=False, action="store_true")


# if cmd_args.task == "model_prediction":
#     model_prediction(config, **(cmd_args_dict))

# if cmd_args.task == "analyse_model_prediction":
#     analyse_model_prediction(config, **(cmd_args_dict))

# if cmd_args.task == "dataset_target_distribution":
#     dataset_target_distribution(config, **(cmd_args_dict))

# if cmd_args.task == "regression_analysis":
#     regression_analysis(config, **(cmd_args_dict))
