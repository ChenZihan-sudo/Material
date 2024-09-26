from .model_prediction import model_prediction, analyse_model_prediction
from .dataset_target_distribution import dataset_target_distribution

# from .sample_data import sample_data


def manager(config, args):
    """
    Args:
        - config: the whole config dictionary comes from the main.py
        - args: extra arguments used for parser in this file
    """
    import argparse

    parser = argparse.ArgumentParser(description="sample and analysis by inference a model")

    parser.add_argument("-C", "--config_path", default="config.yml", type=str, help="config file path (default: config.yml)")
    parser.add_argument("-M", "--model_path", required=True, type=str, help="model path for inference (<model path>/checkpoint.pt)")
    parser.add_argument("-D", "--dataset_name", required=True, type=str, help="dataset type (HypoDataset or OptimizedHypoDataset)")
    parser.add_argument("-B", "--batch_size", required=False, default=500, type=int)
    parser.add_argument("-G", "--generation", default="1G", type=str, help="generation tag of the model (default:1G)")

    # task configuration

    # model_prediction: generate model prediction results from a dataset and save the results to the model path
    # analyse_model_prediction: analyse the model results, generate a distribution histogram
    # dataset_target_distribution: generate dataset target(formation energy) results and save the results to the dataset path. Analyse the target distribution, show it by a histogram.
    # regression_figure: 
    # sample_data: sample the hypothesis compound data from the hypothesis dataset
    parser.add_argument(
        "-T", "--task", default="", type=str, help="task name (model_prediction, analyse_model_prediction, sample_data)", required=True
    )

    # parser.add_argument(
    #     "--sample_data_from_model_path", help="Get sample data indexs from a model instead of generate", required=False, default="", type=str
    # )
    # parser.add_argument("--sample_data_seed", required=False, default=114514, type=int)
    # parser.add_argument("--sample_data_ratio", required=False, default=0.01, type=float)

    # parser.add_argument("--gen-regression-result", required=False, action="store_true")

    cmd_args = parser.parse_args(args)
    cmd_args_dict = cmd_args.__dict__

    if cmd_args.task == "model_prediction":
        model_prediction(config, **(cmd_args_dict))

    if cmd_args.task == "analyse_model_prediction":
        analyse_model_prediction(config, **(cmd_args_dict))

    if cmd_args.task == "dataset_target_distribution":
        dataset_target_distribution(config, **(cmd_args_dict))


__all__ = ["manager", "model_prediction", "analyse_model_prediction"]
