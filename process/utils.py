# make processed file name based on the config parameters
def make_processed_filename(dataset_name: str, args: dict):
    p = args["Process"]
    d = args["Dataset"][dataset_name]
    name_prefix = args["Dataset"][dataset_name]["processed_dir"]
    name_postfix = (
        "dcut_"
        + str("no" if p["max_cutoff_distance"] is None else p["max_cutoff_distance"])
        + "_ncut_"
        + str("no" if p["max_cutoff_neighbors"] is None else p["max_cutoff_neighbors"])
        + "_tnorm_"
        + str("no" if p["target_normalization"] is False else "yes")
        + "_efeat_"
        + str("1" if p["edge"]["normalization"] is False else p["edge"]["edge_feature"])
        + "_gwid_"
        + str("no" if p["edge"]["gaussian_smearing"]["enable"] is False else p["edge"]["gaussian_smearing"]["width"])
        # + str("") # TODO: 添加对继承参数文件数据集的识别符
    )

    # * avoid multiple auto naming
    if name_postfix not in name_prefix:
        args["Dataset"][dataset_name]["processed_dir"] = name_prefix + "_" + name_postfix
        print(f"Dataset auto naming: {name_postfix}")
    return args


# get the PARAMETER file from a processed dataset path
def get_parameter_file_path(dataset_name_from_the_get_parameter_from, args, parameter_file_name="PARAMETERS", **kwargs):
    from os import path as osp

    if args["Process"]["auto_processed_name"] is True:
        args = make_processed_filename(dataset_name_from_the_get_parameter_from, args)

    processed_data_path = args["Dataset"][dataset_name_from_the_get_parameter_from]["processed_dir"]
    return osp.join(processed_data_path, parameter_file_name)
