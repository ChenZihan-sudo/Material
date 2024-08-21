from ray import tune

__all__ = {"make_tune_config"}

custom_func = "fc_dim"


def fc_dim(config):
    res = {"num_layer": tune.choice(config[0]), "dim": tune.choice(config[1])}
    return res


def get_and_modify_nested_value(config, keys, new_value=None):
    if len(keys) == 1:
        if new_value is not None:
            config[keys[0]] = new_value
        return config[keys[0]]
    else:
        return get_and_modify_nested_value(config[keys[0]], keys[1:], new_value)


def parse_tune_hyperparamters(config: list):
    method = config[0]
    params = config[1:]

    if method not in custom_func:
        parsed = getattr(tune, method)(*params)
        return parsed
    else:
        import tuning

        parsed = getattr(tuning.prepare, method)(params)
        return parsed


# The original config.yml settings that overlap with the Tune configurations
# will be replaced by the Tune configurations.
def make_tune_config(original_config, tune_config, key_list: list = []):
    for key, config in tune_config.items():

        if key == "tune":
            parsed_value = parse_tune_hyperparamters(config)
            get_and_modify_nested_value(original_config, key_list, parsed_value)

        if isinstance(config, dict):
            inner_list = key_list.copy()
            inner_list.append(key)
            make_tune_config(original_config, config, inner_list)

    return original_config
