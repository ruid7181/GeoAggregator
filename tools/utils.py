import yaml

import torch
from thop import profile
from tools.classes import Dict2Obj


def get_config():
    """
    Load and process a YAML configuration file,
    returns an object representation of the flattened config dictionary.

    ### Usage Example for access configurations: ###
    ```python
    config = get_config()
    print(config.your_desired_config)
    """
    def __flatten__(dic):
        _dict_ = {}

        def dict_flatten(temp_dic, prefix):
            for key, value in temp_dic.items():
                new_prefix = '_'.join([prefix, key]).strip('_')
                if isinstance(value, dict):
                    dict_flatten(value, new_prefix)
                else:
                    _dict_[new_prefix] = value

        dict_flatten(dic, '')
        return _dict_

    config_search_files = list()
    config_search_files.append(r'.\configurations\config.yml')
    config_search_files.append(r'..\configurations\config.yml')
    config_search_files.append(r'..\..\configurations\config.yml')

    for fp in config_search_files:
        try:
            with open(fp) as f:
                config_data = f.read()
                config = yaml.safe_load(config_data)
                break
        except FileNotFoundError:
            continue
    config = __flatten__(config)

    return Dict2Obj(**config)


def param_stat(model, input_size=None, input_sample=None):
    """Total learnable parameters in a pytorch model."""
    if input_size and input_sample is None:
        raise ValueError
    if input_sample is None:
        input_sample = torch.randn(input_size)
    flops, params = profile(model=model, inputs=input_sample)

    return flops, params
