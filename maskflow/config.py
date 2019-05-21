import collections
from pathlib import Path

import importlib_resources as resources
import yaml


def update_config(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update_config(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_default_config():
    with resources.open_text('maskflow', 'config.yaml') as f:
        config = yaml.load(f.read(), Loader=yaml.SafeLoader)
    return config


def load_config(config_path=None):
    config = get_default_config()
    if config_path:
        with open(config_path) as f:
            loaded_config = yaml.load(f.read(), Loader=yaml.SafeLoader)
            config = update_config(config, loaded_config)
    return config


def save_config(config, config_path):
    with open(config_path, "w") as f:
        yaml.dump(dict(config), f, default_flow_style=False)
