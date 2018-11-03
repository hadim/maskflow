import collections
from pathlib import Path

try:
    import importlib_resources as resources
except:
    import importlib.resources as resources
from maskrcnn_benchmark.config import cfg

import yaml


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def get_default_config():
    with resources.open_text('maskflow', 'config.yaml') as f:
        config = update(cfg, yaml.load(f.read()))
    return config


def load_config(config_path=None):
    config = get_default_config()
    if config_path:
        with open(config_path) as f:
            config = update(config, yaml.load(f.read()))
    # TODo: fix theis ugly hack
    config.PATHS_CATALOG = str(Path(__file__).parent.absolute() / "paths_catalog.py")
    return config


def save_config(config, config_path):
    with open(config_path, "w") as f:
        yaml.dump(dict(config), f, default_flow_style=False)
