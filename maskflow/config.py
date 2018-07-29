import importlib_resources

import yaml


def get_default_config():
    """Get the default configuration of the Maskflow model.
    
    Returns:
        The default configuration as a `dict`.
    """
    with importlib_resources.open_text('maskflow', 'config.yml') as f:
        default_config = yaml.load(f.read())
    return default_config
    

def load_config(config_path=None):
    """Load a Maskflow configuration from a YAML file.
    
    Args:
        config_path: The path to the config file. Missing configuration keys
                     are filled by the default configuration from `get_default_config()`.
                     
    Returns:
        The configuration as a `dict`.
                
    """
    config = get_default_config()
    if config_path:
        with open(config_path) as f:
            config.update(yaml.load(f.read()))
    return config


def save_config(config, config_path):
    """Save a Maskflow configuration to a YAML file.
    
    Args:
        config: The configuration as `dict`.
        config_path: The path to the config file.
    """
    with open(config_path, "w") as f:
        yaml.dump(config, f)

