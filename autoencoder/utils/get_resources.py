import importlib.resources as pkg_resources
import yaml
import os

import autoencoder.config
import autoencoder.input_data_raw


def get_config() -> dict:
    """
    Gets the config as dictionary.
    """
    with pkg_resources.open_text(autoencoder.config, 'config.yml') as config_file:
        config = yaml.safe_load(config_file)

    return config


def get_path_to_working_dir() -> str:
    config = get_config()
    return config["working_dir"]


def get_path_to_input_data() -> str:
    with pkg_resources.path(autoencoder.input_data_raw, '__init__.py') as init_file:
        path = os.path.dirname(init_file)
    return path
