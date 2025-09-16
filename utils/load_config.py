import yaml


def load_config(CONFIG_PATH):
    """Function to load yaml configuration file"""
    with open(CONFIG_PATH ) as file:
        config = yaml.safe_load(file)

    return config

