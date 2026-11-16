import yaml

from src.utils.enums.path_enums import Paths

def load_config(config_name):
    file_path = f"{Paths.CONFIGS}/{config_name}.yaml"
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config