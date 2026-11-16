import yaml
from pathlib import Path

from src.utils.enums.path_enums import Paths

def load_config(config_name):
    candidate = Path(config_name)
    file_path = candidate if candidate.suffix in (".yaml", ".yml") else Path(Paths.CONFIGS) / f"{config_name}.yaml"
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config