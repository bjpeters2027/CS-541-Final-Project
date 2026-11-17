from pathlib import Path

import torch

from src.cnn_embed import initialize_model as initialize_model_embed
from src.cnn_noembed import initialize_model as initialize_model_noembed

from src.utils.enums.path_enums import Paths
from src.utils.enums.embedding_enums import Embedding
from src.utils.enums.model_name_enums import Models
from src.utils.enums.model_save_enums import Saves

def load_model(filename):
    """
    
    Load a saved model's state dictionary into an instantiated architecture.

    Args:
        modelType (nn.Module): An instance of model class.
        filename (str): Name of the saved file.
    """
    input_path = f"{Paths.MODELS}/{filename}.pt"
    
    save = torch.load(input_path, map_location=torch.device('cpu'))
    
    if save[Saves.ARCHITECTURE] == Models.CNN_EMBED:
        model = initialize_model_embed(save[Saves.EMBEDDING], save[Saves.CONFIG])
    else:
        model = initialize_model_noembed(save[Saves.CONFIG])
    
    model.load_state_dict(save[Saves.STATE])
    model.eval()
    
    return model

def save_model(model, filename, modelArchitecture, config=None, experimentName=None, embeddingModel=None):
    """
    Save the model to a specified path.

    Args:
        model (nn.Module): The model to be saved.
        filename (str): Name of the file to save the model. Defaults to "model".
        modelArchitecture (str): Name of the model architecture.
        config (dict): loaded configuration dictionary. Default (None)
        experimentName (str): name of the experiment. Default (None)
        embeddingModel (str): name of the embedding model being used, if applicable. Default (None)
    """
    
    Path(Paths.MODELS).mkdir(parents=True, exist_ok=True)
    output_path=f"{Paths.MODELS}/{filename}.pt"
    # Plain string keys and values, not enum members, so the checkpoint loads with torch.load's default weights_only=True.
    save = {
        Saves.ARCHITECTURE.value: str(modelArchitecture),
        Saves.EMBEDDING.value: None if embeddingModel is None else str(embeddingModel),
        Saves.CONFIG.value: config,
        Saves.EXPERIMENT.value: experimentName,
        Saves.STATE.value: model.state_dict()
    }
    torch.save(save, output_path)