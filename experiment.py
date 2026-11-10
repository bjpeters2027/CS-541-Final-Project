import sys
from pathlib import Path
import torch.nn as nn

from src.cnn_embed import CNNEmbed
from src.cnn_noembed import CNNNoEmbed

from src.cnn_embed import initialize_model as initialize_model_embed
from src.cnn_noembed import initialize_model as initialize_model_noembed

from src.utils.enums.embedding_enums import Embedding
from src.utils.enums.model_name_enums import Models
from src.utils.enums.path_enums import Paths

from src.train import train_experiment
from src.utils.load_config import load_config

def initializer(model_name, embedding_choice, loss_choice):
    embeddingModel = None
    
    if model_name == Models.CNN_EMBED:
        modelArchitecture = Models.CNN_EMBED
        config = load_config(f"{Models.CNN_EMBED}_config")
        
        if embedding_choice == 1:
            embeddingModel = Embedding.WORD2VEC
            model = initialize_model_embed(Embedding.WORD2VEC, config)
        elif embedding_choice == 2:
            embeddingModel = Embedding.BERT
            model = initialize_model_embed(Embedding.BERT, config)
        else:
            print("Invalid embedding choice. Please select either 1 or 2.")
            sys.exit(1)
        
        if loss_choice == 1:
            criterion = nn.CrossEntropyLoss()
        elif loss_choice == 2:
            criterion = nn.CosineEmbeddingLoss()
        else:
            print("Invalid loss choice. Please select either 1 or 2.")
            sys.exit(1)
    
    else:
        modelArchitecture = Models.CNN_NOEMBED
        config = load_config(f"{Models.CNN_NOEMBED}_config")
        
        model = initialize_model_noembed(config)
        
    return criterion, model, config, modelArchitecture, embeddingModel
    

if __name__ == "__main__":
    if (len(sys.argv) < 2):
        print("Usage: python experiment.py <model> [train] [experiment_name]")
        sys.exit(1)
        
    model_choice = sys.argv[1]
    train = sys.argv[2].lower() == "train" if len(sys.argv) > 2 else False
    experiment_name = sys.argv[3] if len(sys.argv) > 3 else Models.CNN_EMBED if model_choice == Models.CNN_EMBED else Models.CNN_NOEMBED if model_choice == Models.CNN_NOEMBED else "experiment"
    
    if model_choice not in [Models.CNN_EMBED, Models.CNN_NOEMBED]:
        print(f"Invalid model. Choose either '{Models.CNN_EMBED}' or '{Models.CNN_NOEMBED}'.")
        sys.exit(1)
    else:
        if train:
            if model_choice == Models.CNN_EMBED:
                print("Select Embedding Model:")
                print("1. Word2Vec")
                print("2. BERT")
                embedding_choice = int(input("Enter your choice (1 or 2): "))
                
                print("Select Loss:")
                print("1. Cross-Entropy Loss")
                print("2. Cosine Similarity Loss")
                loss_choice = int(input("Enter your choice (1 or 2): "))
            train_experiment(experiment_name, *initializer(model_choice, embedding_choice, loss_choice))
            
        else:
            model_path = Path(Paths.CLASSIFIERS)
            print("Select the model to evaluate:")
            for idx, model_file in enumerate(model_path.glob("*.pt")):
                print(f"{idx + 1}. {model_file.name}")
            model_choice = int(input("Enter your choice: ")) - 1
            model_file = list(model_path.glob("*.pt"))[model_choice]

            # TODO: Load model and evaluate it