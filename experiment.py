import sys
from pathlib import Path
import torch.nn as nn

from src.utils.enums.model_name_enums import Models
from src.utils.enums.path_enums import Paths

from src.train import train_experiment
from src.train import initializer

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