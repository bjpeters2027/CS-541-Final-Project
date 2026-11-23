import sys
from pathlib import Path
import argparse
import random

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import json

from src.utils.enums.embedding_enums import Embedding

from src.cnn_embed import CNNEmbed, initialize_model as initialize_model_embed
from src.cnn_noembed import initialize_model as initialize_model_noembed

from src.utils.enums.model_name_enums import Models
from src.utils.enums.path_enums import Paths

from src.utils.load_config import load_config
from src.utils.load_data import close_loaders, load_data
from src.utils.load_model import save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forward_and_loss(model, inputs, labels, criterion):
    if isinstance(model, CNNEmbed) and isinstance(criterion, nn.CosineEmbeddingLoss):
        embeddings = model.encode_images(inputs)
        targets = model.class_embeddings[labels]
        positive_pairs = embeddings.new_ones(labels.shape)
        loss = criterion(embeddings, targets, positive_pairs)
        return model.score_embeddings(embeddings), loss
    logits = model(inputs)
    return logits, criterion(logits, labels)

def test_model(model, test_data, criterion):
    """
    Test the given model using the provided data loader and loss criterion.

    Args:
        model (_type_): _description_
        test_data (_type_): _description_
        criterion (_type_): _description_
    """
    
    running_test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_data:
            inputs, labels = inputs.to(device), labels.to(device)
            
            logits, loss = forward_and_loss(model, inputs, labels, criterion)
            running_test_loss += loss.item() * labels.size(0)
            
            _, predictions = torch.max(torch.softmax(logits, dim=1), 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    if total == 0:
        raise ValueError("Evaluation loader must contain at least one sample.")
    test_loss = running_test_loss / total
    test_accuracy = correct / total
    
    return test_loss, test_accuracy

def train_model(model, train_data, val_data, criterion, optimizer, epochs):
    """
    Train the given model using the provided data loader, loss criterion, and optimizer for a speecified number of epochs.

    Args:
        model (nn.Module): The model we are training
        train_data (dl.DataLoader): The data loader containing the training data
        val_data (dl.DataLoader): Held-out validation data, never final test data
        criterion (nn.Module): The loss function used compute the loss
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters
        epochs (int): The number of epochs we will train over
    """
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    if epochs < 1:
        raise ValueError("epochs must be positive.")
    for epoch in range(epochs):
        running_train_loss = 0
        total = 0
        model.train()
        for inputs, labels in train_data:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            _, loss = forward_and_loss(model, inputs, labels, criterion)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * labels.size(0)
            total += labels.size(0)
            
        if total == 0:
            raise ValueError("Training loader must contain at least one sample.")
        train_losses.append(running_train_loss / total)
        
        val_loss, val_accuracy = test_model(model, val_data, criterion)
                
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
                
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {train_losses[-1]:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    
    return train_losses, val_losses, val_accuracies

def save_metrics(train_losses, val_losses, val_accuracies, test_loss, test_accuracy, filename="metrics"):
    """
    Save the training and testing metrics to a JSON file.

    Args:
        train_losses (list): List of training losses for each epoch.
        val_losses (list): List of validation losses for each epoch.
        val_accuracies (list): List of validation accuracies for each epoch.
        test_loss (float): Final held-out test loss.
        test_accuracy (float): Final held-out test accuracy.
        filename (str, optional): Name of the JSON file to save the metrics. Defaults to "metrics".
    """
    
    output_path=f"{Paths.RESULTS}/{filename}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accuracies": val_accuracies,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f)

def train_experiment(experimentname, criterion, model, config, modelArchitecture, embeddingModel):
    """
    Run the experiment: trains models, saves results.
    """
    
    model = model.to(device)
    
    data_config = config["data"]
    training_args = config["training_args"]
    hyperparameters = config["hyperparameters"]
    
    train_loader, test_loader, val_loader = load_data(
        data_config, training_args["batch_size"], config.get("seed", 541)
    )
    
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
        epochs = training_args["epochs"]
        train_losses, val_losses, val_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
        test_loss, test_accuracy = test_model(model, test_loader, criterion)
    finally:
        close_loaders(train_loader, test_loader, val_loader)
    
    save_metrics(train_losses, val_losses, val_accuracies, test_loss, test_accuracy, filename=f"{experimentname}_metrics")
    
    save_model(model, f"{experimentname}_model", modelArchitecture, config=config, experimentName=experimentname, embeddingModel=embeddingModel)

def initializer(model_name, embedding_choice=None, loss_choice=None, config_path=None):
    if model_name not in (Models.CNN_EMBED, Models.CNN_NOEMBED):
        raise ValueError(f"Unsupported model: {model_name!r}")
    config = load_config(config_path or f"{model_name}_config")
    seed = config.get("seed", 541)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    embeddingModel = None
    
    if model_name == Models.CNN_EMBED:
        modelArchitecture = Models.CNN_EMBED
        
        if str(embedding_choice) == "1":
            embeddingModel = Embedding.WORD2VEC
            model = initialize_model_embed(Embedding.WORD2VEC, config)
        elif str(embedding_choice) == "2":
            embeddingModel = Embedding.BERT
            model = initialize_model_embed(Embedding.BERT, config)
        else:
            print("Invalid embedding choice. Please select either 1 or 2.")
            sys.exit(1)
        
        if str(loss_choice) == "1":
            criterion = nn.CrossEntropyLoss()
        elif str(loss_choice) == "2":
            criterion = nn.CosineEmbeddingLoss()
        else:
            print("Invalid loss choice. Please select either 1 or 2.")
            sys.exit(1)
    
    else:
        modelArchitecture = Models.CNN_NOEMBED
        model = initialize_model_noembed(config)
        criterion = nn.CrossEntropyLoss()
        
    return criterion, model, config, modelArchitecture, embeddingModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model and evaluate the final held-out test set.")
    parser.add_argument("experiment")
    parser.add_argument("model", choices=[Models.CNN_EMBED.value, Models.CNN_NOEMBED.value])
    parser.add_argument("embedding", nargs="?", choices=["1", "2"])
    parser.add_argument("loss", nargs="?", choices=["1", "2"])
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    train_experiment(args.experiment, *initializer(args.model, args.embedding, args.loss, args.config))