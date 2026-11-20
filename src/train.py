import sys
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from src.utils.enums.embedding_enums import Embedding

from src.cnn_embed import CNNEmbed, initialize_model as initialize_model_embed
from src.cnn_noembed import initialize_model as initialize_model_noembed

from src.utils.enums.model_name_enums import Models
from src.utils.enums.path_enums import Paths

from src.utils.load_config import load_config
from src.utils.load_data import load_data
from src.utils.load_model import save_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def forward_and_loss(model, inputs, labels, criterion, temp = 1):
    if isinstance(model, CNNEmbed) and isinstance(criterion, nn.CosineEmbeddingLoss):
        embeddings = model.encode_images(inputs)
        targets = model.class_embeddings[labels]
        positive_pairs = embeddings.new_ones(labels.shape)
        loss = criterion(embeddings, targets, positive_pairs)
        return model.score_embeddings(embeddings), loss
    logits = model(inputs) / temp
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
    running_test_accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in tqdm(test_data, desc="Validation", unit="batch", leave=False, mininterval=1.0):
            inputs, labels = inputs.to(device), labels.to(device)
            
            correct = 0
            total = 0
            
            logits, loss = forward_and_loss(model, inputs, labels, criterion)
            running_test_loss += loss.item()
            
            _, predictions = torch.max(torch.softmax(logits, dim=1), 1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
            running_test_accuracy += correct / total
            
    test_loss = running_test_loss / len(test_data)
    test_accuracy = running_test_accuracy / len(test_data)
    
    return test_loss, test_accuracy

def train_model(model, train_data, test_data, criterion, optimizer, epochs):
    """
    Train the given model using the provided data loader, loss criterion, and optimizer for a speecified number of epochs.

    Args:
        model (nn.Module): The model we are training
        train_data (dl.DataLoader): The data loader containing the training data
        test_data (dl.DataLoader): The data loader evaluated after each epoch (the validation partition)
        criterion (nn.Module): The loss function used compute the loss
        optimizer (torch.optim.Optimizer): The optimizer used to update the model's parameters
        epochs (int): The number of epochs we will train over
    """
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in range(epochs):
        running_train_loss = 0
        model.train()
        for inputs, labels in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}", unit="batch", mininterval=1.0):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            _, loss = forward_and_loss(model, inputs, labels, criterion)
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item()
            
        train_losses.append(running_train_loss / len(train_data))
        
        running_test_loss, running_test_accuracy = test_model(model, test_data, criterion)
                
        test_losses.append(running_test_loss)
        test_accuracies.append(running_test_accuracy)
                
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {running_train_loss/len(train_data):.4f}, Validation Loss: {running_test_loss:.4f}, Validation Accuracy: {running_test_accuracy:.4f}", flush=True)
    
    return train_losses, test_losses, test_accuracies

def save_metrics(train_losses, test_losses, test_accuracies, filename="metrics"):
    """
    Save the training and testing metrics to a JSON file.

    Args:
        train_losses (list): List of training losses for each epoch.
        test_losses (list): List of testing losses for each epoch.
        test_accuracies (list): List of testing accuracies for each epoch.
        filename (str, optional): Name of the JSON file to save the metrics. Defaults to "metrics".
    """
    
    Path(Paths.RESULTS).mkdir(parents=True, exist_ok=True)
    output_path=f"{Paths.RESULTS}/{filename}.json"
    
    metrics = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f)

def train_experiment(experimentname, criterion, model, config, modelArchitecture, embeddingModel):
    """
    Run the experiment: trains models, saves results.
    """
    
    torch.manual_seed(config["seed"])
    model = model.to(device)
    
    data_config = config["data"]
    training_args = config["training_args"]
    hyperparameters = config["hyperparameters"]
    
    # The seen-class test partition is held out for src/evalutate.py.
    print("Loading the Data")
    train_loader, val_loader, _, num_classes = load_data(data_config, training_args["batch_size"], config["seed"])
    print("Data Loaded")
    model_classes = model.class_embeddings.shape[0] if isinstance(model, CNNEmbed) else model.classifier.out_features
    if model_classes != num_classes:
        raise ValueError(f"The model predicts {model_classes} classes but there are {num_classes} seen classes.")

    print(f"Device: {device} | Training batches: {len(train_loader)} | Validation batches: {len(val_loader)}", flush=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparameters["learning_rate"])
    
    epochs = training_args["epochs"]
    
    train_losses, test_losses, test_accuracies = train_model(model, train_loader, val_loader, criterion, optimizer, epochs)
    
    print("Model Trained, Saving Metrics")
    
    save_metrics(train_losses, test_losses, test_accuracies, filename=f"{experimentname}_metrics")
    
    print("Saving Model")
    
    save_model(model, f"{experimentname}_model", modelArchitecture, config=config, experimentName=experimentname, embeddingModel=embeddingModel)

def initializer(model_name, embedding_choice, loss_choice):
    embeddingModel = None
    
    if model_name == Models.CNN_EMBED:
        modelArchitecture = Models.CNN_EMBED
        config = load_config(f"{Models.CNN_EMBED}_config")
        
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
        config = load_config(f"{Models.CNN_NOEMBED}_config")
        
        model = initialize_model_noembed(config)
        criterion = nn.CrossEntropyLoss()
        
    return criterion, model, config, modelArchitecture, embeddingModel

if __name__ == "__main__":
    if len(sys.argv) != 5:
        sys.exit(1)
    else:
        train_experiment(sys.argv[1], *initializer(sys.argv[2], sys.argv[3], sys.argv[4]))
