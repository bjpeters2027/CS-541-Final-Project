import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import dataloader as dl
from sklearn.model_selection import train_test_split
import json

from src.cnn_embed import CNNEmbed
from src.cnn_noembed import CNNNoEmbed

def load_data(data_path, batch_size, test_split=0.2, validation_split=0.1):
    """
    Load data from a CSV file and return a DataLoader for training.

    Args:
        data_path (str): Path to the CSV file containing the data.
        batch_size (int): Number of samples per batch.
        test_split (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        validation_split (float, optional): Proportion of the dataset to include in the validation split. Defaults to 0.1.
    """
    
    data = pd.read_csv(data_path)
    train_data, unseen_data = train_test_split(data, test_size=test_split+validation_split)
    test_data, val_data = train_test_split(unseen_data, test_size=validation_split/(test_split+validation_split))
    train_loader = dl.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = dl.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = dl.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader

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
        for inputs, labels in test_data:
            correct = 0
            total = 0
            
            logits = model(inputs)
            loss = criterion(logits, labels)
            running_test_loss += loss.item()
            
            _, predictions = torch.max(torch.softmax(logits), 1)
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
        test_data (dl.DataLoader): The data loader containing the testing data
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
        for inputs, labels in train_data:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_losses.append(running_loss / len(train_data))
        
        running_test_loss, running_test_accuracy = test_model(model, test_data, criterion)
                
        test_losses.append(running_test_loss / len(test_data))
        test_accuracies.append(running_test_accuracy / len(test_data))
                
        print(f"Epoch [{epoch+1}/{epochs}], Training Loss: {running_train_loss/len(train_data):.4f}, Test Loss: {running_test_loss/len(test_data):.4f}, Test Accuracy: {running_test_accuracy/len(test_data):.4f}")
    
    return train_losses, test_losses, test_accuracies

def save_metrics(train_losses, test_losses, test_accuracies, output_path="src/artifacts/results/metrics.json"):
    """
    Save the training and testing metrics to a JSON file.

    Args:
        train_losses (list): List of training losses for each epoch.
        test_losses (list): List of testing losses for each epoch.
        test_accuracies (list): List of testing accuracies for each epoch.
        output_path (str, optional): Path to save the metrics JSON file. Defaults to "src/artifacts/results/metrics.json".
    """
    
    metrics = {
        "train_losses": train_losses,
        "test_losses": test_losses,
        "test_accuracies": test_accuracies
    }
    
    with open(output_path, 'w') as f:
        json.dump(metrics, f)
        
