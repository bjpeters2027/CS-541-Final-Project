import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.utils.enums.path_enums import Paths

def load_data(filename, batch_size, test_split=0.2, validation_split=0.1):
    """
    Load data from a CSV file and return a DataLoader for training.

    Args:
        filename (str): Name of the CSV file containing the data.
        batch_size (int): Number of samples per batch.
        test_split (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        validation_split (float, optional): Proportion of the dataset to include in the validation split. Defaults to 0.1.
    """
    
    data_path = f"{Paths.DATA}/{filename}"
    data = pd.read_csv(data_path)
    train_data, unseen_data = train_test_split(data, test_size=test_split+validation_split)
    test_data, val_data = train_test_split(unseen_data, test_size=validation_split/(test_split+validation_split))
    train_loader = dl.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = dl.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = dl.DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, val_loader