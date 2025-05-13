import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Enum-like class to represent different modes of the model
class ModelMode:
    TRAIN = 0
    EVAL = 1
    INFERENCE = 2

mode_to_name = {
    ModelMode.TRAIN: 'train',
    ModelMode.EVAL: 'val',
    ModelMode.INFERENCE: 'test'
}

# Custom PyTorch Dataset for stock data
class StockDataset(Dataset):
    def __init__(self, sequences, labels):
        """
        Initialize the dataset with sequences and labels.

        Args:
            sequences (numpy.ndarray): Input features.
            labels (numpy.ndarray): Labels with the first column as targets and the rest as metadata.
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)  # Convert sequences to PyTorch tensors
        self.labels = torch.tensor(labels[:, 0], dtype=torch.float32)  # Extract the first column as labels
        self.metadata = labels[:, 1:]  # Extract remaining columns as metadata
        self.columns = sequences.shape[1]  # Number of feature columns
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            tuple: (sequence, label, metadata) for the given index.
        """
        return self.sequences[idx], self.labels[idx], self.metadata[idx]

# Function to load feature dataframes and create a DataLoader
def format_feature_dataframes(tickers, mode, batch_size=32, shuffle=False):
   
    if len(tickers) > 1:
        raise ValueError("Only one ticker is currently supported.")

    # Directory containing the feature dataframes
    feature_dataframes_dir = 'feature_dataframes/preprocessed/'

    # List all files in the directory
    feature_files = os.listdir(feature_dataframes_dir)

    # Initialize containers for input features (X) and labels (y)
    dataframes = {'X': np.array([]), 'y': np.array([]), 'mean': np.array([]), 'std': np.array([])}

    # Iterate over all files in the directory and process matching tickers
    for file in feature_files:
        with open(os.path.join(feature_dataframes_dir, file), 'rb') as f:
            data = pickle.load(f)
            for key in dataframes:
                if key in data:
                    if dataframes[key].size == 0:
                        dataframes[key] = np.array(data[key])
                    else:
                        dataframes[key] = np.concatenate([dataframes[key], np.array(data[key])])

    
    

    # Create a StockDataset instance
    dataset = StockDataset(np.concatenate(dataframes['X']), np.concatenate(dataframes['y']))
    
    # Create a DataLoader for the dataset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader, dataset.columns



