# dataloader.py

import json
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

class JSONDataset(Dataset):
    """
    Custom Dataset class for handling JSON data.
    Each instance of this class is used to fetch individual data points for the DataLoader.
    """
    def __init__(self, data):
        """
        Initializes the JSONDataset with loaded data.

        Args:
            data (list): List of data items, where each item is a dictionary representing a single data point.
        """
        self.data = data
    
    def __len__(self):
        """Returns the length of the dataset."""
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves a data item by index.

        Args:
            idx (int): Index of the data item to fetch.

        Returns:
            dict: A dictionary representing a data item.
        """
        return self.data[idx]

class CustomDataLoader:
    """
    Custom DataLoader class to handle loading, splitting, and batching JSON data.
    Allows optional addition of percentile fields for citation counts at specified years after publication.
    """
    def __init__(self, directory, train_size=0.7, valid_size=0.15, test_size=0.15, batch_size=32, 
                 random_state=42):
        """
        Initializes the CustomDataLoader with data loading, optional percentile calculation, and data splitting.

        Args:
            directory (str): Path to the directory containing JSON files.
            train_size (float): Proportion of data to use for training.
            valid_size (float): Proportion of data to use for validation.
            test_size (float): Proportion of data to use for testing.
            batch_size (int): Batch size for data loading.
            random_state (int): Random seed for data shuffling.
        
        Raises:
            ValueError: If train, validation, and test sizes do not sum to 1.
        """
        # Check if the splits add up to 1
        if not round(train_size + valid_size + test_size, 5) == 1.0:
            raise ValueError("Train, validation, and test sizes must sum to 1. Please adjust the sizes.")

        self.directory = directory
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size
        self.batch_size = batch_size
        self.random_state = random_state

        # Load data, add optional percentile fields, split, and create DataLoaders
        self.all_data = self._load_json_files()
        self.train_data, self.valid_data, self.test_data = self._split_data()
        self.train_loader, self.valid_loader, self.test_loader = self._create_dataloaders()

    def _load_json_files(self):
        """
        Loads all JSON files from the specified directory into a single list, except the medatadata

        Returns:
            list: A list of data items loaded from JSON files.
        """
        all_data = []
        for filename in os.listdir(self.directory):
            if filename.endswith('.json') and filename != "metadata.json":
                with open(os.path.join(self.directory, filename), 'r') as f:
                    file_data = json.load(f)
                    all_data.extend(file_data)  # Assuming each file contains a list of records
        return all_data

    def _split_data(self):
        """
        Splits data into training, validation, and test sets.

        Returns:
            tuple: Three lists corresponding to the training, validation, and test data.
        """
        train_data, test_data = train_test_split(
            self.all_data, 
            test_size=(1 - self.train_size), 
            random_state=self.random_state
        )
        valid_data, test_data = train_test_split(
            test_data, 
            test_size=(self.test_size / (self.valid_size + self.test_size)), 
            random_state=self.random_state
        )
        return train_data, valid_data, test_data

    def _create_dataloaders(self):
        """
        Creates DataLoader objects for training, validation, and test sets.

        Returns:
            tuple: DataLoader objects for training, validation, and test data.
        """
        train_loader = DataLoader(JSONDataset(self.train_data), batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(JSONDataset(self.valid_data), batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(JSONDataset(self.test_data), batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader

    def get_loaders(self):
        """
        Provides the DataLoader objects for train, validation, and test sets.

        Returns:
            tuple: DataLoader objects for training, validation, and test sets.
        """
        return self.train_loader, self.valid_loader, self.test_loader


if __name__ == "__main__":
    # Usage
    directory = "./data"
    data_loader = CustomDataLoader(
        directory,
        train_size=0.7,
        valid_size=0.15,
        test_size=0.15,
        batch_size=32,
    )
    train_loader, valid_loader, test_loader = data_loader.get_loaders()
