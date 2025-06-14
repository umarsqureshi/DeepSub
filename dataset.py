"""
Dataset implementation for jet image processing.
This module contains the custom dataset class for handling jet images.
"""

import torch
from torch.utils.data import Dataset

class JetImageDataset(Dataset):
    """
    Custom dataset class for jet images.
    
    This dataset handles pairs of input and filtered images for training.
    """
    
    def __init__(self, input_images, truth_images):
        """
        Initialize the dataset with input and filtered images.
        
        Args:
            input_images (numpy.ndarray): Input images array
            filtered_images (numpy.ndarray): Target truth images array
        """
        self.X = torch.tensor(input_images, dtype=torch.float32).unsqueeze(1)
        self.Y = torch.tensor(truth_images, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        """
        Returns the total number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.X)

    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        
        Args:
            idx (int): Index of the sample to return
            
        Returns:
            tuple: (input_image, target_image) pair
        """
        return self.X[idx], self.Y[idx] 
