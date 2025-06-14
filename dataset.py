"""
Dataset implementation for jet image processing.
"""

import torch
from torch.utils.data import Dataset

class JetImageDataset(Dataset):
    """
    Dataset class for jet images.
    """
    
    def __init__(self, input_images, truth_images):
        """
        Initialize the dataset with input and truth images.
        
        Args:
            input_images (numpy.ndarray): Input images array
            truth_images (numpy.ndarray): Target truth images array
        """
        self.X = torch.tensor(input_images, dtype=torch.float32).unsqueeze(1)
        self.Y = torch.tensor(truth_images, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx] 
