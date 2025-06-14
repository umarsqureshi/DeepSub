"""
Main training script for the SwinIR model.
"""

print("Importing libraries...")

import torch
from torch.utils.data import DataLoader
from model import SwinIR as net
from dataset import JetImageDataset
from trainer import train_model
from config import BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, MODEL_CONFIG

def main():
    """
    Main function to run the training process.
    """
    print("Loading datasets...")

    # Load datasets
    train_dataset = torch.load('datasets/train.pt', weights_only=False)
    val_dataset = torch.load('datasets/val.pt', weights_only=False)

    print("Datasets loaded.")
    print("Creating dataloaders...")

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("Dataloaders created.")

    # Setup device
    if torch.cuda.is_available():
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        device = torch.device('cuda')
    else:
        print("No GPUs available.")
        device = torch.device('cpu')

    print("Instantiating model...")

    # Create model
    model = net(**MODEL_CONFIG).to(device)
    
    if torch.cuda.device_count() > 1:
        print("Using multiple GPUs.")
        model = torch.nn.DataParallel(model)

    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print("Model instantiated.")

    # Train model
    train_model(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        device=device
    )

if __name__ == "__main__":
    main()

