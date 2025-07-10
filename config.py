"""
Configuration parameters for training.
"""

# Training parameters
BATCH_SIZE = 128
NUM_EPOCHS = 15
LEARNING_RATE = 7e-5

# Model parameters
MODEL_CONFIG = {
    'upscale': 1,
    'in_chans': 1,
    'img_size': (64, 64),
    'window_size': 8,
    'img_range': 1.,
    'depths': [6, 6, 6, 6, 6, 6],
    'embed_dim': 180,
    'num_heads': [6, 6, 6, 6, 6, 6],
    'mlp_ratio': 2,
    'upsampler': '',
    'resi_connection': '1conv'
} 