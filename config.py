# config.py
import os
import torch
from datetime import datetime

class Config:
    # Dataset parameters
    DATA_ROOT = 'data/Different-Terrain-Types'
    CLASSES = ['desert', 'forest', 'mountain', 'plains']
    NUM_CLASSES = len(CLASSES)
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    
    # Training parameters
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 30
    LEARNING_RATE = 0.001
    MOMENTUM = 0.9
    WEIGHT_DECAY = 1e-4
    
    # Model parameters
    MODEL_NAME = 'resnet50'  # Options: 'resnet50', 'mobilenet_v2', 'efficientnet_b0'
    PRETRAINED = True
    
    # Paths
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    MODEL_SAVE_DIR = os.path.join('models', 'saved_models')
    MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, f'{MODEL_NAME}_{TIMESTAMP}.pth')
    RESULTS_DIR = os.path.join('results', TIMESTAMP)
    
    # Create directories if they don't exist
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)