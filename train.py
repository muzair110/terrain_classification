import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models
import numpy as np
import time
import json
from tqdm import tqdm

from config import Config
from utils.data_utils import load_data
from utils.visualization import (
    plot_loss_accuracy, 
    plot_confusion_matrix, 
    visualize_predictions,
    plot_class_distribution
)

def create_model(model_name, num_classes, pretrained=True):
    """
    Create and return the specified model with the output layer modified for the number of classes.
    """
    if model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=pretrained)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    return model

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for one epoch."""
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # Use tqdm for progress bar
    pbar = tqdm(train_loader, desc="Training")
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)
        
        # Update progress bar
        pbar.set_postfix(loss=loss.item(), acc=running_corrects/total_samples)
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Validating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)
            
            # Save predictions and labels for confusion matrix
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects / total_samples
    
    return epoch_loss, epoch_acc, all_preds, all_labels

def train_model(config):
    """
    Main training function.
    """
    # Set device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader, test_loader, class_names = load_data(
        config.DATA_ROOT, 
        config.IMG_SIZE, 
        config.BATCH_SIZE, 
        config.NUM_WORKERS
    )
    
    # Create model
    print(f"Creating model: {config.MODEL_NAME}")
    model = create_model(config.MODEL_NAME, config.NUM_CLASSES, config.PRETRAINED)
    model = model.to(device)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), 
        lr=config.LEARNING_RATE, 
        momentum=config.MOMENTUM, 
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
    
    # Plot class distribution
    plot_class_distribution(
        train_loader, 
        val_loader, 
        test_loader, 
        class_names, 
        save_path=os.path.join(config.RESULTS_DIR, 'class_distribution.png')
    )
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    start_time = time.time()
    
    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")
        
        # Train for one epoch
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_acc, all_preds, all_labels = validate(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save stats
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print stats
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save model if validation accuracy improved
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Model saved to {config.MODEL_SAVE_PATH}")
            
            # Plot confusion matrix for best model
            plot_confusion_matrix(
                all_labels, 
                all_preds, 
                class_names, 
                save_path=os.path.join(config.RESULTS_DIR, 'confusion_matrix.png')
            )
    
    # Training time
    time_elapsed = time.time() - start_time
    print(f'\nTraining complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_val_acc:.4f}')
    
    # Plot training curves
    plot_loss_accuracy(
        train_losses, 
        val_losses, 
        train_accs, 
        val_accs, 
        save_path=os.path.join(config.RESULTS_DIR, 'training_curves.png')
    )
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    
    # Visualize sample predictions
    visualize_predictions(
        model, 
        test_loader, 
        class_names, 
        device, 
        num_samples=10, 
        save_dir=config.RESULTS_DIR
    )
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
        'best_val_acc': best_val_acc
    }
    
    with open(os.path.join(config.RESULTS_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    return model, history

if __name__ == "__main__":
    import torch
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load configuration
    config = Config()
    
    # Train model
    model, history = train_model(config)
    
    print("Training completed!")
