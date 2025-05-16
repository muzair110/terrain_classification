import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from tqdm import tqdm

from config import Config
from utils.data_utils import load_data
from utils.visualization import plot_confusion_matrix, visualize_predictions
from train import create_model

def evaluate_model(model_path, config):
    """
    Evaluate a trained model on the test set.
    """
    # Set device
    device = torch.device(config.DEVICE)
    
    # Load data
    _, _, test_loader, class_names = load_data(
        config.DATA_ROOT, 
        config.IMG_SIZE, 
        config.BATCH_SIZE, 
        config.NUM_WORKERS
    )
    
    # Create model
    model = create_model(config.MODEL_NAME, config.NUM_CLASSES, pretrained=False)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    
    # Initialize variables for evaluation
    all_preds = []
    all_labels = []
    
    # Evaluate model on test set
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Save predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Classification report
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print("Classification Report:")
    print(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        all_labels, 
        all_preds, 
        class_names, 
        save_path=os.path.join(config.RESULTS_DIR, 'test_confusion_matrix.png')
    )
    
    # Visualize predictions
    visualize_predictions(
        model, 
        test_loader, 
        class_names, 
        device, 
        num_samples=10, 
        save_dir=config.RESULTS_DIR
    )
    
    # Save classification report
    with open(os.path.join(config.RESULTS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    return accuracy, report

if __name__ == "__main__":
    import torch
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load configuration
    config = Config()
    
    # Evaluate model
    model_path = config.MODEL_SAVE_PATH  # or specify a different path to a saved model
    accuracy, report = evaluate_model(model_path, config)
    
    print("Evaluation completed!")