import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
import os

def plot_loss_accuracy(train_losses, val_losses, train_accs, val_accs, save_path=None):
    """
    Plot training and validation loss and accuracy.
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def visualize_predictions(model, test_loader, class_names, device, num_samples=5, save_dir=None):
    """
    Visualize sample predictions.
    """
    model.eval()
    images, labels, preds = [], [], []
    
    with torch.no_grad():
        for batch_images, batch_labels in test_loader:
            batch_images = batch_images.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_images)
            _, batch_preds = torch.max(outputs, 1)
            
            for i in range(len(batch_images)):
                images.append(batch_images[i].cpu())
                labels.append(batch_labels[i].item())
                preds.append(batch_preds[i].item())
                
                if len(images) >= num_samples:
                    break
            
            if len(images) >= num_samples:
                break
    
    # Denormalize images for visualization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    plt.figure(figsize=(15, 3 * num_samples))
    
    for i in range(len(images)):
        denorm_img = images[i] * std + mean
        denorm_img = denorm_img.permute(1, 2, 0).numpy()
        denorm_img = np.clip(denorm_img, 0, 1)
        
        plt.subplot(num_samples, 1, i + 1)
        plt.imshow(denorm_img)
        title = f"True: {class_names[labels[i]]}, Pred: {class_names[preds[i]]}"
        if labels[i] == preds[i]:
            plt.title(title, color='green')
        else:
            plt.title(title, color='red')
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        save_path = os.path.join(save_dir, 'sample_predictions.png')
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_learning_rate(lrs, losses, save_path=None):
    """
    Plot the learning rate finder results.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(lrs, losses)
    plt.xscale('log')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate vs. Loss')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def plot_class_distribution(train_loader, val_loader, test_loader, class_names, save_path=None):
    """
    Plot class distribution in train, validation, and test sets.
    """
    def count_labels(dataloader):
        counter = [0] * len(class_names)
        for _, labels in dataloader:
            for label in labels:
                counter[label.item()] += 1
        return counter
    
    train_counts = count_labels(train_loader)
    val_counts = count_labels(val_loader)
    test_counts = count_labels(test_loader)
    
    width = 0.25
    x = np.arange(len(class_names))
    
    plt.figure(figsize=(12, 6))
    plt.bar(x - width, train_counts, width, label='Train')
    plt.bar(x, val_counts, width, label='Validation')
    plt.bar(x + width, test_counts, width, label='Test')
    
    plt.xlabel('Classes')
    plt.ylabel('Number of Images')
    plt.title('Class Distribution')
    plt.xticks(x, class_names)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()