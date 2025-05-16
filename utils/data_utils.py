import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import glob

class TerrainDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label

def get_transforms(img_size):
    # Define transforms for training data
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Define transforms for validation and test data
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def load_data(data_root, img_size, batch_size, num_workers, val_size=0.15, test_size=0.15):
    # Get class names
    class_names = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
    class_to_idx = {class_name: i for i, class_name in enumerate(class_names)}
    
    # Collect all image paths and labels
    all_image_paths = []
    all_labels = []
    
    for class_name in class_names:
        class_dir = os.path.join(data_root, class_name)
        class_idx = class_to_idx[class_name]
        
        # Get all image paths for this class
        class_image_paths = glob.glob(os.path.join(class_dir, '*.jpg')) + \
                            glob.glob(os.path.join(class_dir, '*.jpeg')) + \
                            glob.glob(os.path.join(class_dir, '*.png'))
        
        all_image_paths.extend(class_image_paths)
        all_labels.extend([class_idx] * len(class_image_paths))
    
    # Split data into train, validation, and test sets
    # First, split into train and temp (val + test)
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        all_image_paths, all_labels, test_size=val_size+test_size, stratify=all_labels, random_state=42
    )
    
    # Then split temp into val and test
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels, test_size=test_size/(val_size+test_size), stratify=temp_labels, random_state=42
    )
    
    # Get transforms
    train_transform, val_transform = get_transforms(img_size)
    
    # Create datasets
    train_dataset = TerrainDataset(train_paths, train_labels, train_transform)
    val_dataset = TerrainDataset(val_paths, val_labels, val_transform)
    test_dataset = TerrainDataset(test_paths, test_labels, val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, class_names