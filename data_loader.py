"""
Data loader module for Plant Disease Detection
Handles loading and preprocessing of PlantVillage dataset
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, datasets
from PIL import Image
import numpy as np
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
import random

class PlantDiseaseDataset(Dataset):
    """Custom dataset for plant disease images"""
    
    def __init__(self, image_paths, labels, class_names, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.class_names = class_names
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class PlantDiseaseDataLoader:
    """Data loader class for plant disease detection"""
    
    def __init__(self, data_dir, img_size=224, batch_size=32, num_workers=4):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.class_names = None
        self.class_to_idx = None
        self.idx_to_class = None
        
    def get_class_mapping(self):
        """Get class names and mappings from directory structure"""
        if not self.data_dir.exists():
            raise ValueError(f"Data directory {self.data_dir} does not exist")
        
        # Get all subdirectories (class names)
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        class_names = sorted([d.name for d in class_dirs])
        
        self.class_names = class_names
        self.class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        self.idx_to_class = {idx: cls for idx, cls in enumerate(class_names)}
        
        print(f"Found {len(class_names)} classes:")
        for i, cls in enumerate(class_names):
            print(f"  {i}: {cls}")
        
        return class_names, self.class_to_idx, self.idx_to_class
    
    def collect_image_paths(self):
        """Collect all image paths and their corresponding labels"""
        if self.class_names is None:
            self.get_class_mapping()
        
        image_paths = []
        labels = []
        
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            if not class_dir.exists():
                print(f"Warning: Class directory {class_dir} does not exist")
                continue
            
            # Get all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
            class_images = []
            for ext in image_extensions:
                class_images.extend(list(class_dir.glob(f'*{ext}')))
            
            print(f"Found {len(class_images)} images in {class_name}")
            
            for img_path in class_images:
                image_paths.append(str(img_path))
                labels.append(self.class_to_idx[class_name])
        
        return image_paths, labels
    
    def get_data_statistics(self, image_paths, labels):
        """Get statistics about the dataset"""
        class_counts = Counter(labels)
        
        print("\nDataset Statistics:")
        print(f"Total images: {len(image_paths)}")
        print(f"Number of classes: {len(self.class_names)}")
        print("\nClass distribution:")
        for idx, count in class_counts.items():
            class_name = self.idx_to_class[idx]
            print(f"  {class_name}: {count} images")
        
        return class_counts
    
    def create_transforms(self, is_training=True):
        """Create data transforms for training and validation"""
        if is_training:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        
        return transform
    
    def create_datasets(self, train_split=0.7, val_split=0.15, test_split=0.15, random_seed=42):
        """Create train, validation, and test datasets"""
        if not (abs(train_split + val_split + test_split - 1.0) < 1e-6):
            raise ValueError("Train, validation, and test splits must sum to 1.0")
        
        # Set random seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # Get class mapping
        self.get_class_mapping()
        
        # Collect image paths and labels
        image_paths, labels = self.collect_image_paths()
        
        # Get dataset statistics
        self.get_data_statistics(image_paths, labels)
        
        # Split data
        X_train, X_temp, y_train, y_temp = train_test_split(
            image_paths, labels, 
            test_size=(val_split + test_split), 
            random_state=random_seed,
            stratify=labels
        )
        
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=test_split/(val_split + test_split),
            random_state=random_seed,
            stratify=y_temp
        )
        
        print(f"\nData splits:")
        print(f"  Training: {len(X_train)} images")
        print(f"  Validation: {len(X_val)} images")
        print(f"  Test: {len(X_test)} images")
        
        # Create transforms
        train_transform = self.create_transforms(is_training=True)
        val_transform = self.create_transforms(is_training=False)
        
        # Create datasets
        train_dataset = PlantDiseaseDataset(X_train, y_train, self.class_names, train_transform)
        val_dataset = PlantDiseaseDataset(X_val, y_val, self.class_names, val_transform)
        test_dataset = PlantDiseaseDataset(X_test, y_test, self.class_names, val_transform)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset, test_dataset):
        """Create PyTorch DataLoaders"""
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader, test_loader
    
    def load_data(self, train_split=0.7, val_split=0.15, test_split=0.15, random_seed=42):
        """Complete data loading pipeline"""
        print("Loading Plant Disease Dataset...")
        print("=" * 50)
        
        # Create datasets
        train_dataset, val_dataset, test_dataset = self.create_datasets(
            train_split, val_split, test_split, random_seed
        )
        
        # Create dataloaders
        train_loader, val_loader, test_loader = self.create_dataloaders(
            train_dataset, val_dataset, test_dataset
        )
        
        print(f"\nData loading complete!")
        print(f"Batch size: {self.batch_size}")
        print(f"Image size: {self.img_size}x{self.img_size}")
        
        return train_loader, val_loader, test_loader

def test_data_loader():
    """Test function for the data loader"""
    # Example usage
    data_dir = "/Users/lesa/Documents/plant_diseaase_detection/data/raw/plantdisease/PlantVillage"
    
    # Create data loader
    loader = PlantDiseaseDataLoader(
        data_dir=data_dir,
        img_size=224,
        batch_size=16,
        num_workers=2
    )
    
    try:
        # Load data
        train_loader, val_loader, test_loader = loader.load_data()
        
        # Test a batch
        print("\nTesting data loader...")
        for images, labels in train_loader:
            print(f"Batch shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
            print(f"Sample labels: {labels[:5].tolist()}")
            break
        
        print("Data loader test successful!")
        return loader
        
    except Exception as e:
        print(f"Error testing data loader: {e}")
        return None

if __name__ == "__main__":
    test_data_loader()
