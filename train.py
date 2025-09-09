"""
Training module for Plant Disease Detection
Handles model training with callbacks, optimization, and monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from tqdm import tqdm
import wandb
import mlflow
import mlflow.pytorch

from model import ModelFactory, get_model_summary
from data_loader import PlantDiseaseDataLoader
from preprocess import ImagePreprocessor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping callback to prevent overfitting"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, restore_best_weights: bool = True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model checkpoint"""
        self.best_weights = model.state_dict().copy()

class ModelCheckpoint:
    """Model checkpoint callback"""
    
    def __init__(self, filepath: str, monitor: str = 'val_accuracy', mode: str = 'max', save_best_only: bool = True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best_score = None
        self.monitor_op = np.greater if mode == 'max' else np.less
        
    def __call__(self, score: float, model: nn.Module, optimizer: optim.Optimizer, epoch: int):
        if self.best_score is None or self.monitor_op(score, self.best_score):
            self.best_score = score
            self.save_model(model, optimizer, epoch, score)
            return True
        return False
    
    def save_model(self, model: nn.Module, optimizer: optim.Optimizer, epoch: int, score: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'score': score,
            'timestamp': datetime.now().isoformat()
        }
        torch.save(checkpoint, self.filepath)
        logger.info(f"Model saved to {self.filepath} with {self.monitor}={score:.4f}")

class TrainingMetrics:
    """Track training metrics and history"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epochs = []
    
    def update(self, epoch: int, train_loss: float, train_acc: float, 
               val_loss: float, val_acc: float, lr: float):
        """Update metrics for current epoch"""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
    
    def get_best_epoch(self, metric: str = 'val_accuracy') -> int:
        """Get epoch with best metric value"""
        if metric == 'val_accuracy':
            return np.argmax(self.val_accuracies)
        elif metric == 'val_loss':
            return np.argmin(self.val_losses)
        else:
            raise ValueError(f"Unknown metric: {metric}")
    
    def plot_history(self, save_path: Optional[str] = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='Training Loss')
        axes[0, 0].plot(self.epochs, self.val_losses, 'r-', label='Validation Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        axes[0, 1].plot(self.epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[1, 0].plot(self.epochs, self.learning_rates, 'g-', label='Learning Rate')
        axes[1, 0].set_title('Learning Rate Schedule')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        axes[1, 0].set_yscale('log')
        
        # Combined plot
        ax2 = axes[1, 1].twinx()
        axes[1, 1].plot(self.epochs, self.train_losses, 'b-', label='Train Loss')
        axes[1, 1].plot(self.epochs, self.val_losses, 'r-', label='Val Loss')
        ax2.plot(self.epochs, self.train_accuracies, 'b--', label='Train Acc')
        ax2.plot(self.epochs, self.val_accuracies, 'r--', label='Val Acc')
        axes[1, 1].set_title('Training Overview')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss', color='b')
        ax2.set_ylabel('Accuracy', color='r')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Training history plot saved to {save_path}")
        
        plt.show()

class PlantDiseaseTrainer:
    """Main trainer class for plant disease detection"""
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        class_names: List[str],
        experiment_name: str = "plant_disease_detection",
        use_wandb: bool = False,
        use_mlflow: bool = False
    ):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.experiment_name = experiment_name
        self.use_wandb = use_wandb
        self.use_mlflow = use_mlflow
        
        # Move model to device
        self.model.to(device)
        
        # Initialize metrics
        self.metrics = TrainingMetrics()
        
        # Setup experiment tracking
        if use_wandb:
            wandb.init(project=experiment_name)
            wandb.watch(self.model)
        
        if use_mlflow:
            mlflow.set_experiment(experiment_name)
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            output = self.model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100. * correct / total:.2f}%'
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = criterion(output, target)
                
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        scheduler_type: str = 'reduce_on_plateau',
        early_stopping_patience: int = 10,
        checkpoint_path: str = 'best_model.pth'
    ) -> Dict[str, Any]:
        """Main training loop"""
        
        # Setup optimizer and criterion
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Setup scheduler
        if scheduler_type == 'reduce_on_plateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        elif scheduler_type == 'cosine':
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler_type == 'onecycle':
            scheduler = OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_loader), epochs=num_epochs)
        else:
            scheduler = None
        
        # Setup callbacks
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', mode='max')
        
        # Training loop
        start_time = time.time()
        best_val_acc = 0.0
        
        logger.info(f"Starting training for {num_epochs} epochs...")
        logger.info(f"Model: {self.model.__class__.__name__}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, criterion)
            
            # Validate
            val_loss, val_acc = self.validate_epoch(val_loader, criterion)
            
            # Update learning rate
            current_lr = optimizer.param_groups[0]['lr']
            if scheduler_type == 'reduce_on_plateau':
                scheduler.step(val_acc)
            elif scheduler_type in ['cosine', 'onecycle']:
                scheduler.step()
            
            # Update metrics
            self.metrics.update(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
            
            # Log metrics
            epoch_time = time.time() - epoch_start
            logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                       f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% - "
                       f"LR: {current_lr:.6f} - Time: {epoch_time:.2f}s")
            
            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': current_lr
                })
            
            # Log to mlflow
            if self.use_mlflow:
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'learning_rate': current_lr
                }, step=epoch)
            
            # Checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_checkpoint(val_acc, self.model, optimizer, epoch)
            
            # Early stopping
            if early_stopping(val_loss, self.model):
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Training completed
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time:.2f}s")
        logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
        
        # Save final model
        final_checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'class_names': self.class_names,
            'best_val_acc': best_val_acc,
            'total_epochs': epoch + 1,
            'training_time': total_time
        }
        torch.save(final_checkpoint, f'final_model_{self.experiment_name}.pth')
        
        # Plot training history
        self.metrics.plot_history(f'training_history_{self.experiment_name}.png')
        
        return {
            'best_val_acc': best_val_acc,
            'total_epochs': epoch + 1,
            'training_time': total_time,
            'metrics': self.metrics
        }

def main_training_pipeline():
    """Main training pipeline"""
    # Configuration
    config = {
        'data_dir': '/Users/lesa/Documents/plant_diseaase_detection/data/raw/plantdisease/PlantVillage',
        'model_type': 'resnet50',
        'num_classes': 15,
        'img_size': 224,
        'batch_size': 32,
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'scheduler_type': 'reduce_on_plateau',
        'early_stopping_patience': 10,
        'use_wandb': False,
        'use_mlflow': False
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load data
        logger.info("Loading data...")
        data_loader = PlantDiseaseDataLoader(
            data_dir=config['data_dir'],
            img_size=config['img_size'],
            batch_size=config['batch_size']
        )
        
        train_loader, val_loader, test_loader = data_loader.load_data()
        class_names = data_loader.class_names
        
        # Create model
        logger.info("Creating model...")
        model = ModelFactory.create_model(
            model_type=config['model_type'],
            num_classes=len(class_names),
            pretrained=True,
            freeze_backbone=False
        )
        
        # Print model summary
        model_summary = get_model_summary(model)
        logger.info(f"Model summary: {model_summary}")
        
        # Create trainer
        trainer = PlantDiseaseTrainer(
            model=model,
            device=device,
            class_names=class_names,
            experiment_name=f"plant_disease_{config['model_type']}",
            use_wandb=config['use_wandb'],
            use_mlflow=config['use_mlflow']
        )
        
        # Train model
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config['num_epochs'],
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            scheduler_type=config['scheduler_type'],
            early_stopping_patience=config['early_stopping_patience']
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Results: {results}")
        
        return trainer, results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main_training_pipeline()
