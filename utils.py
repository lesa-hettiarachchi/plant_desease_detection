"""
Utility functions for Plant Disease Detection
Common helper functions and utilities
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import random
import os
from PIL import Image
import cv2

def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    """Get the best available device"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Metal Performance Shaders (MPS)")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device

def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count model parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': frozen_params
    }

def format_time(seconds: float) -> str:
    """Format time in human-readable format"""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        return f"{seconds/60:.2f}m"
    else:
        return f"{seconds/3600:.2f}h"

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   epoch: int, loss: float, accuracy: float, 
                   filepath: str, **kwargs):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
        'timestamp': datetime.now().isoformat(),
        **kwargs
    }
    
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath: str, model: nn.Module, 
                   optimizer: Optional[torch.optim.Optimizer] = None) -> Dict[str, Any]:
    """Load model checkpoint"""
    if not Path(filepath).exists():
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"Checkpoint loaded from {filepath}")
    return checkpoint

def create_logger(name: str, log_file: Optional[str] = None, 
                 level: str = 'INFO') -> logging.Logger:
    """Create logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger

def plot_training_history(history: Dict[str, List[float]], 
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (15, 5)):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Training Loss', color='blue')
    axes[0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Training Accuracy', color='blue')
    axes[1].plot(history['val_acc'], label='Validation Accuracy', color='red')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning rate plot
    if 'lr' in history:
        axes[2].plot(history['lr'], label='Learning Rate', color='green')
        axes[2].set_title('Learning Rate')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Learning Rate')
        axes[2].legend()
        axes[2].grid(True)
        axes[2].set_yscale('log')
    else:
        axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()

def calculate_model_size(model: nn.Module) -> Dict[str, float]:
    """Calculate model size in MB"""
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        'model_size_mb': model_size,
        'param_size_mb': param_size / (1024 * 1024),
        'buffer_size_mb': buffer_size / (1024 * 1024)
    }

def get_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Calculate class weights for imbalanced datasets"""
    from collections import Counter
    
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    weights = []
    for i in range(num_classes):
        if i in class_counts:
            weight = total_samples / (num_classes * class_counts[i])
            weights.append(weight)
        else:
            weights.append(1.0)
    
    return torch.FloatTensor(weights)

def visualize_predictions(images: torch.Tensor, labels: torch.Tensor, 
                         predictions: torch.Tensor, class_names: List[str],
                         num_samples: int = 8, figsize: Tuple[int, int] = (15, 10)):
    """Visualize model predictions"""
    fig, axes = plt.subplots(2, 4, figsize=figsize)
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        # Denormalize image
        img = images[i].cpu()
        img = img * 0.229 + 0.485  # Reverse ImageNet normalization
        img = torch.clamp(img, 0, 1)
        img = img.permute(1, 2, 0)
        
        # Get prediction
        pred = predictions[i].item()
        true_label = labels[i].item()
        
        # Plot image
        axes[i].imshow(img)
        axes[i].set_title(f'True: {class_names[true_label]}\nPred: {class_names[pred]}')
        axes[i].axis('off')
        
        # Color code based on correctness
        if pred == true_label:
            axes[i].spines['top'].set_color('green')
            axes[i].spines['bottom'].set_color('green')
            axes[i].spines['left'].set_color('green')
            axes[i].spines['right'].set_color('green')
        else:
            axes[i].spines['top'].set_color('red')
            axes[i].spines['bottom'].set_color('red')
            axes[i].spines['left'].set_color('red')
            axes[i].spines['right'].set_color('red')
    
    # Hide unused subplots
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def save_predictions(predictions: np.ndarray, labels: np.ndarray, 
                    class_names: List[str], filepath: str):
    """Save predictions to file"""
    results = []
    for i, (pred, label) in enumerate(zip(predictions, labels)):
        results.append({
            'sample_id': i,
            'true_class': class_names[label],
            'predicted_class': class_names[pred],
            'correct': pred == label
        })
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Predictions saved to {filepath}")

def create_directory_structure(base_dir: str):
    """Create standard directory structure for the project"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'logs',
        'results',
        'evaluation_results',
        'checkpoints',
        'frontend/static',
        'frontend/templates'
    ]
    
    for directory in directories:
        Path(base_dir) / directory
        (Path(base_dir) / directory).mkdir(parents=True, exist_ok=True)
    
    print(f"Directory structure created in {base_dir}")

def get_memory_usage() -> Dict[str, float]:
    """Get memory usage information"""
    import psutil
    
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / (1024 * 1024),  # Resident Set Size
        'vms_mb': memory_info.vms / (1024 * 1024),  # Virtual Memory Size
        'percent': process.memory_percent()
    }

def benchmark_model(model: nn.Module, input_shape: Tuple[int, ...], 
                   device: torch.device, num_runs: int = 100) -> Dict[str, float]:
    """Benchmark model inference speed"""
    model.eval()
    model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(1, *input_shape).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)
    
    # Benchmark
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(dummy_input)
    
    torch.cuda.synchronize() if device.type == 'cuda' else None
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    
    return {
        'total_time': total_time,
        'avg_time_per_inference': avg_time,
        'inferences_per_second': 1.0 / avg_time,
        'num_runs': num_runs
    }

def validate_image_file(file_path: str, max_size: int = 10 * 1024 * 1024) -> Dict[str, Any]:
    """Validate image file"""
    result = {
        'valid': False,
        'error': None,
        'size': 0,
        'format': None,
        'dimensions': None
    }
    
    try:
        # Check file size
        file_size = os.path.getsize(file_path)
        result['size'] = file_size
        
        if file_size > max_size:
            result['error'] = f"File too large: {file_size} bytes (max: {max_size})"
            return result
        
        # Check if it's a valid image
        with Image.open(file_path) as img:
            result['format'] = img.format
            result['dimensions'] = img.size
            result['valid'] = True
            
    except Exception as e:
        result['error'] = str(e)
    
    return result

def create_summary_report(model: nn.Module, config: Dict[str, Any], 
                         results: Dict[str, Any], output_path: str):
    """Create a comprehensive summary report"""
    report = {
        'timestamp': datetime.now().isoformat(),
        'model_info': {
            'type': model.__class__.__name__,
            'parameters': count_parameters(model),
            'size_mb': calculate_model_size(model)['model_size_mb']
        },
        'configuration': config,
        'results': results,
        'system_info': {
            'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            'pytorch_version': torch.__version__,
            'device': str(next(model.parameters()).device)
        }
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Summary report saved to {output_path}")

def main():
    """Main function for utility testing"""
    print("Plant Disease Detection Utilities")
    print("=" * 40)
    
    # Test device detection
    device = get_device()
    print(f"Device: {device}")
    
    # Test memory usage
    memory = get_memory_usage()
    print(f"Memory usage: {memory['rss_mb']:.2f} MB")
    
    # Test directory creation
    create_directory_structure("test_project")
    print("Directory structure created")
    
    print("Utility functions tested successfully!")

if __name__ == "__main__":
    main()
