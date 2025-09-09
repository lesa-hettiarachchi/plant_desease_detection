"""
Configuration module for Plant Disease Detection
Centralized configuration management
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import json
from dataclasses import dataclass, asdict

@dataclass
class DataConfig:
    """Data configuration"""
    data_dir: str = "/Users/lesa/Documents/plant_diseaase_detection/data/raw/plantdisease/PlantVillage"
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    random_seed: int = 42
    class_names: Optional[List[str]] = None

@dataclass
class ModelConfig:
    """Model configuration"""
    model_type: str = "resnet50"
    num_classes: int = 15
    pretrained: bool = True
    freeze_backbone: bool = False
    dropout_rate: float = 0.5
    input_size: List[int] = None
    
    def __post_init__(self):
        if self.input_size is None:
            self.input_size = [3, 224, 224]

@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 50
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    scheduler_type: str = "reduce_on_plateau"
    early_stopping_patience: int = 10
    checkpoint_path: str = "best_model.pth"
    final_model_path: str = "final_model.pth"
    use_wandb: bool = False
    use_mlflow: bool = False
    experiment_name: str = "plant_disease_detection"

@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    test_batch_size: int = 32
    generate_plots: bool = True
    save_predictions: bool = True
    output_dir: str = "evaluation_results"
    include_heatmap: bool = True

@dataclass
class APIConfig:
    """API configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    model_path: str = "best_model.pth"
    device: str = "auto"
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    allowed_extensions: List[str] = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    log_level: str = "INFO"
    log_file: str = "plant_disease.log"
    metrics_file: str = "metrics.json"
    enable_profiling: bool = False
    profile_output: str = "profile.prof"

class Config:
    """Main configuration class"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.data = DataConfig()
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.evaluation = EvaluationConfig()
        self.api = APIConfig()
        self.monitoring = MonitoringConfig()
        
        # Load from file if provided
        if config_file:
            self.load_from_file(config_file)
        
        # Auto-detect device
        if self.api.device == "auto":
            import torch
            self.api.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    def load_from_file(self, config_file: str):
        """Load configuration from YAML or JSON file"""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config_data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config_data = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Update configurations
        self._update_from_dict(config_data)
    
    def _update_from_dict(self, config_data: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section, values in config_data.items():
            if hasattr(self, section) and isinstance(values, dict):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
    
    def save_to_file(self, config_file: str):
        """Save configuration to file"""
        config_data = {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'training': asdict(self.training),
            'evaluation': asdict(self.evaluation),
            'api': asdict(self.api),
            'monitoring': asdict(self.monitoring)
        }
        
        config_path = Path(config_file)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.dump(config_data, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config_data, f, indent=2)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    def get_class_names(self) -> List[str]:
        """Get class names from data directory or default"""
        if self.data.class_names:
            return self.data.class_names
        
        # Try to detect from data directory
        data_path = Path(self.data.data_dir)
        if data_path.exists():
            class_dirs = [d for d in data_path.iterdir() if d.is_dir()]
            class_names = sorted([d.name for d in class_dirs])
            self.data.class_names = class_names
            self.model.num_classes = len(class_names)
            return class_names
        
        # Default class names
        default_classes = [
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites',
            'Tomato_Target_Spot', 'Tomato_YellowLeaf__Curl_Virus', 'Tomato_mosaic_virus',
            'Tomato_healthy', 'Potato_Early_blight', 'Potato_Late_blight',
            'Potato_healthy', 'Pepper_Bacterial_spot', 'Pepper_healthy'
        ]
        self.data.class_names = default_classes
        self.model.num_classes = len(default_classes)
        return default_classes
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Check data directory
        if not Path(self.data.data_dir).exists():
            issues.append(f"Data directory does not exist: {self.data.data_dir}")
        
        # Check splits sum to 1
        total_split = self.data.train_split + self.data.val_split + self.data.test_split
        if abs(total_split - 1.0) > 1e-6:
            issues.append(f"Data splits must sum to 1.0, got {total_split}")
        
        # Check model type
        valid_models = ['cnn', 'resnet50', 'resnet101', 'resnet152', 
                       'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                       'densenet121', 'densenet161', 'densenet201',
                       'vgg16', 'vgg19']
        if self.model.model_type not in valid_models:
            issues.append(f"Invalid model type: {self.model.model_type}. Must be one of {valid_models}")
        
        # Check learning rate
        if self.training.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        
        # Check batch size
        if self.data.batch_size <= 0:
            issues.append("Batch size must be positive")
        
        # Check image size
        if self.data.img_size <= 0:
            issues.append("Image size must be positive")
        
        return issues

# Default configuration
DEFAULT_CONFIG = {
    'data': {
        'data_dir': '/Users/lesa/Documents/plant_diseaase_detection/data/raw/plantdisease/PlantVillage',
        'img_size': 224,
        'batch_size': 32,
        'num_workers': 4,
        'train_split': 0.7,
        'val_split': 0.15,
        'test_split': 0.15,
        'random_seed': 42
    },
    'model': {
        'model_type': 'resnet50',
        'num_classes': 15,
        'pretrained': True,
        'freeze_backbone': False,
        'dropout_rate': 0.5
    },
    'training': {
        'num_epochs': 50,
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'scheduler_type': 'reduce_on_plateau',
        'early_stopping_patience': 10,
        'checkpoint_path': 'best_model.pth',
        'final_model_path': 'final_model.pth',
        'use_wandb': False,
        'use_mlflow': False,
        'experiment_name': 'plant_disease_detection'
    },
    'evaluation': {
        'test_batch_size': 32,
        'generate_plots': True,
        'save_predictions': True,
        'output_dir': 'evaluation_results',
        'include_heatmap': True
    },
    'api': {
        'host': '0.0.0.0',
        'port': 8000,
        'model_path': 'best_model.pth',
        'device': 'auto',
        'max_file_size': 10485760,  # 10MB
        'allowed_extensions': ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    },
    'monitoring': {
        'log_level': 'INFO',
        'log_file': 'plant_disease.log',
        'metrics_file': 'metrics.json',
        'enable_profiling': False,
        'profile_output': 'profile.prof'
    }
}

def create_default_config_file(config_file: str = "config.yaml"):
    """Create a default configuration file"""
    config = Config()
    config.save_to_file(config_file)
    print(f"Default configuration saved to {config_file}")

def load_config(config_file: str = "config.yaml") -> Config:
    """Load configuration from file"""
    if not Path(config_file).exists():
        print(f"Configuration file {config_file} not found. Creating default configuration...")
        create_default_config_file(config_file)
    
    return Config(config_file)

def main():
    """Main function for configuration management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Plant Disease Detection Configuration")
    parser.add_argument("--create-config", action="store_true", help="Create default configuration file")
    parser.add_argument("--config-file", default="config.yaml", help="Configuration file path")
    parser.add_argument("--validate", action="store_true", help="Validate configuration")
    
    args = parser.parse_args()
    
    if args.create_config:
        create_default_config_file(args.config_file)
        return
    
    config = load_config(args.config_file)
    
    if args.validate:
        issues = config.validate()
        if issues:
            print("Configuration issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("Configuration is valid!")
    else:
        print("Configuration loaded successfully!")
        print(f"Data directory: {config.data.data_dir}")
        print(f"Model type: {config.model.model_type}")
        print(f"Number of classes: {config.model.num_classes}")
        print(f"Batch size: {config.data.batch_size}")
        print(f"Learning rate: {config.training.learning_rate}")

if __name__ == "__main__":
    main()
