"""
Model definitions for Plant Disease Detection
Includes CNN architectures and transfer learning models
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import timm
from torch.nn import Dropout, BatchNorm2d, ReLU, AdaptiveAvgPool2d, Linear
from typing import Optional, List, Dict, Any
import math

class PlantDiseaseCNN(nn.Module):
    """Custom CNN architecture for plant disease detection"""
    
    def __init__(self, num_classes: int, img_size: int = 224, dropout_rate: float = 0.5):
        super(PlantDiseaseCNN, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.dropout_rate = dropout_rate
        
        # Feature extraction layers
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Fifth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class TransferLearningModel(nn.Module):
    """Transfer learning model using pre-trained architectures"""
    
    def __init__(
        self, 
        model_name: str = 'resnet50',
        num_classes: int = 15,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5
    ):
        super(TransferLearningModel, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        
        # Load pre-trained model
        if model_name in ['resnet50', 'resnet101', 'resnet152']:
            self.backbone = self._get_resnet(model_name, pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove original classifier
        elif model_name in ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']:
            self.backbone = timm.create_model(model_name, pretrained=pretrained)
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif model_name in ['densenet121', 'densenet161', 'densenet201']:
            self.backbone = self._get_densenet(model_name, pretrained)
            feature_dim = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif model_name in ['vgg16', 'vgg19']:
            self.backbone = self._get_vgg(model_name, pretrained)
            feature_dim = 512  # VGG feature dimension
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes)
        )
        
        # Freeze backbone if specified
        if freeze_backbone:
            self._freeze_backbone()
    
    def _get_resnet(self, model_name: str, pretrained: bool):
        """Get ResNet model"""
        if model_name == 'resnet50':
            return models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'resnet101':
            return models.resnet101(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'resnet152':
            return models.resnet152(weights='IMAGENET1K_V1' if pretrained else None)
    
    def _get_densenet(self, model_name: str, pretrained: bool):
        """Get DenseNet model"""
        if model_name == 'densenet121':
            return models.densenet121(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'densenet161':
            return models.densenet161(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'densenet201':
            return models.densenet201(weights='IMAGENET1K_V1' if pretrained else None)
    
    def _get_vgg(self, model_name: str, pretrained: bool):
        """Get VGG model"""
        if model_name == 'vgg16':
            return models.vgg16(weights='IMAGENET1K_V1' if pretrained else None)
        elif model_name == 'vgg19':
            return models.vgg19(weights='IMAGENET1K_V1' if pretrained else None)
    
    def _freeze_backbone(self):
        """Freeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters"""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

class EnsembleModel(nn.Module):
    """Ensemble model combining multiple architectures"""
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)
        self.weights = weights or [1.0] * len(models)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
    
    def forward(self, x):
        predictions = []
        for model, weight in zip(self.models, self.weights):
            pred = model(x)
            predictions.append(weight * pred)
        
        # Average predictions
        ensemble_pred = torch.stack(predictions, dim=0).sum(dim=0)
        return ensemble_pred

class ModelFactory:
    """Factory class for creating different model architectures"""
    
    @staticmethod
    def create_model(
        model_type: str = 'resnet50',
        num_classes: int = 15,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout_rate: float = 0.5
    ) -> nn.Module:
        """Create a model based on the specified type"""
        
        if model_type == 'cnn':
            return PlantDiseaseCNN(num_classes=num_classes, dropout_rate=dropout_rate)
        
        elif model_type in ['resnet50', 'resnet101', 'resnet152', 
                           'efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2',
                           'densenet121', 'densenet161', 'densenet201',
                           'vgg16', 'vgg19']:
            return TransferLearningModel(
                model_name=model_type,
                num_classes=num_classes,
                pretrained=pretrained,
                freeze_backbone=freeze_backbone,
                dropout_rate=dropout_rate
            )
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def get_available_models() -> Dict[str, List[str]]:
        """Get list of available model types"""
        return {
            'cnn': ['cnn'],
            'resnet': ['resnet50', 'resnet101', 'resnet152'],
            'efficientnet': ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2'],
            'densenet': ['densenet121', 'densenet161', 'densenet201'],
            'vgg': ['vgg16', 'vgg19']
        }

def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_model_summary(model: nn.Module, input_size: tuple = (3, 224, 224)) -> Dict[str, Any]:
    """Get model summary including parameter count and memory usage"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 * 1024)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'frozen_parameters': total_params - trainable_params,
        'model_size_mb': model_size_mb,
        'input_size': input_size
    }

def test_models():
    """Test different model architectures"""
    print("Testing model architectures...")
    
    num_classes = 15
    input_size = (1, 3, 224, 224)
    
    # Test CNN
    print("\n1. Testing Custom CNN:")
    cnn_model = PlantDiseaseCNN(num_classes=num_classes)
    cnn_summary = get_model_summary(cnn_model)
    print(f"   Parameters: {cnn_summary['trainable_parameters']:,}")
    print(f"   Model size: {cnn_summary['model_size_mb']:.2f} MB")
    
    # Test ResNet50
    print("\n2. Testing ResNet50:")
    resnet_model = TransferLearningModel('resnet50', num_classes=num_classes)
    resnet_summary = get_model_summary(resnet_model)
    print(f"   Parameters: {resnet_summary['trainable_parameters']:,}")
    print(f"   Model size: {resnet_summary['model_size_mb']:.2f} MB")
    
    # Test EfficientNet
    print("\n3. Testing EfficientNet-B0:")
    efficientnet_model = TransferLearningModel('efficientnet_b0', num_classes=num_classes)
    efficientnet_summary = get_model_summary(efficientnet_model)
    print(f"   Parameters: {efficientnet_summary['trainable_parameters']:,}")
    print(f"   Model size: {efficientnet_summary['model_size_mb']:.2f} MB")
    
    # Test model factory
    print("\n4. Testing Model Factory:")
    factory_models = ModelFactory.get_available_models()
    print("   Available models:")
    for category, models in factory_models.items():
        print(f"     {category}: {models}")
    
    print("\nModel testing completed successfully!")

if __name__ == "__main__":
    test_models()
