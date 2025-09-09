"""
Preprocessing module for Plant Disease Detection
Handles image preprocessing, augmentation, and normalization
"""

import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from typing import Tuple, List, Optional, Union

class ImagePreprocessor:
    """Image preprocessing and augmentation utilities"""
    
    def __init__(self, img_size: int = 224, mean: List[float] = None, std: List[float] = None):
        self.img_size = img_size
        self.mean = mean or [0.485, 0.456, 0.406]  # ImageNet stats
        self.std = std or [0.229, 0.224, 0.225]    # ImageNet stats
        
    def get_torch_transforms(self, is_training: bool = True) -> transforms.Compose:
        """Get PyTorch transforms for training or validation"""
        if is_training:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std)
            ])
        
        return transform
    
    def get_albumentations_transforms(self, is_training: bool = True) -> A.Compose:
        """Get Albumentations transforms for more advanced augmentation"""
        if is_training:
            transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.3),
                A.RandomRotate90(p=0.3),
                A.Rotate(limit=20, p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit=20,
                    sat_shift_limit=30,
                    val_shift_limit=20,
                    p=0.5
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
                A.Blur(blur_limit=3, p=0.2),
                A.MotionBlur(blur_limit=3, p=0.2),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
                A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.2),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.1),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        else:
            transform = A.Compose([
                A.Resize(self.img_size, self.img_size),
                A.Normalize(mean=self.mean, std=self.std),
                ToTensorV2()
            ])
        
        return transform
    
    def preprocess_image(self, image_path: Union[str, Path], transform: transforms.Compose) -> torch.Tensor:
        """Preprocess a single image"""
        image = Image.open(image_path).convert('RGB')
        return transform(image)
    
    def denormalize_image(self, tensor: torch.Tensor) -> np.ndarray:
        """Denormalize a tensor back to image format for visualization"""
        # Clone tensor to avoid modifying original
        tensor = tensor.clone()
        
        # Denormalize
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        
        # Clamp to [0, 1] range
        tensor = torch.clamp(tensor, 0, 1)
        
        # Convert to numpy and transpose for matplotlib
        return tensor.permute(1, 2, 0).numpy()
    
    def visualize_augmentations(self, image_path: Union[str, Path], num_samples: int = 8):
        """Visualize different augmentations applied to an image"""
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            return
        
        # Load original image
        original = Image.open(image_path).convert('RGB')
        
        # Get training transform
        transform = self.get_torch_transforms(is_training=True)
        
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        # Show original
        axes[0].imshow(original)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Show augmented versions
        for i in range(1, num_samples):
            augmented = transform(original)
            denorm = self.denormalize_image(augmented)
            axes[i].imshow(denorm)
            axes[i].set_title(f'Augmented {i}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()

class DataAugmentation:
    """Advanced data augmentation strategies"""
    
    @staticmethod
    def get_heavy_augmentation(img_size: int = 224) -> A.Compose:
        """Heavy augmentation for small datasets"""
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=30, p=0.7),
            A.RandomBrightnessContrast(
                brightness_limit=0.3,
                contrast_limit=0.3,
                p=0.7
            ),
            A.HueSaturationValue(
                hue_shift_limit=30,
                sat_shift_limit=40,
                val_shift_limit=30,
                p=0.7
            ),
            A.RandomGamma(gamma_limit=(70, 130), p=0.5),
            A.GaussNoise(var_limit=(10.0, 80.0), p=0.4),
            A.Blur(blur_limit=5, p=0.3),
            A.MotionBlur(blur_limit=5, p=0.3),
            A.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), p=0.5),
            A.RandomShadow(shadow_roi=(0, 0.5, 1, 1), p=0.3),
            A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), p=0.2),
            A.CoarseDropout(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_light_augmentation(img_size: int = 224) -> A.Compose:
        """Light augmentation for large datasets"""
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.1,
                contrast_limit=0.1,
                p=0.3
            ),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    
    @staticmethod
    def get_test_time_augmentation(img_size: int = 224) -> List[A.Compose]:
        """Test Time Augmentation (TTA) transforms"""
        return [
            A.Compose([
                A.Resize(img_size, img_size),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(img_size, img_size),
                A.VerticalFlip(p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]),
            A.Compose([
                A.Resize(img_size, img_size),
                A.Rotate(limit=90, p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        ]

class ImageQualityChecker:
    """Check and filter image quality"""
    
    @staticmethod
    def check_image_quality(image_path: Union[str, Path]) -> dict:
        """Check various quality metrics of an image"""
        try:
            image = cv2.imread(str(image_path))
            if image is None:
                return {"valid": False, "error": "Could not load image"}
            
            height, width = image.shape[:2]
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # Check for very small images
            too_small = height < 50 or width < 50
            
            # Check for very dark or bright images
            too_dark = brightness < 30
            too_bright = brightness > 225
            
            # Check for very low contrast
            low_contrast = contrast < 20
            
            # Check for very blurry images
            too_blurry = blur_score < 100
            
            quality_score = 100
            issues = []
            
            if too_small:
                quality_score -= 30
                issues.append("too_small")
            if too_dark:
                quality_score -= 20
                issues.append("too_dark")
            if too_bright:
                quality_score -= 20
                issues.append("too_bright")
            if low_contrast:
                quality_score -= 25
                issues.append("low_contrast")
            if too_blurry:
                quality_score -= 25
                issues.append("too_blurry")
            
            return {
                "valid": quality_score > 50,
                "quality_score": max(0, quality_score),
                "blur_score": blur_score,
                "brightness": brightness,
                "contrast": contrast,
                "size": (height, width),
                "issues": issues
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}
    
    @staticmethod
    def filter_dataset_by_quality(image_paths: List[str], min_quality: int = 50) -> Tuple[List[str], List[dict]]:
        """Filter dataset by image quality"""
        valid_paths = []
        quality_reports = []
        
        for path in image_paths:
            report = ImageQualityChecker.check_image_quality(path)
            quality_reports.append(report)
            
            if report["valid"] and report["quality_score"] >= min_quality:
                valid_paths.append(path)
        
        return valid_paths, quality_reports

def test_preprocessing():
    """Test preprocessing functions"""
    print("Testing preprocessing module...")
    
    # Create preprocessor
    preprocessor = ImagePreprocessor(img_size=224)
    
    # Test transforms
    train_transform = preprocessor.get_torch_transforms(is_training=True)
    val_transform = preprocessor.get_torch_transforms(is_training=False)
    
    print("✓ PyTorch transforms created")
    
    # Test albumentations
    albu_train = preprocessor.get_albumentations_transforms(is_training=True)
    albu_val = preprocessor.get_albumentations_transforms(is_training=False)
    
    print("✓ Albumentations transforms created")
    
    # Test augmentation strategies
    heavy_aug = DataAugmentation.get_heavy_augmentation()
    light_aug = DataAugmentation.get_light_augmentation()
    tta_transforms = DataAugmentation.get_test_time_augmentation()
    
    print("✓ Augmentation strategies created")
    print(f"  - Heavy augmentation: {len(heavy_aug.transforms)} transforms")
    print(f"  - Light augmentation: {len(light_aug.transforms)} transforms")
    print(f"  - TTA: {len(tta_transforms)} transforms")
    
    print("\nPreprocessing module test completed successfully!")

if __name__ == "__main__":
    test_preprocessing()
