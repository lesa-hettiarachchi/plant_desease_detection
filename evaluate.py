"""
Evaluation module for Plant Disease Detection
Handles model evaluation, metrics calculation, and visualization
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.preprocessing import label_binarize
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from tqdm import tqdm
import json
from datetime import datetime

# Import Grad-CAM for explainability
try:
    from grad_cam import GradCAM
    from grad_cam.utils import visualize_cam
    GRAD_CAM_AVAILABLE = True
except ImportError:
    GRAD_CAM_AVAILABLE = False
    print("Grad-CAM not available. Install with: pip install grad-cam")

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, model: nn.Module, device: torch.device, class_names: List[str]):
        self.model = model
        self.device = device
        self.class_names = class_names
        self.num_classes = len(class_names)
        
        # Move model to device
        self.model.to(device)
        self.model.eval()
        
        # Initialize Grad-CAM if available
        if GRAD_CAM_AVAILABLE:
            self.grad_cam = GradCAM(model, target_layer='layer4')  # For ResNet
        else:
            self.grad_cam = None
    
    def predict(self, data_loader: DataLoader, return_probabilities: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions on a dataset"""
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in tqdm(data_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                
                # Get predictions
                _, predicted = torch.max(outputs, 1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(target.cpu().numpy())
                
                if return_probabilities:
                    probabilities = torch.softmax(outputs, dim=1)
                    all_probabilities.extend(probabilities.cpu().numpy())
        
        predictions = np.array(all_predictions)
        labels = np.array(all_labels)
        probabilities = np.array(all_probabilities) if return_probabilities else None
        
        return predictions, labels, probabilities
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate comprehensive evaluation metrics"""
        metrics = {}
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        metrics['accuracy'] = accuracy
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro averages
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro', zero_division=0
        )
        
        # Weighted averages
        precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
            y_true, y_pred, average='weighted', zero_division=0
        )
        
        metrics.update({
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted
        })
        
        # Per-class detailed metrics
        class_metrics = {}
        for i, class_name in enumerate(self.class_names):
            class_metrics[class_name] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1[i],
                'support': support[i]
            }
        metrics['per_class'] = class_metrics
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # ROC-AUC for multi-class (if probabilities available)
        if y_prob is not None:
            try:
                # Binarize labels for multi-class ROC
                y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
                
                # Calculate ROC-AUC for each class
                roc_auc_scores = []
                for i in range(self.num_classes):
                    if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists
                        roc_auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                        roc_auc_scores.append(roc_auc)
                    else:
                        roc_auc_scores.append(0.0)
                
                metrics['roc_auc_macro'] = np.mean(roc_auc_scores)
                metrics['roc_auc_per_class'] = dict(zip(self.class_names, roc_auc_scores))
                
            except Exception as e:
                logger.warning(f"Could not calculate ROC-AUC: {e}")
                metrics['roc_auc_macro'] = 0.0
                metrics['roc_auc_per_class'] = {name: 0.0 for name in self.class_names}
        
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 10)):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=figsize)
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Count'}
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 save_path: Optional[str] = None):
        """Plot classification report as heatmap"""
        report = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        
        # Extract metrics for plotting
        metrics = ['precision', 'recall', 'f1-score']
        data = []
        class_names = []
        
        for class_name in self.class_names:
            if class_name in report:
                class_names.append(class_name)
                row = [report[class_name][metric] for metric in metrics]
                data.append(row)
        
        data = np.array(data)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            data,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            xticklabels=metrics,
            yticklabels=class_names,
            cbar_kws={'label': 'Score'}
        )
        
        plt.title('Classification Report', fontsize=16, fontweight='bold')
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Classes', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Classification report saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, y_true: np.ndarray, y_prob: np.ndarray, 
                       save_path: Optional[str] = None, figsize: Tuple[int, int] = (12, 8)):
        """Plot ROC curves for each class"""
        if y_prob is None:
            logger.warning("No probabilities provided for ROC curve plotting")
            return
        
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
        
        plt.figure(figsize=figsize)
        
        # Plot ROC curve for each class
        for i, class_name in enumerate(self.class_names):
            if len(np.unique(y_true_bin[:, i])) > 1:  # Check if class exists
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = roc_auc_score(y_true_bin[:, i], y_prob[:, i])
                plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curves', fontsize=16, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"ROC curves saved to {save_path}")
        
        plt.show()
    
    def generate_grad_cam(self, image: torch.Tensor, target_class: int, 
                         save_path: Optional[str] = None) -> np.ndarray:
        """Generate Grad-CAM visualization for model explainability"""
        if self.grad_cam is None:
            logger.warning("Grad-CAM not available")
            return None
        
        try:
            # Generate Grad-CAM
            mask, _ = self.grad_cam(image.unsqueeze(0).to(self.device), target_class)
            
            # Convert to numpy for visualization
            mask = mask.squeeze().cpu().numpy()
            
            # Normalize mask
            mask = (mask - mask.min()) / (mask.max() - mask.min())
            
            if save_path:
                plt.figure(figsize=(8, 6))
                plt.imshow(mask, cmap='jet')
                plt.colorbar()
                plt.title(f'Grad-CAM for {self.class_names[target_class]}')
                plt.axis('off')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Grad-CAM saved to {save_path}")
            
            return mask
            
        except Exception as e:
            logger.error(f"Error generating Grad-CAM: {e}")
            return None
    
    def evaluate_dataset(self, data_loader: DataLoader, dataset_name: str = "Test") -> Dict[str, Any]:
        """Comprehensive evaluation of a dataset"""
        logger.info(f"Evaluating {dataset_name} dataset...")
        
        # Make predictions
        predictions, labels, probabilities = self.predict(data_loader, return_probabilities=True)
        
        # Calculate metrics
        metrics = self.calculate_metrics(labels, predictions, probabilities)
        
        # Print summary
        logger.info(f"\n{dataset_name} Results:")
        logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Precision (Macro): {metrics['precision_macro']:.4f}")
        logger.info(f"Recall (Macro): {metrics['recall_macro']:.4f}")
        logger.info(f"F1-Score (Macro): {metrics['f1_macro']:.4f}")
        logger.info(f"F1-Score (Weighted): {metrics['f1_weighted']:.4f}")
        
        if 'roc_auc_macro' in metrics:
            logger.info(f"ROC-AUC (Macro): {metrics['roc_auc_macro']:.4f}")
        
        # Generate visualizations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Confusion matrix
        cm_path = f"confusion_matrix_{dataset_name.lower()}_{timestamp}.png"
        self.plot_confusion_matrix(labels, predictions, save_path=cm_path)
        
        # Classification report
        cr_path = f"classification_report_{dataset_name.lower()}_{timestamp}.png"
        self.plot_classification_report(labels, predictions, save_path=cr_path)
        
        # ROC curves
        if probabilities is not None:
            roc_path = f"roc_curves_{dataset_name.lower()}_{timestamp}.png"
            self.plot_roc_curves(labels, probabilities, save_path=roc_path)
        
        # Save detailed metrics
        metrics_path = f"metrics_{dataset_name.lower()}_{timestamp}.json"
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Detailed metrics saved to {metrics_path}")
        
        return metrics
    
    def compare_models(self, models: Dict[str, nn.Module], data_loader: DataLoader) -> pd.DataFrame:
        """Compare multiple models on the same dataset"""
        results = []
        
        for model_name, model in models.items():
            logger.info(f"Evaluating {model_name}...")
            
            # Temporarily replace model
            original_model = self.model
            self.model = model.to(self.device)
            self.model.eval()
            
            # Evaluate
            predictions, labels, probabilities = self.predict(data_loader, return_probabilities=True)
            metrics = self.calculate_metrics(labels, predictions, probabilities)
            
            # Store results
            results.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision (Macro)': metrics['precision_macro'],
                'Recall (Macro)': metrics['recall_macro'],
                'F1-Score (Macro)': metrics['f1_macro'],
                'F1-Score (Weighted)': metrics['f1_weighted'],
                'ROC-AUC (Macro)': metrics.get('roc_auc_macro', 0.0)
            })
            
            # Restore original model
            self.model = original_model
        
        # Create comparison DataFrame
        df = pd.DataFrame(results)
        df = df.sort_values('Accuracy', ascending=False)
        
        logger.info("\nModel Comparison:")
        logger.info(df.to_string(index=False))
        
        return df

def evaluate_single_image(model: nn.Module, image_path: str, class_names: List[str], 
                         device: torch.device, img_size: int = 224) -> Dict[str, Any]:
    """Evaluate a single image"""
    from preprocess import ImagePreprocessor
    
    # Load and preprocess image
    preprocessor = ImagePreprocessor(img_size=img_size)
    transform = preprocessor.get_torch_transforms(is_training=False)
    
    image = preprocessor.preprocess_image(image_path, transform)
    image = image.unsqueeze(0).to(device)
    
    # Make prediction
    model.eval()
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get top-5 predictions
    top5_prob, top5_indices = torch.topk(probabilities, 5)
    
    result = {
        'predicted_class': class_names[predicted.item()],
        'confidence': confidence.item(),
        'top5_predictions': [
            {
                'class': class_names[idx.item()],
                'probability': prob.item()
            }
            for idx, prob in zip(top5_indices[0], top5_prob[0])
        ]
    }
    
    return result

def main_evaluation():
    """Main evaluation pipeline"""
    # Configuration
    config = {
        'model_path': 'best_model.pth',
        'data_dir': '/Users/lesa/Documents/plant_diseaase_detection/data/raw/plantdisease/PlantVillage',
        'img_size': 224,
        'batch_size': 32
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    try:
        # Load model
        logger.info("Loading model...")
        checkpoint = torch.load(config['model_path'], map_location=device)
        model = checkpoint['model']  # Assuming model is saved in checkpoint
        class_names = checkpoint['class_names']
        
        # Load data
        logger.info("Loading data...")
        from data_loader import PlantDiseaseDataLoader
        data_loader = PlantDiseaseDataLoader(
            data_dir=config['data_dir'],
            img_size=config['img_size'],
            batch_size=config['batch_size']
        )
        
        _, _, test_loader = data_loader.load_data()
        
        # Create evaluator
        evaluator = ModelEvaluator(model, device, class_names)
        
        # Evaluate test set
        test_metrics = evaluator.evaluate_dataset(test_loader, "Test")
        
        logger.info("Evaluation completed successfully!")
        
        return evaluator, test_metrics
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main_evaluation()
