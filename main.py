"""
Main entry point for Plant Disease Detection System
Provides command-line interface for training, evaluation, and serving
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Optional

# Import our modules
from config import Config, load_config
from data_loader import PlantDiseaseDataLoader
from model import ModelFactory
from train import PlantDiseaseTrainer
from evaluate import ModelEvaluator
from serve import create_app
from utils import set_seed, get_device, create_logger

def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Setup logging configuration"""
    logger = create_logger("plant_disease", log_file, log_level)
    return logger

def train_model(config_file: str, resume: bool = False):
    """Train the plant disease detection model"""
    logger = setup_logging()
    logger.info("Starting model training...")
    
    # Load configuration
    config = load_config(config_file)
    
    # Set random seed
    set_seed(config.data.random_seed)
    
    # Get device
    device = get_device()
    
    try:
        # Load data
        logger.info("Loading data...")
        data_loader = PlantDiseaseDataLoader(
            data_dir=config.data.data_dir,
            img_size=config.data.img_size,
            batch_size=config.data.batch_size,
            num_workers=config.data.num_workers
        )
        
        train_loader, val_loader, test_loader = data_loader.load_data(
            train_split=config.data.train_split,
            val_split=config.data.val_split,
            test_split=config.data.test_split,
            random_seed=config.data.random_seed
        )
        
        class_names = data_loader.class_names
        config.model.num_classes = len(class_names)
        
        # Create model
        logger.info("Creating model...")
        model = ModelFactory.create_model(
            model_type=config.model.model_type,
            num_classes=config.model.num_classes,
            pretrained=config.model.pretrained,
            freeze_backbone=config.model.freeze_backbone,
            dropout_rate=config.model.dropout_rate
        )
        
        # Create trainer
        trainer = PlantDiseaseTrainer(
            model=model,
            device=device,
            class_names=class_names,
            experiment_name=config.training.experiment_name,
            use_wandb=config.training.use_wandb,
            use_mlflow=config.training.use_mlflow
        )
        
        # Train model
        results = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=config.training.num_epochs,
            learning_rate=config.training.learning_rate,
            weight_decay=config.training.weight_decay,
            scheduler_type=config.training.scheduler_type,
            early_stopping_patience=config.training.early_stopping_patience,
            checkpoint_path=config.training.checkpoint_path
        )
        
        logger.info("Training completed successfully!")
        logger.info(f"Best validation accuracy: {results['best_val_acc']:.2f}%")
        
        return trainer, results
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

def evaluate_model(config_file: str, model_path: str):
    """Evaluate the trained model"""
    logger = setup_logging()
    logger.info("Starting model evaluation...")
    
    # Load configuration
    config = load_config(config_file)
    
    # Get device
    device = get_device()
    
    try:
        # Load model
        logger.info("Loading model...")
        checkpoint = torch.load(model_path, map_location=device)
        
        if 'model_state_dict' in checkpoint:
            model = ModelFactory.create_model(
                model_type=config.model.model_type,
                num_classes=config.model.num_classes,
                pretrained=False
            )
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model = checkpoint
        
        model.eval()
        model.to(device)
        
        # Get class names
        class_names = checkpoint.get('class_names', config.data.class_names)
        
        # Load data
        logger.info("Loading data...")
        data_loader = PlantDiseaseDataLoader(
            data_dir=config.data.data_dir,
            img_size=config.data.img_size,
            batch_size=config.evaluation.test_batch_size,
            num_workers=config.data.num_workers
        )
        
        _, _, test_loader = data_loader.load_data()
        
        # Create evaluator
        evaluator = ModelEvaluator(model, device, class_names)
        
        # Evaluate model
        results = evaluator.evaluate_dataset(test_loader, "Test")
        
        logger.info("Evaluation completed successfully!")
        logger.info(f"Test accuracy: {results['accuracy']:.4f}")
        
        return evaluator, results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def serve_model(config_file: str, model_path: str, host: str = "0.0.0.0", port: int = 8000):
    """Serve the model via FastAPI"""
    logger = setup_logging()
    logger.info("Starting model serving...")
    
    # Load configuration
    config = load_config(config_file)
    
    try:
        # Get class names
        class_names = config.data.class_names or [
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites',
            'Tomato_Target_Spot', 'Tomato_YellowLeaf__Curl_Virus', 'Tomato_mosaic_virus',
            'Tomato_healthy', 'Potato_Early_blight', 'Potato_Late_blight',
            'Potato_healthy', 'Pepper_Bacterial_spot', 'Pepper_healthy'
        ]
        
        # Create FastAPI app
        app = create_app(
            model_path=model_path,
            class_names=class_names,
            device=config.api.device
        )
        
        # Run server
        import uvicorn
        logger.info(f"Starting server on {host}:{port}")
        uvicorn.run(app, host=host, port=port, log_level="info")
        
    except Exception as e:
        logger.error(f"Serving failed: {e}")
        raise

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Plant Disease Detection System")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train the model")
    train_parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate the model")
    eval_parser.add_argument("--model", required=True, help="Path to trained model")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Serve the model via API")
    serve_parser.add_argument("--model", required=True, help="Path to trained model")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_parser.add_argument("--create", action="store_true", help="Create default configuration")
    config_parser.add_argument("--validate", action="store_true", help="Validate configuration")
    
    args = parser.parse_args()
    
    if args.command == "train":
        train_model(args.config, args.resume)
    elif args.command == "evaluate":
        evaluate_model(args.config, args.model)
    elif args.command == "serve":
        serve_model(args.config, args.model, args.host, args.port)
    elif args.command == "config":
        if args.create:
            from config import create_default_config_file
            create_default_config_file(args.config)
            print(f"Default configuration created: {args.config}")
        elif args.validate:
            config = load_config(args.config)
            issues = config.validate()
            if issues:
                print("Configuration issues found:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print("Configuration is valid!")
        else:
            print("Use --create to create default config or --validate to validate existing config")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
