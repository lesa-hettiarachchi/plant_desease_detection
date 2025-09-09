"""
FastAPI serving module for Plant Disease Detection
Provides REST API for model inference and monitoring
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import base64
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# Import our modules
from model import ModelFactory
from preprocess import ImagePreprocessor, ImageQualityChecker
from evaluate import evaluate_single_image

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for API
class PredictionRequest(BaseModel):
    image_base64: str
    return_heatmap: bool = False

class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    all_predictions: List[Dict[str, float]]
    processing_time: float
    image_quality: Optional[Dict[str, Any]] = None
    heatmap_base64: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_info: Dict[str, Any]
    timestamp: str

class ModelInfoResponse(BaseModel):
    model_name: str
    num_classes: int
    class_names: List[str]
    input_size: List[int]
    model_size_mb: float

class PlantDiseaseAPI:
    """Plant Disease Detection API class"""
    
    def __init__(self, model_path: str, class_names: List[str], device: str = "cpu"):
        self.model_path = model_path
        self.class_names = class_names
        self.device = torch.device(device)
        self.model = None
        self.preprocessor = ImagePreprocessor(img_size=224)
        self.transform = self.preprocessor.get_torch_transforms(is_training=False)
        self.quality_checker = ImageQualityChecker()
        
        # Load model
        self.load_model()
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Plant Disease Detection API",
            description="API for detecting plant diseases using deep learning",
            version="1.0.0"
        )
        
        # Add CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Setup routes
        self.setup_routes()
    
    def load_model(self):
        """Load the trained model"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Extract model info
            if 'model_state_dict' in checkpoint:
                # Load model architecture
                model_type = checkpoint.get('model_type', 'resnet50')
                num_classes = len(self.class_names)
                
                self.model = ModelFactory.create_model(
                    model_type=model_type,
                    num_classes=num_classes,
                    pretrained=False
                )
                
                # Load weights
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                self.model.to(self.device)
                
                logger.info(f"Model loaded successfully: {model_type}")
            else:
                # Direct model loading
                self.model = checkpoint
                self.model.eval()
                self.model.to(self.device)
                logger.info("Model loaded successfully (direct)")
            
            # Get model info
            self.model_info = self.get_model_info()
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if self.model is None:
            return {}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # Calculate model size
        param_size = sum(p.numel() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.model.buffers())
        model_size_mb = (param_size + buffer_size) / (1024 * 1024)
        
        return {
            'model_name': self.model.__class__.__name__,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'input_size': [3, 224, 224],
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb
        }
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for inference"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = self.transform(image)
        return image_tensor.unsqueeze(0).to(self.device)
    
    def predict_image(self, image: Image.Image, return_heatmap: bool = False) -> Dict[str, Any]:
        """Predict disease for a single image"""
        start_time = datetime.now()
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            # Get all predictions
            all_predictions = []
            for i, prob in enumerate(probabilities[0]):
                all_predictions.append({
                    'class': self.class_names[i],
                    'probability': prob.item()
                })
            
            # Sort by probability
            all_predictions.sort(key=lambda x: x['probability'], reverse=True)
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'predicted_class': self.class_names[predicted.item()],
                'confidence': confidence.item(),
                'all_predictions': all_predictions,
                'processing_time': processing_time
            }
            
            # Add image quality check
            try:
                # Convert PIL to numpy for quality check
                img_array = np.array(image)
                quality_report = self.quality_checker.check_image_quality_from_array(img_array)
                result['image_quality'] = quality_report
            except Exception as e:
                logger.warning(f"Quality check failed: {e}")
                result['image_quality'] = None
            
            # Generate heatmap if requested
            if return_heatmap:
                try:
                    heatmap = self.generate_heatmap(image_tensor, predicted.item())
                    if heatmap is not None:
                        result['heatmap_base64'] = heatmap
                except Exception as e:
                    logger.warning(f"Heatmap generation failed: {e}")
                    result['heatmap_base64'] = None
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def generate_heatmap(self, image_tensor: torch.Tensor, target_class: int) -> Optional[str]:
        """Generate Grad-CAM heatmap"""
        try:
            # This is a simplified heatmap generation
            # In practice, you'd use Grad-CAM or similar methods
            from evaluate import ModelEvaluator
            
            evaluator = ModelEvaluator(self.model, self.device, self.class_names)
            heatmap = evaluator.generate_grad_cam(image_tensor, target_class)
            
            if heatmap is not None:
                # Convert heatmap to base64
                import matplotlib.pyplot as plt
                import io
                
                plt.figure(figsize=(8, 6))
                plt.imshow(heatmap, cmap='jet')
                plt.colorbar()
                plt.axis('off')
                
                # Save to bytes
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
                buffer.seek(0)
                
                # Convert to base64
                heatmap_base64 = base64.b64encode(buffer.getvalue()).decode()
                plt.close()
                
                return heatmap_base64
            
        except Exception as e:
            logger.warning(f"Heatmap generation failed: {e}")
        
        return None
    
    def setup_routes(self):
        """Setup API routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def root():
            """Root endpoint with API documentation"""
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Plant Disease Detection API</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    .container { max-width: 800px; margin: 0 auto; }
                    .endpoint { background: #f5f5f5; padding: 15px; margin: 10px 0; border-radius: 5px; }
                    .method { color: #007bff; font-weight: bold; }
                    .description { color: #666; margin-top: 5px; }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>🌱 Plant Disease Detection API</h1>
                    <p>Welcome to the Plant Disease Detection API! This service uses deep learning to identify plant diseases from images.</p>
                    
                    <h2>Available Endpoints:</h2>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /health
                        <div class="description">Check API health and model status</div>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">GET</span> /model/info
                        <div class="description">Get model information and class names</div>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">POST</span> /predict
                        <div class="description">Predict plant disease from uploaded image</div>
                    </div>
                    
                    <div class="endpoint">
                        <span class="method">POST</span> /predict/base64
                        <div class="description">Predict plant disease from base64 encoded image</div>
                    </div>
                    
                    <h2>Usage Examples:</h2>
                    <p>Check the <a href="/docs">interactive API documentation</a> for detailed usage examples.</p>
                </div>
            </body>
            </html>
            """
            return HTMLResponse(content=html_content)
        
        @self.app.get("/health", response_model=HealthResponse)
        async def health_check():
            """Health check endpoint"""
            return HealthResponse(
                status="healthy",
                model_loaded=self.model is not None,
                model_info=self.model_info,
                timestamp=datetime.now().isoformat()
            )
        
        @self.app.get("/model/info", response_model=ModelInfoResponse)
        async def get_model_info():
            """Get model information"""
            if self.model is None:
                raise HTTPException(status_code=500, detail="Model not loaded")
            
            return ModelInfoResponse(
                model_name=self.model_info['model_name'],
                num_classes=self.model_info['num_classes'],
                class_names=self.model_info['class_names'],
                input_size=self.model_info['input_size'],
                model_size_mb=self.model_info['model_size_mb']
            )
        
        @self.app.post("/predict", response_model=PredictionResponse)
        async def predict_from_file(
            file: UploadFile = File(...),
            return_heatmap: bool = False
        ):
            """Predict plant disease from uploaded image file"""
            try:
                # Validate file type
                if not file.content_type.startswith('image/'):
                    raise HTTPException(status_code=400, detail="File must be an image")
                
                # Read image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                
                # Make prediction
                result = self.predict_image(image, return_heatmap)
                
                return PredictionResponse(**result)
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/predict/base64", response_model=PredictionResponse)
        async def predict_from_base64(request: PredictionRequest):
            """Predict plant disease from base64 encoded image"""
            try:
                # Decode base64 image
                image_data = base64.b64decode(request.image_base64)
                image = Image.open(io.BytesIO(image_data))
                
                # Make prediction
                result = self.predict_image(image, request.return_heatmap)
                
                return PredictionResponse(**result)
                
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/classes")
        async def get_classes():
            """Get list of supported plant disease classes"""
            return {
                "classes": self.class_names,
                "num_classes": len(self.class_names)
            }

def create_app(model_path: str, class_names: List[str], device: str = "cpu") -> FastAPI:
    """Create FastAPI application"""
    api = PlantDiseaseAPI(model_path, class_names, device)
    return api.app

def main():
    """Main function to run the API server"""
    # Configuration
    config = {
        'model_path': 'best_model.pth',
        'class_names': [
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites',
            'Tomato_Target_Spot', 'Tomato_YellowLeaf__Curl_Virus', 'Tomato_mosaic_virus',
            'Tomato_healthy', 'Potato_Early_blight', 'Potato_Late_blight',
            'Potato_healthy', 'Pepper_Bacterial_spot', 'Pepper_healthy'
        ],
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'host': '0.0.0.0',
        'port': 8000
    }
    
    logger.info("Starting Plant Disease Detection API...")
    logger.info(f"Model path: {config['model_path']}")
    logger.info(f"Device: {config['device']}")
    logger.info(f"Number of classes: {len(config['class_names'])}")
    
    # Create app
    app = create_app(
        model_path=config['model_path'],
        class_names=config['class_names'],
        device=config['device']
    )
    
    # Run server
    uvicorn.run(
        app,
        host=config['host'],
        port=config['port'],
        log_level="info"
    )

if __name__ == "__main__":
    main()
