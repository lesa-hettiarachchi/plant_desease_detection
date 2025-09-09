# Plant Disease Detection System

A comprehensive deep learning system for detecting plant diseases in tomato, potato, and bell pepper plants using computer vision and transfer learning. Built with PyTorch, FastAPI, and modern ML practices.

## 🌟 Features

- **Multiple Model Architectures**: CNN, ResNet, EfficientNet, DenseNet, VGG
- **Advanced Data Augmentation**: Albumentations with 20+ transformations
- **Transfer Learning**: Pre-trained models with fine-tuning
- **REST API**: FastAPI-based prediction service with Swagger docs
- **Web Interface**: Modern, responsive frontend with drag-and-drop upload
- **Model Explainability**: Grad-CAM visualization for interpretability
- **Comprehensive Evaluation**: Detailed metrics, confusion matrices, ROC curves
- **Experiment Tracking**: MLflow and Weights & Biases integration
- **Production Ready**: Docker support, monitoring, and logging

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Clone and navigate to the project
cd /Users/lesa/Documents/plant_diseaase_detection

# Create conda environment (automated)
./setup_environment.sh

# Or manually
conda env create -f environment_macos.yml
conda activate plant_disease_detection
```

### 2. Create Configuration

```bash
# Create default configuration
python main.py config --create

# Validate configuration
python main.py config --validate
```

### 3. Train Model

```bash
# Train with default settings
python main.py train

# Train with custom config
python main.py train --config custom_config.yaml
```

### 4. Evaluate Model

```bash
# Evaluate trained model
python main.py evaluate --model best_model.pth
```

### 5. Serve Model

```bash
# Start API server
python main.py serve --model best_model.pth --host 0.0.0.0 --port 8000
```

### 6. Use Web Interface

```bash
# Start frontend server
cd frontend
python -m http.server 3000

# Open browser to http://localhost:3000
```

## 📁 Project Structure

```
plant_diseaase_detection/
├── data/                           # Data directory
│   └── raw/plantdisease/PlantVillage/
│       ├── Tomato_* (disease classes)
│       ├── Potato_* (disease classes)
│       └── Pepper__bell___* (disease classes)
├── frontend/                       # Web interface
│   ├── index.html                  # Main frontend
│   ├── package.json               # Frontend dependencies
│   └── README.md                  # Frontend documentation
├── data_loader.py                 # Data loading and preprocessing
├── preprocess.py                  # Image preprocessing and augmentation
├── model.py                       # Model architectures
├── train.py                       # Training pipeline
├── evaluate.py                    # Model evaluation
├── serve.py                       # FastAPI server
├── config.py                      # Configuration management
├── utils.py                       # Utility functions
├── main.py                        # Main entry point
├── environment_macos.yml          # macOS conda environment
├── requirements.txt               # Python dependencies
├── setup_environment.sh           # Environment setup script
└── README.md                      # This file
```

## 🔧 Configuration

The system uses YAML configuration files. Create a custom config:

```yaml
data:
  data_dir: "/path/to/plant/dataset"
  img_size: 224
  batch_size: 32
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15

model:
  model_type: "resnet50"  # cnn, resnet50, efficientnet_b0, etc.
  num_classes: 15
  pretrained: true
  freeze_backbone: false
  dropout_rate: 0.5

training:
  num_epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  scheduler_type: "reduce_on_plateau"
  early_stopping_patience: 10

api:
  host: "0.0.0.0"
  port: 8000
  device: "auto"
```

## 🧠 Supported Models

### Custom CNN
- 5-layer convolutional architecture
- Batch normalization and dropout
- Global average pooling
- Fully connected layers

### Transfer Learning Models
- **ResNet**: ResNet50, ResNet101, ResNet152
- **EfficientNet**: EfficientNet-B0, B1, B2
- **DenseNet**: DenseNet121, DenseNet161, DenseNet201
- **VGG**: VGG16, VGG19

## 📊 Supported Plant Diseases

### Tomato (10 classes)
- Bacterial Spot, Early Blight, Late Blight
- Leaf Mold, Septoria Leaf Spot, Spider Mites
- Target Spot, Yellow Leaf Curl Virus, Mosaic Virus
- Healthy

### Potato (3 classes)
- Early Blight, Late Blight, Healthy

### Pepper (2 classes)
- Bacterial Spot, Healthy

## 🎯 Usage Examples

### Training

```python
from main import train_model

# Train with custom configuration
trainer, results = train_model("config.yaml")
print(f"Best accuracy: {results['best_val_acc']:.2f}%")
```

### Evaluation

```python
from main import evaluate_model

# Evaluate trained model
evaluator, results = evaluate_model("config.yaml", "best_model.pth")
print(f"Test accuracy: {results['accuracy']:.4f}")
```

### API Usage

```python
import requests
import base64

# Load image
with open("plant_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode()

# Make prediction
response = requests.post("http://localhost:8000/predict/base64", json={
    "image_base64": image_data,
    "return_heatmap": True
})

result = response.json()
print(f"Predicted: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🔍 Model Evaluation

The system provides comprehensive evaluation metrics:

- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and macro averages
- **Confusion Matrix**: Visual representation of predictions
- **ROC Curves**: Multi-class ROC analysis
- **Grad-CAM**: Model explainability visualization

## 🚀 Deployment

### Local Deployment

```bash
# Start API server
python main.py serve --model best_model.pth

# Start frontend
cd frontend && python -m http.server 3000
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["python", "main.py", "serve", "--model", "best_model.pth"]
```

## 📈 Monitoring

### MLflow Integration

```python
# Enable MLflow tracking
config.training.use_mlflow = True
config.training.experiment_name = "plant_disease_experiment"
```

### Weights & Biases

```python
# Enable W&B tracking
config.training.use_wandb = True
```

## 🛠️ Development

### Code Quality

```bash
# Format code
black *.py

# Lint code
flake8 *.py

# Run tests
pytest tests/
```

### Adding New Models

```python
# Add custom model to model.py
class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Your architecture here
    
    def forward(self, x):
        # Your forward pass here
        return x

# Register in ModelFactory
ModelFactory.register_model('custom', CustomModel)
```

## 🐛 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```python
# Reduce batch size
config.data.batch_size = 16

# Use gradient accumulation
config.training.gradient_accumulation_steps = 2
```

**Model Loading Issues:**
```python
# Check model compatibility
model = torch.load('model.pth', map_location='cpu')
print(model.keys())
```

**API Connection Issues:**
```bash
# Check if server is running
curl http://localhost:8000/health

# Check CORS settings
# Ensure CORS is enabled in serve.py
```

## 📚 Documentation

- [API Documentation](http://localhost:8000/docs) - Interactive Swagger docs
- [Frontend Guide](frontend/README.md) - Web interface documentation
- [Configuration Reference](config.py) - Complete config options

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- [PlantVillage Dataset](https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset)
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - Web framework
- [Albumentations](https://albumentations.ai/) - Data augmentation
- [timm](https://github.com/rwightman/pytorch-image-models) - Pre-trained models
