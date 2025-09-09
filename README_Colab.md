# Plant Disease Detection - Google Colab Setup

This guide will help you run the Plant Disease Detection system on Google Colab with GPU acceleration.

## 🚀 Quick Start

### 1. Upload to GitHub

```bash
# Initialize Git repository
git init
git add .
git commit -m "Initial commit: Plant Disease Detection System"

# Create GitHub repository and push
git remote add origin https://github.com/yourusername/plant-disease-detection.git
git branch -M main
git push -u origin main
```

### 2. Open in Google Colab

1. Go to [Google Colab](https://colab.research.google.com/)
2. Click "File" → "Open notebook" → "GitHub"
3. Enter your GitHub repository URL
4. Select `Plant_Disease_Detection_Colab.ipynb`

### 3. Enable GPU

1. In Colab, go to "Runtime" → "Change runtime type"
2. Set "Hardware accelerator" to "GPU"
3. Choose "T4" or "V100" if available
4. Click "Save"

### 4. Run the Notebook

1. Run all cells in sequence
2. Upload your PlantVillage dataset when prompted
3. The system will automatically train and evaluate the model

## 📊 Expected Performance

### GPU Memory Usage
- **T4 GPU (15GB)**: Can handle batch size 32-64
- **V100 GPU (16GB)**: Can handle batch size 64-128
- **CPU**: Batch size 8-16 (much slower)

### Training Time
- **ResNet50**: ~2-3 hours for 20 epochs
- **EfficientNet-B0**: ~1-2 hours for 20 epochs
- **Custom CNN**: ~1 hour for 20 epochs

## 🔧 Configuration

### Recommended Settings for Colab

```python
CONFIG = {
    'batch_size': 32,        # Adjust based on GPU memory
    'num_epochs': 20,        # Reduced for demo
    'learning_rate': 1e-3,   # Standard learning rate
    'model_type': 'resnet50' # or 'efficientnet_b0'
}
```

### Memory Optimization

```python
# For limited GPU memory
CONFIG['batch_size'] = 16
CONFIG['num_workers'] = 2

# For more GPU memory
CONFIG['batch_size'] = 64
CONFIG['num_workers'] = 4
```

## 📁 Dataset Upload

### Option 1: Direct Upload
1. Zip your PlantVillage dataset
2. Upload via Colab's file upload widget
3. The notebook will automatically extract it

### Option 2: Google Drive
1. Upload dataset to Google Drive
2. Mount Google Drive in Colab
3. Update the `DATA_DIR` path

### Option 3: GitHub
1. Upload dataset to GitHub (use Git LFS for large files)
2. Clone the repository in Colab
3. Update the `DATA_DIR` path

## 🎯 Model Options

### Available Models
- **ResNet50**: Good balance of accuracy and speed
- **EfficientNet-B0**: Fast and efficient
- **EfficientNet-B1**: Better accuracy, slower
- **Custom CNN**: Lightweight, good for small datasets

### Model Selection
```python
# Change this in the configuration cell
CONFIG['model_type'] = 'resnet50'  # or 'efficientnet_b0', 'cnn'
```

## 📈 Monitoring

### Weights & Biases (Optional)
```python
# Enable in configuration
CONFIG['use_wandb'] = True

# Login to W&B
import wandb
wandb.login()
```

### MLflow (Optional)
```python
# Enable in configuration
CONFIG['use_mlflow'] = True
```

## 🔍 Troubleshooting

### Common Issues

**Out of Memory Error:**
```python
# Reduce batch size
CONFIG['batch_size'] = 16
# or
CONFIG['batch_size'] = 8
```

**Slow Training:**
```python
# Reduce number of workers
CONFIG['num_workers'] = 2
# or
CONFIG['num_workers'] = 1
```

**Dataset Not Found:**
```python
# Check the data directory path
print(os.listdir('data/'))
# Update DATA_DIR accordingly
```

### Performance Tips

1. **Use GPU**: Always enable GPU for training
2. **Batch Size**: Start with 32, adjust based on memory
3. **Epochs**: Start with 20, increase if needed
4. **Model**: ResNet50 is a good starting point
5. **Data**: Ensure dataset is properly structured

## 📊 Expected Results

### Accuracy Targets
- **ResNet50**: 85-95% accuracy
- **EfficientNet-B0**: 80-90% accuracy
- **Custom CNN**: 75-85% accuracy

### Training Progress
- **Epoch 1-5**: Rapid improvement
- **Epoch 5-15**: Steady improvement
- **Epoch 15+**: Fine-tuning, potential overfitting

## 🚀 Next Steps

After training in Colab:

1. **Download Model**: The notebook will download `best_model.pth`
2. **Local Deployment**: Use the downloaded model in your local system
3. **API Deployment**: Deploy using the FastAPI server
4. **Web Interface**: Use the frontend for predictions

## 📞 Support

If you encounter issues:

1. Check the Colab logs for error messages
2. Verify your dataset structure
3. Adjust configuration parameters
4. Check GPU memory usage
5. Try reducing batch size if out of memory

## 🎉 Success!

Once training completes, you'll have:
- A trained model (`best_model.pth`)
- Training history plots
- Evaluation metrics
- Confusion matrix
- Classification report

The model is ready for deployment and inference!
