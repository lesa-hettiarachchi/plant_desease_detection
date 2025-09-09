#!/bin/bash

# Plant Disease Detection Environment Setup Script
# This script creates a conda environment with all required dependencies

set -e  # Exit on any error

echo "🌱 Setting up Plant Disease Detection Environment..."

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install Anaconda or Miniconda first."
    echo "   Download from: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# Detect OS and use appropriate environment file
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "🍎 Detected macOS - using macOS-optimized environment"
    ENV_FILE="environment_macos.yml"
else
    echo "🐧 Detected Linux/Windows - using standard environment"
    ENV_FILE="environment.yml"
fi

# Create conda environment from appropriate environment file
echo "📦 Creating conda environment from $ENV_FILE..."
conda env create -f $ENV_FILE

# Activate the environment
echo "🔄 Activating environment..."
conda activate plant_disease_detection

# Verify installation
echo "✅ Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'TorchVision version: {torchvision.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
python -c "import cv2; print(f'OpenCV version: {cv2.__version__}')"

echo ""
echo "🎉 Environment setup complete!"
echo ""
echo "To activate the environment, run:"
echo "  conda activate plant_disease_detection"
echo ""
echo "To deactivate the environment, run:"
echo "  conda deactivate"
echo ""
echo "To remove the environment, run:"
echo "  conda env remove -n plant_disease_detection"
