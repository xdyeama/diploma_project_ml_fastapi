"""
Configuration settings for the application.
"""

from pathlib import Path
import torch

# Paths
BASE_DIR = Path(__file__).parent
MODEL_WEIGHTS_DIR = BASE_DIR / "model_weights"
UPLOADED_MODELS_DIR = BASE_DIR / "uploaded_models"
UPLOADED_MODELS_DIR.mkdir(exist_ok=True)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model configuration
MODEL_IN_CHANNELS = 1
MODEL_OUT_CHANNELS = 4  # background, edema, non-enhancing, enhancing

# Image preprocessing configuration
TARGET_IMAGE_SIZE = (256, 256)

# Logging configuration
LOG_LEVEL = "INFO"
