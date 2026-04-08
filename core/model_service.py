"""
Singleton service for U-Net model loading and management.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import logging
from threading import Lock
from config import MODEL_WEIGHTS_DIR, DEVICE, MODEL_IN_CHANNELS, MODEL_OUT_CHANNELS
from models import unet as unet_module
from models import unet_legacy

logger = logging.getLogger(__name__)


def _register_unet_for_unpickle() -> None:
    """Register legacy UNet/DoubleConv/center_crop so notebook checkpoints can unpickle."""
    for mod_name in ("__main__", "__mp_main__"):
        if mod_name not in sys.modules:
            continue
        mod = sys.modules[mod_name]
        for name in ("DoubleConv", "UNet", "center_crop"):
            if hasattr(unet_legacy, name):
                setattr(mod, name, getattr(unet_legacy, name))


class ModelService:
    """
    Singleton service for managing U-Net model instance.
    Thread-safe implementation for model loading and access.
    """
    
    _instance: Optional['ModelService'] = None
    _lock: Lock = Lock()
    
    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the service (only once)."""
        if self._initialized:
            return
        
        self._model: Optional[nn.Module] = None
        self._model_path: Optional[str] = None
        self._initialized = True
    
    def load_model(self, model_path: str) -> nn.Module:
        """
        Load U-Net model from weights file.
        
        Args:
            model_path: Path to the model weights file
            
        Returns:
            Loaded U-Net model
            
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading model from {model_path}")
            
            # Initialize model architecture
            unet_model = unet_module.UNet(
                in_channels=MODEL_IN_CHANNELS,
                out_channels=MODEL_OUT_CHANNELS
            )
            unet_model.to(DEVICE)
            
            # Full-model checkpoints were pickled with UNet from __main__; under uvicorn
            # the main module is __mp_main__, which has no UNet. Register ours so unpickle works.
            _register_unet_for_unpickle()
            # PyTorch 2.6+ defaults to weights_only=True; full-model checkpoints need weights_only=False.
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            
            # Handle different checkpoint formats (full model vs state_dict)
            if isinstance(checkpoint, nn.Module):
                unet_model = checkpoint
                unet_model.to(DEVICE)
            elif isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    unet_model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    unet_model.load_state_dict(checkpoint['state_dict'])
                else:
                    unet_model.load_state_dict(checkpoint)
            else:
                unet_model.load_state_dict(checkpoint)
            
            unet_model.eval()
            
            # Update singleton instance
            with self._lock:
                self._model = unet_model
                self._model_path = model_path
            
            logger.info("Model loaded successfully")
            return unet_model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def load_default_model(self) -> Optional[nn.Module]:
        try:
            model_files = list(MODEL_WEIGHTS_DIR.glob("*.pt")) + list(MODEL_WEIGHTS_DIR.glob("*.pth"))
            print(f"Model files: {model_files}")

            if not model_files:
                logger.warning("No model weights found. Please upload a model.")
                return None

            # Приоритет: сначала entire_model (полная модель), потом weights
            preferred = ["unet_entire_model.pt", "unet_entire_model.pth",
                         "unet_weights.pt", "unet_weights.pth"]

            model_path = None
            for name in preferred:
                candidate = MODEL_WEIGHTS_DIR / name
                if candidate.exists():
                    model_path = candidate
                    break

            if model_path is None:
                model_path = model_files[0]

            print(f"Loading model from {model_path}")
            return self.load_model(str(model_path))
        except Exception as e:
            logger.error(f"Error loading default model: {e}")
            return None
    
    def get_model(self) -> Optional[nn.Module]:
        """
        Get the current model instance.
        
        Returns:
            Current model instance or None if not loaded
        """
        return self._model
    
    def is_model_loaded(self) -> bool:
        """
        Check if model is loaded.
        
        Returns:
            True if model is loaded, False otherwise
        """
        return self._model is not None
    
    def get_model_path(self) -> Optional[str]:
        """
        Get the path of the currently loaded model.
        
        Returns:
            Model path or None if not loaded
        """
        return self._model_path
    
    def reload_model(self, model_path: str) -> nn.Module:
        """
        Reload model from a new path.
        
        Args:
            model_path: Path to the new model weights file
            
        Returns:
            Reloaded model
        """
        return self.load_model(model_path)
