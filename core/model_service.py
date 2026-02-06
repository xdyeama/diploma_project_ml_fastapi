"""
Singleton service for U-Net model loading and management.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import logging
from threading import Lock

from models.unet import UNet
from config import MODEL_WEIGHTS_DIR, DEVICE, MODEL_IN_CHANNELS, MODEL_OUT_CHANNELS

logger = logging.getLogger(__name__)


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
            unet_model = UNet(
                in_channels=MODEL_IN_CHANNELS,
                out_channels=MODEL_OUT_CHANNELS
            )
            unet_model.to(DEVICE)
            
            # Load weights
            checkpoint = torch.load(model_path, map_location=DEVICE)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, dict):
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
        """
        Load default model from model_weights directory.
        
        Returns:
            Loaded model if found, None otherwise
        """
        try:
            # Try to load from model_weights directory
            model_files = list(MODEL_WEIGHTS_DIR.glob("*.pt")) + list(MODEL_WEIGHTS_DIR.glob("*.pth"))
            if model_files:
                model_path = model_files[0]
                return self.load_model(str(model_path))
            else:
                logger.warning("No model weights found. Please upload a model.")
                return None
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
