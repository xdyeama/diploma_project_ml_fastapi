"""
Singleton service for SMG-Net model loading and management.
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional
import logging
from threading import Lock
from config import DEVICE
from models.smgnet import SMGNet

logger = logging.getLogger(__name__)


def _register_smgnet_for_unpickle() -> None:
    """Register SMGNet so notebook checkpoints can unpickle."""
    for mod_name in ("__main__", "__mp_main__"):
        if mod_name not in sys.modules:
            continue
        mod = sys.modules[mod_name]
        if hasattr(SMGNet, "__name__"):
            setattr(mod, "SMGNet", SMGNet)


class SMGNetService:
    """
    Singleton service for managing SMG-Net model instance.
    Thread-safe implementation for model loading and access.
    """
    
    _instance: Optional['SMGNetService'] = None
    _lock: Lock = Lock()
    
    def __new__(cls):
        """Create singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(SMGNetService, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the service (only once)."""
        if self._initialized:
            return
        
        self._model: Optional[nn.Module] = None
        self._model_path: Optional[str] = None
        self._num_classes: int = 4
        self._initialized = True
    
    def load_model(self, model_path: str, num_classes: int = 4, in_channels: int = 1) -> nn.Module:
        """
        Load SMG-Net model from weights file.
        
        Args:
            model_path: Path to the model weights file
            num_classes: Number of classification classes (default: 4)
            in_channels: Number of input channels (default: 1)
            
        Returns:
            Loaded SMG-Net model
            
        Raises:
            Exception: If model loading fails
        """
        try:
            logger.info(f"Loading SMG-Net model from {model_path}")
            
            # Initialize model architecture
            smgnet_model = SMGNet(num_classes=num_classes, in_channels=in_channels)
            smgnet_model.to(DEVICE)
            
            # Register for unpickling if needed
            _register_smgnet_for_unpickle()
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            
            # Handle different checkpoint formats
            if isinstance(checkpoint, nn.Module):
                smgnet_model = checkpoint
                smgnet_model.to(DEVICE)
            elif isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    smgnet_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                elif 'state_dict' in checkpoint:
                    smgnet_model.load_state_dict(checkpoint['state_dict'], strict=False)
                elif 'model' in checkpoint:
                    smgnet_model.load_state_dict(checkpoint['model'], strict=False)
                else:
                    # Try loading as state_dict directly
                    smgnet_model.load_state_dict(checkpoint, strict=False)
            else:
                smgnet_model.load_state_dict(checkpoint, strict=False)
            
            smgnet_model.eval()
            
            # Update singleton instance
            with self._lock:
                self._model = smgnet_model
                self._model_path = model_path
                self._num_classes = num_classes
            
            logger.info(f"SMG-Net model loaded successfully (num_classes={num_classes})")
            return smgnet_model
            
        except Exception as e:
            logger.error(f"Error loading SMG-Net model: {e}")
            raise
    
    def load_default_model(self, model_path: Optional[str] = None) -> Optional[nn.Module]:
        """
        Load default SMG-Net model.
        
        Args:
            model_path: Optional path to model file. If None, uses model_weights/smgnet_best.pth
            
        Returns:
            Loaded model if found, None otherwise
        """
        try:
            if model_path is None:
                model_path = Path(__file__).parent.parent / "model_weights" / "smgnet_best.pth"
            else:
                model_path = Path(model_path)
            
            if not model_path.exists():
                logger.warning(f"SMG-Net model not found at {model_path}")
                return None
            
            return self.load_model(str(model_path))
        except Exception as e:
            logger.error(f"Error loading default SMG-Net model: {e}")
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
    
    def get_num_classes(self) -> int:
        """
        Get the number of classes for the loaded model.
        
        Returns:
            Number of classes
        """
        return self._num_classes
