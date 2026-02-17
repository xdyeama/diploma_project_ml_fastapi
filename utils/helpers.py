import os
import torch
import numpy as np

from core.model_service import ModelService

def _get_model():
    """Get model instance or raise exception if not loaded."""
    model_service = ModelService()
    model = model_service.get_model()
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded. Please upload a model first."
        )
    
    return model


def _adapt_input_channels(tensor: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
    """Repeat input channel if model expects more channels (e.g. legacy 2-channel UNet)."""
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            expected = m.in_channels
            if tensor.shape[1] != expected and tensor.shape[1] == 1:
                tensor = tensor.repeat(1, expected, 1, 1)
            break
    return tensor
