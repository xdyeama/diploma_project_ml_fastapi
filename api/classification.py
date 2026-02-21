"""
Classification endpoints for SMG-Net model.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from pathlib import Path
from datetime import datetime
import torch
import numpy as np
from PIL import Image
import io
import logging
import torchvision.transforms as transforms

from schemas import ClassificationResponse
from core.smgnet_service import SMGNetService
from config import DEVICE

router = APIRouter(prefix="/classification", tags=["classification"])
logger = logging.getLogger(__name__)

# Default class names (can be customized)
DEFAULT_CLASS_NAMES = [
    "Normal",
    "Alzheimer's Disease",
    "Mild Cognitive Impairment",
    "Vascular Dementia"
]


def _get_smgnet_model():
    """Get SMG-Net model instance or raise exception if not loaded."""
    service = SMGNetService()
    model = service.get_model()
    
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="SMG-Net model not loaded. Please load the model first."
        )
    
    return model, service


def _preprocess_for_classification(image: Image.Image, target_size: tuple = (512, 512)) -> torch.Tensor:
    """
    Preprocess image for SMG-Net classification.
    
    Args:
        image: PIL Image
        target_size: Target size (height, width)
        
    Returns:
        Preprocessed tensor [1, 1, H, W]
    """

    transform_classification = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])
    
    tensor = transform_classification(image)
    return tensor.unsqueeze(0)  # Add batch dimension


@router.post("/classify_image", response_model=ClassificationResponse)
async def classify_image(
    file: UploadFile = File(...),
    class_names: str = "Alzheimer's, MS, Normal, Tumor"
):
    """
    Classify an MRI image using SMG-Net.
    
    Accepts PNG, JPEG, or TIFF images. Returns classification results
    with predicted class and probabilities.
    
    Args:
        file: Image file to classify
        class_names: Optional comma-separated class names (e.g., "Normal,AD,MCI,VD")
    """
    model, service = _get_smgnet_model()
    
    try:
        # Parse class names if provided
        if class_names:
            names = [name.strip() for name in class_names.split(",")]
        else:
            names = DEFAULT_CLASS_NAMES[:service.get_num_classes()]
        
        # Read and preprocess image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L") 
        
        input_tensor = _preprocess_for_classification(image)
        input_tensor = input_tensor.to(DEVICE)
        
        model.eval()
        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Convert to lists
        prob_list = probabilities[0].cpu().numpy().tolist()
        
        return ClassificationResponse(
            status="success",
            message="Classification completed successfully",
            predicted_class=int(predicted_class),
            predicted_class_name=names[predicted_class] if predicted_class < len(names) else f"Class {predicted_class}",
            class_probabilities=prob_list,
            class_names=names,
            confidence=float(confidence)
        )
    
    except Exception as e:
        logger.error(f"Error during classification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Classification failed: {str(e)}"
        )


@router.post("/classify_batch", response_model=list[ClassificationResponse])
async def classify_batch(
    files: list[UploadFile] = File(...),
    class_names: str = "Alzheimer's, MS, Normal, Tumor"
):
    """
    Classify multiple MRI images in batch.
    
    Args:
        files: List of image files to classify
        class_names: Optional comma-separated class names
    """
    model, service = _get_smgnet_model()
    
    try:
        # Parse class names
        if class_names:
            names = [name.strip() for name in class_names.split(",")]
        else:
            names = DEFAULT_CLASS_NAMES[:service.get_num_classes()]
        
        results = []
        
        for file in files:
            try:
                # Read and preprocess
                image_bytes = await file.read()
                image = Image.open(io.BytesIO(image_bytes)).convert("L") 
                input_tensor = _preprocess_for_classification(image)
                input_tensor = input_tensor.to(DEVICE)
                
                model.eval()
                # Inference
                with torch.no_grad():
                    logits = model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_class].item()
                
                prob_list = probabilities[0].cpu().numpy().tolist()
                
                results.append(ClassificationResponse(
                    status="success",
                    message=f"Classification completed for {file.filename}",
                    predicted_class=int(predicted_class),
                    predicted_class_name=names[predicted_class] if predicted_class < len(names) else f"Class {predicted_class}",
                    class_probabilities=prob_list,
                    class_names=names,
                    confidence=float(confidence)
                ))
            except Exception as e:
                logger.error(f"Error classifying {file.filename}: {e}")
                results.append(ClassificationResponse(
                    status="error",
                    message=f"Failed to classify {file.filename}: {str(e)}",
                    predicted_class=-1,
                    predicted_class_name=None,
                    class_probabilities=[],
                    class_names=names,
                    confidence=0.0
                ))
        
        return results
    
    except Exception as e:
        logger.error(f"Error during batch classification: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch classification failed: {str(e)}"
        )
