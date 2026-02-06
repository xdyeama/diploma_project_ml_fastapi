"""
Model management endpoints.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from pathlib import Path
from datetime import datetime
import logging

from schemas import ModelUploadResponse
from core.model_service import ModelService
from config import UPLOADED_MODELS_DIR

router = APIRouter(prefix="/upload-model", tags=["model"])
logger = logging.getLogger(__name__)


@router.post("", response_model=ModelUploadResponse)
async def upload_model(file: UploadFile = File(...)):
    """
    Upload and load a new model weights file.
    
    Accepts .pt or .pth files containing PyTorch model weights.
    """
    model_service = ModelService()
    
    if not file.filename.endswith(('.pt', '.pth')):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Please upload a .pt or .pth file."
        )
    
    try:
        # Save uploaded file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = UPLOADED_MODELS_DIR / filename
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Load the model using singleton service
        logger.info(f"Loading uploaded model from {file_path}")
        model_service.load_model(str(file_path))
        
        return ModelUploadResponse(
            status="success",
            message="Model uploaded and loaded successfully",
            filename=filename,
            model_path=str(file_path)
        )
    
    except Exception as e:
        logger.error(f"Error uploading model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload model: {str(e)}"
        )
