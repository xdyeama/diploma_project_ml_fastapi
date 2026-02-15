"""
Inference endpoints for image and NIfTI file processing.
"""

from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path
from datetime import datetime
import tempfile
import os
import torch
import numpy as np
from PIL import Image
import nibabel as nib
import io
import logging

from schemas import InferenceResponse
from core.model_service import ModelService
from utils.preprocessing import preprocess_image, preprocess_nifti
from utils.postprocessing import postprocess_segmentation
from config import UPLOADED_MODELS_DIR, DEVICE

router = APIRouter(prefix="/inference", tags=["inference"])
logger = logging.getLogger(__name__)


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


@router.post("/image", response_model=InferenceResponse)
async def inference_image(file: UploadFile = File(...)):
    """
    Perform inference on a grayscale image.
    
    Accepts PNG, JPEG, or TIFF images. The image will be preprocessed
    and segmented using the U-Net model.
    """
    model = _get_model()
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Convert to grayscale
        
        # Preprocess
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.to(DEVICE)
        input_tensor = _adapt_input_channels(input_tensor, model)
        
        # Inference
        with torch.no_grad():
            output = model(input_tensor)
            segmentation = torch.argmax(output, dim=1).cpu().numpy()[0]
        
        # Postprocess
        processed_segmentation = postprocess_segmentation(segmentation)
        
        # Convert to image and save
        seg_image = Image.fromarray(
            (processed_segmentation * 255 / processed_segmentation.max()).astype(np.uint8)
        )
        output_filename = f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        output_path = UPLOADED_MODELS_DIR / output_filename
        seg_image.save(output_path, format="PNG")
        
        return InferenceResponse(
            status="success",
            message="Inference completed successfully",
            shape=list(segmentation.shape),
            classes_detected=list(np.unique(segmentation)),
            output_format="image",
            output_path=output_filename
        )
    
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference failed: {str(e)}"
        )


@router.post("/nifti", response_model=InferenceResponse)
async def inference_nifti(file: UploadFile = File(...)):
    """
    Perform inference on a NIfTI file (.nii or .nii.gz).
    
    Processes the entire 3D volume and returns segmentation results.
    """
    model = _get_model()
    
    try:
        # Read NIfTI file; nib.load() requires a path, so write upload to a temp file
        nifti_bytes = await file.read()
        suffix = Path(file.filename).suffix if file.filename else ".nii"
        if suffix == ".gz" and file.filename and file.filename.endswith(".nii.gz"):
            suffix = ".nii.gz"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(nifti_bytes)
            tmp_path = tmp.name
        try:
            nifti_img = nib.load(tmp_path)
            volume_data = nifti_img.get_fdata()
        finally:
            os.unlink(tmp_path)
        
        # Preprocess volume
        processed_volume = preprocess_nifti(volume_data)
        
        # Process slice by slice
        segmentation_volume = []
        for i in range(processed_volume.shape[2]):
            slice_data = processed_volume[:, :, i]
            input_tensor = torch.from_numpy(slice_data).unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            input_tensor = _adapt_input_channels(input_tensor, model)
            
            with torch.no_grad():
                output = model(input_tensor)
                seg_slice = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            segmentation_volume.append(seg_slice)
        
        segmentation_volume = np.stack(segmentation_volume, axis=2)
        
        # Create output NIfTI
        seg_nifti = nib.Nifti1Image(
            segmentation_volume.astype(np.uint8),
            nifti_img.affine,
            nifti_img.header
        )
        
        # Save to temporary file
        output_filename = f"segmentation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.nii.gz"
        output_path = UPLOADED_MODELS_DIR / output_filename
        nib.save(seg_nifti, output_path)
        
        return InferenceResponse(
            status="success",
            message="NIfTI inference completed successfully",
            shape=list(segmentation_volume.shape),
            classes_detected=list(np.unique(segmentation_volume)),
            output_format="nifti",
            output_path=output_filename
        )
    
    except Exception as e:
        logger.error(f"Error during NIfTI inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NIfTI inference failed: {str(e)}"
        )


@router.post("/nifti/flair", response_model=InferenceResponse)
async def inference_nifti_flair(file: UploadFile = File(...)):
    """
    Perform inference on a FLAIR NIfTI file.
    
    FLAIR (Fluid Attenuated Inversion Recovery) is a common MRI sequence
    for brain tumor detection.
    """
    return await inference_nifti(file)


@router.post("/nifti/t1ce", response_model=InferenceResponse)
async def inference_nifti_t1ce(file: UploadFile = File(...)):
    """
    Perform inference on a T1CE (T1-weighted contrast-enhanced) NIfTI file.
    
    T1CE is commonly used for brain tumor segmentation as it highlights
    enhancing tumor regions.
    """
    return await inference_nifti(file)


