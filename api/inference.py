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
import base64
import shutil
import math
from schemas import InferenceResponse
from core.model_service import ModelService
from utils.preprocessing import preprocess_image, preprocess_nifti
from utils.postprocessing import postprocess_segmentation
from utils.visualization import build_overlay_grid
from utils.dicom_handler import _build_overlay_grid, _load_dicom_volume, _crop_black_borders
from utils.helpers import _get_model, _adapt_input_channels
from config import UPLOADED_MODELS_DIR, DEVICE
import dicom2nifti
import dicom2nifti.settings as dicom2nifti_settings

router = APIRouter(prefix="/inference", tags=["inference"])
logger = logging.getLogger(__name__)


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

    Processes the entire 3D volume slice-by-slice, then combines every axial
    slice into a single coloured-overlay PNG grid (FLAIR + segmentation mask)
    matching the showPredictsById colour scheme:
        red   → necrotic core
        green → edema
        blue  → enhancing tumour
    """
    model = _get_model()

    try:
        # ── 1. Write upload to a temp file so nibabel can load it ─────────
        nifti_bytes = await file.read()
        suffix = Path(file.filename).suffix if file.filename else ".nii"
        if suffix == ".gz" and file.filename and file.filename.endswith(".nii.gz"):
            suffix = ".nii.gz"

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(nifti_bytes)
            tmp_path = tmp.name

        try:
            nifti_img = nib.load(tmp_path)
            volume_data = nifti_img.get_fdata()      # raw FLAIR voxels (H, W, D)
        finally:
            os.unlink(tmp_path)

        # ── 2. Preprocess & run slice-by-slice inference ──────────────────
        processed_volume = preprocess_nifti(volume_data)  # (H, W, D)

        segmentation_volume = []
        for i in range(processed_volume.shape[2]):
            slice_data = processed_volume[:, :, i]
            input_tensor = (
                torch.from_numpy(slice_data)
                .unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            )
            input_tensor = _adapt_input_channels(input_tensor, model)

            with torch.no_grad():
                output = model(input_tensor)
                seg_slice = torch.argmax(output, dim=1).cpu().numpy()[0]

            segmentation_volume.append(seg_slice)

        segmentation_volume = np.stack(segmentation_volume, axis=2)   # (H, W, D)
        seg_volume_uint8 = segmentation_volume.astype(np.uint8)

        # ── 3. Save full 3-D segmentation as NIfTI ────────────────────────
        seg_nifti = nib.Nifti1Image(
            seg_volume_uint8,
            nifti_img.affine,
            nifti_img.header,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nifti_filename = f"segmentation_{timestamp}.nii.gz"
        nib.save(seg_nifti, UPLOADED_MODELS_DIR / nifti_filename)

        # ── 4. Build combined overlay PNG (all slices in a grid) ──────────
        png_bytes = build_overlay_grid(
            flair_volume=volume_data,          # original, un-preprocessed FLAIR
            seg_volume=seg_volume_uint8,
            img_size=128,
            alpha=0.5,
            max_slices=100,
            cols=10,
        )
        png_filename = f"segmentation_{timestamp}_overlay.png"
        png_output_path = UPLOADED_MODELS_DIR / png_filename
        png_output_path.write_bytes(png_bytes)

        # ── 5. Encode PNG as base64 for the response ──────────────────────
        png_base64 = base64.b64encode(png_bytes).decode("utf-8")

        return InferenceResponse(
            status="success",
            message="NIfTI inference completed successfully",
            shape=list(segmentation_volume.shape),
            classes_detected=list(map(int, np.unique(segmentation_volume))),
            output_format="image",
            output_path=png_filename,
            nifti_path=nifti_filename,
            # Callers can decode this directly: data:image/png;base64,<png_base64>
            image_base64=png_base64,
        )

    except Exception as e:
        logger.error(f"Error during NIfTI inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"NIfTI inference failed: {str(e)}",
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


@router.post("/dicom", response_model=InferenceResponse)
async def inference_dicom(files: list[UploadFile] = File(...)):
    """
    Perform inference on one or more DICOM files (.dcm).

    Accepts a single file or a series of slices (uploaded together).
    Multiple files are sorted by InstanceNumber and stacked into a 3-D volume
    before segmentation, producing a much richer overlay grid.

    Pipeline:
        1. Save all uploaded .dcm files to a temp directory
        2. Load + sort + stack into a (H, W, D) volume via pydicom
        3. Convert to NIfTI with nibabel
        4. Run slice-by-slice segmentation (same as /nifti endpoint)
        5. Build overlay grid PNG, crop black borders, return as base64
    """
    model = _get_model()

    tmp_dicom_dir = None
    tmp_nifti_dir = None

    try:
        # ── 1. Save all uploaded .dcm files ──────────────────────────────
        tmp_dicom_dir = tempfile.mkdtemp(prefix="dcm_in_")
        dcm_paths = []
        for upload in files:
            dest = os.path.join(tmp_dicom_dir, upload.filename or f"{len(dcm_paths)}.dcm")
            content = await upload.read()
            with open(dest, "wb") as f:
                f.write(content)
            dcm_paths.append(dest)

        if not dcm_paths:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="No DICOM files received.",
            )

        # ── 2. Load + stack into a (H, W, D) volume ──────────────────────
        try:
            volume_data, affine = _load_dicom_volume(dcm_paths)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=f"DICOM loading failed: {e}",
            )

        # Handle 4-D (e.g. fMRI): take first time-point
        if volume_data.ndim == 4:
            volume_data = volume_data[..., 0]

        # ── 3. Wrap in a NIfTI image (no file I/O needed) ────────────────
        nifti_img = nib.Nifti1Image(volume_data, affine)

        # ── 4. Preprocess & run slice-by-slice inference ──────────────────
        processed_volume = preprocess_nifti(volume_data)

        segmentation_volume = []
        for i in range(processed_volume.shape[2]):
            slice_data   = processed_volume[:, :, i]
            input_tensor = (
                torch.from_numpy(slice_data)
                .unsqueeze(0).unsqueeze(0).float().to(DEVICE)
            )
            input_tensor = _adapt_input_channels(input_tensor, model)

            with torch.no_grad():
                output    = model(input_tensor)
                seg_slice = torch.argmax(output, dim=1).cpu().numpy()[0]

            segmentation_volume.append(seg_slice)

        segmentation_volume = np.stack(segmentation_volume, axis=2)  # (H, W, D)
        seg_volume_uint8    = segmentation_volume.astype(np.uint8)

        # ── 5. Save full 3-D segmentation as NIfTI ───────────────────────
        seg_nifti = nib.Nifti1Image(seg_volume_uint8, affine)
        timestamp      = datetime.now().strftime("%Y%m%d_%H%M%S")
        nifti_filename = f"segmentation_dicom_{timestamp}.nii.gz"
        nib.save(seg_nifti, UPLOADED_MODELS_DIR / nifti_filename)

        # ── 6. Build overlay grid PNG + crop black borders ────────────────
        png_bytes    = _build_overlay_grid(
            flair_volume=volume_data,
            seg_volume=seg_volume_uint8,
            img_size=128,
            alpha=0.5,
            max_slices=100,
            cols=10,
        )
        png_filename = f"segmentation_dicom_{timestamp}_overlay.png"
        (UPLOADED_MODELS_DIR / png_filename).write_bytes(png_bytes)

        # ── 7. Base64-encode PNG for JSON response ────────────────────────
        png_base64 = base64.b64encode(png_bytes).decode("utf-8")

        return InferenceResponse(
            status="success",
            message=f"DICOM inference completed ({len(dcm_paths)} file(s), "
                    f"{segmentation_volume.shape[2]} slice(s))",
            shape=list(segmentation_volume.shape),
            classes_detected=list(map(int, np.unique(segmentation_volume))),
            output_format="image",
            output_path=png_filename,
            nifti_path=nifti_filename,
            image_base64=png_base64,
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Error during DICOM inference: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DICOM inference failed: {str(e)}",
        )

    finally:
        for tmp_dir in (tmp_dicom_dir, tmp_nifti_dir):
            if tmp_dir and os.path.isdir(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
