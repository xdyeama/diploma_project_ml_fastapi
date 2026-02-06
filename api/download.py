"""
Download endpoints for segmentation results.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse
from pathlib import Path

from config import UPLOADED_MODELS_DIR

router = APIRouter(prefix="", tags=["download"])


@router.get("/download-segmentation/{filename}")
async def download_segmentation(filename: str):
    """
    Download a segmentation result file.
    
    Files are stored temporarily after inference.
    """
    file_path = UPLOADED_MODELS_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Segmentation file not found"
        )
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/octet-stream"
    )
