"""
Pydantic schemas for request/response models.
"""

from pydantic import BaseModel
from typing import List, Optional



class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    message: str
    model_loaded: bool
    device: str


class ModelUploadResponse(BaseModel):
    """Model upload response."""
    status: str
    message: str
    filename: str
    model_path: str


class InferenceResponse(BaseModel):
    """Inference response."""
    status: str
    message: str
    shape: List[int]
    classes_detected: List[int]
    output_format: str
    output_path: Optional[str] = None
    nifti_path: Optional[str] = None
    image_base64: Optional[str] = None 