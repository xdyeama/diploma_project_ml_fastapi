"""
Health check endpoints.
"""

from fastapi import APIRouter
from schemas import HealthResponse
from core.model_service import ModelService
from config import DEVICE

router = APIRouter(prefix="", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    model_service = ModelService()
    return HealthResponse(
        status="healthy",
        message="Brain Tumor Segmentation API is running",
        model_loaded=model_service.is_model_loaded(),
        device=str(DEVICE)
    )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    model_service = ModelService()
    return HealthResponse(
        status="healthy",
        message="Service is operational",
        model_loaded=model_service.is_model_loaded(),
        device=str(DEVICE)
    )
