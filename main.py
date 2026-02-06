"""
FastAPI application entry point.
U-Net CNN brain tumor segmentation model service.
"""

from fastapi import FastAPI
import logging

from api import health, model, inference, download
from core.model_service import ModelService
from config import LOG_LEVEL

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Brain Tumor Segmentation API",
    description="U-Net CNN model for brain tumor segmentation from MRI images",
    version="1.0.0"
)

# Include routers
app.include_router(health.router)
app.include_router(model.router)
app.include_router(inference.router)
app.include_router(download.router)


@app.on_event("startup")
async def startup_event():
    """Load default model on startup."""
    logger.info("Starting up application...")
    model_service = ModelService()
    model_service.load_default_model()
    logger.info("Startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
