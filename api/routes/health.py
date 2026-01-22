"""
Health check and system information endpoints
"""

from fastapi import APIRouter
from datetime import datetime
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from api.schemas import HealthResponse, ModelInfoResponse
from api import __version__
from src.config import DEFAULT_EXTRACTIVE_MODEL, DEFAULT_ABSTRACTIVE_MODEL, DEVICE

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns system health status and basic information.
    """
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=False,  # Will be updated when model manager is implemented
        version=__version__
    )


@router.get("/models/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about loaded models.
    
    Returns details about the models currently in use.
    """
    return ModelInfoResponse(
        extractive_model=DEFAULT_EXTRACTIVE_MODEL,
        abstractive_model=DEFAULT_ABSTRACTIVE_MODEL,
        device=DEVICE,
        models_loaded=False  # Will be updated when model manager is implemented
    )
