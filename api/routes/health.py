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
from src.model_manager import get_model_manager

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns system health status and basic information.
    """
    model_manager = get_model_manager()
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        models_loaded=model_manager.models_loaded,
        version=__version__
    )


@router.get("/models/info", response_model=ModelInfoResponse)
async def get_model_info():
    """
    Get information about loaded models.
    
    Returns details about the models currently in use.
    """
    model_manager = get_model_manager()
    model_info = model_manager.get_model_info()
    
    return ModelInfoResponse(
        extractive_model=DEFAULT_EXTRACTIVE_MODEL,
        abstractive_model=DEFAULT_ABSTRACTIVE_MODEL,
        device=DEVICE,
        models_loaded=model_info['models_loaded']
    )
