"""
FastAPI Dependencies
Shared dependencies for API endpoints
"""

from fastapi import Header, HTTPException, status
from typing import Optional

from src.model_manager import get_model_manager, ModelManager


async def get_models() -> ModelManager:
    """
    Dependency to get model manager instance.
    
    Returns:
        ModelManager instance
    """
    return get_model_manager()


async def verify_api_key(x_api_key: Optional[str] = Header(None)) -> str:
    """
    Simple API key verification (optional).
    
    Args:
        x_api_key: API key from header
        
    Returns:
        API key if valid
        
    Raises:
        HTTPException: If API key is invalid
    """
    # For now, this is optional - can be enabled by setting environment variable
    # In production, implement proper JWT authentication
    
    # Example: Check if API key matches environment variable
    # if x_api_key != os.getenv("API_KEY"):
    #     raise HTTPException(
    #         status_code=status.HTTP_401_UNAUTHORIZED,
    #         detail="Invalid API key"
    #     )
    
    return x_api_key or "public"
