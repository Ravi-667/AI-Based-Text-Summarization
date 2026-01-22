"""
Model Manager - Singleton pattern for efficient model management
Loads models once and reuses across requests
"""

import logging
from typing import Optional
from threading import Lock

from src.extractive_summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.config import DEVICE

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton model manager for efficient model caching.
    Ensures models are loaded only once and reused across requests.
    """
    
    _instance: Optional['ModelManager'] = None
    _lock: Lock = Lock()
    
    def __new__(cls):
        """Singleton pattern implementation."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize model manager (only once)."""
        if self._initialized:
            return
        
        self._extractive_model: Optional[ExtractiveSummarizer] = None
        self._abstractive_model: Optional[AbstractiveSummarizer] = None
        self._device = DEVICE
        self._models_loaded = False
        self._initialized = True
        
        logger.info("ModelManager initialized")
    
    @property
    def extractive_model(self) -> ExtractiveSummarizer:
        """
        Get extractive summarizer (lazy loading).
        
        Returns:
            ExtractiveSummarizer instance
        """
        if self._extractive_model is None:
            with self._lock:
                if self._extractive_model is None:
                    logger.info("Loading extractive model...")
                    self._extractive_model = ExtractiveSummarizer(device=self._device)
                    logger.info("Extractive model loaded successfully")
        
        return self._extractive_model
    
    @property
    def abstractive_model(self) -> AbstractiveSummarizer:
        """
        Get abstractive summarizer (lazy loading).
        
        Returns:
            AbstractiveSummarizer instance
        """
        if self._abstractive_model is None:
            with self._lock:
                if self._abstractive_model is None:
                    logger.info("Loading abstractive model...")
                    self._abstractive_model = AbstractiveSummarizer(device=self._device)
                    logger.info("Abstractive model loaded successfully")
        
        return self._abstractive_model
    
    @property
    def models_loaded(self) -> bool:
        """Check if any models are loaded."""
        return (self._extractive_model is not None or 
                self._abstractive_model is not None)
    
    def preload_models(self) -> None:
        """
        Preload both models on startup.
        Useful for faster first request response.
        """
        logger.info("Preloading models...")
        _ = self.extractive_model
        _ = self.abstractive_model
        self._models_loaded = True
        logger.info("All models preloaded successfully")
    
    def get_model_info(self) -> dict:
        """
        Get information about loaded models.
        
        Returns:
            Dictionary with model information
        """
        return {
            'extractive_loaded': self._extractive_model is not None,
            'abstractive_loaded': self._abstractive_model is not None,
            'device': self._device,
            'models_loaded': self.models_loaded
        }
    
    def clear_cache(self) -> None:
        """
        Clear model cache (for testing or memory management).
        WARNING: This will force models to reload on next request.
        """
        logger.warning("Clearing model cache...")
        self._extractive_model = None
        self._abstractive_model = None
        self._models_loaded = False
        logger.info("Model cache cleared")


# Global instance
_model_manager = ModelManager()


def get_model_manager() -> ModelManager:
    """
    Get the global model manager instance.
    
    Returns:
        ModelManager singleton instance
    """
    return _model_manager
