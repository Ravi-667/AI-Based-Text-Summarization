"""
Model Loader and Manager for AI-Based Text Summarization
"""

import os
import torch
from typing import Tuple, Optional
from transformers import (
    AutoTokenizer, 
    AutoModel,
    T5Tokenizer,
    T5ForConditionalGeneration
)
from src.config import (
    DEFAULT_EXTRACTIVE_MODEL,
    DEFAULT_ABSTRACTIVE_MODEL,
    MODEL_CACHE_DIR,
    DEVICE,
    USE_GPU
)
from src.utils import setup_logger

logger = setup_logger(__name__)


class ModelLoader:
    """Manages loading and caching of pre-trained models."""
    
    def __init__(self, device: Optional[str] = None):
        """
        Initialize model loader.
        
        Args:
            device: Device to use ('cuda', 'cpu', or 'auto')
        """
        self.device = self._get_device(device)
        self.cache_dir = MODEL_CACHE_DIR
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info(f"ModelLoader initialized with device: {self.device}")
    
    def _get_device(self, device: Optional[str] = None) -> str:
        """
        Determine which device to use.
        
        Args:
            device: Requested device
            
        Returns:
            Device string ('cuda' or 'cpu')
        """
        if device is None:
            device = DEVICE
        
        if device == "auto":
            if USE_GPU and torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Using CPU.")
            return "cpu"
        
        return device
    
    def load_bert_model(
        self, 
        model_name: Optional[str] = None
    ) -> Tuple[AutoModel, AutoTokenizer]:
        """
        Load BERT model for extractive summarization.
        
        Args:
            model_name: Name of the BERT model to load
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name is None:
            model_name = DEFAULT_EXTRACTIVE_MODEL
        
        logger.info(f"Loading BERT model: {model_name}")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            model = AutoModel.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            logger.info(f"Successfully loaded {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading BERT model: {e}")
            raise
    
    def load_t5_model(
        self,
        model_name: Optional[str] = None
    ) -> Tuple[T5ForConditionalGeneration, T5Tokenizer]:
        """
        Load T5 model for abstractive summarization.
        
        Args:
            model_name: Name of the T5 model to load
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if model_name is None:
            model_name = DEFAULT_ABSTRACTIVE_MODEL
        
        logger.info(f"Loading T5 model: {model_name}")
        
        try:
            tokenizer = T5Tokenizer.from_pretrained(
                model_name,
                cache_dir=self.cache_dir,
                legacy=False
            )
            
            model = T5ForConditionalGeneration.from_pretrained(
                model_name,
                cache_dir=self.cache_dir
            )
            
            model = model.to(self.device)
            model.eval()  # Set to evaluation mode
            
            logger.info(f"Successfully loaded {model_name}")
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Error loading T5 model: {e}")
            raise
    
    def clear_cache(self) -> None:
        """Clear model cache directory."""
        import shutil
        
        if os.path.exists(self.cache_dir):
            logger.info(f"Clearing cache directory: {self.cache_dir}")
            shutil.rmtree(self.cache_dir)
            os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_model_info(self) -> dict:
        """
        Get information about available models and device.
        
        Returns:
            Dictionary with system information
        """
        info = {
            'device': self.device,
            'cuda_available': torch.cuda.is_available(),
            'cache_dir': self.cache_dir,
            'default_extractive': DEFAULT_EXTRACTIVE_MODEL,
            'default_abstractive': DEFAULT_ABSTRACTIVE_MODEL
        }
        
        if torch.cuda.is_available():
            info['cuda_device_name'] = torch.cuda.get_device_name(0)
            info['cuda_memory_allocated'] = f"{torch.cuda.memory_allocated(0) / 1e9:.2f} GB"
        
        return info


def get_device() -> str:
    """
    Get the appropriate device for model inference.
    
    Returns:
        Device string ('cuda' or 'cpu')
    """
    loader = ModelLoader()
    return loader.device
