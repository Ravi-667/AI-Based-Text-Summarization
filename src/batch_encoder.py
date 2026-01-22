"""
Batch encoding utilities for performance optimization
"""

import numpy as np
import torch
from typing import List
from tqdm import tqdm


def batch_encode_sentences(
    sentences: List[str],
    model,
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    device: str = 'cpu',
    show_progress: bool = False
) -> np.ndarray:
    """
    Encode multiple sentences in batches for better performance.
    
    Args:
        sentences: List of sentences to encode
        model: BERT model for encoding
        tokenizer: Tokenizer for the model
        batch_size: Number of sentences per batch
        max_length: Maximum sequence length
        device: Device for computation
        show_progress: Show progress bar
        
    Returns:
        Array of sentence embeddings
    """
    if len(sentences) == 0:
        return np.array([])
    
    embeddings = []
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    
    iterator = range(0, len(sentences), batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches, desc="Encoding batches")
    
    for i in iterator:
        batch_sentences = sentences[i:i + batch_size]
        
        # Tokenize batch
        inputs = tokenizer(
            batch_sentences,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use [CLS] token embeddings
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.extend(batch_embeddings)
    
    return np.array(embeddings)


def optimize_batch_size(num_sentences: int, max_batch_size: int = 64) -> int:
    """
    Determine optimal batch size based on number of sentences.
    
    Args:
        num_sentences: Number of sentences to process
        max_batch_size: Maximum allowed batch size
        
    Returns:
        Optimal batch size
    """
    if num_sentences <= 8:
        return num_sentences
    elif num_sentences <= 32:
        return 8
    elif num_sentences <= 128:
        return 16
    else:
        return min(32, max_batch_size)
