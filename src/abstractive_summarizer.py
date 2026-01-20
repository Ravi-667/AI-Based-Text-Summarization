"""
Abstractive Summarization Module using T5
"""

import torch
from typing import List, Optional
from src.preprocessing import TextPreprocessor
from src.model_loader import ModelLoader
from src.config import (
    ABSTRACTIVE_MAX_LENGTH,
    ABSTRACTIVE_MIN_LENGTH,
    ABSTRACTIVE_LENGTH_PENALTY,
    ABSTRACTIVE_NUM_BEAMS,
    ABSTRACTIVE_EARLY_STOPPING,
    MAX_INPUT_LENGTH
)
from src.utils import setup_logger

logger = setup_logger(__name__)


class AbstractiveSummarizer:
    """T5-based abstractive summarization."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize abstractive summarizer.
        
        Args:
            model_name: Name of T5 model to use
            device: Device for inference ('cuda', 'cpu', or 'auto')
        """
        self.preprocessor = TextPreprocessor()
        self.model_loader = ModelLoader(device=device)
        
        # Load T5 model
        self.model, self.tokenizer = self.model_loader.load_t5_model(model_name)
        
        logger.info("AbstractiveSummarizer initialized")
    
    def generate_summary(
        self,
        text: str,
        max_length: Optional[int] = None,
        min_length: Optional[int] = None,
        length_penalty: Optional[float] = None,
        num_beams: Optional[int] = None,
        early_stopping: Optional[bool] = None
    ) -> str:
        """
        Generate abstractive summary using T5.
        
        Args:
            text: Input text to summarize
            max_length: Maximum summary length in tokens
            min_length: Minimum summary length in tokens
            length_penalty: Length penalty factor
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early
            
        Returns:
            Summary text
        """
        # Use default values if not provided
        max_length = max_length or ABSTRACTIVE_MAX_LENGTH
        min_length = min_length or ABSTRACTIVE_MIN_LENGTH
        length_penalty = length_penalty or ABSTRACTIVE_LENGTH_PENALTY
        num_beams = num_beams or ABSTRACTIVE_NUM_BEAMS
        early_stopping = early_stopping if early_stopping is not None else ABSTRACTIVE_EARLY_STOPPING
        
        # Preprocess text
        cleaned_text, _ = self.preprocessor.preprocess(text)
        
        if not cleaned_text:
            logger.warning("No valid text found")
            return ""
        
        # Add T5 prefix for summarization task
        input_text = f"summarize: {cleaned_text}"
        
        logger.info(f"Generating abstractive summary (max_length={max_length})")
        
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=MAX_INPUT_LENGTH,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.model_loader.device) for k, v in inputs.items()}
        
        # Generate summary
        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs['input_ids'],
                max_length=max_length,
                min_length=min_length,
                length_penalty=length_penalty,
                num_beams=num_beams,
                early_stopping=early_stopping,
                no_repeat_ngram_size=3  # Prevent repetition
            )
        
        # Decode summary
        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        logger.info("Summary generated successfully")
        
        return summary
    
    def generate_summary_chunks(
        self,
        text: str,
        chunk_size: int = 400,
        **kwargs
    ) -> str:
        """
        Generate summary for long text by chunking.
        
        Args:
            text: Input text
            chunk_size: Maximum words per chunk
            **kwargs: Arguments for generate_summary
            
        Returns:
            Combined summary
        """
        # Split into chunks
        chunks = self.preprocessor.chunk_text(text, max_length=chunk_size)
        
        if len(chunks) == 1:
            # Text is short enough, summarize directly
            return self.generate_summary(text, **kwargs)
        
        logger.info(f"Processing {len(chunks)} chunks")
        
        # Summarize each chunk
        chunk_summaries = []
        for i, (chunk_text, _, _) in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}")
            summary = self.generate_summary(chunk_text, **kwargs)
            chunk_summaries.append(summary)
        
        # Combine chunk summaries
        combined_summary_text = ' '.join(chunk_summaries)
        
        # Summarize the combined summaries
        logger.info("Generating final summary from chunk summaries")
        final_summary = self.generate_summary(combined_summary_text, **kwargs)
        
        return final_summary
    
    def batch_summarize(
        self,
        texts: List[str],
        **kwargs
    ) -> List[str]:
        """
        Generate summaries for multiple texts.
        
        Args:
            texts: List of input texts
            **kwargs: Arguments passed to generate_summary
            
        Returns:
            List of summaries
        """
        summaries = []
        
        for i, text in enumerate(texts):
            logger.info(f"Summarizing text {i+1}/{len(texts)}")
            summary = self.generate_summary(text, **kwargs)
            summaries.append(summary)
        
        return summaries
    
    def interactive_summary(
        self,
        text: str,
        length_options: List[str] = ['short', 'medium', 'long']
    ) -> dict:
        """
        Generate summaries with different length options.
        
        Args:
            text: Input text
            length_options: List of length options
            
        Returns:
            Dictionary mapping length option to summary
        """
        length_configs = {
            'short': {'max_length': 80, 'min_length': 30},
            'medium': {'max_length': 150, 'min_length': 50},
            'long': {'max_length': 250, 'min_length': 100}
        }
        
        summaries = {}
        
        for option in length_options:
            if option in length_configs:
                logger.info(f"Generating {option} summary")
                config = length_configs[option]
                summaries[option] = self.generate_summary(text, **config)
        
        return summaries


def summarize_abstractive(
    text: str,
    max_length: int = 150,
    min_length: int = 50
) -> str:
    """
    Convenience function for abstractive summarization.
    
    Args:
        text: Input text
        max_length: Maximum summary length
        min_length: Minimum summary length
        
    Returns:
        Summary text
    """
    summarizer = AbstractiveSummarizer()
    return summarizer.generate_summary(
        text,
        max_length=max_length,
        min_length=min_length
    )
