"""
Extractive Summarization Module using BERT
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
from src.preprocessing import TextPreprocessor
from src.model_loader import ModelLoader
from src.sentence_ranker import SentenceRanker
from src.config import (
    EXTRACTIVE_RATIO,
    EXTRACTIVE_MIN_SENTENCES,
    EXTRACTIVE_MAX_SENTENCES,
    MAX_INPUT_LENGTH
)
from src.utils import setup_logger

logger = setup_logger(__name__)


class ExtractiveSummarizer:
    """BERT-based extractive summarization."""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize extractive summarizer.
        
        Args:
            model_name: Name of BERT model to use
            device: Device for inference ('cuda', 'cpu', or 'auto')
        """
        self.preprocessor = TextPreprocessor()
        self.ranker = SentenceRanker()
        self.model_loader = ModelLoader(device=device)
        
        # Load BERT model
        self.model, self.tokenizer = self.model_loader.load_bert_model(model_name)
        
        logger.info("ExtractiveSummarizer initialized")
    
    def encode_sentences(self, sentences: List[str]) -> np.ndarray:
        """
        Encode sentences using BERT to get embeddings.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Sentence embeddings matrix (num_sentences x embedding_dim)
        """
        embeddings = []
        
        for sentence in sentences:
            # Tokenize
            inputs = self.tokenizer(
                sentence,
                return_tensors='pt',
                truncation=True,
                max_length=MAX_INPUT_LENGTH,
                padding=True
            )
            
            # Move to device
            inputs = {k: v.to(self.model_loader.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use [CLS] token embedding as sentence representation
                sentence_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.append(sentence_embedding[0])
        
        return np.array(embeddings)
    
    def score_sentences(
        self,
        sentences: List[str],
        method: str = 'combined',
        use_embeddings: bool = True
    ) -> np.ndarray:
        """
        Score sentences for importance.
        
        Args:
            sentences: List of sentences
            method: Scoring method ('tfidf', 'textrank', 'lexrank', 'combined')
            use_embeddings: Whether to use BERT embeddings
            
        Returns:
            Array of sentence scores
        """
        if len(sentences) == 0:
            return np.array([])
        
        embeddings = None
        if use_embeddings and method in ['textrank', 'combined']:
            logger.info(f"Encoding {len(sentences)} sentences with BERT...")
            embeddings = self.encode_sentences(sentences)
        
        logger.info(f"Scoring sentences using method: {method}")
        
        if method == 'tfidf':
            scores = self.ranker.tfidf_score(sentences)
        elif method == 'textrank':
            if embeddings is None:
                logger.warning("TextRank requires embeddings. Falling back to LexRank.")
                scores = self.ranker.lexrank(sentences)
            else:
                scores = self.ranker.textrank(sentences, embeddings)
        elif method == 'lexrank':
            scores = self.ranker.lexrank(sentences)
        elif method == 'combined':
            scores = self.ranker.combined_score(sentences, embeddings)
        else:
            raise ValueError(f"Unknown scoring method: {method}")
        
        return scores
    
    def generate_summary(
        self,
        text: str,
        ratio: Optional[float] = None,
        num_sentences: Optional[int] = None,
        method: str = 'combined',
        use_embeddings: bool = True,
        return_indices: bool = False
    ) -> str:
        """
        Generate extractive summary.
        
        Args:
            text: Input text to summarize
            ratio: Ratio of sentences to extract (0-1)
            num_sentences: Exact number of sentences to extract
            method: Scoring method to use
            use_embeddings: Whether to use BERT embeddings
            return_indices: If True, also return indices of selected sentences
            
        Returns:
            Summary text (and indices if return_indices=True)
        """
        # Preprocess text
        cleaned_text, sentences = self.preprocessor.preprocess(text)
        
        if len(sentences) == 0:
            logger.warning("No valid sentences found in text")
            return ""
        
        logger.info(f"Processing {len(sentences)} sentences")
        
        # Determine number of sentences to extract
        if num_sentences is None:
            if ratio is None:
                ratio = EXTRACTIVE_RATIO
            
            num_sentences = max(
                EXTRACTIVE_MIN_SENTENCES,
                min(
                    EXTRACTIVE_MAX_SENTENCES,
                    int(len(sentences) * ratio)
                )
            )
        
        # Don't extract more sentences than available
        num_sentences = min(num_sentences, len(sentences))
        
        logger.info(f"Extracting {num_sentences} sentences")
        
        # Score sentences
        scores = self.score_sentences(sentences, method, use_embeddings)
        
        # Select top sentences
        top_indices = np.argsort(scores)[-num_sentences:]
        
        # Sort indices to maintain original order
        top_indices = sorted(top_indices)
        
        # Extract summary sentences
        summary_sentences = [sentences[i] for i in top_indices]
        summary = ' '.join(summary_sentences)
        
        logger.info("Summary generated successfully")
        
        if return_indices:
            return summary, top_indices
        
        return summary
    
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


def summarize_extractive(
    text: str,
    ratio: float = 0.3,
    method: str = 'combined'
) -> str:
    """
    Convenience function for extractive summarization.
    
    Args:
        text: Input text
        ratio: Extraction ratio
        method: Scoring method
        
    Returns:
        Summary text
    """
    summarizer = ExtractiveSummarizer()
    return summarizer.generate_summary(text, ratio=ratio, method=method)
