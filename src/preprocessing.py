"""
Text Preprocessing Module for AI-Based Text Summarization
"""

import re
import nltk
from typing import List, Tuple
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextPreprocessor:
    """Handles text cleaning and preprocessing for summarization."""
    
    def __init__(self, remove_stopwords: bool = False):
        """
        Initialize text preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords (for some scoring methods)
        """
        self.remove_stopwords = remove_stopwords
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text
        """
        # Remove multiple spaces
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep sentence structure
        text = re.sub(r'[^\w\s.,!?;:\-\']', '', text)
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([.,!?;:])', r'\1', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n+', '\n', text)
        
        return text.strip()
    
    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Input text
            
        Returns:
            List of sentences
        """
        sentences = sent_tokenize(text)
        
        # Filter out very short or very long sentences
        filtered_sentences = []
        for sent in sentences:
            word_count = len(sent.split())
            if 5 <= word_count <= 100:  # Configurable limits
                filtered_sentences.append(sent.strip())
        
        return filtered_sentences
    
    def tokenize_words(self, text: str, remove_stop: bool = None) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            remove_stop: Override instance setting for stopword removal
            
        Returns:
            List of word tokens
        """
        words = word_tokenize(text.lower())
        
        # Remove stopwords if specified
        if remove_stop or (remove_stop is None and self.remove_stopwords):
            words = [w for w in words if w not in self.stop_words and w.isalnum()]
        else:
            words = [w for w in words if w.isalnum()]
        
        return words
    
    def chunk_text(
        self, 
        text: str, 
        max_length: int = 512, 
        overlap: int = 50
    ) -> List[Tuple[str, int, int]]:
        """
        Split long text into overlapping chunks.
        
        Args:
            text: Input text
            max_length: Maximum tokens per chunk
            overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of tuples (chunk_text, start_pos, end_pos)
        """
        sentences = self.tokenize_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0
        start_idx = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence.split())
            
            if current_length + sentence_length > max_length and current_chunk:
                # Save current chunk
                chunk_text = ' '.join(current_chunk)
                chunks.append((chunk_text, start_idx, i))
                
                # Start new chunk with overlap
                overlap_sentences = current_chunk[-min(2, len(current_chunk)):]
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s.split()) for s in current_chunk)
                start_idx = i - len(overlap_sentences)
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add remaining chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append((chunk_text, start_idx, len(sentences)))
        
        return chunks
    
    def preprocess(self, text: str) -> Tuple[str, List[str]]:
        """
        Complete preprocessing pipeline.
        
        Args:
            text: Raw input text
            
        Returns:
            Tuple of (cleaned_text, sentences)
        """
        cleaned_text = self.clean_text(text)
        sentences = self.tokenize_sentences(cleaned_text)
        
        return cleaned_text, sentences


def preprocess_text(text: str, remove_stopwords: bool = False) -> Tuple[str, List[str]]:
    """
    Convenience function for text preprocessing.
    
    Args:
        text: Input text
        remove_stopwords: Whether to remove stopwords
        
    Returns:
        Tuple of (cleaned_text, sentences)
    """
    preprocessor = TextPreprocessor(remove_stopwords=remove_stopwords)
    return preprocessor.preprocess(text)
