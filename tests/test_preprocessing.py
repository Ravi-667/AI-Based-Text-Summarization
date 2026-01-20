"""
Unit tests for text preprocessing module
"""

import pytest
from src.preprocessing import TextPreprocessor, preprocess_text


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.preprocessor = TextPreprocessor()
    
    def test_clean_text(self):
        """Test text cleaning."""
        text = "This  is   a    test.   Multiple   spaces!"
        cleaned = self.preprocessor.clean_text(text)
        
        assert "  " not in cleaned
        assert cleaned.strip() == cleaned
    
    def test_tokenize_sentences(self):
        """Test sentence tokenization."""
        text = "This is sentence one. This is sentence two. This is sentence three."
        sentences = self.preprocessor.tokenize_sentences(text)
        
        assert len(sentences) == 3
        assert all(isinstance(s, str) for s in sentences)
    
    def test_tokenize_words(self):
        """Test word tokenization."""
        text = "This is a test sentence."
        words = self.preprocessor.tokenize_words(text)
        
        assert isinstance(words, list)
        assert len(words) > 0
        assert all(isinstance(w, str) for w in words)
    
    def test_remove_stopwords(self):
        """Test stopword removal."""
        preprocessor = TextPreprocessor(remove_stopwords=True)
        text = "This is a test with stopwords."
        words = preprocessor.tokenize_words(text)
        
        # Common stopwords should be removed
        assert "is" not in words
        assert "a" not in words
    
    def test_chunk_text(self):
        """Test text chunking."""
        text = " ".join(["This is sentence number {}.".format(i) for i in range(100)])
        chunks = self.preprocessor.chunk_text(text, max_length=50)
        
        assert len(chunks) > 1
        assert all(isinstance(chunk, tuple) for chunk in chunks)
    
    def test_preprocess_empty_text(self):
        """Test preprocessing with empty text."""
        text = ""
        cleaned, sentences = self.preprocessor.preprocess(text)
        
        assert cleaned == ""
        assert len(sentences) == 0
    
    def test_preprocess_valid_text(self):
        """Test complete preprocessing pipeline."""
        text = "First sentence. Second sentence. Third sentence."
        cleaned, sentences = self.preprocessor.preprocess(text)
        
        assert isinstance(cleaned, str)
        assert isinstance(sentences, list)
        assert len(sentences) == 3


def test_preprocess_text_function():
    """Test the convenience function."""
    text = "This is a test. It has multiple sentences."
    cleaned, sentences = preprocess_text(text)
    
    assert isinstance(cleaned, str)
    assert isinstance(sentences, list)
    assert len(sentences) == 2
