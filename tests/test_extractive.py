"""
Unit tests for extractive summarization module
"""

import pytest
from src.extractive_summarizer import ExtractiveSummarizer


class TestExtractiveSummarizer:
    """Test cases for ExtractiveSummarizer class."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        Artificial intelligence is transforming many industries.
        Machine learning algorithms can now perform complex tasks.
        Deep learning has led to breakthroughs in computer vision.
        Natural language processing enables computers to understand text.
        AI applications are expanding rapidly across various sectors.
        """
    
    @pytest.fixture
    def summarizer(self):
        """Create summarizer instance."""
        return ExtractiveSummarizer(device='cpu')
    
    def test_generate_summary(self, summarizer, sample_text):
        """Test basic summary generation."""
        summary = summarizer.generate_summary(sample_text, num_sentences=2)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) < len(sample_text)
    
    def test_summary_with_ratio(self, summarizer, sample_text):
        """Test summary with extraction ratio."""
        summary = summarizer.generate_summary(sample_text, ratio=0.4)
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_different_scoring_methods(self, summarizer, sample_text):
        """Test different scoring methods."""
        methods = ['tfidf', 'lexrank', 'combined']
        
        for method in methods:
            summary = summarizer.generate_summary(
                sample_text,
                num_sentences=2,
                method=method,
                use_embeddings=False  # Faster for testing
            )
            assert isinstance(summary, str)
            assert len(summary) > 0
    
    def test_empty_text(self, summarizer):
        """Test with empty text."""
        summary = summarizer.generate_summary("", num_sentences=2)
        assert summary == ""
    
    def test_short_text(self, summarizer):
        """Test with very short text."""
        text = "This is a single sentence."
        summary = summarizer.generate_summary(text, num_sentences=2)
        
        assert isinstance(summary, str)
        # Should return the original text if it's shorter than requested
        assert len(summary) > 0
