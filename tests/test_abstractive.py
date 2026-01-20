"""
Unit tests for abstractive summarization module
"""

import pytest
from src.abstractive_summarizer import AbstractiveSummarizer


class TestAbstractiveSummarizer:
    """Test cases for AbstractiveSummarizer class."""
    
    @pytest.fixture
    def sample_text(self):
        """Sample text for testing."""
        return """
        Climate change is one of the biggest challenges of our time.
        Rising temperatures are affecting ecosystems worldwide.
        Renewable energy sources offer hope for reducing emissions.
        International cooperation is essential for addressing this crisis.
        Individual actions also play an important role in mitigation.
        """
    
    @pytest.fixture
    def summarizer(self):
        """Create summarizer instance."""
        return AbstractiveSummarizer(device='cpu')
    
    def test_generate_summary(self, summarizer, sample_text):
        """Test basic summary generation."""
        summary = summarizer.generate_summary(
            sample_text,
            max_length=50,
            min_length=20
        )
        
        assert isinstance(summary, str)
        assert len(summary) > 0
        assert len(summary) < len(sample_text)
    
    def test_different_lengths(self, summarizer, sample_text):
        """Test summaries with different length parameters."""
        short_summary = summarizer.generate_summary(
            sample_text,
            max_length=30,
            min_length=10
        )
        
        long_summary = summarizer.generate_summary(
            sample_text,
            max_length=100,
            min_length=50
        )
        
        assert isinstance(short_summary, str)
        assert isinstance(long_summary, str)
        # Note: Actual lengths may vary due to tokenization
    
    def test_empty_text(self, summarizer):
        """Test with empty text."""
        summary = summarizer.generate_summary("")
        assert summary == ""
    
    def test_interactive_summary(self, summarizer, sample_text):
        """Test interactive summary with multiple lengths."""
        summaries = summarizer.interactive_summary(
            sample_text,
            length_options=['short', 'medium']
        )
        
        assert isinstance(summaries, dict)
        assert 'short' in summaries
        assert 'medium' in summaries
        assert all(isinstance(s, str) for s in summaries.values())
