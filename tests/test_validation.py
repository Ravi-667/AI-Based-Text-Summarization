"""
Tests for Custom Exceptions and Validation
"""

import pytest
import tempfile
from pathlib import Path

from src.exceptions import (
    SummarizationError,
    InputValidationError,
    EmptyTextError,
    TextTooShortError,
    TextTooLongError,
    ModelLoadError,
    FileFormatError,
    FileReadError,
    ConfigurationError,
    DeviceError
)
from src.utils import (
    validate_text_input,
    validate_file_path,
    validate_summarization_params,
    safe_read_document
)


class TestCustomExceptions:
    """Test custom exception classes."""
    
    def test_summarization_error(self):
        """Test base SummarizationError."""
        error = SummarizationError("Test error", {'key': 'value'})
        assert "Test error" in str(error)
        assert error.details == {'key': 'value'}
    
    def test_empty_text_error(self):
        """Test EmptyTextError."""
        error = EmptyTextError()
        assert "empty" in str(error).lower()
    
    def test_text_too_short_error(self):
        """Test TextTooShortError."""
        error = TextTooShortError(sentence_count=2, minimum_required=3)
        assert "2" in str(error)
        assert "3" in str(error)
        assert error.details['sentence_count'] == 2
    
    def test_text_too_long_error(self):
        """Test TextTooLongError."""
        error = TextTooLongError(word_count=60000, maximum_allowed=50000)
        assert "60000" in str(error)
        assert "50000" in str(error)
    
    def test_model_load_error(self):
        """Test ModelLoadError."""
        original_error = Exception("Connection timeout")
        error = ModelLoadError("bert-base", original_error)
        assert "bert-base" in str(error)
        assert "Connection timeout" in str(error)
    
    def test_file_format_error(self):
        """Test FileFormatError."""
        error = FileFormatError("document.xyz", ".xyz")
        assert ".xyz" in str(error)
        assert ".txt" in str(error)  # Should list supported formats
    
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("ratio", 1.5, "Must be between 0 and 1")
        assert "ratio" in str(error)
        assert "1.5" in str(error)


class TestValidation:
    """Test validation functions."""
    
    def test_validate_text_input_valid(self):
        """Test validation with valid text."""
        text = "This is a valid text. It has multiple sentences. And enough words."
        is_valid, error = validate_text_input(text)
        assert is_valid is True
        assert error is None
    
    def test_validate_text_input_empty(self):
        """Test validation with empty text."""
        is_valid, error = validate_text_input("")
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_validate_text_input_whitespace(self):
        """Test validation with whitespace only."""
        is_valid, error = validate_text_input("   \n\t  ")
        assert is_valid is False
        assert "empty" in error.lower()
    
    def test_validate_text_input_too_short(self):
        """Test validation with too few words."""
        text = "Short."
        is_valid, error = validate_text_input(text, min_words=10)
        assert is_valid is False
        assert "too short" in error.lower()
    
    def test_validate_text_input_too_long(self):
        """Test validation with too many words."""
        text = " ".join(["word"] * 100)
        is_valid, error = validate_text_input(text, max_words=50)
        assert is_valid is False
        assert "too long" in error.lower()
    
    def test_validate_text_input_too_few_sentences(self):
        """Test validation with too few sentences."""
        text = "Only one sentence"
        is_valid, error = validate_text_input(text, min_sentences=3)
        assert is_valid is False
        assert "Too few sentences" in error
    
    def test_validate_file_path_valid(self):
        """Test file path validation with valid file."""
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            is_valid, error = validate_file_path(temp_path)
            assert is_valid is True
            assert error is None
        finally:
            Path(temp_path).unlink()
    
    def test_validate_file_path_not_found(self):
        """Test file path validation with nonexistent file."""
        is_valid, error = validate_file_path("/nonexistent/file.txt")
        assert is_valid is False
        assert "not found" in error.lower()
    
    def test_validate_file_path_unsupported_format(self):
        """Test file path validation with unsupported format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
            temp_path = f.name
        
        try:
            is_valid, error = validate_file_path(temp_path)
            assert is_valid is False
            assert "unsupported" in error.lower()
        finally:
            Path(temp_path).unlink()
    
    def test_validate_summarization_params_valid(self):
        """Test parameter validation with valid params."""
        is_valid, error = validate_summarization_params(
            method='extractive',
            ratio=0.3,
            num_sentences=5
        )
        assert is_valid is True
        assert error is None
    
    def test_validate_summarization_params_invalid_method(self):
        """Test parameter validation with invalid method."""
        is_valid, error = validate_summarization_params(method='invalid')
        assert is_valid is False
        assert "invalid method" in error.lower()
    
    def test_validate_summarization_params_invalid_ratio(self):
        """Test parameter validation with invalid ratio."""
        is_valid, error = validate_summarization_params(
            method='extractive',
            ratio=1.5
        )
        assert is_valid is False
        assert "ratio" in error.lower()
    
    def test_validate_summarization_params_invalid_num_sentences(self):
        """Test parameter validation with invalid num_sentences."""
        is_valid, error = validate_summarization_params(
            method='extractive',
            num_sentences=-1
        )
        assert is_valid is False
        assert "num_sentences" in error.lower()
    
    def test_validate_summarization_params_length_mismatch(self):
        """Test parameter validation with min_length >= max_length."""
        is_valid, error = validate_summarization_params(
            method='abstractive',
            min_length=100,
            max_length=50
        )
        assert is_valid is False
        assert "min_length" in error.lower()
    
    def test_safe_read_document_valid(self):
        """Test safe document reading with valid file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is test content. With multiple sentences.")
            temp_path = f.name
        
        try:
            text, error = safe_read_document(temp_path)
            assert text is not None
            assert error is None
            assert "test content" in text
        finally:
            Path(temp_path).unlink()
    
    def test_safe_read_document_not_found(self):
        """Test safe document reading with nonexistent file."""
        text, error = safe_read_document("/nonexistent/file.txt")
        assert text is None
        assert error is not None
        assert "not found" in error.lower()
    
    def test_safe_read_document_empty(self):
        """Test safe document reading with empty file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            temp_path = f.name
        
        try:
            text, error = safe_read_document(temp_path)
            assert text is None
            assert error is not None
        finally:
            Path(temp_path).unlink()
