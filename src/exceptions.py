"""
Custom Exception Classes for AI-Based Text Summarization
Provides specialized exceptions for different error scenarios
"""


class SummarizationError(Exception):
    """Base exception for all summarization-related errors."""
    
    def __init__(self, message: str, details: dict = None):
        """
        Initialize summarization error.
        
        Args:
            message: Error message
            details: Additional error details (optional)
        """
        self.message = message
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class InputValidationError(SummarizationError):
    """Raised when input validation fails."""
    pass


class EmptyTextError(InputValidationError):
    """Raised when input text is empty."""
    
    def __init__(self):
        super().__init__("Input text is empty or contains only whitespace")


class TextTooShortError(InputValidationError):
    """Raised when text is too short for meaningful summarization."""
    
    def __init__(self, sentence_count: int, minimum_required: int = 3):
        message = (
            f"Text is too short for extractive summarization. "
            f"Found {sentence_count} sentence(s), minimum {minimum_required} recommended. "
            f"Consider using abstractive summarization instead."
        )
        super().__init__(message, {'sentence_count': sentence_count, 'minimum': minimum_required})


class TextTooLongError(InputValidationError):
    """Raised when text exceeds maximum length limits."""
    
    def __init__(self, word_count: int, maximum_allowed: int):
        message = (
            f"Text exceeds maximum length. "
            f"Found {word_count} words, maximum {maximum_allowed} allowed. "
            f"Consider splitting the text or using chunked processing."
        )
        super().__init__(message, {'word_count': word_count, 'maximum': maximum_allowed})


class ModelLoadError(SummarizationError):
    """Raised when model loading fails."""
    
    def __init__(self, model_name: str, original_error: Exception = None):
        message = f"Failed to load model: {model_name}"
        details = {'model_name': model_name}
        
        if original_error:
            details['original_error'] = str(original_error)
            message += f" | Error: {str(original_error)}"
        
        super().__init__(message, details)


class ProcessingError(SummarizationError):
    """Raised when text processing fails."""
    pass


class SummarizationFailedError(ProcessingError):
    """Raised when summarization process fails."""
    
    def __init__(self, method: str, original_error: Exception = None):
        message = f"{method.capitalize()} summarization failed"
        details = {'method': method}
        
        if original_error:
            details['original_error'] = str(original_error)
            message += f": {str(original_error)}"
        
        super().__init__(message, details)


class FileFormatError(SummarizationError):
    """Raised when file format is not supported."""
    
    def __init__(self, file_path: str, extension: str = None):
        supported_formats = ['.txt', '.pdf', '.docx']
        
        if extension is None:
            from pathlib import Path
            extension = Path(file_path).suffix
        
        message = (
            f"Unsupported file format: {extension}. "
            f"Supported formats: {', '.join(supported_formats)}"
        )
        super().__init__(message, {
            'file_path': file_path,
            'extension': extension,
            'supported_formats': supported_formats
        })


class FileReadError(SummarizationError):
    """Raised when file cannot be read."""
    
    def __init__(self, file_path: str, original_error: Exception = None):
        message = f"Failed to read file: {file_path}"
        details = {'file_path': file_path}
        
        if original_error:
            details['original_error'] = str(original_error)
            message += f" | Error: {str(original_error)}"
        
        super().__init__(message, details)


class ConfigurationError(SummarizationError):
    """Raised when configuration is invalid."""
    
    def __init__(self, parameter: str, value, reason: str):
        message = f"Invalid configuration for '{parameter}': {value}. {reason}"
        super().__init__(message, {
            'parameter': parameter,
            'value': value,
            'reason': reason
        })


class DeviceError(SummarizationError):
    """Raised when device configuration is invalid."""
    
    def __init__(self, requested_device: str, reason: str):
        message = f"Cannot use device '{requested_device}': {reason}"
        super().__init__(message, {
            'requested_device': requested_device,
            'reason': reason
        })
