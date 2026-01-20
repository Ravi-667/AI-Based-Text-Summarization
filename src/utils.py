"""
Utility functions for AI-Based Text Summarization System
"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import PyPDF2
from docx import Document


def setup_logger(
    name: str = "summarizer",
    log_file: Optional[str] = None,
    level: str = "INFO"
) -> logging.Logger:
    """
    Set up logger for the application.
    
    Args:
        name: Logger name
        log_file: Path to log file (optional)
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def read_document(file_path: str) -> str:
    """
    Read text from various document formats.
    
    Args:
        file_path: Path to document file
        
    Returns:
        Extracted text content
        
    Raises:
        ValueError: If file format is not supported
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    extension = file_path.suffix.lower()
    
    if extension == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    elif extension == '.pdf':
        return read_pdf(str(file_path))
    
    elif extension in ['.docx', '.doc']:
        return read_docx(str(file_path))
    
    else:
        raise ValueError(
            f"Unsupported file format: {extension}. "
            f"Supported formats: .txt, .pdf, .docx"
        )


def read_pdf(file_path: str) -> str:
    """
    Extract text from PDF file.
    
    Args:
        file_path: Path to PDF file
        
    Returns:
        Extracted text
    """
    text = ""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        raise Exception(f"Error reading PDF: {e}")
    
    return text


def read_docx(file_path: str) -> str:
    """
    Extract text from Word document.
    
    Args:
        file_path: Path to Word file
        
    Returns:
        Extracted text
    """
    try:
        doc = Document(file_path)
        text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
        return text
    except Exception as e:
        raise Exception(f"Error reading DOCX: {e}")


def save_summary(
    summary: str,
    output_path: str,
    original_text: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save summary to file with optional metadata.
    
    Args:
        summary: Generated summary text
        output_path: Path to save summary
        original_text: Original text (optional)
        metadata: Additional metadata (optional)
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        if metadata:
            f.write("=" * 80 + "\n")
            f.write("SUMMARY METADATA\n")
            f.write("=" * 80 + "\n")
            for key, value in metadata.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(summary)
        f.write("\n\n")
        
        if original_text:
            f.write("=" * 80 + "\n")
            f.write("ORIGINAL TEXT\n")
            f.write("=" * 80 + "\n")
            f.write(original_text)


def get_text_stats(text: str) -> Dict[str, Any]:
    """
    Calculate statistics about the text.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with text statistics
    """
    words = text.split()
    sentences = text.split('.')
    
    # Remove empty strings
    words = [w for w in words if w.strip()]
    sentences = [s for s in sentences if s.strip()]
    
    # Calculate reading time (average reading speed: 200-250 wpm)
    reading_time_minutes = len(words) / 225
    
    stats = {
        'character_count': len(text),
        'word_count': len(words),
        'sentence_count': len(sentences),
        'avg_sentence_length': len(words) / max(len(sentences), 1),
        'reading_time_minutes': round(reading_time_minutes, 1)
    }
    
    return stats


def create_directories() -> None:
    """Create necessary project directories if they don't exist."""
    directories = [
        'data/samples',
        'data/outputs',
        'models/cache',
        'logs'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def format_summary_output(
    summary: str,
    original_stats: Dict[str, Any],
    summary_stats: Dict[str, Any],
    method: str
) -> str:
    """
    Format summary output with statistics.
    
    Args:
        summary: Generated summary
        original_stats: Statistics of original text
        summary_stats: Statistics of summary
        method: Summarization method used
        
    Returns:
        Formatted output string
    """
    compression_ratio = (
        1 - (summary_stats['word_count'] / original_stats['word_count'])
    ) * 100
    
    output = f"""
{'=' * 80}
SUMMARY ({method.upper()})
{'=' * 80}

{summary}

{'=' * 80}
STATISTICS
{'=' * 80}
Original Text:
  - Words: {original_stats['word_count']}
  - Sentences: {original_stats['sentence_count']}
  - Reading Time: {original_stats['reading_time_minutes']} min

Summary:
  - Words: {summary_stats['word_count']}
  - Sentences: {summary_stats['sentence_count']}
  - Reading Time: {summary_stats['reading_time_minutes']} min
  
Compression: {compression_ratio:.1f}%
{'=' * 80}
"""
    return output
