"""AI-Based Text Summarization Package"""

__version__ = "1.0.0"

from src.extractive_summarizer import ExtractiveSummarizer, summarize_extractive
from src.abstractive_summarizer import AbstractiveSummarizer, summarize_abstractive
from src.evaluation import SummaryEvaluator, evaluate_summary
from src.preprocessing import TextPreprocessor, preprocess_text
from src.utils import read_document, save_summary, get_text_stats

__all__ = [
    'ExtractiveSummarizer',
    'AbstractiveSummarizer',
    'SummaryEvaluator',
    'TextPreprocessor',
    'summarize_extractive',
    'summarize_abstractive',
    'evaluate_summary',
    'preprocess_text',
    'read_document',
    'save_summary',
    'get_text_stats'
]
