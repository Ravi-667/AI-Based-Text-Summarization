"""
Summarization API endpoints
"""

from fastapi import APIRouter, HTTPException, status
from datetime import datetime
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from api.schemas import (
    ExtractiveSummarizationRequest,
    AbstractiveSummarizationRequest,
    BothSummarizationRequest,
    BatchSummarizationRequest,
    SummarizationResponse,
    BothSummarizationResponse,
    BatchSummarizationResponse
)
from src.extractive_summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.utils import get_text_stats
from src.exceptions import SummarizationError

router = APIRouter()

# Model cache (will be replaced with proper singleton in optimization phase)
_extractive_model = None
_abstractive_model = None


def get_extractive_model():
    """Get or create extractive model."""
    global _extractive_model
    if _extractive_model is None:
        _extractive_model = ExtractiveSummarizer(device='cpu')
    return _extractive_model


def get_abstractive_model():
    """Get or create abstractive model."""
    global _abstractive_model
    if _abstractive_model is None:
        _abstractive_model = AbstractiveSummarizer(device='cpu')
    return _abstractive_model


@router.post("/summarize/extractive", response_model=SummarizationResponse)
async def summarize_extractive(request: ExtractiveSummarizationRequest):
    """
    Perform extractive summarization.
    
    Selects the most important sentences from the original text.
    """
    start_time = time.time()
    
    try:
        # Get model
        summarizer = get_extractive_model()
        
        # Get statistics
        original_stats = get_text_stats(request.text)
        
        # Generate summary
        summary = summarizer.generate_summary(
            request.text,
            ratio=request.ratio,
            num_sentences=request.num_sentences,
            method=request.method
        )
        
        # Get summary statistics
        summary_stats = get_text_stats(summary)
        
        # Calculate compression
        compression_ratio = (
            1 - (summary_stats['word_count'] / original_stats['word_count'])
        ) * 100
        
        processing_time = time.time() - start_time
        
        return SummarizationResponse(
            summary=summary,
            original_length=original_stats['word_count'],
            summary_length=summary_stats['word_count'],
            compression_ratio=round(compression_ratio, 2),
            method="extractive",
            processing_time=round(processing_time, 3)
        )
        
    except SummarizationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )


@router.post("/summarize/abstractive", response_model=SummarizationResponse)
async def summarize_abstractive(request: AbstractiveSummarizationRequest):
    """
    Perform abstractive summarization.
    
    Generates a new paraphrased summary using T5 model.
    """
    start_time = time.time()
    
    try:
        # Get model
        summarizer = get_abstractive_model()
        
        # Get statistics
        original_stats = get_text_stats(request.text)
        
        # Generate summary
        summary = summarizer.generate_summary(
            request.text,
            max_length=request.max_length,
            min_length=request.min_length
        )
        
        # Get summary statistics
        summary_stats = get_text_stats(summary)
        
        # Calculate compression
        compression_ratio = (
            1 - (summary_stats['word_count'] / original_stats['word_count'])
        ) * 100
        
        processing_time = time.time() - start_time
        
        return SummarizationResponse(
            summary=summary,
            original_length=original_stats['word_count'],
            summary_length=summary_stats['word_count'],
            compression_ratio=round(compression_ratio, 2),
            method="abstractive",
            processing_time=round(processing_time, 3)
        )
        
    except SummarizationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )


@router.post("/summarize/both", response_model=BothSummarizationResponse)
async def summarize_both(request: BothSummarizationRequest):
    """
    Perform both extractive and abstractive summarization.
    
    Returns results from both methods for comparison.
    """
    start_time = time.time()
    
    try:
        # Get models
        extractive_model = get_extractive_model()
        abstractive_model = get_abstractive_model()
        
        # Get statistics
        original_stats = get_text_stats(request.text)
        
        # Extractive summary
        ext_params = request.extractive_params or {}
        extractive_summary = extractive_model.generate_summary(
            request.text,
            ratio=ext_params.get('ratio', 0.3),
            method=ext_params.get('method', 'combined')
        )
        ext_stats = get_text_stats(extractive_summary)
        ext_compression = (1 - (ext_stats['word_count'] / original_stats['word_count'])) * 100
        
        # Abstractive summary
        abs_params = request.abstractive_params or {}
        abstractive_summary = abstractive_model.generate_summary(
            request.text,
            max_length=abs_params.get('max_length', 150),
            min_length=abs_params.get('min_length', 50)
        )
        abs_stats = get_text_stats(abstractive_summary)
        abs_compression = (1 - (abs_stats['word_count'] / original_stats['word_count'])) * 100
        
        processing_time = time.time() - start_time
        
        return BothSummarizationResponse(
            extractive=SummarizationResponse(
                summary=extractive_summary,
                original_length=original_stats['word_count'],
                summary_length=ext_stats['word_count'],
                compression_ratio=round(ext_compression, 2),
                method="extractive",
                processing_time=round(processing_time / 2, 3)
            ),
            abstractive=SummarizationResponse(
                summary=abstractive_summary,
                original_length=original_stats['word_count'],
                summary_length=abs_stats['word_count'],
                compression_ratio=round(abs_compression, 2),
                method="abstractive",
                processing_time=round(processing_time / 2, 3)
            )
        )
        
    except SummarizationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Summarization failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchSummarizationResponse)
async def batch_summarize(request: BatchSummarizationRequest):
    """
    Perform batch summarization on multiple texts.
    
    Processes multiple texts in sequence and returns all summaries.
    """
    start_time = time.time()
    
    try:
        summaries = []
        successful = 0
        failed = 0
        
        # Get appropriate model
        if request.method == 'extractive':
            model = get_extractive_model()
        elif request.method == 'abstractive':
            model = get_abstractive_model()
        else:  # both
            ext_model = get_extractive_model()
            abs_model = get_abstractive_model()
        
        # Process each text
        for text in request.texts:
            try:
                original_stats = get_text_stats(text)
                
                if request.method == 'extractive':
                    summary = model.generate_summary(text, **request.params)
                elif request.method == 'abstractive':
                    summary = model.generate_summary(text, **request.params)
                else:  # both - use extractive for batch
                    summary = ext_model.generate_summary(text, **request.params)
                
                summary_stats = get_text_stats(summary)
                compression = (1 - (summary_stats['word_count'] / original_stats['word_count'])) * 100
                
                summaries.append(SummarizationResponse(
                    summary=summary,
                    original_length=original_stats['word_count'],
                    summary_length=summary_stats['word_count'],
                    compression_ratio=round(compression, 2),
                    method=request.method,
                    processing_time=0  # Individual time not tracked in batch
                ))
                successful += 1
                
            except Exception as e:
                failed += 1
                # Continue processing other texts
                continue
        
        total_time = time.time() - start_time
        
        return BatchSummarizationResponse(
            summaries=summaries,
            total=len(request.texts),
            successful=successful,
            failed=failed,
            total_time=round(total_time, 3)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch processing failed: {str(e)}"
        )
