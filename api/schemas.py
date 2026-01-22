"""
Pydantic schemas for API request/response validation
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator


class ExtractiveSummarizationRequest(BaseModel):
    """Request model for extractive summarization."""
    
    text: str = Field(..., min_length=10, description="Text to summarize")
    ratio: Optional[float] = Field(0.3, ge=0.1, le=0.8, description="Extraction ratio")
    num_sentences: Optional[int] = Field(None, ge=1, description="Number of sentences to extract")
    method: str = Field("combined", description="Scoring method")
    
    @validator('method')
    def validate_method(cls, v):
        allowed = ['tfidf', 'textrank', 'lexrank', 'combined']
        if v not in allowed:
            raise ValueError(f"Method must be one of: {', '.join(allowed)}")
        return v


class AbstractiveSummarizationRequest(BaseModel):
    """Request model for abstractive summarization."""
    
    text: str = Field(..., min_length=10, description="Text to summarize")
    max_length: Optional[int] = Field(150, ge=10, le=500, description="Maximum summary length")
    min_length: Optional[int] = Field(50, ge=10, le=200, description="Minimum summary length")
    
    @validator('min_length')
    def validate_lengths(cls, v, values):
        if 'max_length' in values and v >= values['max_length']:
            raise ValueError("min_length must be less than max_length")
        return v


class BothSummarizationRequest(BaseModel):
    """Request model for both summarization methods."""
    
    text: str = Field(..., min_length=10, description="Text to summarize")
    extractive_params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    abstractive_params: Optional[Dict[str, Any]] = Field(default_factory=dict)


class BatchSummarizationRequest(BaseModel):
    """Request model for batch summarization."""
    
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts")
    method: str = Field("extractive", description="Summarization method")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    
    @validator('method')
    def validate_method(cls, v):
        allowed = ['extractive', 'abstractive', 'both']
        if v not in allowed:
            raise ValueError(f"Method must be one of: {', '.join(allowed)}")
        return v


class SummarizationResponse(BaseModel):
    """Response model for summarization."""
    
    summary: str = Field(..., description="Generated summary")
    original_length: int = Field(..., description="Original text word count")
    summary_length: int = Field(..., description="Summary word count")
    compression_ratio: float = Field(..., description="Compression percentage")
    method: str = Field(..., description="Method used")
    processing_time: float = Field(..., description="Processing time in seconds")


class BothSummarizationResponse(BaseModel):
    """Response model for both methods."""
    
    extractive: SummarizationResponse
    abstractive: SummarizationResponse


class BatchSummarizationResponse(BaseModel):
    """Response model for batch summarization."""
    
    summaries: List[SummarizationResponse]
    total: int = Field(..., description="Total texts processed")
    successful: int = Field(..., description="Successfully summarized")
    failed: int = Field(..., description="Failed summarizations")
    total_time: float = Field(..., description="Total processing time")


class HealthResponse(BaseModel):
    """Response model for health check."""
    
    status: str = Field(..., description="Health status")
    timestamp: str = Field(..., description="Current timestamp")
    models_loaded: bool = Field(..., description="Whether models are loaded")
    version: str = Field(..., description="API version")


class ModelInfoResponse(BaseModel):
    """Response model for model information."""
    
    extractive_model: str = Field(..., description="Extractive model name")
    abstractive_model: str = Field(..., description="Abstractive model name")
    device: str = Field(..., description="Computation device")
    models_loaded: bool = Field(..., description="Model loading status")


class ErrorResponse(BaseModel):
    """Response model for errors."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    status_code: int = Field(..., description="HTTP status code")
