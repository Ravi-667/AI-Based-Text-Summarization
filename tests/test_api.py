"""
Tests for FastAPI endpoints
"""

import pytest
from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


class TestHealthEndpoints:
    """Test health and system endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data
    
    def test_model_info(self):
        """Test model information endpoint."""
        response = client.get("/api/v1/models/info")
        assert response.status_code == 200
        data = response.json()
        assert "extractive_model" in data
        assert "abstractive_model" in data
        assert "device" in data


class TestSummarizationEndpoints:
    """Test summarization endpoints."""
    
    def test_extractive_summarization(self):
        """Test extractive summarization endpoint."""
        payload = {
            "text": "This is a test document. It has multiple sentences. " * 10,
            "ratio": 0.3,
            "method": "combined"
        }
        
        response = client.post("/api/v1/summarize/extractive", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "compression_ratio" in data
        assert data["method"] == "extractive"
    
    def test_abstractive_summarization(self):
        """Test abstractive summarization endpoint."""
        payload = {
            "text": "This is a test document for abstractive summarization. " * 10,
            "max_length": 100,
            "min_length": 30
        }
        
        response = client.post("/api/v1/summarize/abstractive", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert "compression_ratio" in data
        assert data["method"] == "abstractive"
    
    def test_both_summarization(self):
        """Test both methods endpoint."""
        payload = {
            "text": "This is a comprehensive test document. " * 15,
            "extractive_params": {"ratio": 0.3},
            "abstractive_params": {"max_length": 80}
        }
        
        response = client.post("/api/v1/summarize/both", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "extractive" in data
        assert "abstractive" in data
        assert data["extractive"]["method"] == "extractive"
        assert data["abstractive"]["method"] == "abstractive"
    
    def test_batch_summarization(self):
        """Test batch summarization endpoint."""
        payload = {
            "texts": [
                "First document for testing. " * 5,
                "Second document for testing. " * 5,
                "Third document for testing. " * 5
            ],
            "method": "extractive",
            "params": {"ratio": 0.3}
        }
        
        response = client.post("/api/v1/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["summaries"]) == data["successful"]
    
    def test_invalid_text_too_short(self):
        """Test validation with text too short."""
        payload = {
            "text": "Short",
            "ratio": 0.3
        }
        
        response = client.post("/api/v1/summarize/extractive", json=payload)
        assert response.status_code == 422  # Validation error
    
    def test_invalid_ratio(self):
        """Test validation with invalid ratio."""
        payload = {
            "text": "Valid text for testing summarization. " * 10,
            "ratio": 1.5  # Invalid: > 1
        }
        
        response = client.post("/api/v1/summarize/extractive", json=payload)
        assert response.status_code == 422
    
    def test_invalid_method(self):
        """Test validation with invalid method."""
        payload = {
            "text": "Valid text for testing. " * 10,
            "method": "invalid_method"
        }
        
        response = client.post("/api/v1/summarize/extractive", json=payload)
        assert response.status_code == 422


class TestAPIDocs:
    """Test API documentation endpoints."""
    
    def test_openapi_docs(self):
        """Test that OpenAPI docs are available."""
        response = client.get("/docs")
        assert response.status_code == 200
    
    def test_redoc(self):
        """Test that ReDoc documentation is available."""
        response = client.get("/redoc")
        assert response.status_code == 200
