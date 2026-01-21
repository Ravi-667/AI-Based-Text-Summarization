"""
Tests for Batch Processing Module
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.batch_processor import BatchProcessor


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_documents(temp_dir):
    """Create sample documents for testing."""
    docs = []
    
    # Create sample text files
    for i in range(3):
        file_path = Path(temp_dir) / f"document_{i+1}.txt"
        content = f"This is test document {i+1}. " * 10
        content += "It contains multiple sentences for testing. "
        content += "The summarization should work on this text. "
        content += "We are testing batch processing functionality. "
        content += "This helps ensure the system works correctly."
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        docs.append(file_path)
    
    return docs


@pytest.fixture
def output_dir(temp_dir):
    """Create output directory."""
    output_path = Path(temp_dir) / "output"
    output_path.mkdir()
    return str(output_path)


class TestBatchProcessor:
    """Test BatchProcessor class."""
    
    def test_initialization_extractive(self):
        """Test initialization with extractive method."""
        processor = BatchProcessor(method='extractive', device='cpu')
        assert processor.method == 'extractive'
        assert processor.extractive_summarizer is not None
        assert processor.abstractive_summarizer is None
    
    def test_initialization_abstractive(self):
        """Test initialization with abstractive method."""
        processor = BatchProcessor(method='abstractive', device='cpu')
        assert processor.method == 'abstractive'
        assert processor.extractive_summarizer is None
        assert processor.abstractive_summarizer is not None
    
    def test_initialization_both(self):
        """Test initialization with both methods."""
        processor = BatchProcessor(method='both', device='cpu')
        assert processor.method == 'both'
        assert processor.extractive_summarizer is not None
        assert processor.abstractive_summarizer is not None
    
    def test_find_documents(self, temp_dir, sample_documents):
        """Test finding documents in directory."""
        processor = BatchProcessor(method='extractive', device='cpu')
        documents = processor.find_documents(temp_dir, extensions=['.txt'])
        
        assert len(documents) == 3
        assert all(doc.suffix == '.txt' for doc in documents)
    
    def test_find_documents_nonexistent(self):
        """Test finding documents in nonexistent directory."""
        processor = BatchProcessor(method='extractive', device='cpu')
        
        with pytest.raises(FileNotFoundError):
            processor.find_documents('/nonexistent/directory')
    
    def test_process_single_document(self, sample_documents):
        """Test processing a single document."""
        processor = BatchProcessor(method='extractive', device='cpu', ratio=0.3)
        result = processor.process_single_document(sample_documents[0])
        
        assert result['status'] == 'success'
        assert result['extractive_summary'] is not None
        assert result['original_stats'] is not None
        assert result['error'] is None
        assert result['processing_time'] is not None
    
    def test_process_single_document_empty(self, temp_dir):
        """Test processing empty document."""
        empty_file = Path(temp_dir) / "empty.txt"
        empty_file.touch()
        
        processor = BatchProcessor(method='extractive', device='cpu')
        result = processor.process_single_document(empty_file)
        
        assert result['status'] == 'failed'
        assert result['error'] == 'Empty document'
    
    def test_process_directory(self, temp_dir, sample_documents, output_dir):
        """Test processing entire directory."""
        processor = BatchProcessor(method='extractive', device='cpu', ratio=0.3)
        
        batch_results = processor.process_directory(
            directory=temp_dir,
            output_dir=output_dir,
            extensions=['.txt'],
            save_summaries=True
        )
        
        assert batch_results['total'] == 3
        assert batch_results['successful'] == 3
        assert batch_results['failed'] == 0
        assert len(batch_results['results']) == 3
    
    def test_process_directory_with_failures(self, temp_dir, sample_documents, output_dir):
        """Test processing directory with some failures."""
        # Create an empty file
        empty_file = Path(temp_dir) / "empty.txt"
        empty_file.touch()
        
        processor = BatchProcessor(method='extractive', device='cpu', ratio=0.3)
        
        batch_results = processor.process_directory(
            directory=temp_dir,
            output_dir=output_dir,
            extensions=['.txt'],
            save_summaries=True
        )
        
        assert batch_results['total'] == 4  # 3 good + 1 empty
        assert batch_results['successful'] == 3
        assert batch_results['failed'] == 1
    
    def test_process_files(self, sample_documents, output_dir):
        """Test processing specific list of files."""
        processor = BatchProcessor(method='extractive', device='cpu', ratio=0.3)
        
        file_paths = [str(doc) for doc in sample_documents[:2]]
        batch_results = processor.process_files(
            file_paths=file_paths,
            output_dir=output_dir,
            save_summaries=True
        )
        
        assert batch_results['total'] == 2
        assert batch_results['successful'] == 2
        assert batch_results['failed'] == 0
    
    def test_generate_batch_report_json(self, temp_dir, sample_documents, output_dir):
        """Test generating JSON batch report."""
        processor = BatchProcessor(method='extractive', device='cpu', ratio=0.3)
        
        batch_results = processor.process_directory(
            directory=temp_dir,
            output_dir=output_dir,
            extensions=['.txt']
        )
        
        report_path = Path(output_dir) / "report.json"
        processor.generate_batch_report(batch_results, str(report_path), format='json')
        
        assert report_path.exists()
    
    def test_generate_batch_report_csv(self, temp_dir, sample_documents, output_dir):
        """Test generating CSV batch report."""
        processor = BatchProcessor(method='extractive', device='cpu', ratio=0.3)
        
        batch_results = processor.process_directory(
            directory=temp_dir,
            output_dir=output_dir,
            extensions=['.txt']
        )
        
        report_path = Path(output_dir) / "report.csv"
        processor.generate_batch_report(batch_results, str(report_path), format='csv')
        
        assert report_path.exists()
    
    def test_both_methods(self, sample_documents, output_dir):
        """Test processing with both summarization methods."""
        processor = BatchProcessor(
            method='both',
            device='cpu',
            ratio=0.3,
            max_length=80,
            min_length=20
        )
        
        result = processor.process_single_document(sample_documents[0])
        
        assert result['status'] == 'success'
        assert result['extractive_summary'] is not None
        assert result['abstractive_summary'] is not None
