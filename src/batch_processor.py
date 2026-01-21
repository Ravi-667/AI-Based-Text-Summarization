"""
Batch Processing Module for AI-Based Text Summarization
Handles processing multiple documents in batch mode
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import json
import csv
from tqdm import tqdm

from src.extractive_summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.utils import read_document, save_summary, get_text_stats, setup_logger

logger = setup_logger(__name__)


class BatchProcessor:
    """Process multiple documents in batch mode."""
    
    def __init__(
        self,
        method: str = 'extractive',
        device: str = 'auto',
        **summarization_kwargs
    ):
        """
        Initialize batch processor.
        
        Args:
            method: Summarization method ('extractive', 'abstractive', 'both')
            device: Device for inference ('cuda', 'cpu', or 'auto')
            **summarization_kwargs: Additional arguments for summarization
        """
        self.method = method
        self.device = device
        self.summarization_kwargs = summarization_kwargs
        
        # Initialize summarizers based on method
        self.extractive_summarizer = None
        self.abstractive_summarizer = None
        
        if method in ['extractive', 'both']:
            logger.info("Loading extractive summarizer...")
            self.extractive_summarizer = ExtractiveSummarizer(device=device)
        
        if method in ['abstractive', 'both']:
            logger.info("Loading abstractive summarizer...")
            self.abstractive_summarizer = AbstractiveSummarizer(device=device)
        
        logger.info(f"BatchProcessor initialized with method: {method}")
    
    def find_documents(
        self,
        directory: str,
        extensions: List[str] = ['.txt', '.pdf', '.docx'],
        recursive: bool = True
    ) -> List[Path]:
        """
        Find all documents in directory.
        
        Args:
            directory: Directory path to search
            extensions: List of file extensions to include
            recursive: Whether to search recursively
            
        Returns:
            List of Path objects for found documents
        """
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        
        documents = []
        
        if recursive:
            for ext in extensions:
                documents.extend(directory_path.rglob(f"*{ext}"))
        else:
            for ext in extensions:
                documents.extend(directory_path.glob(f"*{ext}"))
        
        logger.info(f"Found {len(documents)} documents in {directory}")
        return sorted(documents)
    
    def process_single_document(
        self,
        file_path: Path
    ) -> Dict:
        """
        Process a single document.
        
        Args:
            file_path: Path to document
            
        Returns:
            Dictionary with processing results
        """
        result = {
            'file': str(file_path),
            'filename': file_path.name,
            'status': 'pending',
            'error': None,
            'extractive_summary': None,
            'abstractive_summary': None,
            'original_stats': None,
            'processing_time': None
        }
        
        start_time = datetime.now()
        
        try:
            # Read document
            logger.info(f"Processing: {file_path.name}")
            text = read_document(str(file_path))
            
            if not text or not text.strip():
                result['status'] = 'failed'
                result['error'] = 'Empty document'
                return result
            
            # Get original stats
            result['original_stats'] = get_text_stats(text)
            
            # Generate summaries based on method
            if self.method in ['extractive', 'both'] and self.extractive_summarizer:
                result['extractive_summary'] = self.extractive_summarizer.generate_summary(
                    text,
                    **self.summarization_kwargs
                )
            
            if self.method in ['abstractive', 'both'] and self.abstractive_summarizer:
                result['abstractive_summary'] = self.abstractive_summarizer.generate_summary(
                    text,
                    **self.summarization_kwargs
                )
            
            result['status'] = 'success'
            
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {str(e)}")
            result['status'] = 'failed'
            result['error'] = str(e)
        
        finally:
            processing_time = (datetime.now() - start_time).total_seconds()
            result['processing_time'] = processing_time
        
        return result
    
    def process_directory(
        self,
        directory: str,
        output_dir: Optional[str] = None,
        extensions: List[str] = ['.txt', '.pdf', '.docx'],
        recursive: bool = True,
        save_summaries: bool = True
    ) -> Dict:
        """
        Process all documents in a directory.
        
        Args:
            directory: Directory containing documents
            output_dir: Directory to save summaries (optional)
            extensions: File extensions to process
            recursive: Process subdirectories recursively
            save_summaries: Whether to save individual summaries
            
        Returns:
            Dictionary with batch processing results
        """
        # Find all documents
        documents = self.find_documents(directory, extensions, recursive)
        
        if len(documents) == 0:
            logger.warning(f"No documents found in {directory}")
            return {
                'total': 0,
                'successful': 0,
                'failed': 0,
                'results': []
            }
        
        # Create output directory if needed
        if save_summaries and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process all documents with progress bar
        results = []
        successful = 0
        failed = 0
        
        for doc_path in tqdm(documents, desc="Processing documents"):
            result = self.process_single_document(doc_path)
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
                
                # Save summaries if requested
                if save_summaries and output_dir:
                    self._save_document_summary(result, output_dir)
            else:
                failed += 1
        
        # Create batch summary
        batch_results = {
            'total': len(documents),
            'successful': successful,
            'failed': failed,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'method': self.method
        }
        
        logger.info(f"Batch processing complete: {successful}/{len(documents)} successful")
        
        return batch_results
    
    def process_files(
        self,
        file_paths: List[str],
        output_dir: Optional[str] = None,
        save_summaries: bool = True
    ) -> Dict:
        """
        Process a list of specific files.
        
        Args:
            file_paths: List of file paths to process
            output_dir: Directory to save summaries
            save_summaries: Whether to save individual summaries
            
        Returns:
            Dictionary with batch processing results
        """
        # Convert to Path objects
        documents = [Path(fp) for fp in file_paths]
        
        # Validate all files exist
        for doc in documents:
            if not doc.exists():
                raise FileNotFoundError(f"File not found: {doc}")
        
        # Create output directory if needed
        if save_summaries and output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Process all documents
        results = []
        successful = 0
        failed = 0
        
        for doc_path in tqdm(documents, desc="Processing files"):
            result = self.process_single_document(doc_path)
            results.append(result)
            
            if result['status'] == 'success':
                successful += 1
                
                if save_summaries and output_dir:
                    self._save_document_summary(result, output_dir)
            else:
                failed += 1
        
        batch_results = {
            'total': len(documents),
            'successful': successful,
            'failed': failed,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'method': self.method
        }
        
        return batch_results
    
    def _save_document_summary(self, result: Dict, output_dir: str):
        """Save summary for a single document."""
        output_path = Path(output_dir)
        base_name = Path(result['filename']).stem
        
        # Save extractive summary
        if result['extractive_summary']:
            ext_file = output_path / f"{base_name}_extractive.txt"
            with open(ext_file, 'w', encoding='utf-8') as f:
                f.write(result['extractive_summary'])
        
        # Save abstractive summary
        if result['abstractive_summary']:
            abs_file = output_path / f"{base_name}_abstractive.txt"
            with open(abs_file, 'w', encoding='utf-8') as f:
                f.write(result['abstractive_summary'])
    
    def generate_batch_report(
        self,
        batch_results: Dict,
        output_path: str,
        format: str = 'json'
    ):
        """
        Generate batch processing report.
        
        Args:
            batch_results: Results from batch processing
            output_path: Path to save report
            format: Report format ('json' or 'csv')
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(batch_results, f, indent=2, ensure_ascii=False)
        
        elif format == 'csv':
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'Filename',
                    'Status',
                    'Error',
                    'Original Words',
                    'Original Sentences',
                    'Processing Time (s)'
                ])
                
                # Data rows
                for result in batch_results['results']:
                    writer.writerow([
                        result['filename'],
                        result['status'],
                        result.get('error', ''),
                        result['original_stats']['word_count'] if result['original_stats'] else '',
                        result['original_stats']['sentence_count'] if result['original_stats'] else '',
                        f"{result['processing_time']:.2f}" if result['processing_time'] else ''
                    ])
        
        logger.info(f"Batch report saved to: {output_path}")
    
    def print_batch_summary(self, batch_results: Dict):
        """Print human-readable batch summary."""
        print("\n" + "="*80)
        print("BATCH PROCESSING SUMMARY")
        print("="*80)
        print(f"Total documents: {batch_results['total']}")
        print(f"Successful: {batch_results['successful']}")
        print(f"Failed: {batch_results['failed']}")
        print(f"Success rate: {batch_results['successful']/batch_results['total']*100:.1f}%")
        print(f"Method: {batch_results['method']}")
        print(f"Timestamp: {batch_results['timestamp']}")
        
        if batch_results['failed'] > 0:
            print("\nFailed documents:")
            for result in batch_results['results']:
                if result['status'] == 'failed':
                    print(f"  - {result['filename']}: {result['error']}")
        
        print("="*80 + "\n")
