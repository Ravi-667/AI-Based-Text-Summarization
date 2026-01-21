"""
Main CLI Application for AI-Based Text Summarization
"""

import argparse
import sys
import os
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.extractive_summarizer import ExtractiveSummarizer
from src.abstractive_summarizer import AbstractiveSummarizer
from src.evaluation import SummaryEvaluator
from src.utils import (
    read_document,
    save_summary,
    get_text_stats,
    format_summary_output,
    setup_logger,
    create_directories
)
from src.config import OUTPUT_DIR

logger = setup_logger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="AI-Based Text Summarization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extractive summarization with 30% ratio
  python main.py --input article.txt --method extractive --ratio 0.3
  
  # Abstractive summarization with max 200 words
  python main.py --input research.txt --method abstractive --max-length 200
  
  # Use both methods and compare
  python main.py --input document.txt --method both --output summary.txt
  
  # Batch processing
  python main.py --batch data/samples/ --method extractive
        """
    )
    
    # Input options
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input file path (txt, pdf, or docx)'
    )
    
    parser.add_argument(
        '--text', '-t',
        type=str,
        help='Direct text input (alternative to --input)'
    )
    
    parser.add_argument(
        '--batch', '-b',
        type=str,
        help='Directory for batch processing'
    )
    
    # Summarization method
    parser.add_argument(
        '--method', '-m',
        type=str,
        choices=['extractive', 'abstractive', 'both'],
        default='extractive',
        help='Summarization method (default: extractive)'
    )
    
    # Extractive options
    parser.add_argument(
        '--ratio', '-r',
        type=float,
        default=0.3,
        help='Extraction ratio for extractive method (default: 0.3)'
    )
    
    parser.add_argument(
        '--num-sentences', '-n',
        type=int,
        help='Exact number of sentences to extract'
    )
    
    parser.add_argument(
        '--scoring',
        type=str,
        choices=['tfidf', 'textrank', 'lexrank', 'combined'],
        default='combined',
        help='Sentence scoring method (default: combined)'
    )
    
    # Abstractive options
    parser.add_argument(
        '--max-length',
        type=int,
        default=150,
        help='Maximum summary length for abstractive method (default: 150)'
    )
    
    parser.add_argument(
        '--min-length',
        type=int,
        default=50,
        help='Minimum summary length for abstractive method (default: 50)'
    )
    
    # Output options
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file path (optional)'
    )
    
    parser.add_argument(
        '--format',
        type=str,
        choices=['text', 'json'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '--evaluate',
        action='store_true',
        help='Show evaluation metrics'
    )
    
    # Model options
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'cpu'],
        default='auto',
        help='Device to use (default: auto)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    return parser.parse_args()


def summarize_text(text: str, args) -> dict:
    """
    Summarize text using specified method.
    
    Args:
        text: Input text
        args: Command line arguments
        
    Returns:
        Dictionary with summaries and metadata
    """
    results = {'original_text': text}
    
    # Get original text statistics
    original_stats = get_text_stats(text)
    results['original_stats'] = original_stats
    
    logger.info(f"Original text: {original_stats['word_count']} words, "
                f"{original_stats['sentence_count']} sentences")
    
    # Extractive summarization
    if args.method in ['extractive', 'both']:
        logger.info("Generating extractive summary...")
        summarizer = ExtractiveSummarizer(device=args.device)
        
        extractive_summary = summarizer.generate_summary(
            text,
            ratio=args.ratio,
            num_sentences=args.num_sentences,
            method=args.scoring
        )
        
        results['extractive_summary'] = extractive_summary
        results['extractive_stats'] = get_text_stats(extractive_summary)
    
    # Abstractive summarization
    if args.method in ['abstractive', 'both']:
        logger.info("Generating abstractive summary...")
        summarizer = AbstractiveSummarizer(device=args.device)
        
        abstractive_summary = summarizer.generate_summary(
            text,
            max_length=args.max_length,
            min_length=args.min_length
        )
        
        results['abstractive_summary'] = abstractive_summary
        results['abstractive_stats'] = get_text_stats(abstractive_summary)
    
    # Evaluation
    if args.evaluate:
        logger.info("Evaluating summaries...")
        evaluator = SummaryEvaluator()
        
        if 'extractive_summary' in results:
            results['extractive_eval'] = evaluator.evaluate_summary(
                results['extractive_summary'],
                text
            )
        
        if 'abstractive_summary' in results:
            results['abstractive_eval'] = evaluator.evaluate_summary(
                results['abstractive_summary'],
                text
            )
    
    return results


def format_output(results: dict, args) -> str:
    """Format results for output."""
    output = ""
    
    if args.format == 'json':
        import json
        return json.dumps(results, indent=2)
    
    # Text format
    output += "=" * 80 + "\n"
    output += "AI-BASED TEXT SUMMARIZATION RESULTS\n"
    output += "=" * 80 + "\n\n"
    
    # Original text stats
    stats = results['original_stats']
    output += f"ORIGINAL TEXT STATISTICS:\n"
    output += f"  Words: {stats['word_count']}\n"
    output += f"  Sentences: {stats['sentence_count']}\n"
    output += f"  Reading Time: {stats['reading_time_minutes']} minutes\n\n"
    
    # Extractive summary
    if 'extractive_summary' in results:
        output += format_summary_output(
            results['extractive_summary'],
            results['original_stats'],
            results['extractive_stats'],
            'extractive'
        )
        
        if 'extractive_eval' in results:
            evaluator = SummaryEvaluator()
            output += evaluator.format_evaluation_report(
                results['extractive_eval'],
                "Extractive"
            )
    
    # Abstractive summary
    if 'abstractive_summary' in results:
        output += format_summary_output(
            results['abstractive_summary'],
            results['original_stats'],
            results['abstractive_stats'],
            'abstractive'
        )
        
        if 'abstractive_eval' in results:
            evaluator = SummaryEvaluator()
            output += evaluator.format_evaluation_report(
                results['abstractive_eval'],
                "Abstractive"
            )
    
    return output


def main():
    """Main application entry point."""
    # Create necessary directories
    create_directories()
    
    # Parse arguments
    args = parse_arguments()
    
    # Set log level
    if args.verbose:
        logger.setLevel('DEBUG')
    
    # Get input text
    text = None
    
    if args.text:
        text = args.text
        logger.info("Using direct text input")
    
    elif args.input:
        logger.info(f"Reading from file: {args.input}")
        try:
            text = read_document(args.input)
        except Exception as e:
            logger.error(f"Error reading input file: {e}")
            sys.exit(1)
    
    
    elif args.batch:
        logger.info(f"Batch processing directory: {args.batch}")
        
        try:
            from src.batch_processor import BatchProcessor
            
            # Create batch processor
            processor = BatchProcessor(
                method=args.method,
                device=args.device,
                ratio=args.ratio if args.method in ['extractive', 'both'] else None,
                num_sentences=args.num_sentences if args.method in ['extractive', 'both'] else None,
                scoring=args.scoring if args.method in ['extractive', 'both'] else None,
                max_length=args.max_length if args.method in ['abstractive', 'both'] else None,
                min_length=args.min_length if args.method in ['abstractive', 'both'] else None
            )
            
            # Determine output directory
            if args.output:
                output_dir = args.output
            else:
                output_dir = os.path.join(OUTPUT_DIR, f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Process directory
            batch_results = processor.process_directory(
                directory=args.batch,
                output_dir=output_dir,
                recursive=True,
                save_summaries=True
            )
            
            # Print summary
            processor.print_batch_summary(batch_results)
            
            # Save batch report
            report_path = os.path.join(output_dir, 'batch_report.json')
            processor.generate_batch_report(batch_results, report_path, format='json')
            
            # Also save CSV report
            csv_path = os.path.join(output_dir, 'batch_report.csv')
            processor.generate_batch_report(batch_results, csv_path, format='csv')
            
            print(f"✓ Summaries saved to: {output_dir}")
            print(f"✓ Reports saved: batch_report.json, batch_report.csv")
            
            sys.exit(0)
            
        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    else:
        logger.error("No input provided. Use --input, --text, or --batch")
        sys.exit(1)
    
    if not text or not text.strip():
        logger.error("Input text is empty")
        sys.exit(1)
    
    # Generate summaries
    try:
        results = summarize_text(text, args)
    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)
    
    # Format output
    output_text = format_output(results, args)
    
    # Save or print output
    if args.output:
        output_path = args.output
        logger.info(f"Saving results to: {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(output_text)
        
        print(f"✓ Summary saved to: {output_path}")
    else:
        print(output_text)
    
    logger.info("Summarization complete!")


if __name__ == "__main__":
    main()
