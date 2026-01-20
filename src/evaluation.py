"""
Evaluation Module for Summary Quality Assessment
"""

import numpy as np
from typing import List, Dict, Optional
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import setup_logger

logger = setup_logger(__name__)


class SummaryEvaluator:
    """Evaluates summary quality using various metrics."""
    
    def __init__(self):
        """Initialize summary evaluator."""
        self.rouge_scorer = rouge_scorer.RougeScorer(
            ['rouge1', 'rouge2', 'rougeL'],
            use_stemmer=True
        )
    
    def calculate_rouge(
        self,
        summary: str,
        reference: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate ROUGE scores.
        
        Args:
            summary: Generated summary
            reference: Reference summary
            
        Returns:
            Dictionary with ROUGE scores
        """
        scores = self.rouge_scorer.score(reference, summary)
        
        # Convert to dictionary format
        rouge_dict = {}
        for metric, score in scores.items():
            rouge_dict[metric] = {
                'precision': score.precision,
                'recall': score.recall,
                'fmeasure': score.fmeasure
            }
        
        logger.info(f"ROUGE-L F1: {rouge_dict['rougeL']['fmeasure']:.4f}")
        
        return rouge_dict
    
    def calculate_bleu(
        self,
        summary: str,
        reference: str,
        n: int = 4
    ) -> float:
        """
        Calculate BLEU score (simplified version).
        
        Args:
            summary: Generated summary
            reference: Reference summary
            n: Maximum n-gram size
            
        Returns:
            BLEU score
        """
        summary_tokens = summary.lower().split()
        reference_tokens = reference.lower().split()
        
        if len(summary_tokens) == 0:
            return 0.0
        
        # Calculate precision for each n-gram
        precisions = []
        
        for i in range(1, n + 1):
            summary_ngrams = self._get_ngrams(summary_tokens, i)
            reference_ngrams = self._get_ngrams(reference_tokens, i)
            
            if len(summary_ngrams) == 0:
                continue
            
            # Count matches
            matches = sum(min(summary_ngrams.count(ng), reference_ngrams.count(ng))
                         for ng in set(summary_ngrams))
            
            precision = matches / len(summary_ngrams)
            precisions.append(precision)
        
        if len(precisions) == 0:
            return 0.0
        
        # Geometric mean of precisions
        bleu = np.exp(np.mean([np.log(p + 1e-10) for p in precisions]))
        
        # Brevity penalty
        bp = min(1.0, np.exp(1 - len(reference_tokens) / len(summary_tokens)))
        
        bleu_score = bp * bleu
        
        logger.info(f"BLEU: {bleu_score:.4f}")
        
        return bleu_score
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Get n-grams from tokens."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def semantic_similarity(
        self,
        summary: str,
        original: str
    ) -> float:
        """
        Calculate semantic similarity using TF-IDF.
        
        Args:
            summary: Generated summary
            original: Original text
            
        Returns:
            Cosine similarity score (0-1)
        """
        vectorizer = TfidfVectorizer()
        
        try:
            vectors = vectorizer.fit_transform([original, summary])
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            
            logger.info(f"Semantic Similarity: {similarity:.4f}")
            
            return float(similarity)
            
        except ValueError:
            logger.warning("Could not calculate semantic similarity")
            return 0.0
    
    def readability_score(self, text: str) -> Dict[str, float]:
        """
        Calculate readability metrics.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with readability metrics
        """
        sentences = text.split('.')
        words = text.split()
        
        sentences = [s for s in sentences if s.strip()]
        words = [w for w in words if w.strip()]
        
        if len(sentences) == 0 or len(words) == 0:
            return {
                'avg_sentence_length': 0,
                'avg_word_length': 0
            }
        
        avg_sentence_length = len(words) / len(sentences)
        avg_word_length = np.mean([len(word) for word in words])
        
        # Simple readability score (lower is more readable)
        readability = avg_sentence_length * avg_word_length / 100
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'avg_word_length': avg_word_length,
            'readability_score': readability
        }
    
    def evaluate_summary(
        self,
        summary: str,
        original_text: str,
        reference_summary: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Comprehensive summary evaluation.
        
        Args:
            summary: Generated summary
            original_text: Original text
            reference_summary: Reference summary for comparison (optional)
            
        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("Evaluating summary...")
        
        evaluation = {
            'semantic_similarity': self.semantic_similarity(summary, original_text),
            'readability': self.readability_score(summary)
        }
        
        # Calculate ROUGE and BLEU if reference is provided
        if reference_summary:
            evaluation['rouge'] = self.calculate_rouge(summary, reference_summary)
            evaluation['bleu'] = self.calculate_bleu(summary, reference_summary)
        
        # Calculate compression ratio
        original_words = len(original_text.split())
        summary_words = len(summary.split())
        
        if original_words > 0:
            evaluation['compression_ratio'] = 1 - (summary_words / original_words)
        else:
            evaluation['compression_ratio'] = 0
        
        return evaluation
    
    def compare_summaries(
        self,
        summaries: Dict[str, str],
        original_text: str,
        reference_summary: Optional[str] = None
    ) -> Dict[str, Dict]:
        """
        Compare multiple summaries.
        
        Args:
            summaries: Dictionary mapping method name to summary
            original_text: Original text
            reference_summary: Reference summary (optional)
            
        Returns:
            Comparison results for each method
        """
        results = {}
        
        for method, summary in summaries.items():
            logger.info(f"Evaluating {method} summary")
            results[method] = self.evaluate_summary(
                summary,
                original_text,
                reference_summary
            )
        
        return results
    
    def format_evaluation_report(
        self,
        evaluation: Dict[str, any],
        method_name: str = "Summary"
    ) -> str:
        """
        Format evaluation results as a readable report.
        
        Args:
            evaluation: Evaluation results
            method_name: Name of summarization method
            
        Returns:
            Formatted report string
        """
        report = f"\n{'=' * 70}\n"
        report += f"EVALUATION REPORT: {method_name}\n"
        report += f"{'=' * 70}\n\n"
        
        # Semantic similarity
        report += f"Semantic Similarity: {evaluation['semantic_similarity']:.4f}\n"
        
        # Compression
        report += f"Compression Ratio: {evaluation['compression_ratio']:.2%}\n\n"
        
        # Readability
        report += "Readability Metrics:\n"
        for metric, value in evaluation['readability'].items():
            report += f"  {metric.replace('_', ' ').title()}: {value:.2f}\n"
        
        # ROUGE scores (if available)
        if 'rouge' in evaluation:
            report += "\nROUGE Scores:\n"
            for metric, scores in evaluation['rouge'].items():
                report += f"  {metric.upper()}:\n"
                report += f"    Precision: {scores['precision']:.4f}\n"
                report += f"    Recall: {scores['recall']:.4f}\n"
                report += f"    F1-Score: {scores['fmeasure']:.4f}\n"
        
        # BLEU score (if available)
        if 'bleu' in evaluation:
            report += f"\nBLEU Score: {evaluation['bleu']:.4f}\n"
        
        report += f"\n{'=' * 70}\n"
        
        return report


def evaluate_summary(
    summary: str,
    original_text: str,
    reference: Optional[str] = None
) -> Dict[str, any]:
    """
    Convenience function for summary evaluation.
    
    Args:
        summary: Generated summary
        original_text: Original text
        reference: Reference summary (optional)
        
    Returns:
        Evaluation metrics
    """
    evaluator = SummaryEvaluator()
    return evaluator.evaluate_summary(summary, original_text, reference)
