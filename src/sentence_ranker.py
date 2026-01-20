"""
Sentence Ranking Algorithms for Extractive Summarization
"""

import numpy as np
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class SentenceRanker:
    """Implements various sentence ranking algorithms."""
    
    def __init__(self):
        """Initialize sentence ranker."""
        self.vectorizer = None
    
    def tfidf_score(self, sentences: List[str]) -> np.ndarray:
        """
        Score sentences using TF-IDF.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Array of scores for each sentence
        """
        if len(sentences) == 0:
            return np.array([])
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            tfidf_matrix = self.vectorizer.fit_transform(sentences)
            
            # Sum TF-IDF scores for each sentence
            scores = np.array(tfidf_matrix.sum(axis=1)).flatten()
            
        except ValueError:
            # If all sentences are empty or have no valid tokens
            scores = np.zeros(len(sentences))
        
        return scores
    
    def position_score(self, num_sentences: int) -> np.ndarray:
        """
        Score sentences based on position (first and last sentences are important).
        
        Args:
            num_sentences: Total number of sentences
            
        Returns:
            Array of position scores
        """
        scores = np.zeros(num_sentences)
        
        if num_sentences == 0:
            return scores
        
        # First sentences are most important
        for i in range(min(3, num_sentences)):
            scores[i] = 1.0 - (i * 0.2)
        
        # Last sentences are also important
        for i in range(max(num_sentences - 2, 0), num_sentences):
            scores[i] += 0.5
        
        return scores
    
    def textrank(self, sentences: List[str], embeddings: np.ndarray) -> np.ndarray:
        """
        Implement TextRank algorithm using sentence embeddings.
        
        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings matrix
            
        Returns:
            Array of TextRank scores
        """
        if len(sentences) == 0 or embeddings.shape[0] == 0:
            return np.array([])
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Set diagonal to 0 (no self-loops)
        np.fill_diagonal(similarity_matrix, 0)
        
        # Normalize similarity matrix
        row_sums = similarity_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Avoid division by zero
        similarity_matrix = similarity_matrix / row_sums
        
        # PageRank algorithm
        damping_factor = 0.85
        num_sentences = len(sentences)
        scores = np.ones(num_sentences) / num_sentences
        
        for _ in range(30):  # Iterate until convergence
            prev_scores = scores.copy()
            scores = (1 - damping_factor) / num_sentences + \
                     damping_factor * similarity_matrix.T.dot(prev_scores)
            
            # Check for convergence
            if np.allclose(scores, prev_scores, atol=1e-6):
                break
        
        return scores
    
    def lexrank(self, sentences: List[str]) -> np.ndarray:
        """
        Implement LexRank algorithm using TF-IDF.
        
        Args:
            sentences: List of sentences
            
        Returns:
            Array of LexRank scores
        """
        if len(sentences) == 0:
            return np.array([])
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        
        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
            
            # Calculate cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Apply threshold
            threshold = 0.1
            similarity_matrix[similarity_matrix < threshold] = 0
            
            # Normalize
            row_sums = similarity_matrix.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            similarity_matrix = similarity_matrix / row_sums
            
            # PageRank
            damping_factor = 0.85
            num_sentences = len(sentences)
            scores = np.ones(num_sentences) / num_sentences
            
            for _ in range(30):
                prev_scores = scores.copy()
                scores = (1 - damping_factor) / num_sentences + \
                         damping_factor * similarity_matrix.T.dot(prev_scores)
                
                if np.allclose(scores, prev_scores, atol=1e-6):
                    break
            
        except ValueError:
            # If vectorization fails
            scores = np.zeros(len(sentences))
        
        return scores
    
    def combined_score(
        self,
        sentences: List[str],
        embeddings: np.ndarray = None,
        weights: Dict[str, float] = None
    ) -> np.ndarray:
        """
        Combine multiple scoring methods.
        
        Args:
            sentences: List of sentences
            embeddings: Sentence embeddings (optional)
            weights: Weights for each method
            
        Returns:
            Combined scores
        """
        if weights is None:
            weights = {
                'tfidf': 0.4,
                'position': 0.3,
                'textrank': 0.3
            }
        
        num_sentences = len(sentences)
        combined = np.zeros(num_sentences)
        
        # TF-IDF scores
        if weights.get('tfidf', 0) > 0:
            tfidf_scores = self.tfidf_score(sentences)
            if len(tfidf_scores) > 0:
                tfidf_scores = tfidf_scores / (np.max(tfidf_scores) + 1e-10)
                combined += weights['tfidf'] * tfidf_scores
        
        # Position scores
        if weights.get('position', 0) > 0:
            position_scores = self.position_score(num_sentences)
            combined += weights['position'] * position_scores
        
        # TextRank scores
        if weights.get('textrank', 0) > 0 and embeddings is not None:
            textrank_scores = self.textrank(sentences, embeddings)
            if len(textrank_scores) > 0:
                textrank_scores = textrank_scores / (np.max(textrank_scores) + 1e-10)
                combined += weights['textrank'] * textrank_scores
        
        return combined
