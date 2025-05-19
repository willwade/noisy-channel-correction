#!/usr/bin/env python3
"""
Word-level N-gram model for context-aware language modeling.

This module implements a word-level n-gram model for calculating
the probability of a word given its context (previous words).
"""

import os
import sys
import logging
import math
import pickle
from typing import List, Dict, Tuple, Set, Optional, Any
from collections import defaultdict, Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WordNGramModel:
    """
    Word-level N-gram model for context-aware language modeling.
    
    This class implements a word-level n-gram model for calculating
    the probability of a word given its context (previous words).
    """
    
    def __init__(self, max_order: int = 3, smoothing_method: str = "kneser_ney"):
        """
        Initialize a word-level N-gram model.
        
        Args:
            max_order: Maximum n-gram order (e.g., 3 for trigrams)
            smoothing_method: Smoothing method to use ("kneser_ney", "laplace", "witten_bell")
        """
        self.max_order = max_order
        self.smoothing_method = smoothing_method
        
        # Initialize n-gram counts
        self.ngram_counts = [defaultdict(int) for _ in range(max_order + 1)]
        
        # Initialize vocabulary
        self.vocabulary = set()
        
        # Smoothing parameters
        self.discount = 0.1  # Discount parameter for Kneser-Ney smoothing
        self.alpha = 0.1  # Laplace smoothing parameter
        
        # Model ready flag
        self.model_ready = False
        
    def train(self, text: str) -> bool:
        """
        Train the model on the given text.
        
        Args:
            text: Text to train on
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Training word n-gram model on text ({len(text)} characters)")
            
            # Tokenize the text into words
            words = self._tokenize(text)
            
            # Update vocabulary
            self.vocabulary.update(words)
            
            # Count n-grams
            self._count_ngrams(words)
            
            # Set model ready flag
            self.model_ready = True
            
            logger.info(f"Word n-gram model trained successfully. Vocabulary size: {len(self.vocabulary)}")
            return True
        except Exception as e:
            logger.error(f"Error training word n-gram model: {e}")
            return False
    
    def update(self, text: str) -> bool:
        """
        Update the model with new text.
        
        Args:
            text: Text to update the model with
            
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Updating word n-gram model with text ({len(text)} characters)")
            
            # Tokenize the text into words
            words = self._tokenize(text)
            
            # Update vocabulary
            self.vocabulary.update(words)
            
            # Count n-grams
            self._count_ngrams(words)
            
            # Set model ready flag
            self.model_ready = True
            
            logger.info(f"Word n-gram model updated successfully. Vocabulary size: {len(self.vocabulary)}")
            return True
        except Exception as e:
            logger.error(f"Error updating word n-gram model: {e}")
            return False
    
    def probability(self, word: str, context: List[str]) -> float:
        """
        Calculate the probability of a word given its context.
        
        Args:
            word: The word to calculate probability for
            context: List of previous words
            
        Returns:
            Probability of the word given the context
        """
        if not self.model_ready:
            logger.warning("Word n-gram model not ready. Cannot calculate probability.")
            return 1e-10  # Return a small probability
        
        # Convert to lowercase
        word = word.lower()
        context = [w.lower() for w in context]
        
        # Limit context to max_order - 1
        context = context[-(self.max_order - 1):]
        
        # Calculate probability based on the smoothing method
        if self.smoothing_method == "kneser_ney":
            return self._kneser_ney_probability(word, context)
        elif self.smoothing_method == "laplace":
            return self._laplace_probability(word, context)
        elif self.smoothing_method == "witten_bell":
            return self._witten_bell_probability(word, context)
        else:
            return self._kneser_ney_probability(word, context)  # Default to Kneser-Ney
    
    def save(self, model_path: str) -> bool:
        """
        Save the model to a file.
        
        Args:
            model_path: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create a dictionary with the model state
            model_state = {
                "max_order": self.max_order,
                "smoothing_method": self.smoothing_method,
                "ngram_counts": self.ngram_counts,
                "vocabulary": self.vocabulary,
                "discount": self.discount,
                "alpha": self.alpha,
                "model_ready": self.model_ready,
            }
            
            # Save the model state to a file
            with open(model_path, "wb") as f:
                pickle.dump(model_state, f)
            
            logger.info(f"Word n-gram model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving word n-gram model: {e}")
            return False
    
    def load(self, model_path: str) -> bool:
        """
        Load the model from a file.
        
        Args:
            model_path: Path to load the model from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the model state from a file
            with open(model_path, "rb") as f:
                model_state = pickle.load(f)
            
            # Restore the model state
            self.max_order = model_state["max_order"]
            self.smoothing_method = model_state["smoothing_method"]
            self.ngram_counts = model_state["ngram_counts"]
            self.vocabulary = model_state["vocabulary"]
            self.discount = model_state["discount"]
            self.alpha = model_state["alpha"]
            self.model_ready = model_state["model_ready"]
            
            logger.info(f"Word n-gram model loaded from {model_path}. Vocabulary size: {len(self.vocabulary)}")
            return True
        except Exception as e:
            logger.error(f"Error loading word n-gram model: {e}")
            return False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize the text into words.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of words
        """
        # Simple tokenization by splitting on whitespace and removing punctuation
        words = []
        for word in text.split():
            # Remove punctuation at the beginning and end of the word
            word = word.strip(".,;:!?\"'()[]{}")
            if word:
                words.append(word.lower())
        
        return words
    
    def _count_ngrams(self, words: List[str]) -> None:
        """
        Count n-grams in the given list of words.
        
        Args:
            words: List of words to count n-grams from
        """
        # Add sentence boundary markers
        padded_words = ["<s>"] * (self.max_order - 1) + words + ["</s>"]
        
        # Count n-grams for each order
        for order in range(1, self.max_order + 1):
            for i in range(len(padded_words) - order + 1):
                ngram = tuple(padded_words[i:i+order])
                self.ngram_counts[order][ngram] += 1
    
    def _kneser_ney_probability(self, word: str, context: List[str]) -> float:
        """
        Calculate the probability of a word given its context using Kneser-Ney smoothing.
        
        Args:
            word: The word to calculate probability for
            context: List of previous words
            
        Returns:
            Probability of the word given the context
        """
        # Start with the highest order n-gram
        for order in range(min(len(context) + 1, self.max_order), 0, -1):
            # Get the context for this order
            current_context = tuple(context[-(order-1):]) if order > 1 else tuple()
            
            # Get the n-gram
            ngram = current_context + (word,)
            
            # Get the count of the n-gram
            ngram_count = self.ngram_counts[order].get(ngram, 0)
            
            # Get the count of the context
            context_count = sum(count for ng, count in self.ngram_counts[order].items() 
                               if ng[:-1] == current_context)
            
            # If we have seen this n-gram, calculate the probability
            if ngram_count > 0 and context_count > 0:
                # Calculate the probability with discounting
                prob = max(ngram_count - self.discount, 0) / context_count
                
                # Calculate the backoff weight
                backoff_weight = (self.discount / context_count) * len(self.ngram_counts[order])
                
                # Calculate the backoff probability
                backoff_prob = self._kneser_ney_probability(word, context[-(order-2):]) if order > 1 else 1.0 / len(self.vocabulary)
                
                # Return the interpolated probability
                return prob + backoff_weight * backoff_prob
        
        # If we haven't seen this word, return a small probability
        return 1.0 / (len(self.vocabulary) + 1)
    
    def _laplace_probability(self, word: str, context: List[str]) -> float:
        """
        Calculate the probability of a word given its context using Laplace smoothing.
        
        Args:
            word: The word to calculate probability for
            context: List of previous words
            
        Returns:
            Probability of the word given the context
        """
        # Start with the highest order n-gram
        for order in range(min(len(context) + 1, self.max_order), 0, -1):
            # Get the context for this order
            current_context = tuple(context[-(order-1):]) if order > 1 else tuple()
            
            # Get the n-gram
            ngram = current_context + (word,)
            
            # Get the count of the n-gram
            ngram_count = self.ngram_counts[order].get(ngram, 0)
            
            # Get the count of the context
            context_count = sum(count for ng, count in self.ngram_counts[order].items() 
                               if ng[:-1] == current_context)
            
            # If we have seen this context, calculate the probability
            if context_count > 0:
                # Calculate the probability with Laplace smoothing
                prob = (ngram_count + self.alpha) / (context_count + self.alpha * len(self.vocabulary))
                
                return prob
        
        # If we haven't seen this context, return a uniform probability
        return 1.0 / len(self.vocabulary) if self.vocabulary else 0.1
    
    def _witten_bell_probability(self, word: str, context: List[str]) -> float:
        """
        Calculate the probability of a word given its context using Witten-Bell smoothing.
        
        Args:
            word: The word to calculate probability for
            context: List of previous words
            
        Returns:
            Probability of the word given the context
        """
        # Start with the highest order n-gram
        for order in range(min(len(context) + 1, self.max_order), 0, -1):
            # Get the context for this order
            current_context = tuple(context[-(order-1):]) if order > 1 else tuple()
            
            # Get the n-gram
            ngram = current_context + (word,)
            
            # Get the count of the n-gram
            ngram_count = self.ngram_counts[order].get(ngram, 0)
            
            # Get the count of the context
            context_count = sum(count for ng, count in self.ngram_counts[order].items() 
                               if ng[:-1] == current_context)
            
            # Count the number of unique words following this context
            unique_words = len(set(ng[-1] for ng in self.ngram_counts[order].keys() 
                                if ng[:-1] == current_context))
            
            # If we have seen this context, calculate the probability
            if context_count > 0:
                # Calculate lambda for Witten-Bell smoothing
                lambda_wb = context_count / (context_count + unique_words)
                
                # Calculate the probability with Witten-Bell smoothing
                prob = lambda_wb * (ngram_count / context_count)
                
                # Calculate the backoff probability
                backoff_prob = self._witten_bell_probability(word, context[-(order-2):]) if order > 1 else 1.0 / len(self.vocabulary)
                
                # Return the interpolated probability
                return prob + (1 - lambda_wb) * backoff_prob
        
        # If we haven't seen this word, return a small probability
        return 1.0 / (len(self.vocabulary) + 1)
