"""
Correction Engine (Noisy Channel Model) for AAC input correction.

This module implements a noisy channel model for correcting noisy AAC input.
It combines a PPM language model for P(intended) and a confusion matrix for
P(noisy | intended) to rank candidate corrections.

The main components are:
1. NoisyChannelCorrector: Class for correcting noisy input
2. correct: Function to correct a noisy input
"""

import os
import sys
import logging
import math
from typing import List, Dict, Tuple, Set, Optional, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the PPM model
from module4.ppm.enhanced_ppm_predictor import EnhancedPPMPredictor

# Import the confusion matrix
from module2.confusion_matrix import ConfusionMatrix, get_error_probability

# Import the candidate generator
from module3.candidate_generator import generate_candidates, CandidateGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NoisyChannelCorrector:
    """
    Class for correcting noisy input using a noisy channel model.

    This class combines a PPM language model for P(intended) and a confusion matrix
    for P(noisy | intended) to rank candidate corrections.
    """

    def __init__(
        self,
        ppm_model: Optional[EnhancedPPMPredictor] = None,
        confusion_model: Optional[ConfusionMatrix] = None,
        lexicon: Optional[Set[str]] = None,
        max_candidates: int = 10,
    ):
        """
        Initialize a noisy channel corrector.

        Args:
            ppm_model: PPM language model for P(intended)
            confusion_model: Confusion matrix for P(noisy | intended)
            lexicon: Set of valid words to use for filtering candidates
            max_candidates: Maximum number of candidates to return
        """
        # Initialize the PPM model
        self.ppm_model = ppm_model if ppm_model is not None else EnhancedPPMPredictor()
        
        # Initialize the confusion matrix
        self.confusion_model = confusion_model
        
        # Initialize the lexicon
        self.lexicon = lexicon if lexicon is not None else set()
        
        # Initialize the candidate generator
        self.candidate_generator = CandidateGenerator(
            lexicon=self.lexicon, max_candidates=max_candidates
        )
        
        # Maximum number of candidates to return
        self.max_candidates = max_candidates
        
        # Check if the models are ready
        self.models_ready = self.ppm_model.model_ready and self.confusion_model is not None
        
        if not self.models_ready:
            logger.warning("Models not fully initialized. Some functionality may be limited.")

    def load_ppm_model(self, model_path: str) -> bool:
        """
        Load a PPM model from a file.

        Args:
            model_path: Path to the PPM model file

        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.ppm_model.load_model(model_path)
            if success:
                logger.info(f"Loaded PPM model from {model_path}")
            else:
                logger.error(f"Failed to load PPM model from {model_path}")
            return success
        except Exception as e:
            logger.error(f"Error loading PPM model from {model_path}: {e}")
            return False

    def load_confusion_model(self, model_path: str) -> bool:
        """
        Load a confusion matrix from a file.

        Args:
            model_path: Path to the confusion matrix file

        Returns:
            True if successful, False otherwise
        """
        try:
            self.confusion_model = ConfusionMatrix.load(model_path)
            logger.info(f"Loaded confusion matrix from {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading confusion matrix from {model_path}: {e}")
            return False

    def load_lexicon_from_file(self, file_path: str) -> bool:
        """
        Load a lexicon from a file.

        Args:
            file_path: Path to the lexicon file (one word per line)

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, "r") as f:
                self.lexicon = set(line.strip().lower() for line in f if line.strip())
            
            # Update the candidate generator's lexicon
            self.candidate_generator.lexicon = self.lexicon
            
            logger.info(f"Loaded lexicon with {len(self.lexicon)} words from {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading lexicon from {file_path}: {e}")
            return False

    def correct(
        self,
        noisy_input: str,
        context: str = "",
        max_edit_distance: int = 2,
        use_keyboard_adjacency: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Correct a noisy input using the noisy channel model.

        Args:
            noisy_input: The noisy input text
            context: Optional context for the PPM model
            max_edit_distance: Maximum edit distance to consider
            use_keyboard_adjacency: Whether to use keyboard adjacency for candidate generation

        Returns:
            List of (correction, score) tuples, sorted by score (highest first)
        """
        # Check if the models are ready
        if not self.models_ready:
            logger.warning("Models not fully initialized. Using fallback correction.")
            return [(noisy_input, 1.0)]  # Return the input as is

        # Generate candidates
        candidates = self.candidate_generator.generate_candidates(
            noisy_input, max_edit_distance, use_keyboard_adjacency
        )

        # If no candidates were found, return the input as is
        if not candidates:
            return [(noisy_input, 1.0)]

        # Score candidates using the noisy channel model
        scored_candidates = self._score_candidates(noisy_input, candidates, context)

        # Return the top N candidates
        return scored_candidates[: self.max_candidates]

    def _score_candidates(
        self, noisy_input: str, candidates: List[Tuple[str, float]], context: str = ""
    ) -> List[Tuple[str, float]]:
        """
        Score candidates using the noisy channel model.

        Args:
            noisy_input: The noisy input text
            candidates: List of (candidate, initial_score) tuples
            context: Optional context for the PPM model

        Returns:
            List of (candidate, score) tuples, sorted by score (highest first)
        """
        scored_candidates = []

        for candidate, initial_score in candidates:
            # Calculate P(intended) using the PPM model
            p_intended = self._get_p_intended(candidate, context)
            
            # Calculate P(noisy | intended) using the confusion matrix
            p_noisy_given_intended = self._get_p_noisy_given_intended(noisy_input, candidate)
            
            # Calculate the final score using the noisy channel model formula
            # score = log(P(intended)) + log(P(noisy | intended))
            log_p_intended = math.log(p_intended) if p_intended > 0 else -float("inf")
            log_p_noisy_given_intended = math.log(p_noisy_given_intended) if p_noisy_given_intended > 0 else -float("inf")
            
            # Combine the scores
            score = log_p_intended + log_p_noisy_given_intended
            
            # Convert back to a probability (for easier interpretation)
            final_score = math.exp(score) if score != -float("inf") else 0.0
            
            # Add to the list of scored candidates
            scored_candidates.append((candidate, final_score))

        # Sort by score (highest first)
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    def _get_p_intended(self, candidate: str, context: str = "") -> float:
        """
        Calculate P(intended) using the PPM model.

        Args:
            candidate: The candidate correction
            context: Optional context for the PPM model

        Returns:
            The probability P(intended)
        """
        # If the PPM model is not available, use a simple frequency-based approach
        if not self.ppm_model.model_ready:
            # Return a small probability for all candidates
            return 0.01
        
        # Use the PPM model to get the probability of the candidate
        # For simplicity, we'll use the word frequency as a proxy for P(intended)
        # In a more sophisticated implementation, we would use the PPM model's
        # probability calculation directly
        
        # Get the word frequency from the PPM model
        freq = self.ppm_model.word_frequencies.get(candidate.lower(), 0)
        
        # Normalize by the total number of words
        total_words = sum(self.ppm_model.word_frequencies.values())
        
        # Calculate the probability (with smoothing to avoid zero probabilities)
        p_intended = (freq + 1) / (total_words + len(self.ppm_model.word_frequencies))
        
        return p_intended

    def _get_p_noisy_given_intended(self, noisy: str, intended: str) -> float:
        """
        Calculate P(noisy | intended) using the confusion matrix.

        Args:
            noisy: The noisy input text
            intended: The candidate correction

        Returns:
            The probability P(noisy | intended)
        """
        # If the confusion matrix is not available, use a simple edit distance approach
        if self.confusion_model is None:
            # Calculate edit distance
            distance = self.candidate_generator._levenshtein_distance(noisy, intended)
            
            # Convert distance to a probability (higher for smaller distances)
            max_len = max(len(noisy), len(intended))
            p_noisy_given_intended = 1.0 - (distance / max_len if max_len > 0 else 0)
            
            return p_noisy_given_intended
        
        # Use the confusion matrix to get the probability
        return get_error_probability(noisy, intended, self.confusion_model)


def correct(
    noisy_input: str,
    ppm_model_path: Optional[str] = None,
    confusion_matrix_path: Optional[str] = None,
    lexicon_path: Optional[str] = None,
    max_candidates: int = 5,
) -> List[Tuple[str, float]]:
    """
    Correct a noisy input using the noisy channel model.

    This is a convenience function that creates a NoisyChannelCorrector and uses it
    to correct a noisy input.

    Args:
        noisy_input: The noisy input text
        ppm_model_path: Path to the PPM model file
        confusion_matrix_path: Path to the confusion matrix file
        lexicon_path: Path to the lexicon file
        max_candidates: Maximum number of candidates to return

    Returns:
        List of (correction, score) tuples, sorted by score (highest first)
    """
    # Create a corrector
    corrector = NoisyChannelCorrector(max_candidates=max_candidates)
    
    # Load the PPM model if provided
    if ppm_model_path:
        corrector.load_ppm_model(ppm_model_path)
    
    # Load the confusion matrix if provided
    if confusion_matrix_path:
        corrector.load_confusion_model(confusion_matrix_path)
    
    # Load the lexicon if provided
    if lexicon_path:
        corrector.load_lexicon_from_file(lexicon_path)
    
    # Correct the input
    return corrector.correct(noisy_input)
