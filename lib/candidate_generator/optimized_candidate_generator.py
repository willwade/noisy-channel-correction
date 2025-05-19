"""
Optimized Candidate Generator for AAC input correction.

This module provides an optimized version of the CandidateGenerator class
that limits the number of candidates generated to improve performance.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Set, Optional, Any
import heapq
from collections import defaultdict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the keyboard layout model for adjacency calculations
from lib.noise_model.keyboard_error_model import KeyboardLayoutModel
from lib.candidate_generator.candidate_generator import CandidateGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class OptimizedCandidateGenerator(CandidateGenerator):
    """
    Optimized version of the CandidateGenerator class.

    This class limits the number of candidates generated to improve performance.
    """

    def __init__(self, lexicon: Optional[Set[str]] = None, max_candidates: int = 10, max_edits: int = 1000):
        """
        Initialize an optimized candidate generator.

        Args:
            lexicon: Set of valid words to use for filtering candidates
            max_candidates: Maximum number of candidates to return
            max_edits: Maximum number of edit candidates to generate
        """
        super().__init__(lexicon, max_candidates)
        self.max_edits = max_edits

    def _get_edit_distance_candidates(self, word: str, max_distance: int) -> Set[str]:
        """
        Generate candidates within a certain edit distance, with a limit.

        Args:
            word: The input word
            max_distance: Maximum edit distance to consider

        Returns:
            Set of candidate words
        """
        # For edit distance 1
        if max_distance == 1:
            edits1 = self._edit_distance_1(word)
            # Limit the number of candidates
            if len(edits1) > self.max_edits:
                logger.warning(f"Limiting edit distance 1 candidates from {len(edits1)} to {self.max_edits}")
                edits1 = set(list(edits1)[:self.max_edits])
            return edits1

        # For edit distance 2, apply edit distance 1 twice
        elif max_distance == 2:
            # Get all candidates at edit distance 1
            edits1 = self._edit_distance_1(word)
            
            # Limit the number of edit distance 1 candidates
            if len(edits1) > self.max_edits // 10:
                logger.warning(f"Limiting edit distance 1 candidates from {len(edits1)} to {self.max_edits // 10}")
                edits1 = set(list(edits1)[:self.max_edits // 10])
            
            # Apply edit distance 1 to each candidate to get edit distance 2
            edits2 = set()
            for edit1 in edits1:
                # Check if we've reached the limit
                if len(edits2) >= self.max_edits:
                    logger.warning(f"Reached maximum number of edit distance 2 candidates ({self.max_edits})")
                    break
                
                # Get edit distance 1 candidates for this word
                edit1_candidates = self._edit_distance_1(edit1)
                
                # Limit the number of candidates for this word
                if len(edit1_candidates) > 100:
                    edit1_candidates = set(list(edit1_candidates)[:100])
                
                # Add to the overall set
                edits2.update(edit1_candidates)
            
            # Final limit on the total number of candidates
            if len(edits2) > self.max_edits:
                logger.warning(f"Limiting edit distance 2 candidates from {len(edits2)} to {self.max_edits}")
                edits2 = set(list(edits2)[:self.max_edits])
            
            return edits2

        # For higher edit distances (not typically used)
        else:
            candidates = self._edit_distance_1(word)
            for _ in range(max_distance - 1):
                # Limit the number of candidates at each step
                if len(candidates) > self.max_edits // 10:
                    candidates = set(list(candidates)[:self.max_edits // 10])
                
                new_candidates = set()
                for candidate in candidates:
                    # Check if we've reached the limit
                    if len(new_candidates) >= self.max_edits:
                        break
                    
                    # Get edit distance 1 candidates for this word
                    candidate_edits = self._edit_distance_1(candidate)
                    
                    # Limit the number of candidates for this word
                    if len(candidate_edits) > 100:
                        candidate_edits = set(list(candidate_edits)[:100])
                    
                    # Add to the overall set
                    new_candidates.update(candidate_edits)
                
                candidates = new_candidates
                
                # Final limit on the total number of candidates
                if len(candidates) > self.max_edits:
                    candidates = set(list(candidates)[:self.max_edits])
            
            return candidates
