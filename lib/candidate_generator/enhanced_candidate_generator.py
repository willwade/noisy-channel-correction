"""
Enhanced Candidate Generator for AAC input correction.

This module provides an enhanced version of the OptimizedCandidateGenerator class
that increases the number of candidates generated and improves the ranking algorithm.
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
from lib.candidate_generator.optimized_candidate_generator import (
    OptimizedCandidateGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedCandidateGenerator(OptimizedCandidateGenerator):
    """
    Enhanced version of the OptimizedCandidateGenerator class.

    This class increases the number of candidates generated and improves the ranking algorithm.
    It also adds support for multi-word inputs and more lenient filtering.
    """

    def __init__(
        self,
        lexicon: Optional[Set[str]] = None,
        max_candidates: int = 20,  # Increased from 10
        max_edits: int = 5000,  # Increased from 1000
        keyboard_boost: float = 0.2,  # Increased from 0.1
        strict_filtering: bool = True,  # Default to strict filtering to ensure only valid words are returned
    ):
        """
        Initialize an enhanced candidate generator.

        Args:
            lexicon: Set of valid words to use for filtering candidates
            max_candidates: Maximum number of candidates to return
            max_edits: Maximum number of edit candidates to generate
            keyboard_boost: Boost factor for keyboard-adjacent substitutions
            strict_filtering: Whether to strictly filter candidates by lexicon
        """
        super().__init__(lexicon, max_candidates, max_edits)
        self.keyboard_boost = keyboard_boost
        self.strict_filtering = strict_filtering

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
                logger.warning(
                    f"Limiting edit distance 1 candidates from {len(edits1)} to {self.max_edits}"
                )
                edits1 = set(list(edits1)[: self.max_edits])
            return edits1

        # For edit distance 2, apply edit distance 1 twice
        elif max_distance == 2:
            # Get all candidates at edit distance 1
            edits1 = self._edit_distance_1(word)

            # Limit the number of edit distance 1 candidates
            # Use a larger limit for edit distance 1 candidates
            ed1_limit = self.max_edits // 5  # Increased from 10
            if len(edits1) > ed1_limit:
                logger.warning(
                    f"Limiting edit distance 1 candidates from {len(edits1)} to {ed1_limit}"
                )
                edits1 = set(list(edits1)[:ed1_limit])

            # Apply edit distance 1 to each candidate to get edit distance 2
            edits2 = set()
            for edit1 in edits1:
                # Check if we've reached the limit
                if len(edits2) >= self.max_edits:
                    logger.warning(
                        f"Reached maximum number of edit distance 2 candidates ({self.max_edits})"
                    )
                    break

                # Get edit distance 1 candidates for this word
                edit1_candidates = self._edit_distance_1(edit1)

                # Limit the number of candidates for this word
                # Use a larger limit for each word's candidates
                word_limit = 200  # Increased from 100
                if len(edit1_candidates) > word_limit:
                    edit1_candidates = set(list(edit1_candidates)[:word_limit])

                # Add to the overall set
                edits2.update(edit1_candidates)

            # Final limit on the total number of candidates
            if len(edits2) > self.max_edits:
                logger.warning(
                    f"Limiting edit distance 2 candidates from {len(edits2)} to {self.max_edits}"
                )
                edits2 = set(list(edits2)[: self.max_edits])

            return edits2

        # For higher edit distances (not typically used)
        else:
            candidates = self._edit_distance_1(word)
            for _ in range(max_distance - 1):
                # Limit the number of candidates at each step
                # Use a larger limit for intermediate candidates
                step_limit = self.max_edits // 5  # Increased from 10
                if len(candidates) > step_limit:
                    candidates = set(list(candidates)[:step_limit])

                new_candidates = set()
                for candidate in candidates:
                    # Check if we've reached the limit
                    if len(new_candidates) >= self.max_edits:
                        break

                    # Get edit distance 1 candidates for this word
                    candidate_edits = self._edit_distance_1(candidate)

                    # Limit the number of candidates for this word
                    # Use a larger limit for each word's candidates
                    word_limit = 200  # Increased from 100
                    if len(candidate_edits) > word_limit:
                        candidate_edits = set(list(candidate_edits)[:word_limit])

                    # Add to the overall set
                    new_candidates.update(candidate_edits)

                candidates = new_candidates

                # Final limit on the total number of candidates
                if len(candidates) > self.max_edits:
                    candidates = set(list(candidates)[: self.max_edits])

            return candidates

    def _filter_candidates(self, candidates: Set[str]) -> Set[str]:
        """
        Filter candidates to only include valid words, with optional leniency.

        Args:
            candidates: Set of candidate words

        Returns:
            Set of filtered candidate words
        """
        # If no lexicon is provided, return all candidates
        if not self.lexicon:
            return candidates

        # If strict filtering is enabled, use the parent class implementation
        if self.strict_filtering:
            return super()._filter_candidates(candidates)

        # Lenient filtering: include candidates that are in the lexicon
        # but also include candidates that are close to words in the lexicon
        filtered_candidates = set()

        # First, include all candidates that are in the lexicon
        for candidate in candidates:
            if candidate in self.lexicon:
                filtered_candidates.add(candidate)

        # If we have enough candidates, return them
        if len(filtered_candidates) >= self.max_candidates:
            return filtered_candidates

        # Otherwise, include candidates that are close to words in the lexicon
        # This helps with very noisy inputs that might not be in the lexicon
        remaining_slots = self.max_candidates - len(filtered_candidates)
        close_candidates = []

        for candidate in candidates:
            if candidate not in filtered_candidates:
                # Find the closest word in the lexicon
                min_distance = float("inf")
                for word in self.lexicon:
                    distance = self._levenshtein_distance(candidate, word)
                    if distance < min_distance:
                        min_distance = distance

                # Add the candidate if it's close enough to a word in the lexicon
                if min_distance <= 3:  # Allow up to 3 edits from a valid word
                    close_candidates.append((candidate, min_distance))

        # Sort by distance (closest first) and add the top remaining_slots
        close_candidates.sort(key=lambda x: x[1])
        for candidate, _ in close_candidates[:remaining_slots]:
            filtered_candidates.add(candidate)

        # If we still don't have enough candidates, include some unfiltered ones
        if len(filtered_candidates) < self.max_candidates // 2:
            # Add some of the original candidates to ensure we have some options
            unfiltered = list(candidates - filtered_candidates)
            if unfiltered:
                # Take a sample of unfiltered candidates
                sample_size = min(
                    self.max_candidates - len(filtered_candidates), len(unfiltered)
                )
                filtered_candidates.update(unfiltered[:sample_size])

        return filtered_candidates

    def generate_candidates(
        self,
        noisy_input: str,
        max_edit_distance: int = 2,
        use_keyboard_adjacency: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Generate candidates for a noisy input, with support for multi-word inputs.

        Args:
            noisy_input: The noisy input text
            max_edit_distance: Maximum edit distance to consider
            use_keyboard_adjacency: Whether to use keyboard adjacency for candidate generation

        Returns:
            List of (candidate, score) tuples, sorted by score (highest first)
        """
        # Convert to lowercase for case-insensitive comparison
        noisy_input = noisy_input.lower()

        # Check if this is a multi-word input
        words = noisy_input.split()
        if len(words) > 1:
            return self._handle_multi_word_input(
                words, max_edit_distance, use_keyboard_adjacency
            )

        # For single-word inputs, use the parent class implementation
        return super().generate_candidates(
            noisy_input, max_edit_distance, use_keyboard_adjacency
        )

    def _handle_multi_word_input(
        self,
        words: List[str],
        max_edit_distance: int = 2,
        use_keyboard_adjacency: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Handle multi-word inputs by correcting each word separately.

        Args:
            words: List of words in the input
            max_edit_distance: Maximum edit distance to consider
            use_keyboard_adjacency: Whether to use keyboard adjacency for candidate generation

        Returns:
            List of (candidate, score) tuples, sorted by score (highest first)
        """
        # Correct each word separately
        word_corrections = []
        for word in words:
            # Skip very short words (likely not meaningful)
            if len(word) <= 1:
                word_corrections.append([(word, 1.0)])
                continue

            # Generate candidates for this word
            candidates = super().generate_candidates(
                word, max_edit_distance, use_keyboard_adjacency
            )

            # If no candidates were found, use the original word
            if not candidates:
                word_corrections.append([(word, 1.0)])
            else:
                word_corrections.append(candidates)

        # Combine the corrections to form complete sentence candidates
        sentence_candidates = []

        # Use the top candidate for each word to form the most likely sentence
        top_sentence = " ".join(corrections[0][0] for corrections in word_corrections)
        top_score = sum(corrections[0][1] for corrections in word_corrections) / len(
            word_corrections
        )
        sentence_candidates.append((top_sentence, top_score))

        # Try some variations using the top 2 candidates for each word
        # This is a simplified approach to avoid combinatorial explosion
        for i, corrections in enumerate(word_corrections):
            if len(corrections) > 1:
                # Create a variation using the second-best candidate for this word
                variation_words = [
                    corrections[1 if j == i else 0][0]
                    for j, _ in enumerate(word_corrections)
                ]
                variation_sentence = " ".join(variation_words)
                variation_score = sum(
                    corrections[1 if j == i else 0][1]
                    for j, corrections in enumerate(word_corrections)
                ) / len(word_corrections)
                sentence_candidates.append((variation_sentence, variation_score))

        # Sort by score (highest first) and return the top candidates
        sentence_candidates.sort(key=lambda x: x[1], reverse=True)
        return sentence_candidates[: self.max_candidates]

    def _rank_candidates(
        self, candidates: Set[str], noisy_input: str
    ) -> List[Tuple[str, float]]:
        """
        Rank candidates by likelihood.

        Args:
            candidates: Set of candidate words
            noisy_input: The original noisy input

        Returns:
            List of (candidate, score) tuples, sorted by score (highest first)
        """
        # Calculate edit distance for each candidate
        scored_candidates = []

        for candidate in candidates:
            # Calculate edit distance
            distance = self._levenshtein_distance(noisy_input, candidate)

            # Convert distance to a similarity score (higher is better)
            max_len = max(len(noisy_input), len(candidate))
            score = 1.0 - (distance / max_len if max_len > 0 else 0)

            # Boost score for keyboard-adjacent substitutions
            # This helps prioritize likely typos like 'k' -> 'c' over other edits
            if len(noisy_input) == len(candidate):
                # Count character positions that differ
                diff_positions = [
                    (i, noisy_input[i], candidate[i])
                    for i in range(len(noisy_input))
                    if noisy_input[i] != candidate[i]
                ]

                # Check if the differing characters are adjacent on the keyboard
                for _, n_char, c_char in diff_positions:
                    adjacent_keys = [
                        k.lower()
                        for k in self.keyboard_layout.get_adjacent_keys(n_char)
                    ]
                    if c_char in adjacent_keys:
                        # Boost the score for keyboard-adjacent substitutions
                        score += (
                            self.keyboard_boost
                        )  # Using the configurable boost factor

            # Boost score for candidates with similar length to the input
            # This helps prioritize candidates that are close in length to the input
            length_diff = abs(len(noisy_input) - len(candidate))
            length_penalty = length_diff / max(len(noisy_input), 1)
            score -= length_penalty * 0.1  # Small penalty for length differences

            # Boost score for candidates that preserve the first and last letters
            # This helps prioritize candidates that preserve the shape of the word
            if (
                len(noisy_input) > 1
                and len(candidate) > 1
                and noisy_input[0] == candidate[0]
                and noisy_input[-1] == candidate[-1]
            ):
                score += 0.1  # Boost for preserving first and last letters

            scored_candidates.append((candidate, score))

        # Sort by score (highest first)
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)
