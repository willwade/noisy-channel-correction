"""
Improved Candidate Generator for AAC input correction.

This module provides an improved version of the CandidateGenerator class
that combines all the best features from the optimized, enhanced, and advanced versions.
"""

import os
import sys
import logging
from typing import List, Tuple, Set, Optional, Generator
import itertools

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the candidate generator
from lib.candidate_generator.candidate_generator import CandidateGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ImprovedCandidateGenerator(CandidateGenerator):
    """
    Improved version of the CandidateGenerator class.

    This class combines the best features from all previous versions:
    - Efficient candidate generation with smart limits
    - Improved ranking algorithm with keyboard adjacency boost
    - Smart filtering based on multiple factors
    - Support for multi-word correction
    - Support for word frequency information
    """

    def __init__(
        self,
        lexicon: Optional[Set[str]] = None,
        max_candidates: int = 30,
        max_edits: int = 200,  # Reduced from 20000 to 200
        keyboard_boost: float = 0.3,
        strict_filtering: bool = True,
        smart_filtering: bool = True,
        use_frequency_info: bool = True,
    ):
        """
        Initialize an improved candidate generator.

        Args:
            lexicon: Set of valid words to use for filtering candidates
            max_candidates: Maximum number of candidates to return
            max_edits: Maximum number of edit candidates to generate (default: 200)
            keyboard_boost: Boost factor for keyboard-adjacent substitutions
            strict_filtering: Whether to strictly filter candidates by lexicon
            smart_filtering: Whether to use smart filtering for candidates
            use_frequency_info: Whether to use word frequency information
        """
        super().__init__(lexicon, max_candidates)
        self.max_edits = max_edits
        self.keyboard_boost = keyboard_boost
        self.strict_filtering = strict_filtering
        self.smart_filtering = smart_filtering
        self.use_frequency_info = use_frequency_info
        self.word_frequencies = {}  # Word frequency dictionary (populated if available)

    def load_word_frequencies(self, file_path: str) -> bool:
        """
        Load word frequencies from a file.

        Args:
            file_path: Path to the word frequencies file (format: word frequency)

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        word = parts[0].lower()
                        freq = float(parts[1])
                        self.word_frequencies[word] = freq
            logger.info(f"Loaded {len(self.word_frequencies)} word frequencies")
            return True
        except Exception as e:
            logger.error(f"Error loading word frequencies: {e}")
            return False

    def _get_edit_distance_candidates(self, word: str, max_distance: int) -> Set[str]:
        """
        Generate candidates within a certain edit distance, with a smarter approach.

        Args:
            word: The input word
            max_distance: Maximum edit distance to consider

        Returns:
            Set of candidate words
        """
        # For edit distance 1, use the parent implementation but with higher limits
        if max_distance == 1:
            edits1 = self._edit_distance_1(word)
            # Only log a warning if we're significantly over the limit
            if len(edits1) > self.max_edits * 1.5:
                logger.warning(
                    f"Large number of edit distance 1 candidates: {len(edits1)}"
                )
            return edits1

        # For edit distance 2, use a more efficient approach
        elif max_distance == 2:
            # Get all candidates at edit distance 1
            edits1 = self._edit_distance_1(word)

            # Instead of limiting by count, prioritize by likelihood
            # For longer words, we can be more selective with edit distance 1 candidates
            if len(word) > 5 and len(edits1) > self.max_edits // 2:
                # Prioritize candidates that preserve the first and last letters
                # This is based on research showing these are less likely to be mistyped
                edits1_prioritized = [
                    e
                    for e in edits1
                    if (
                        len(e) > 1
                        and len(word) > 1
                        and e[0] == word[0]
                        and e[-1] == word[-1]
                    )
                ]

                # If we have enough prioritized candidates, use those
                if len(edits1_prioritized) >= self.max_edits // 4:
                    edits1 = set(edits1_prioritized)
                    # Add some non-prioritized candidates to ensure diversity
                    non_prioritized = list(set(edits1) - set(edits1_prioritized))
                    edits1.update(non_prioritized[: self.max_edits // 8])

            # Apply edit distance 1 to each candidate to get edit distance 2
            edits2 = set()

            # Process candidates in batches to avoid memory issues
            batch_size = min(1000, len(edits1))
            for i in range(0, len(edits1), batch_size):
                batch = list(edits1)[i : i + batch_size]

                for edit1 in batch:
                    # Get edit distance 1 candidates for this word
                    edit1_candidates = self._edit_distance_1(edit1)

                    # Add to the overall set
                    edits2.update(edit1_candidates)

                    # If we've generated enough candidates, stop early
                    if len(edits2) > self.max_edits:
                        logger.info(
                            f"Early stopping: Generated sufficient candidates ({len(edits2)})"
                        )
                        break

            return edits2

        # For higher edit distances, use a more selective approach
        else:
            candidates = self._edit_distance_1(word)
            for _ in range(max_distance - 1):
                new_candidates = set()

                # Process in batches
                batch_size = min(1000, len(candidates))
                for i in range(0, len(candidates), batch_size):
                    batch = list(candidates)[i : i + batch_size]

                    for candidate in batch:
                        # Get edit distance 1 candidates for this word
                        candidate_edits = self._edit_distance_1(candidate)

                        # Add to the overall set
                        new_candidates.update(candidate_edits)

                        # If we've generated enough candidates, stop early
                        if len(new_candidates) > self.max_edits:
                            logger.info(f"Early stopping at edit distance {_+2}")
                            break

                candidates = new_candidates

            return candidates

    def _filter_candidates(self, candidates: Set[str]) -> Set[str]:
        """
        Filter candidates using a smarter approach.

        Args:
            candidates: Set of candidate words

        Returns:
            Set of filtered candidate words
        """
        # If no lexicon is provided, return all candidates
        if not self.lexicon:
            return candidates

        # If smart filtering is disabled, use the parent implementation
        if not self.smart_filtering:
            return super()._filter_candidates(candidates)

        # Smart filtering: prioritize candidates based on multiple factors
        filtered_candidates = set()
        candidate_scores = {}

        # First, include all candidates that are in the lexicon
        for candidate in candidates:
            if candidate in self.lexicon:
                # Calculate a base score for the candidate
                score = 1.0

                # Boost score based on word frequency if available
                if self.use_frequency_info and candidate in self.word_frequencies:
                    # Normalize frequency to a value between 0 and 1
                    freq = self.word_frequencies[candidate]
                    max_freq = (
                        max(self.word_frequencies.values())
                        if self.word_frequencies
                        else 1.0
                    )
                    freq_score = min(freq / max_freq, 1.0)
                    score += freq_score

                candidate_scores[candidate] = score
                filtered_candidates.add(candidate)

        # If we have enough candidates, return the top ones
        if len(filtered_candidates) >= self.max_candidates:
            # Sort by score and return the top candidates
            sorted_candidates = sorted(
                candidate_scores.items(), key=lambda x: x[1], reverse=True
            )
            return set(c for c, _ in sorted_candidates[: self.max_candidates])

        # If strict filtering is enabled and we have some candidates, return them
        if self.strict_filtering and filtered_candidates:
            return filtered_candidates

        # Otherwise, include candidates that are close to words in the lexicon
        # This helps with very noisy inputs that might not be in the lexicon
        for candidate in candidates:
            if candidate not in filtered_candidates:
                # Find the closest words in the lexicon
                closest_words = []
                for word in self.lexicon:
                    distance = self._levenshtein_distance(candidate, word)
                    if distance <= 2:  # Only consider words within edit distance 2
                        closest_words.append((word, distance))

                # Sort by distance (closest first)
                closest_words.sort(key=lambda x: x[1])

                # If we found close matches, add the candidate with a score
                if closest_words:
                    closest_word, distance = closest_words[0]
                    # Score based on distance to the closest word
                    score = 1.0 - (distance / 5.0)  # Normalize to 0-1 range

                    # Boost score based on word frequency if available
                    if (
                        self.use_frequency_info
                        and closest_word in self.word_frequencies
                    ):
                        freq = self.word_frequencies[closest_word]
                        max_freq = (
                            max(self.word_frequencies.values())
                            if self.word_frequencies
                            else 1.0
                        )
                        freq_score = min(freq / max_freq, 1.0)
                        score += (
                            freq_score * 0.5
                        )  # Half weight for frequency of similar word

                    candidate_scores[candidate] = score
                    filtered_candidates.add(candidate)

        # Return the filtered candidates
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
            use_keyboard_adjacency: Whether to use keyboard adjacency for generation

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

        # For single-word inputs, use the parent class implementation with our overrides
        # If the input is already in the lexicon, return it with a perfect score
        if noisy_input in self.lexicon:
            return [(noisy_input, 1.0)]

        candidates = set()

        # Generate candidates based on edit distance
        for distance in range(1, max_edit_distance + 1):
            edit_candidates = self._get_edit_distance_candidates(noisy_input, distance)
            candidates.update(edit_candidates)

        # Generate candidates based on keyboard adjacency
        if use_keyboard_adjacency:
            adjacency_candidates = self._get_keyboard_adjacency_candidates(noisy_input)
            candidates.update(adjacency_candidates)

        # Filter candidates to only include valid words
        filtered_candidates = self._filter_candidates(candidates)

        # Rank candidates by likelihood
        ranked_candidates = self._rank_candidates(filtered_candidates, noisy_input)

        # Return the top N candidates
        return ranked_candidates[: self.max_candidates]

    def _handle_multi_word_input(
        self,
        words: List[str],
        max_edit_distance: int = 2,
        use_keyboard_adjacency: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Handle multi-word inputs with improved context awareness.

        Args:
            words: List of words in the input
            max_edit_distance: Maximum edit distance to consider
            use_keyboard_adjacency: Whether to use keyboard adjacency

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
            candidates = self.generate_candidates(
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

        # Try more combinations for shorter sentences
        if len(words) <= 5:
            # Generate combinations using the top N candidates for each word
            # where N depends on the position in the sentence
            max_candidates_per_position = [
                min(3, len(corrections)) for corrections in word_corrections
            ]

            # Generate a limited number of combinations to avoid combinatorial explosion
            max_combinations = 20  # Limit to 20 combinations

            # Generate combinations
            combinations_count = 0
            for combo in self._generate_combinations(
                word_corrections, max_candidates_per_position
            ):
                if combinations_count >= max_combinations:
                    break

                combo_sentence = " ".join(word for word, _ in combo)
                combo_score = sum(score for _, score in combo) / len(combo)

                # Only add if it's different from what we already have
                if combo_sentence != top_sentence:
                    sentence_candidates.append((combo_sentence, combo_score))
                    combinations_count += 1
        else:
            # For longer sentences, just try variations with one word changed at a time
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

    def _generate_combinations(
        self,
        word_corrections: List[List[Tuple[str, float]]],
        max_candidates_per_position: List[int],
    ) -> Generator[List[Tuple[str, float]], None, None]:
        """
        Generate combinations of word corrections.

        Args:
            word_corrections: List of corrections for each word
            max_candidates_per_position: Max candidates to consider for each position

        Yields:
            Combinations of word corrections
        """
        # Limit the number of candidates per position
        limited_corrections = [
            corrections[:max_candidates]
            for corrections, max_candidates in zip(
                word_corrections, max_candidates_per_position
            )
        ]

        # Generate all combinations
        for combo in itertools.product(*limited_corrections):
            yield combo

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
                        score += self.keyboard_boost

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
