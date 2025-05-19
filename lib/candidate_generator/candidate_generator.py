"""
Candidate Generator for AAC input correction.

This module provides functionality to generate plausible candidates for a given
noisy input. It uses edit distance, keyboard adjacency, and lexicon-based filtering
to generate and rank candidates.

The main components are:
1. CandidateGenerator: Class for generating and ranking candidates
2. generate_candidates: Function to generate candidates for a noisy input
3. Various helper functions for edit distance, keyboard adjacency, etc.
"""

import os
import sys
import logging
from typing import List, Tuple, Set, Optional

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the keyboard layout model for adjacency calculations
from lib.noise_model.keyboard_error_model import KeyboardLayoutModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class CandidateGenerator:
    """
    Class for generating plausible candidates for a noisy input.

    This class uses edit distance, keyboard adjacency, and lexicon-based filtering
    to generate and rank candidates for a given noisy input.
    """

    def __init__(self, lexicon: Optional[Set[str]] = None, max_candidates: int = 10):
        """
        Initialize a candidate generator.

        Args:
            lexicon: Set of valid words to use for filtering candidates
            max_candidates: Maximum number of candidates to return
        """
        self.lexicon = lexicon if lexicon is not None else set()
        self.max_candidates = max_candidates
        self.keyboard_layout = KeyboardLayoutModel(layout_name="en")

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
            logger.info(
                f"Loaded lexicon with {len(self.lexicon)} words from {file_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading lexicon from {file_path}: {e}")
            return False

    def generate_candidates(
        self,
        noisy_input: str,
        max_edit_distance: int = 2,
        use_keyboard_adjacency: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Generate candidates for a noisy input.

        Args:
            noisy_input: The noisy input text
            max_edit_distance: Maximum edit distance to consider
            use_keyboard_adjacency: Whether to use keyboard adjacency for candidate generation

        Returns:
            List of (candidate, score) tuples, sorted by score (highest first)
        """
        # Convert to lowercase for case-insensitive comparison
        noisy_input = noisy_input.lower()

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

    def _get_edit_distance_candidates(self, word: str, max_distance: int) -> Set[str]:
        """
        Generate candidates within a certain edit distance.

        Args:
            word: The input word
            max_distance: Maximum edit distance to consider

        Returns:
            Set of candidate words
        """
        # For edit distance 1
        if max_distance == 1:
            return self._edit_distance_1(word)

        # For edit distance 2, apply edit distance 1 twice
        elif max_distance == 2:
            # Get all candidates at edit distance 1
            edits1 = self._edit_distance_1(word)

            # Apply edit distance 1 to each candidate to get edit distance 2
            edits2 = set()
            for edit1 in edits1:
                edits2.update(self._edit_distance_1(edit1))

            return edits2

        # For higher edit distances (not typically used)
        else:
            candidates = self._edit_distance_1(word)
            for _ in range(max_distance - 1):
                new_candidates = set()
                for candidate in candidates:
                    new_candidates.update(self._edit_distance_1(candidate))
                candidates = new_candidates

            return candidates

    def _edit_distance_1(self, word: str) -> Set[str]:
        """
        Generate all candidates with edit distance 1.

        This includes:
        - Deletions: Remove one character
        - Transpositions: Swap adjacent characters
        - Replacements: Replace one character with another
        - Insertions: Insert one character

        Args:
            word: The input word

        Returns:
            Set of candidate words with edit distance 1
        """
        letters = "abcdefghijklmnopqrstuvwxyz"
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]

        # Deletions
        deletes = [L + R[1:] for L, R in splits if R]

        # Transpositions
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]

        # Replacements
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]

        # Insertions
        inserts = [L + c + R for L, R in splits for c in letters]

        return set(deletes + transposes + replaces + inserts)

    def _get_keyboard_adjacency_candidates(self, word: str) -> Set[str]:
        """
        Generate candidates based on keyboard adjacency.

        Args:
            word: The input word

        Returns:
            Set of candidate words based on keyboard adjacency
        """
        candidates = set()

        # For each position in the word
        for i in range(len(word)):
            char = word[i]

            # Get adjacent keys on the keyboard
            adjacent_keys = self.keyboard_layout.get_adjacent_keys(char)

            # Replace the character with each adjacent key (convert to lowercase)
            for adj_key in adjacent_keys:
                candidate = word[:i] + adj_key.lower() + word[i + 1 :]
                candidates.add(candidate)

        return candidates

    def _filter_candidates(self, candidates: Set[str]) -> Set[str]:
        """
        Filter candidates to only include valid words.

        Args:
            candidates: Set of candidate words

        Returns:
            Set of filtered candidate words
        """
        # If no lexicon is provided, return all candidates
        if not self.lexicon:
            return candidates

        # Filter candidates to only include words in the lexicon
        return {candidate for candidate in candidates if candidate in self.lexicon}

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
                        score += 0.1

            scored_candidates.append((candidate, score))

        # Sort by score (highest first)
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein distance between two strings.

        Args:
            s1: First string
            s2: Second string

        Returns:
            The Levenshtein distance
        """
        # Create a matrix of size (len(s1)+1) x (len(s2)+1)
        dp = [[0 for _ in range(len(s2) + 1)] for _ in range(len(s1) + 1)]

        # Initialize the first row and column
        for i in range(len(s1) + 1):
            dp[i][0] = i
        for j in range(len(s2) + 1):
            dp[0][j] = j

        # Fill in the rest of the matrix
        for i in range(1, len(s1) + 1):
            for j in range(1, len(s2) + 1):
                cost = 0 if s1[i - 1] == s2[j - 1] else 1
                dp[i][j] = min(
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,  # Insertion
                    dp[i - 1][j - 1] + cost,  # Substitution
                )

                # Check for transposition
                if (
                    i > 1
                    and j > 1
                    and s1[i - 1] == s2[j - 2]
                    and s1[i - 2] == s2[j - 1]
                ):
                    dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + cost)

        return dp[len(s1)][len(s2)]


def generate_candidates(
    noisy_input: str, lexicon: Optional[Set[str]] = None, max_edit_distance: int = 2
) -> List[Tuple[str, float]]:
    """
    Generate candidates for a noisy input.

    This is a convenience function that creates a CandidateGenerator and uses it
    to generate candidates for a noisy input.

    Args:
        noisy_input: The noisy input text
        lexicon: Set of valid words to use for filtering candidates
        max_edit_distance: Maximum edit distance to consider

    Returns:
        List of (candidate, score) tuples, sorted by score (highest first)
    """
    generator = CandidateGenerator(lexicon=lexicon)
    return generator.generate_candidates(
        noisy_input, max_edit_distance=max_edit_distance
    )
