"""
Keyboard-Specific Confusion Matrix Builder for AAC input error modeling.

This module provides functionality to build keyboard-specific confusion matrices
for different keyboard layouts (QWERTY, ABC, frequency-based). These matrices
represent the probability of observing a noisy character given an intended character,
taking into account the physical layout of the keyboard.

The main components are:
1. KeyboardConfusionMatrix: Class for storing and managing keyboard-specific confusion matrices
2. build_keyboard_confusion_matrices: Function to build confusion matrices for different layouts
3. get_keyboard_error_probability: Function to get the probability of an error for a specific layout
"""

import os
import sys
import json
import logging
import math
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Set, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the confusion matrix
from lib.confusion_matrix.confusion_matrix import ConfusionMatrix, EPSILON, PHI

# Import keyboard layouts
from lib.noise_model.language_keyboards import (
    get_keyboard_layout,
    KEYBOARD_LAYOUTS,
    LANGUAGE_NAMES,
)

# Import keyboard layout model
from lib.noise_model.keyboard_error_model import KeyboardLayoutModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KeyboardConfusionMatrix:
    """
    Class for storing and managing keyboard-specific confusion matrices.

    This class extends the ConfusionMatrix class to support multiple keyboard layouts
    and provides methods to build and use keyboard-specific confusion matrices.
    """

    def __init__(self):
        """Initialize an empty keyboard confusion matrix collection."""
        # Dictionary of confusion matrices for different layouts
        self.matrices = {
            "qwerty": ConfusionMatrix(),
            "abc": ConfusionMatrix(),
            "frequency": ConfusionMatrix(),
        }

        # Keyboard layout models for different layouts
        self.keyboard_models = {
            "qwerty": KeyboardLayoutModel(layout_name="en"),
            "abc": KeyboardLayoutModel(layout_name="abc"),
            "frequency": KeyboardLayoutModel(layout_name="frequency"),
        }

        # Statistics about different error types for each layout
        self.stats = {
            "qwerty": {
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "transpositions": 0,
                "correct": 0,
                "total": 0,
            },
            "abc": {
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "transpositions": 0,
                "correct": 0,
                "total": 0,
            },
            "frequency": {
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
                "transpositions": 0,
                "correct": 0,
                "total": 0,
            },
        }

    def add_pair(self, intended: str, noisy: str, layout: str = "qwerty") -> None:
        """
        Add a pair of (intended, noisy) characters to the confusion matrix for a specific layout.

        Args:
            intended: The intended character
            noisy: The observed noisy character
            layout: The keyboard layout to use ('qwerty', 'abc', 'frequency')
        """
        if layout not in self.matrices:
            logger.warning(f"Unknown layout '{layout}'. Using 'qwerty' instead.")
            layout = "qwerty"

        self.matrices[layout].add_pair(intended, noisy)

        # Update statistics
        if intended == noisy:
            self.stats[layout]["correct"] += 1
        else:
            self.stats[layout]["substitutions"] += 1

        self.stats[layout]["total"] += 1

    def add_deletion(self, intended: str, layout: str = "qwerty") -> None:
        """
        Add a deletion error to the confusion matrix for a specific layout.

        Args:
            intended: The intended character that was deleted
            layout: The keyboard layout to use ('qwerty', 'abc', 'frequency')
        """
        if layout not in self.matrices:
            logger.warning(f"Unknown layout '{layout}'. Using 'qwerty' instead.")
            layout = "qwerty"

        self.matrices[layout].add_deletion(intended)

        # Update statistics
        self.stats[layout]["deletions"] += 1
        self.stats[layout]["total"] += 1

    def add_insertion(self, noisy: str, layout: str = "qwerty") -> None:
        """
        Add an insertion error to the confusion matrix for a specific layout.

        Args:
            noisy: The noisy character that was inserted
            layout: The keyboard layout to use ('qwerty', 'abc', 'frequency')
        """
        if layout not in self.matrices:
            logger.warning(f"Unknown layout '{layout}'. Using 'qwerty' instead.")
            layout = "qwerty"

        self.matrices[layout].add_insertion(noisy)

        # Update statistics
        self.stats[layout]["insertions"] += 1
        self.stats[layout]["total"] += 1

    def add_transposition(
        self, intended1: str, intended2: str, layout: str = "qwerty"
    ) -> None:
        """
        Add a transposition error to the confusion matrix for a specific layout.

        Args:
            intended1: The first intended character
            intended2: The second intended character
            layout: The keyboard layout to use ('qwerty', 'abc', 'frequency')
        """
        if layout not in self.matrices:
            logger.warning(f"Unknown layout '{layout}'. Using 'qwerty' instead.")
            layout = "qwerty"

        self.matrices[layout].add_transposition(intended1, intended2)

        # Update statistics
        self.stats[layout]["transpositions"] += 1
        self.stats[layout]["total"] += 1

    def normalize(self, layout: str = None) -> None:
        """
        Normalize the confusion matrix probabilities for one or all layouts.

        Args:
            layout: The keyboard layout to normalize, or None to normalize all layouts
        """
        if layout is None:
            # Normalize all layouts
            for layout_name in self.matrices:
                self.matrices[layout_name].normalize()
        elif layout in self.matrices:
            # Normalize the specified layout
            self.matrices[layout].normalize()
        else:
            logger.warning(f"Unknown layout '{layout}'. Cannot normalize.")

    def get_probability(
        self, intended: str, noisy: str, layout: str = "qwerty"
    ) -> float:
        """
        Get the probability of observing a noisy character given an intended character.

        Args:
            intended: The intended character
            noisy: The observed noisy character
            layout: The keyboard layout to use ('qwerty', 'abc', 'frequency')

        Returns:
            The probability P(noisy | intended) for the specified layout
        """
        if layout not in self.matrices:
            logger.warning(f"Unknown layout '{layout}'. Using 'qwerty' instead.")
            layout = "qwerty"

        return self.matrices[layout].get_probability(intended, noisy)

    def save(self, file_path: str) -> bool:
        """
        Save the keyboard confusion matrices to a JSON file.

        Args:
            file_path: Path to the output JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert each matrix to a serializable format
            data = {}
            for layout, matrix in self.matrices.items():
                data[layout] = {
                    "counts": {k: dict(v) for k, v in matrix.counts.items()},
                    "probabilities": {
                        k: dict(v) for k, v in matrix.probabilities.items()
                    },
                    "vocabulary": list(matrix.vocabulary),
                    "total_pairs": matrix.total_pairs,
                    "stats": self.stats[layout],
                }

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)

            # Write to JSON file
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.info(f"Saved keyboard confusion matrices to {file_path}")
            return True
        except Exception as e:
            logger.error(
                f"Error saving keyboard confusion matrices to {file_path}: {e}"
            )
            return False

    @classmethod
    def load(cls, file_path: str) -> "KeyboardConfusionMatrix":
        """
        Load keyboard confusion matrices from a JSON file.

        Args:
            file_path: Path to the input JSON file

        Returns:
            A KeyboardConfusionMatrix object
        """
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            keyboard_matrix = cls()

            # Load each layout's matrix
            for layout, matrix_data in data.items():
                # Create a new confusion matrix for this layout
                matrix = ConfusionMatrix()

                # Load counts
                for intended, noisy_dict in matrix_data["counts"].items():
                    for noisy, count in noisy_dict.items():
                        matrix.counts[intended][noisy] = count

                # Load probabilities
                for intended, noisy_dict in matrix_data["probabilities"].items():
                    for noisy, prob in noisy_dict.items():
                        matrix.probabilities[intended][noisy] = prob

                # Load other attributes
                matrix.vocabulary = set(matrix_data["vocabulary"])
                matrix.total_pairs = matrix_data["total_pairs"]

                # Add to the keyboard matrix collection
                keyboard_matrix.matrices[layout] = matrix

                # Load statistics if available
                if "stats" in matrix_data:
                    keyboard_matrix.stats[layout] = matrix_data["stats"]

            return keyboard_matrix

        except Exception as e:
            logger.error(
                f"Error loading keyboard confusion matrices from {file_path}: {e}"
            )
            return cls()  # Return an empty matrix collection

    def get_stats(self, layout: str = None) -> Dict[str, Any]:
        """
        Get statistics about the confusion matrix for one or all layouts.

        Args:
            layout: The keyboard layout to get statistics for, or None to get all layouts

        Returns:
            A dictionary of statistics
        """
        if layout is None:
            # Return stats for all layouts
            return self.stats
        elif layout in self.stats:
            # Return stats for the specified layout
            return self.stats[layout]
        else:
            logger.warning(f"Unknown layout '{layout}'. Cannot get statistics.")
            return {}

    def __str__(self) -> str:
        """Return a string representation of the keyboard confusion matrices."""
        lines = ["Keyboard Confusion Matrices:"]

        for layout, matrix in self.matrices.items():
            lines.append(f"\n=== {layout.upper()} Layout ===")
            lines.append(str(matrix))

        return "\n".join(lines)


def build_keyboard_confusion_matrices(
    pairs: List[Tuple[str, str]],
) -> KeyboardConfusionMatrix:
    """
    Build keyboard-specific confusion matrices from pairs of (intended, noisy) text.

    Args:
        pairs: List of (intended, noisy) text pairs

    Returns:
        A KeyboardConfusionMatrix object with matrices for different layouts
    """
    # Create a new keyboard confusion matrix
    keyboard_matrix = KeyboardConfusionMatrix()

    # Create keyboard layout models for different layouts
    qwerty_model = KeyboardLayoutModel(layout_name="en")
    abc_model = KeyboardLayoutModel(layout_name="abc")
    frequency_model = KeyboardLayoutModel(layout_name="frequency")

    # Store the models in the keyboard matrix
    keyboard_matrix.keyboard_models = {
        "qwerty": qwerty_model,
        "abc": abc_model,
        "frequency": frequency_model,
    }

    # Process each pair
    for intended, noisy in pairs:
        # Align the strings to identify substitutions, insertions, and deletions
        alignment = align_strings(intended, noisy)

        # Process each aligned pair
        for i, (intended_char, noisy_char) in enumerate(alignment):
            # Check for different error types
            if intended_char == "" and noisy_char != "":
                # Insertion
                for layout in keyboard_matrix.matrices:
                    keyboard_matrix.add_insertion(noisy_char, layout)
            elif intended_char != "" and noisy_char == "":
                # Deletion
                for layout in keyboard_matrix.matrices:
                    keyboard_matrix.add_deletion(intended_char, layout)
            elif intended_char != noisy_char:
                # Substitution
                for layout in keyboard_matrix.matrices:
                    keyboard_matrix.add_pair(intended_char, noisy_char, layout)
            else:
                # Correct character
                for layout in keyboard_matrix.matrices:
                    keyboard_matrix.add_pair(intended_char, noisy_char, layout)

        # Check for transpositions
        for i in range(len(intended) - 1):
            if (
                i < len(noisy) - 1
                and intended[i] == noisy[i + 1]
                and intended[i + 1] == noisy[i]
            ):
                # Transposition
                for layout in keyboard_matrix.matrices:
                    keyboard_matrix.add_transposition(
                        intended[i], intended[i + 1], layout
                    )

    # Normalize the matrices
    keyboard_matrix.normalize()

    return keyboard_matrix


def align_strings(s1: str, s2: str) -> List[Tuple[str, str]]:
    """
    Align two strings to identify substitutions, insertions, and deletions.

    Args:
        s1: First string (intended)
        s2: Second string (noisy)

    Returns:
        List of aligned character pairs
    """
    # Simple implementation using dynamic programming
    # For a more sophisticated alignment, consider using the Needleman-Wunsch algorithm

    # Initialize the alignment matrix
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Fill the first row and column
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the rest of the matrix
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(
                    dp[i - 1][j - 1] + 1,  # Substitution
                    dp[i - 1][j] + 1,  # Deletion
                    dp[i][j - 1] + 1,
                )  # Insertion

    # Backtrack to find the alignment
    alignment = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0 and s1[i - 1] == s2[j - 1]:
            # Match
            alignment.append((s1[i - 1], s2[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            # Substitution
            alignment.append((s1[i - 1], s2[j - 1]))
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            # Deletion
            alignment.append((s1[i - 1], ""))
            i -= 1
        else:
            # Insertion
            alignment.append(("", s2[j - 1]))
            j -= 1

    # Reverse the alignment
    alignment.reverse()
    return alignment


def get_keyboard_error_probability(
    noisy: str, intended: str, matrix: KeyboardConfusionMatrix, layout: str = "qwerty"
) -> float:
    """
    Get the probability of observing a noisy string given an intended string.

    Args:
        noisy: The observed noisy string
        intended: The intended string
        matrix: The keyboard confusion matrix
        layout: The keyboard layout to use ('qwerty', 'abc', 'frequency')

    Returns:
        The probability P(noisy | intended) for the specified layout
    """
    if layout not in matrix.matrices:
        logger.warning(f"Unknown layout '{layout}'. Using 'qwerty' instead.")
        layout = "qwerty"

    # Align the strings
    alignment = align_strings(intended, noisy)

    # Calculate the probability as the product of individual character probabilities
    log_prob = 0.0
    for intended_char, noisy_char in alignment:
        if intended_char == "" and noisy_char != "":
            # Insertion
            prob = matrix.matrices[layout].get_probability(PHI, noisy_char)
        elif intended_char != "" and noisy_char == "":
            # Deletion
            prob = matrix.matrices[layout].get_probability(intended_char, EPSILON)
        else:
            # Substitution or correct character
            prob = matrix.matrices[layout].get_probability(intended_char, noisy_char)

        # Use log probabilities to avoid underflow
        log_prob += math.log(prob) if prob > 0 else -float("inf")

    # Convert back to probability
    return math.exp(log_prob) if log_prob != -float("inf") else 0.0
