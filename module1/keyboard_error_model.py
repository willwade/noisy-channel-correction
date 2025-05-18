"""
Keyboard layout-based error model for AAC data augmentation.

This module provides classes and functions to simulate typing errors based on
keyboard layouts in different languages. It supports various error types:

1. Proximity errors: Typing a key adjacent to the intended key
2. Missed hits: Missing a key entirely (deletion)
3. Double hits: Typing a key twice (insertion)
4. Transpositions: Swapping adjacent characters

The module is designed to be configurable for different error rates and
keyboard layouts, supporting multiple languages and input methods.
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
import math
import os
import sys

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import keyboard layouts from language_keyboards.py
from module1.language_keyboards import (
    get_keyboard_layout,
    KEYBOARD_LAYOUTS,
    LANGUAGE_NAMES,
)

# Error types
ERROR_TYPES = {
    "proximity": "Typing a key adjacent to the intended key",
    "deletion": "Missing a key entirely",
    "insertion": "Typing a key twice or inserting a random key",
    "transposition": "Swapping adjacent characters",
}


class KeyboardLayoutModel:
    """
    A model of a physical keyboard layout that can calculate adjacency between keys.

    This class represents a keyboard layout as a 2D grid and provides methods to
    find adjacent keys, calculate distances between keys, and simulate typing errors
    based on physical proximity.
    """

    def __init__(self, layout_name: str = "en", rows: int = 3, cols: int = 10):
        """
        Initialize a keyboard layout model.

        Args:
            layout_name: Language code for the keyboard layout (e.g., 'en', 'fr', 'de')
            rows: Number of rows in the keyboard grid
            cols: Number of columns in the keyboard grid
        """
        self.layout_name = layout_name
        self.rows = rows
        self.cols = cols

        # Get the keyboard layout
        self.keys = get_keyboard_layout(layout_name)

        # Create a 2D grid representation of the keyboard
        self.grid = self._create_grid()

        # Create a mapping from keys to their positions
        self.key_positions = self._create_key_positions()

        # Create an adjacency map for each key
        self.adjacency_map = self._create_adjacency_map()

    def _create_grid(self) -> np.ndarray:
        """
        Create a 2D grid representation of the keyboard.

        Returns:
            A 2D numpy array representing the keyboard layout
        """
        # Ensure we have enough keys to fill the grid
        grid_size = self.rows * self.cols
        if len(self.keys) < grid_size:
            # Pad with empty strings if needed
            self.keys = self.keys + [""] * (grid_size - len(self.keys))
        elif len(self.keys) > grid_size:
            # Truncate if too many keys
            self.keys = self.keys[:grid_size]

        # Reshape into a 2D grid
        return np.array(self.keys).reshape(self.rows, self.cols)

    def _create_key_positions(self) -> Dict[str, Tuple[int, int]]:
        """
        Create a mapping from keys to their positions in the grid.

        Returns:
            A dictionary mapping each key to its (row, col) position
        """
        positions = {}
        for row in range(self.rows):
            for col in range(self.cols):
                key = self.grid[row, col]
                if key:  # Only add non-empty keys
                    positions[key] = (row, col)
        return positions

    def _create_adjacency_map(self) -> Dict[str, List[str]]:
        """
        Create an adjacency map for each key.

        Returns:
            A dictionary mapping each key to a list of adjacent keys
        """
        adjacency = {}
        for key, (row, col) in self.key_positions.items():
            adjacent_keys = []

            # Check all 8 adjacent positions
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # Skip the key itself

                    new_row, new_col = row + dr, col + dc
                    if 0 <= new_row < self.rows and 0 <= new_col < self.cols:
                        adjacent_key = self.grid[new_row, new_col]
                        if adjacent_key:  # Only add non-empty keys
                            adjacent_keys.append(adjacent_key)

            adjacency[key] = adjacent_keys

        return adjacency

    def get_adjacent_keys(self, key: str) -> List[str]:
        """
        Get all keys adjacent to the given key.

        Args:
            key: The key to find adjacents for

        Returns:
            A list of adjacent keys
        """
        key = key.upper()  # Normalize to uppercase
        return self.adjacency_map.get(key, [])

    def get_key_distance(self, key1: str, key2: str) -> float:
        """
        Calculate the Euclidean distance between two keys on the keyboard.

        Args:
            key1: First key
            key2: Second key

        Returns:
            The Euclidean distance between the keys, or float('inf') if either key is not found
        """
        key1, key2 = key1.upper(), key2.upper()  # Normalize to uppercase

        if key1 not in self.key_positions or key2 not in self.key_positions:
            return float("inf")

        pos1 = self.key_positions[key1]
        pos2 = self.key_positions[key2]

        # Calculate Euclidean distance
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def get_closest_keys(self, key: str, n: int = 3) -> List[str]:
        """
        Get the n closest keys to the given key.

        Args:
            key: The key to find closest keys for
            n: Number of closest keys to return

        Returns:
            A list of the n closest keys
        """
        key = key.upper()  # Normalize to uppercase

        if key not in self.key_positions:
            return []

        # Calculate distances to all other keys
        distances = [
            (other_key, self.get_key_distance(key, other_key))
            for other_key in self.key_positions
            if other_key != key
        ]

        # Sort by distance and return the n closest
        closest = sorted(distances, key=lambda x: x[1])[:n]
        return [k for k, _ in closest]


class KeyboardErrorModel:
    """
    A model for simulating typing errors based on keyboard layouts.

    This class provides methods to introduce realistic typing errors into text,
    based on the physical properties of keyboard layouts and different input methods.
    """

    def __init__(
        self,
        layout_name: str = "en",
        error_rates: Dict[str, float] = None,
        rows: int = 3,
        cols: int = 10,
        input_method: str = "direct",
    ):
        """
        Initialize a keyboard error model.

        Args:
            layout_name: Language code for the keyboard layout (e.g., 'en', 'fr', 'de')
            error_rates: Dictionary of error rates for different error types
                         (proximity, deletion, insertion, transposition)
            rows: Number of rows in the keyboard grid
            cols: Number of columns in the keyboard grid
            input_method: Input method ('direct', 'scanning', 'eyegaze')
        """
        self.layout_name = layout_name
        self.rows = rows
        self.cols = cols
        self.input_method = input_method

        # Set default error rates if not provided
        self.error_rates = {
            "proximity": 0.05,  # 5% chance of hitting an adjacent key
            "deletion": 0.03,  # 3% chance of missing a key
            "insertion": 0.02,  # 2% chance of hitting a key twice
            "transposition": 0.01,  # 1% chance of swapping adjacent characters
        }

        # Update with user-provided error rates
        if error_rates:
            self.error_rates.update(error_rates)

        # Initialize the keyboard layout model
        self.keyboard = KeyboardLayoutModel(layout_name, rows, cols)

        # Adjust error rates based on input method
        self._adjust_error_rates_for_input_method()

    def _adjust_error_rates_for_input_method(self):
        """
        Adjust error rates based on the input method.

        Different input methods have different error characteristics:
        - Direct access (e.g., typing with fingers): More proximity errors
        - Scanning (e.g., switch access): More timing errors (deletions/insertions)
        - Eyegaze: More precision errors (proximity)
        """
        if self.input_method == "scanning":
            # Scanning has more timing errors
            self.error_rates["deletion"] *= 2.0
            self.error_rates["insertion"] *= 1.5
            self.error_rates["proximity"] *= 0.5

        elif self.input_method == "eyegaze":
            # Eyegaze has more precision errors
            self.error_rates["proximity"] *= 2.0
            self.error_rates["deletion"] *= 1.2
            self.error_rates["insertion"] *= 0.8

    def apply_errors(self, text: str) -> str:
        """
        Apply random typing errors to the input text based on the error model.

        Args:
            text: The input text to add errors to

        Returns:
            The text with simulated typing errors
        """
        if not text:
            return text

        # Store the original case of each character
        original_case = [c.isupper() for c in text]

        # Convert to uppercase for processing (to match keyboard layout)
        text = text.upper()
        result = list(text)
        i = 0

        while i < len(result):
            # Determine if we should introduce an error at this position
            error_type = self._select_error_type()

            if error_type == "proximity" and i < len(result):
                # Replace with an adjacent key
                result[i] = self._apply_proximity_error(result[i])
                i += 1

            elif error_type == "deletion" and i < len(result):
                # Delete the current character
                result.pop(i)
                # Don't increment i since we removed a character

            elif error_type == "insertion" and i < len(result):
                # Insert a duplicate or random character
                char_to_insert = self._apply_insertion_error(result[i])
                result.insert(i, char_to_insert)
                i += 2  # Skip over both the original and inserted character

            elif error_type == "transposition" and i < len(result) - 1:
                # Swap with the next character
                result[i], result[i + 1] = result[i + 1], result[i]
                i += 2  # Skip over both transposed characters

            else:
                # No error, move to the next character
                i += 1

        # Convert result back to the original case
        final_result = []
        for i, char in enumerate(result):
            if i < len(original_case) and not original_case[i]:
                final_result.append(char.lower())
            else:
                final_result.append(char)

        return "".join(final_result)

    def _select_error_type(self) -> Optional[str]:
        """
        Randomly select an error type based on the configured error rates.

        Returns:
            The selected error type, or None if no error should be applied
        """
        # Calculate the total error probability
        total_error_prob = sum(self.error_rates.values())

        # Generate a random number
        r = random.random()

        # No error case
        if r >= total_error_prob:
            return None

        # Normalize the random number to the total error probability
        r /= total_error_prob

        # Select an error type based on the normalized random number
        cumulative_prob = 0
        for error_type, prob in self.error_rates.items():
            cumulative_prob += prob / total_error_prob
            if r < cumulative_prob:
                return error_type

        # Fallback (should not reach here)
        return None

    def _apply_proximity_error(self, char: str) -> str:
        """
        Apply a proximity error by replacing with an adjacent key.

        Args:
            char: The character to replace

        Returns:
            An adjacent character on the keyboard
        """
        adjacent_keys = self.keyboard.get_adjacent_keys(char)

        if not adjacent_keys:
            return char  # No adjacent keys, keep the original

        return random.choice(adjacent_keys)

    def _apply_insertion_error(self, char: str) -> str:
        """
        Apply an insertion error by duplicating or inserting a random character.

        Args:
            char: The character that triggered the insertion

        Returns:
            The character to insert
        """
        # 80% chance to duplicate the character, 20% chance for a random adjacent key
        if random.random() < 0.8:
            return char  # Duplicate
        else:
            adjacent_keys = self.keyboard.get_adjacent_keys(char)
            if adjacent_keys:
                return random.choice(adjacent_keys)
            else:
                return char  # Fallback to duplication
