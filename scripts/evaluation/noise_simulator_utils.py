"""
Utility functions for using the noise simulator in the evaluation module.

This module provides functions to add noise to text using the noise simulator models.
"""

import os
import sys
import logging
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import path utilities from config
from lib.config import resolve_path

# Import noise simulator models
from lib.noise_model.noise_model import (
    KeyboardNoiseModel,
    DwellNoiseModel,
    TranspositionNoiseModel,
    CompositeNoiseModel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Define noise levels and their corresponding error rates
NOISE_LEVELS = {
    "minimal": {
        "proximity": 0.02,
        "deletion": 0.01,
        "insertion": 0.01,
        "transposition": 0.005,
    },
    "light": {
        "proximity": 0.05,
        "deletion": 0.02,
        "insertion": 0.02,
        "transposition": 0.01,
    },
    "moderate": {
        "proximity": 0.1,
        "deletion": 0.03,
        "insertion": 0.03,
        "transposition": 0.02,
    },
    "severe": {
        "proximity": 0.15,
        "deletion": 0.05,
        "insertion": 0.05,
        "transposition": 0.03,
    },
}

# Define noise types and their corresponding keyboard layouts
NOISE_TYPES = {
    "qwerty": "en",  # English QWERTY keyboard
    "abc": "abc",  # ABC layout (alphabetical)
    "frequency": "frequency",  # Frequency-based layout
}


def create_noise_model(
    noise_type: str = "qwerty", noise_level: str = "moderate"
) -> CompositeNoiseModel:
    """
    Create a noise model with the specified type and level.

    Args:
        noise_type: Type of noise ('qwerty', 'abc', or 'frequency')
        noise_level: Level of noise ('minimal', 'light', 'moderate', or 'severe')

    Returns:
        A configured composite noise model
    """
    # Validate noise type and level
    if noise_type not in NOISE_TYPES:
        logger.warning(f"Invalid noise type: {noise_type}. Using 'qwerty'.")
        noise_type = "qwerty"

    if noise_level not in NOISE_LEVELS:
        logger.warning(f"Invalid noise level: {noise_level}. Using 'moderate'.")
        noise_level = "moderate"

    # Get the keyboard layout and error rates
    layout_name = NOISE_TYPES[noise_type]
    error_rates = NOISE_LEVELS[noise_level]

    # Create the keyboard noise model
    keyboard_model = KeyboardNoiseModel(
        layout_name=layout_name,
        error_rates={
            "proximity": error_rates["proximity"],
            "deletion": 0.0,  # We'll use the DwellNoiseModel for deletions
            "insertion": 0.0,  # We'll use the DwellNoiseModel for insertions
            "transposition": 0.0,  # We'll use the TranspositionNoiseModel for transpositions
        },
        input_method="direct",
    )

    # Create the dwell noise model
    dwell_model = DwellNoiseModel(
        deletion_rate=error_rates["deletion"],
        insertion_rate=error_rates["insertion"],
    )

    # Create the transposition noise model
    transposition_model = TranspositionNoiseModel(
        transposition_rate=error_rates["transposition"],
    )

    # Create the composite noise model
    return CompositeNoiseModel(
        models=[keyboard_model, dwell_model, transposition_model]
    )


def add_noise(
    text: str,
    noise_type: str = "qwerty",
    noise_level: str = "moderate",
) -> str:
    """
    Add noise to the input text using module1's noise models.

    Args:
        text: The clean input text
        noise_type: Type of noise ('qwerty', 'abc', or 'frequency')
        noise_level: Level of noise ('minimal', 'light', 'moderate', or 'severe')

    Returns:
        The noisy output text
    """
    # Create the noise model
    noise_model = create_noise_model(noise_type, noise_level)

    # Apply the noise model to the input text
    return noise_model.apply(text)


def generate_noisy_text(
    text: str,
    noise_type: str = "qwerty",
    noise_level: str = "moderate",
) -> str:
    """
    Generate a noisy version of the input text.
    This is a convenience function that wraps add_noise.

    Args:
        text: The clean input text
        noise_type: Type of noise ('qwerty', 'abc', or 'frequency')
        noise_level: Level of noise ('minimal', 'light', 'moderate', or 'severe')

    Returns:
        The noisy output text
    """
    return add_noise(text, noise_type, noise_level)


def generate_noisy_pairs(
    texts: List[str],
    noise_type: str = "qwerty",
    noise_level: str = "moderate",
    num_variants: int = 1,
) -> List[Dict[str, Any]]:
    """
    Generate pairs of (clean, noisy) text using module1's noise models.

    Args:
        texts: List of clean input texts
        noise_type: Type of noise ('qwerty', 'abc', or 'frequency')
        noise_level: Level of noise ('minimal', 'light', 'moderate', or 'severe')
        num_variants: Number of noisy variants to generate for each input

    Returns:
        List of dictionaries with clean and noisy text
    """
    # Create the noise model
    noise_model = create_noise_model(noise_type, noise_level)

    # Generate noisy pairs
    pairs = []
    for text in texts:
        for _ in range(num_variants):
            noisy = noise_model.apply(text)
            pairs.append(
                {
                    "intended": text,
                    "noisy": noisy,
                    "noise_type": noise_type,
                    "noise_level": noise_level,
                }
            )

    return pairs


def load_wordlist(file_path: str) -> List[str]:
    """
    Load a wordlist from a file.

    Args:
        file_path: Path to the wordlist file (can be relative to project root, data dir, or absolute)

    Returns:
        List of words
    """
    try:
        # Resolve the path to ensure it's correctly found
        resolved_path = resolve_path(file_path)
        logger.info(f"Resolving wordlist path: {file_path} -> {resolved_path}")

        with open(resolved_path, "r", encoding="utf-8") as f:
            words = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(words)} words from {resolved_path}")
        return words

    except Exception as e:
        logger.error(
            f"Error loading wordlist from {file_path} (resolved to {resolve_path(file_path)}): {e}"
        )
        return []
