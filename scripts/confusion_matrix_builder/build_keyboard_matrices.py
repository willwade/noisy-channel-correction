#!/usr/bin/env python3
"""
Build keyboard-specific confusion matrices from noisy text data.

This script builds keyboard-specific confusion matrices for different keyboard layouts
(QWERTY, ABC, frequency-based) from pairs of (intended, noisy) text data,
with support for language-specific customization (default: en-GB).
"""

import os
import sys
import json
import argparse
import logging
import random
from typing import List, Tuple, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the keyboard confusion matrix and keyboard layout model
from lib.confusion_matrix.keyboard_confusion_matrix import (
    KeyboardConfusionMatrix,
    build_keyboard_confusion_matrices,
)
from lib.noise_model.noise_model import KeyboardNoiseModel


def load_noisy_pairs(input_path: str) -> List[Tuple[str, str]]:
    """
    Load noisy pairs from a JSON file.

    Args:
        input_path: Path to the input JSON file

    Returns:
        List of (clean, noisy) text pairs
    """
    try:
        with open(input_path, "r") as f:
            data = json.load(f)

        pairs = [(item["clean"], item["noisy"]) for item in data]

        logger.info(f"Loaded {len(pairs)} noisy pairs from {input_path}")
        return pairs

    except Exception as e:
        logger.error(f"Error loading noisy pairs from {input_path}: {e}")
        return []


def load_language_lexicon(language: str, layout: str) -> List[str]:
    """
    Load a language-specific lexicon for a specific keyboard layout.

    Args:
        language: Language code (e.g., "en-GB", "en-US")
        layout: Keyboard layout (e.g., "qwerty", "abc", "frequency")

    Returns:
        List of words in the lexicon
    """
    # Normalize the language code to use in the file path
    language_path = language.lower().replace("-", "_")

    # Define the path to the lexicon file
    lexicon_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "data",
        f"keyboard_lexicons_{language_path}",
        f"{layout}_lexicon.txt",
    )

    # Check if the lexicon file exists
    if not os.path.exists(lexicon_path):
        logger.warning(f"Lexicon file not found: {lexicon_path}")
        logger.warning("Falling back to default lexicon")

        # Fall back to the default lexicon
        lexicon_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "data",
            "keyboard_lexicons",
            f"{layout}_lexicon.txt",
        )

    # Load the lexicon
    words = []
    try:
        with open(lexicon_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    word = parts[0].strip()
                    if word and not word.isdigit():
                        words.append(word)
        logger.info(f"Loaded {len(words)} words from {lexicon_path}")
    except Exception as e:
        logger.error(f"Error loading lexicon: {e}")

    return words


def generate_synthetic_data(num_pairs: int = 1000, language: str = "en-GB") -> List[Tuple[str, str]]:
    """
    Generate synthetic noisy pairs for testing.

    Args:
        num_pairs: Number of pairs to generate
        language: Language code (e.g., "en-GB", "en-US")

    Returns:
        List of (clean, noisy) text pairs
    """
    # Generate synthetic noisy pairs
    pairs = []
    pairs_per_layout = num_pairs // 3

    # Define layout-specific error patterns
    layout_error_rates = {
        "qwerty": {
            "proximity": 0.08,    # QWERTY has more proximity errors due to key arrangement
            "deletion": 0.03,
            "insertion": 0.02,
            "transposition": 0.02,
        },
        "abc": {
            "proximity": 0.05,    # ABC has fewer proximity errors but more insertions
            "deletion": 0.03,
            "insertion": 0.06,    # Higher insertion rate for ABC layout
            "transposition": 0.01,
        },
        "frequency": {
            "proximity": 0.04,    # Frequency layout has fewer proximity errors
            "deletion": 0.05,    # But more deletions and transpositions
            "insertion": 0.02,
            "transposition": 0.04, # Higher transposition rate for frequency layout
        },
    }

    # Generate data for each layout
    for layout, error_rates in layout_error_rates.items():
        # Load layout-specific lexicon
        words = load_language_lexicon(language, layout)
        
        # Create keyboard noise model for this layout
        model = KeyboardNoiseModel(
            layout_name=layout,
            error_rates=error_rates,
            input_method="direct",
        )
        
        layout_pairs = []
        for _ in range(pairs_per_layout):
            # Select a random word
            word = random.choice(words)

            # Generate a noisy version of the word
            noisy = model.apply(word)

            # Add the pair to the list
            layout_pairs.append((word, noisy))

        logger.info(f"Generated {len(layout_pairs)} pairs for {layout} layout")
        pairs.extend(layout_pairs)

    return pairs


def visualize_matrices(matrices: KeyboardConfusionMatrix) -> None:
    """
    Visualize the keyboard confusion matrices.

    Args:
        matrices: The keyboard confusion matrices to visualize
    """
    # Print statistics for each layout
    for layout in matrices.matrices.keys():
        print(f"\n{layout.upper()} Layout:")
        stats = matrices.get_stats(layout)
        total = stats["total"]

        if total > 0:
            print(f"  Total pairs: {total}")
            for key, value in stats.items():
                print(f"  {key.capitalize()}: {value} ({value / total * 100:.2f}%)")
        else:
            print("  No data available")


def test_matrices(
    matrices: KeyboardConfusionMatrix, test_pairs: List[Tuple[str, str]]
) -> Dict[str, Dict[str, float]]:
    """
    Test the keyboard confusion matrices on a set of test pairs.

    Args:
        matrices: The keyboard confusion matrices to test
        test_pairs: List of (clean, noisy) text pairs for testing

    Returns:
        Dictionary of performance metrics for each layout
    """
    # Initialize metrics for each layout
    metrics = {}
    for layout in matrices.matrices.keys():
        metrics[layout] = {
            "total": 0,
            "correct": 0,
            "log_likelihood": 0.0,
            "avg_log_likelihood": 0.0,
        }

    # Test each pair
    for clean, noisy in test_pairs:
        for layout in matrices.matrices.keys():
            # Calculate the log likelihood of the noisy text given the clean text
            log_likelihood = matrices.get_log_likelihood(clean, noisy, layout)

            # Update metrics
            metrics[layout]["total"] += 1
            metrics[layout]["log_likelihood"] += log_likelihood

            # Check if the noisy text is correctly predicted
            # (This would require implementing a correction function)

    # Calculate average log likelihood
    for layout in metrics:
        if metrics[layout]["total"] > 0:
            metrics[layout]["avg_log_likelihood"] = (
                metrics[layout]["log_likelihood"] / metrics[layout]["total"]
            )
        else:
            metrics[layout]["avg_log_likelihood"] = 0.0

    return metrics


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Build keyboard-specific confusion matrices from noisy text data."
    )

    parser.add_argument(
        "--input", "-i", type=str, help="Path to the input JSON file with noisy pairs"
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="models/keyboard_confusion_matrices.json",
        help="Path to save the keyboard confusion matrices",
    )

    parser.add_argument(
        "--synthetic",
        "-s",
        action="store_true",
        help="Generate synthetic data instead of loading from a file",
    )

    parser.add_argument(
        "--num-pairs",
        "-n",
        type=int,
        default=1000,
        help="Number of synthetic pairs to generate (if --synthetic is used)",
    )

    parser.add_argument(
        "--language",
        "-l",
        type=str,
        default="en-GB",
        help="Language code for language-specific customization (default: en-GB)",
    )

    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Visualize the confusion matrices",
    )

    args = parser.parse_args()

    # Create a keyboard confusion matrix to store all layouts
    keyboard_matrix = KeyboardConfusionMatrix()

    # Define layout-specific error patterns
    layout_error_rates = {
        "qwerty": {
            "proximity": 0.08,    # QWERTY has more proximity errors due to key arrangement
            "deletion": 0.03,
            "insertion": 0.02,
            "transposition": 0.02,
        },
        "abc": {
            "proximity": 0.05,    # ABC has fewer proximity errors but more insertions
            "deletion": 0.03,
            "insertion": 0.06,    # Higher insertion rate for ABC layout
            "transposition": 0.01,
        },
        "frequency": {
            "proximity": 0.04,    # Frequency layout has fewer proximity errors
            "deletion": 0.05,    # But more deletions and transpositions
            "insertion": 0.02,
            "transposition": 0.04, # Higher transposition rate for frequency layout
        },
    }

    # Process each layout separately
    for layout, error_rates in layout_error_rates.items():
        logger.info(f"Processing {layout.upper()} layout")

        # Generate layout-specific pairs
        if args.synthetic:
            # Load layout-specific lexicon
            words = load_language_lexicon(args.language, layout)
            
            # Create keyboard noise model for this layout
            model = KeyboardNoiseModel(
                layout_name=layout,
                error_rates=error_rates,
                input_method="direct",
            )
            
            # Generate pairs for this layout
            layout_pairs = []
            for _ in range(args.num_pairs // 3):
                word = random.choice(words)
                noisy = model.apply(word)
                layout_pairs.append((word, noisy))
                
            logger.info(f"Generated {len(layout_pairs)} pairs for {layout} layout")
        elif args.input:
            layout_pairs = load_noisy_pairs(args.input)
            logger.info(f"Loaded {len(layout_pairs)} pairs for {layout} layout")
        else:
            logger.error("Either --input or --synthetic must be specified")
            return

        # Build confusion matrix for this layout only
        layout_matrix = build_keyboard_confusion_matrices(layout_pairs)
        
        # Copy the matrix for this layout to the combined matrix
        keyboard_matrix.matrices[layout] = layout_matrix.matrices[layout]
        keyboard_matrix.stats[layout] = layout_matrix.stats[layout]

    # Visualize the matrices if requested
    if args.visualize:
        visualize_matrices(keyboard_matrix)

    # Save the matrices
    if args.output:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        # Save the matrices
        success = keyboard_matrix.save(args.output)

        if success:
            logger.info(f"Saved keyboard confusion matrices to {args.output}")
        else:
            logger.error(f"Failed to save keyboard confusion matrices to {args.output}")


if __name__ == "__main__":
    main()
