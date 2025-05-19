#!/usr/bin/env python3
"""
Build keyboard-specific confusion matrices from noisy text data.

This script builds keyboard-specific confusion matrices for different keyboard layouts
(QWERTY, ABC, frequency-based) from pairs of (intended, noisy) text data.
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Tuple, Dict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the keyboard confusion matrix
from lib.confusion_matrix.keyboard_confusion_matrix import (
    KeyboardConfusionMatrix,
    build_keyboard_confusion_matrices,
    get_keyboard_error_probability,
)

# Import the keyboard noise model
from lib.noise_model.noise_model import KeyboardNoiseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def generate_synthetic_data(num_pairs: int = 1000) -> List[Tuple[str, str]]:
    """
    Generate synthetic noisy pairs for testing.

    Args:
        num_pairs: Number of pairs to generate

    Returns:
        List of (clean, noisy) text pairs
    """
    import random

    # Sample words
    words = [
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "I",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
    ]

    # Create keyboard noise models for different layouts
    keyboard_models = {
        "qwerty": KeyboardNoiseModel(
            layout_name="en",
            error_rates={
                "proximity": 0.05,
                "deletion": 0.03,
                "insertion": 0.02,
                "transposition": 0.01,
            },
        ),
        "abc": KeyboardNoiseModel(
            layout_name="abc",  # Use the ABC layout
            error_rates={
                "proximity": 0.05,
                "deletion": 0.03,
                "insertion": 0.02,
                "transposition": 0.01,
            },
        ),
        "frequency": KeyboardNoiseModel(
            layout_name="frequency",  # Use the frequency layout
            error_rates={
                "proximity": 0.05,
                "deletion": 0.03,
                "insertion": 0.02,
                "transposition": 0.01,
            },
        ),
    }

    # Generate pairs
    pairs = []
    for _ in range(num_pairs):
        # Select a random word
        clean = random.choice(words)

        # Select a random keyboard model
        layout = random.choice(list(keyboard_models.keys()))
        model = keyboard_models[layout]

        # Generate a noisy version
        noisy = model.apply(clean)

        # Add to pairs
        pairs.append((clean, noisy))

    logger.info(f"Generated {len(pairs)} synthetic noisy pairs")
    return pairs


def visualize_matrices(matrices: KeyboardConfusionMatrix) -> None:
    """
    Visualize the keyboard confusion matrices.

    Args:
        matrices: The keyboard confusion matrices to visualize
    """
    # Print statistics for each layout
    print("\n=== Keyboard Confusion Matrix Statistics ===")
    for layout, stats in matrices.get_stats().items():
        print(f"\n{layout.upper()} Layout:")
        print(f"  Total pairs: {stats['total']}")
        print(
            f"  Correct: {stats['correct']} ({stats['correct']/stats['total']*100:.2f}%)"
        )
        print(
            f"  Substitutions: {stats['substitutions']} ({stats['substitutions']/stats['total']*100:.2f}%)"
        )
        print(
            f"  Deletions: {stats['deletions']} ({stats['deletions']/stats['total']*100:.2f}%)"
        )
        print(
            f"  Insertions: {stats['insertions']} ({stats['insertions']/stats['total']*100:.2f}%)"
        )
        print(
            f"  Transpositions: {stats['transpositions']} ({stats['transpositions']/stats['total']*100:.2f}%)"
        )

    # Print a sample of the matrices
    print("\n=== Sample Confusion Probabilities ===")
    sample_chars = ["a", "e", "i", "o", "u", "t", "n", "s", "r", "h"]
    for layout in matrices.matrices:
        print(f"\n{layout.upper()} Layout:")
        for intended in sample_chars:
            for noisy in sample_chars:
                prob = matrices.get_probability(intended, noisy, layout)
                if prob > 0.01:  # Only show significant probabilities
                    print(f"  P(noisy='{noisy}' | intended='{intended}') = {prob:.4f}")


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
    # Initialize metrics
    metrics = {
        "qwerty": {"total": 0, "correct": 0, "log_likelihood": 0.0},
        "abc": {"total": 0, "correct": 0, "log_likelihood": 0.0},
        "frequency": {"total": 0, "correct": 0, "log_likelihood": 0.0},
    }

    # Test each pair
    for intended, noisy in test_pairs:
        # Test each layout
        for layout in metrics:
            # Calculate probability
            prob = get_keyboard_error_probability(noisy, intended, matrices, layout)

            # Update metrics
            metrics[layout]["total"] += 1
            metrics[layout]["log_likelihood"] += (
                math.log(prob) if prob > 0 else -float("inf")
            )

            # Check if the most likely correction is correct
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
        "--visualize",
        "-v",
        action="store_true",
        help="Visualize the confusion matrices",
    )

    args = parser.parse_args()

    # Load or generate noisy pairs
    if args.synthetic:
        pairs = generate_synthetic_data(args.num_pairs)
    elif args.input:
        pairs = load_noisy_pairs(args.input)
    else:
        logger.error("Either --input or --synthetic must be specified")
        return

    # Build the keyboard confusion matrices
    matrices = build_keyboard_confusion_matrices(pairs)

    # Visualize the matrices if requested
    if args.visualize:
        visualize_matrices(matrices)

    # Save the matrices
    if args.output:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

        # Save the matrices
        success = matrices.save(args.output)

        if success:
            logger.info(f"Saved keyboard confusion matrices to {args.output}")
        else:
            logger.error(f"Failed to save keyboard confusion matrices to {args.output}")


if __name__ == "__main__":
    import math  # Import here to avoid circular import

    main()
