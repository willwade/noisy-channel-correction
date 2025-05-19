#!/usr/bin/env python3
"""
Test script for keyboard-specific confusion matrices.

This script demonstrates how to use keyboard-specific confusion matrices
for different keyboard layouts (QWERTY, ABC, frequency-based).
"""

import os
import sys
import argparse
import logging
from typing import List, Tuple, Dict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the keyboard confusion matrix
from lib.confusion_matrix.keyboard_confusion_matrix import (
    KeyboardConfusionMatrix,
    build_keyboard_confusion_matrices,
)

# Import the keyboard noise model
from lib.noise_model.noise_model import KeyboardNoiseModel
from lib.corrector.corrector import NoisyChannelCorrector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_test_data(num_samples: int = 10) -> Dict[str, List[Tuple[str, str]]]:
    """
    Generate test data for different keyboard layouts.

    Args:
        num_samples: Number of samples to generate per layout

    Returns:
        Dictionary mapping layout names to lists of (clean, noisy) pairs
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
    ]

    # Create keyboard error models for different layouts
    keyboard_models = {
        "qwerty": KeyboardNoiseModel(
            layout_name="en",
            error_rates={
                "proximity": 0.1,
                "deletion": 0.05,
                "insertion": 0.05,
                "transposition": 0.02,
            },
        ),
        "abc": KeyboardNoiseModel(
            layout_name="abc",
            error_rates={
                "proximity": 0.1,
                "deletion": 0.05,
                "insertion": 0.05,
                "transposition": 0.02,
            },
        ),
        "frequency": KeyboardNoiseModel(
            layout_name="frequency",
            error_rates={
                "proximity": 0.1,
                "deletion": 0.05,
                "insertion": 0.05,
                "transposition": 0.02,
            },
        ),
    }

    # Generate data for each layout
    data = {}
    for layout, model in keyboard_models.items():
        pairs = []
        for _ in range(num_samples):
            # Select a random word
            clean = random.choice(words)

            # Generate a noisy version
            noisy = model.apply(clean)

            # Add to pairs
            pairs.append((clean, noisy))

        data[layout] = pairs

    return data


def build_and_save_matrices(
    data: Dict[str, List[Tuple[str, str]]], output_path: str
) -> KeyboardConfusionMatrix:
    """
    Build and save keyboard-specific confusion matrices.

    Args:
        data: Dictionary mapping layout names to lists of (clean, noisy) pairs
        output_path: Path to save the matrices

    Returns:
        The built keyboard confusion matrices
    """
    # Combine all pairs
    all_pairs = []
    for layout, pairs in data.items():
        all_pairs.extend(pairs)

    # Build the matrices
    matrices = build_keyboard_confusion_matrices(all_pairs)

    # Save the matrices
    if output_path:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Save the matrices
        success = matrices.save(output_path)

        if success:
            logger.info(f"Saved keyboard confusion matrices to {output_path}")
        else:
            logger.error(f"Failed to save keyboard confusion matrices to {output_path}")

    return matrices


def test_correction(
    matrices: KeyboardConfusionMatrix,
    test_data: Dict[str, List[Tuple[str, str]]],
    ppm_model_path: str = None,
    lexicon_path: str = None,
) -> Dict[str, Dict[str, float]]:
    """
    Test correction with different keyboard layouts.

    Args:
        matrices: The keyboard confusion matrices
        test_data: Dictionary mapping layout names to lists of (clean, noisy) pairs
        ppm_model_path: Path to the PPM model file
        lexicon_path: Path to the lexicon file

    Returns:
        Dictionary of performance metrics for each layout
    """
    # Initialize metrics
    metrics = {
        "qwerty": {"total": 0, "correct": 0, "top3": 0, "avg_rank": 0},
        "abc": {"total": 0, "correct": 0, "top3": 0, "avg_rank": 0},
        "frequency": {"total": 0, "correct": 0, "top3": 0, "avg_rank": 0},
    }

    # Create a corrector for each layout
    correctors = {}
    for layout in metrics:
        corrector = NoisyChannelCorrector(
            confusion_model=matrices,
            keyboard_layout=layout,
        )

        # Load the PPM model if provided
        if ppm_model_path and os.path.exists(ppm_model_path):
            corrector.load_ppm_model(ppm_model_path)

        # Load the lexicon if provided
        if lexicon_path and os.path.exists(lexicon_path):
            corrector.load_lexicon_from_file(lexicon_path)

        correctors[layout] = corrector

    # Test each layout
    for layout, pairs in test_data.items():
        corrector = correctors[layout]

        for clean, noisy in pairs:
            # Correct the noisy input
            corrections = corrector.correct(noisy)

            # Update metrics
            metrics[layout]["total"] += 1

            # Check if the correct word is in the top corrections
            correct_found = False
            correct_rank = -1

            for i, (correction, _) in enumerate(corrections):
                if correction.lower() == clean.lower():
                    correct_found = True
                    correct_rank = i + 1
                    break

            if correct_found:
                if correct_rank == 1:
                    metrics[layout]["correct"] += 1
                if correct_rank <= 3:
                    metrics[layout]["top3"] += 1
                metrics[layout]["avg_rank"] += correct_rank
            else:
                metrics[layout]["avg_rank"] += (
                    len(corrections) + 1
                )  # Penalize for not finding

    # Calculate averages
    for layout in metrics:
        if metrics[layout]["total"] > 0:
            metrics[layout]["accuracy"] = (
                metrics[layout]["correct"] / metrics[layout]["total"]
            )
            metrics[layout]["top3_accuracy"] = (
                metrics[layout]["top3"] / metrics[layout]["total"]
            )
            metrics[layout]["avg_rank"] = (
                metrics[layout]["avg_rank"] / metrics[layout]["total"]
            )

    return metrics


def print_metrics(metrics: Dict[str, Dict[str, float]]) -> None:
    """
    Print performance metrics.

    Args:
        metrics: Dictionary of performance metrics for each layout
    """
    print("\n=== Correction Performance by Keyboard Layout ===")

    for layout, layout_metrics in metrics.items():
        print(f"\n{layout.upper()} Layout:")
        print(f"  Total samples: {layout_metrics['total']}")
        print(f"  Accuracy (top 1): {layout_metrics.get('accuracy', 0):.4f}")
        print(f"  Accuracy (top 3): {layout_metrics.get('top3_accuracy', 0):.4f}")
        print(f"  Average rank: {layout_metrics.get('avg_rank', 0):.2f}")


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Test keyboard-specific confusion matrices."
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="models/keyboard_confusion_matrices.json",
        help="Path to save the keyboard confusion matrices",
    )

    parser.add_argument(
        "--num-samples",
        "-n",
        type=int,
        default=50,
        help="Number of samples to generate per layout",
    )

    parser.add_argument(
        "--ppm-model",
        "-p",
        type=str,
        default="models/ppm_model.pkl",
        help="Path to the PPM model file",
    )

    parser.add_argument(
        "--lexicon",
        "-l",
        type=str,
        default="data/wordlist.txt",
        help="Path to the lexicon file",
    )

    args = parser.parse_args()

    # Generate test data
    print("Generating test data...")
    test_data = generate_test_data(args.num_samples)

    # Build and save matrices
    print("Building keyboard-specific confusion matrices...")
    matrices = build_and_save_matrices(test_data, args.output)

    # Test correction with different layouts
    print("Testing correction with different keyboard layouts...")
    metrics = test_correction(matrices, test_data, args.ppm_model, args.lexicon)

    # Print metrics
    print_metrics(metrics)


if __name__ == "__main__":
    main()
