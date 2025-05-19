#!/usr/bin/env python3
"""
Build a confusion matrix from noisy pairs.

This script builds a confusion matrix from pairs of (intended, noisy) text
and saves it to a JSON file for later use.

Usage:
    python build_matrix.py --input data/noisy_pairs.json --output data/confusion_matrix.json
    python build_matrix.py --input data/noisy_pairs.json --output data/confusion_matrix.json --visualize
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the confusion matrix module
from lib.confusion_matrix.confusion_matrix import (
    ConfusionMatrix,
    build_confusion_matrix,
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build a confusion matrix from noisy pairs")
    
    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input JSON file containing noisy pairs",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the output JSON file for the confusion matrix",
    )
    
    # Optional arguments
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Visualize the confusion matrix",
    )
    
    return parser.parse_args()


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


def visualize_matrix(matrix: ConfusionMatrix) -> None:
    """
    Visualize the confusion matrix.
    
    Args:
        matrix: The confusion matrix to visualize
    """
    # Print the matrix as a string
    print(matrix)
    
    # Print some statistics
    stats = matrix.get_stats()
    print("\nError Statistics:")
    print(f"Total pairs: {stats['total']}")
    print(f"Correct: {stats['correct']} ({stats['correct'] / stats['total'] * 100:.2f}%)")
    print(f"Substitutions: {stats['substitutions']} ({stats['substitutions'] / stats['total'] * 100:.2f}%)")
    print(f"Deletions: {stats['deletions']} ({stats['deletions'] / stats['total'] * 100:.2f}%)")
    print(f"Insertions: {stats['insertions']} ({stats['insertions'] / stats['total'] * 100:.2f}%)")
    print(f"Transpositions: {stats['transpositions']} ({stats['transpositions'] / stats['total'] * 100:.2f}%)")
    
    # Print some example probabilities
    print("\nExample Probabilities:")
    for intended in "abcdefghijklmnopqrstuvwxyz":
        for noisy in "abcdefghijklmnopqrstuvwxyz":
            prob = matrix.get_probability(noisy, intended)
            if prob > 0.1 and intended != noisy:
                print(f"P(noisy='{noisy}' | intended='{intended}') = {prob:.4f}")


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Load noisy pairs
    pairs = load_noisy_pairs(args.input)
    if not pairs:
        logger.error("No noisy pairs loaded. Exiting.")
        sys.exit(1)
    
    # Build the confusion matrix
    logger.info("Building confusion matrix...")
    matrix = build_confusion_matrix(pairs)
    
    # Save the confusion matrix
    success = matrix.save(args.output)
    if not success:
        logger.error("Failed to save confusion matrix. Exiting.")
        sys.exit(1)
    
    logger.info("Successfully built confusion matrix.")
    logger.info(f"Output saved to {args.output}")
    
    # Visualize the matrix if requested
    if args.visualize:
        visualize_matrix(matrix)


if __name__ == "__main__":
    main()
