#!/usr/bin/env python3
"""
Convert keyboard-specific confusion matrices to a standard confusion matrix.

This script converts the keyboard-specific confusion matrices to a standard
confusion matrix format that can be used by the correction engine.
"""

import os
import sys
import json
import argparse
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the confusion matrix classes after adding parent directory to path
from lib.confusion_matrix.keyboard_confusion_matrix import KeyboardConfusionMatrix  # noqa

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def convert_keyboard_to_standard(
    keyboard_matrix_path: str, output_path: str, layout: str = "qwerty"
) -> bool:
    """
    Convert a keyboard-specific confusion matrix to a standard confusion matrix.

    Args:
        keyboard_matrix_path: Path to the keyboard-specific confusion matrix file
        output_path: Path to save the standard confusion matrix
        layout: Keyboard layout to use (default: qwerty)

    Returns:
        True if successful, False otherwise
    """
    # Load the keyboard-specific confusion matrix
    try:
        keyboard_matrix = KeyboardConfusionMatrix.load(keyboard_matrix_path)
        logger.info(f"Loaded keyboard confusion matrix from {keyboard_matrix_path}")
    except Exception as e:
        logger.error(f"Error loading keyboard confusion matrix: {e}")
        return False

    # Check if the layout exists in the keyboard matrix
    if layout not in keyboard_matrix.matrices:
        logger.error(f"Layout {layout} not found in keyboard confusion matrix")
        return False

    # Get the confusion matrix for the specified layout
    layout_matrix = keyboard_matrix.matrices[layout]

    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save the standard confusion matrix
    try:
        # Extract the data from the confusion matrix
        matrix_data = {
            "counts": dict(layout_matrix.counts),
            "probabilities": {},  # Add empty probabilities field
            "stats": layout_matrix.stats,
            "vocabulary": [],  # Add empty vocabulary field as a list
            "total_pairs": 0  # Initialize total_pairs
        }

        # Convert defaultdicts to regular dicts for JSON serialization
        total_count = 0
        for char, char_counts in matrix_data["counts"].items():
            matrix_data["counts"][char] = dict(char_counts)
            # Add source character to vocabulary
            if char not in matrix_data["vocabulary"]:
                matrix_data["vocabulary"].append(char)

            # Add target characters to vocabulary and calculate total counts
            for target_char, count in char_counts.items():
                if target_char not in matrix_data["vocabulary"]:
                    matrix_data["vocabulary"].append(target_char)
                total_count += count

        # Set the total_pairs field
        matrix_data["total_pairs"] = total_count

        # Calculate probabilities from counts
        for char, char_counts in matrix_data["counts"].items():
            matrix_data["probabilities"][char] = {}
            total = sum(char_counts.values())
            if total > 0:
                for target_char, count in char_counts.items():
                    matrix_data["probabilities"][char][target_char] = count / total

        # Save the matrix to a JSON file
        with open(output_path, "w") as f:
            json.dump(matrix_data, f, indent=2)

        logger.info(f"Saved standard confusion matrix to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving standard confusion matrix: {e}")
        return False


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Convert keyboard-specific matrix to standard format"
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to the keyboard-specific confusion matrix file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        required=True,
        help="Path to save the standard confusion matrix",
    )

    parser.add_argument(
        "--layout",
        "-l",
        type=str,
        default="qwerty",
        choices=["qwerty", "abc", "frequency"],
        help="Keyboard layout to use (default: qwerty)",
    )

    args = parser.parse_args()

    # Convert the keyboard-specific confusion matrix to a standard confusion matrix
    success = convert_keyboard_to_standard(args.input, args.output, args.layout)

    if success:
        print("Successfully converted keyboard confusion matrix to standard format")
        print(f"Saved to {args.output}")
    else:
        print("Failed to convert keyboard confusion matrix")


if __name__ == "__main__":
    main()
