#!/usr/bin/env python3
"""
Extract the QWERTY layout confusion matrix from a keyboard-specific confusion matrix file.

This script extracts the QWERTY layout confusion matrix from a keyboard-specific
confusion matrix file and saves it as a standard confusion matrix file.
"""

import os
import sys
import json
import argparse
import logging

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_dir)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def extract_qwerty_matrix(input_path: str, output_path: str) -> bool:
    """
    Extract the QWERTY layout confusion matrix from a keyboard-specific confusion matrix file.

    Args:
        input_path: Path to the keyboard-specific confusion matrix file
        output_path: Path to save the standard confusion matrix

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the keyboard-specific confusion matrix
        with open(input_path, "r") as f:
            data = json.load(f)

        # Extract the QWERTY layout matrix
        if "qwerty" not in data:
            logger.error(f"QWERTY layout not found in {input_path}")
            return False

        qwerty_matrix = data["qwerty"]

        # Save the QWERTY matrix as a standard confusion matrix
        with open(output_path, "w") as f:
            json.dump(qwerty_matrix, f, indent=2)

        logger.info(f"Extracted QWERTY matrix from {input_path} and saved to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error extracting QWERTY matrix: {e}")
        return False


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract the QWERTY layout confusion matrix from a keyboard-specific confusion matrix file."
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="models/keyboard_confusion_matrices_en_gb.json",
        help="Path to the keyboard-specific confusion matrix file",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="models/qwerty_confusion_matrix_en_gb.json",
        help="Path to save the standard confusion matrix",
    )

    args = parser.parse_args()

    # Extract the QWERTY matrix
    success = extract_qwerty_matrix(args.input, args.output)

    if success:
        print(f"Successfully extracted QWERTY matrix to {args.output}")
    else:
        print("Failed to extract QWERTY matrix")
        sys.exit(1)


if __name__ == "__main__":
    main()
