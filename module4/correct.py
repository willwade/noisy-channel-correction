#!/usr/bin/env python3
"""
Command-line interface for the noisy channel corrector.

This script provides a command-line interface for correcting noisy input
using the noisy channel model.
"""

import os
import sys
import argparse
import logging
from typing import List, Tuple, Optional

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the corrector
from module4.corrector import NoisyChannelCorrector, correct

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Correct noisy input using a noisy channel model."
    )
    
    # Input arguments
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Noisy input text to correct"
    )
    parser.add_argument(
        "--output", "-o", type=str, help="Output file for corrected text"
    )
    
    # Model arguments
    parser.add_argument(
        "--ppm-model", type=str, help="Path to the PPM model file"
    )
    parser.add_argument(
        "--confusion-matrix", type=str, help="Path to the confusion matrix file"
    )
    parser.add_argument(
        "--lexicon", type=str, help="Path to the lexicon file"
    )
    
    # Correction parameters
    parser.add_argument(
        "--max-candidates", type=int, default=5, help="Maximum number of candidates to return"
    )
    parser.add_argument(
        "--max-edit-distance", type=int, default=2, help="Maximum edit distance to consider"
    )
    parser.add_argument(
        "--no-keyboard-adjacency", action="store_true", help="Disable keyboard adjacency for candidate generation"
    )
    
    # Context
    parser.add_argument(
        "--context", type=str, default="", help="Context for the PPM model"
    )
    
    # Verbose output
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create a corrector
    corrector = NoisyChannelCorrector(max_candidates=args.max_candidates)
    
    # Load the PPM model if provided
    if args.ppm_model:
        logger.info(f"Loading PPM model from {args.ppm_model}")
        success = corrector.load_ppm_model(args.ppm_model)
        if not success:
            logger.warning(f"Failed to load PPM model from {args.ppm_model}")
    
    # Load the confusion matrix if provided
    if args.confusion_matrix:
        logger.info(f"Loading confusion matrix from {args.confusion_matrix}")
        success = corrector.load_confusion_model(args.confusion_matrix)
        if not success:
            logger.warning(f"Failed to load confusion matrix from {args.confusion_matrix}")
    
    # Load the lexicon if provided
    if args.lexicon:
        logger.info(f"Loading lexicon from {args.lexicon}")
        success = corrector.load_lexicon_from_file(args.lexicon)
        if not success:
            logger.warning(f"Failed to load lexicon from {args.lexicon}")
    
    # Correct the input
    logger.info(f"Correcting input: {args.input}")
    corrections = corrector.correct(
        args.input,
        context=args.context,
        max_edit_distance=args.max_edit_distance,
        use_keyboard_adjacency=not args.no_keyboard_adjacency,
    )
    
    # Print the corrections
    print("\nCorrections:")
    for i, (correction, score) in enumerate(corrections):
        print(f"{i+1}. {correction} (score: {score:.4f})")
    
    # Save the top correction to the output file if provided
    if args.output and corrections:
        try:
            with open(args.output, "w") as f:
                f.write(corrections[0][0])
            logger.info(f"Saved top correction to {args.output}")
        except Exception as e:
            logger.error(f"Error saving correction to {args.output}: {e}")
    
    # Return the top correction
    if corrections:
        return corrections[0][0]
    else:
        return args.input


if __name__ == "__main__":
    main()
