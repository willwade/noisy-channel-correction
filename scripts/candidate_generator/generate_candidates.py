#!/usr/bin/env python3
"""
Script to demonstrate the usage of the candidate generator.

This script loads a lexicon from a file and generates candidates for a given noisy input.
"""

import os
import sys
import argparse
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the candidate generator
from lib.candidate_generator.improved_candidate_generator import (
    ImprovedCandidateGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function to demonstrate the usage of the candidate generator."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Generate candidates for a noisy input."
    )
    parser.add_argument("--input", type=str, required=True, help="The noisy input text")
    parser.add_argument(
        "--lexicon",
        type=str,
        default="data/wordlist.txt",
        help="Path to the lexicon file",
    )
    parser.add_argument(
        "--max-edit-distance",
        type=int,
        default=2,
        help="Maximum edit distance to consider",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help="Maximum number of candidates to return",
    )
    parser.add_argument(
        "--no-keyboard-adjacency",
        action="store_true",
        help="Disable keyboard adjacency for candidate generation",
    )
    args = parser.parse_args()

    # Create a candidate generator
    generator = ImprovedCandidateGenerator(max_candidates=args.max_candidates)

    # Load the lexicon
    lexicon_path = os.path.abspath(args.lexicon)
    if not os.path.exists(lexicon_path):
        logger.error(f"Lexicon file not found: {lexicon_path}")
        return 1

    logger.info(f"Loading lexicon from {lexicon_path}")
    if not generator.load_lexicon_from_file(lexicon_path):
        logger.error(f"Failed to load lexicon from {lexicon_path}")
        return 1

    # Generate candidates
    logger.info(f"Generating candidates for '{args.input}'")
    candidates = generator.generate_candidates(
        args.input,
        max_edit_distance=args.max_edit_distance,
        use_keyboard_adjacency=not args.no_keyboard_adjacency,
    )

    # Print the candidates
    print(f"\nCandidates for '{args.input}':")
    print("-" * 40)
    for i, (candidate, score) in enumerate(candidates, 1):
        print(f"{i}. {candidate} (score: {score:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
