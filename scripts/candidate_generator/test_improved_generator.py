#!/usr/bin/env python3
"""
Test script for the improved candidate generator.

This script tests the improved candidate generator with various inputs.
"""

import os
import sys
import logging
import time
from typing import List, Tuple, Set, Dict, Optional

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the candidate generator
from lib.candidate_generator.improved_candidate_generator import ImprovedCandidateGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_lexicon(file_path: str) -> Set[str]:
    """
    Load a lexicon from a file.

    Args:
        file_path: Path to the lexicon file (one word per line)

    Returns:
        Set of words in the lexicon
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lexicon = set(line.strip().lower() for line in f if line.strip())
        logger.info(f"Loaded lexicon with {len(lexicon)} words from {file_path}")
        return lexicon
    except Exception as e:
        logger.error(f"Error loading lexicon from {file_path}: {e}")
        return set()


def test_single_word(
    generator: ImprovedCandidateGenerator,
    word: str,
    max_edit_distance: int = 2,
) -> None:
    """
    Test the generator with a single word.

    Args:
        generator: The candidate generator
        word: The word to test
        max_edit_distance: Maximum edit distance to consider
    """
    logger.info(f"Testing with word: '{word}'")

    # Test the generator
    start_time = time.time()
    candidates = generator.generate_candidates(
        word, max_edit_distance=max_edit_distance
    )
    elapsed_time = time.time() - start_time
    logger.info(
        f"Generated {len(candidates)} candidates in {elapsed_time:.4f} seconds"
    )

    # Show the top candidates
    logger.info("Top 5 candidates:")
    for i, (candidate, score) in enumerate(candidates[:5]):
        logger.info(f"  {i+1}. {candidate} ({score:.4f})")


def test_multi_word(
    generator: ImprovedCandidateGenerator,
    text: str,
    max_edit_distance: int = 2,
) -> None:
    """
    Test the generator with multi-word text.

    Args:
        generator: The candidate generator
        text: The text to test
        max_edit_distance: Maximum edit distance to consider
    """
    logger.info(f"Testing with text: '{text}'")

    # Test the generator
    start_time = time.time()
    candidates = generator.generate_candidates(
        text, max_edit_distance=max_edit_distance
    )
    elapsed_time = time.time() - start_time
    logger.info(
        f"Generated {len(candidates)} candidates in {elapsed_time:.4f} seconds"
    )

    # Show the top candidates
    logger.info("Top 5 candidates:")
    for i, (candidate, score) in enumerate(candidates[:5]):
        logger.info(f"  {i+1}. {candidate} ({score:.4f})")


def main():
    """Main function to test the candidate generator."""
    # Load the lexicon
    lexicon_path = os.path.abspath("data/wordlist.txt")
    if not os.path.exists(lexicon_path):
        logger.error(f"Lexicon file not found: {lexicon_path}")
        return 1

    lexicon = load_lexicon(lexicon_path)
    if not lexicon:
        logger.error("Failed to load lexicon")
        return 1

    # Create the generator
    generator = ImprovedCandidateGenerator(
        lexicon=lexicon,
        max_candidates=30,
        max_edits=20000,
        keyboard_boost=0.3,
        strict_filtering=True,
        smart_filtering=True,
        use_frequency_info=True,
    )

    # Test with single words
    test_words = [
        "helo",  # Simple typo
        "teh",   # Common transposition
        "wrld",  # Missing vowel
        "compter",  # Missing letter
        "recieve",  # Common misspelling
        "xylophne",  # Uncommon word with typo
    ]

    for word in test_words:
        test_single_word(generator, word)
        print()

    # Test with multi-word text
    test_texts = [
        "helo wrld",  # Simple multi-word
        "teh quik brwn fox",  # Longer multi-word
        "i wnat to go hmoe",  # Common sentence with typos
    ]

    for text in test_texts:
        test_multi_word(generator, text)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
