#!/usr/bin/env python3
"""
Test script for the advanced candidate generator.

This script tests the advanced candidate generator with various inputs
and compares it to the enhanced candidate generator.
"""

import os
import sys
import logging
import time
from typing import List, Tuple, Set, Dict, Optional

# Add the parent directory to the Python path
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Import the candidate generators
from lib.candidate_generator.enhanced_candidate_generator import (
    EnhancedCandidateGenerator,
)
from lib.candidate_generator.advanced_candidate_generator import (
    AdvancedCandidateGenerator,
)

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
    enhanced_generator: EnhancedCandidateGenerator,
    advanced_generator: AdvancedCandidateGenerator,
    word: str,
    max_edit_distance: int = 2,
) -> None:
    """
    Test the generators with a single word.

    Args:
        enhanced_generator: The enhanced candidate generator
        advanced_generator: The advanced candidate generator
        word: The word to test
        max_edit_distance: Maximum edit distance to consider
    """
    logger.info(f"Testing with word: '{word}'")

    # Test the enhanced generator
    start_time = time.time()
    enhanced_candidates = enhanced_generator.generate_candidates(
        word, max_edit_distance=max_edit_distance
    )
    enhanced_time = time.time() - start_time
    logger.info(
        f"Enhanced generator: {len(enhanced_candidates)} candidates in {enhanced_time:.4f} seconds"
    )

    # Test the advanced generator
    start_time = time.time()
    advanced_candidates = advanced_generator.generate_candidates(
        word, max_edit_distance=max_edit_distance
    )
    advanced_time = time.time() - start_time
    logger.info(
        f"Advanced generator: {len(advanced_candidates)} candidates in {advanced_time:.4f} seconds"
    )

    # Compare the top candidates
    logger.info("Top 5 candidates from enhanced generator:")
    for i, (candidate, score) in enumerate(enhanced_candidates[:5]):
        logger.info(f"  {i+1}. {candidate} ({score:.4f})")

    logger.info("Top 5 candidates from advanced generator:")
    for i, (candidate, score) in enumerate(advanced_candidates[:5]):
        logger.info(f"  {i+1}. {candidate} ({score:.4f})")


def test_multi_word(
    enhanced_generator: EnhancedCandidateGenerator,
    advanced_generator: AdvancedCandidateGenerator,
    text: str,
    max_edit_distance: int = 2,
) -> None:
    """
    Test the generators with multi-word text.

    Args:
        enhanced_generator: The enhanced candidate generator
        advanced_generator: The advanced candidate generator
        text: The text to test
        max_edit_distance: Maximum edit distance to consider
    """
    logger.info(f"Testing with text: '{text}'")

    # Test the enhanced generator
    start_time = time.time()
    enhanced_candidates = enhanced_generator.generate_candidates(
        text, max_edit_distance=max_edit_distance
    )
    enhanced_time = time.time() - start_time
    logger.info(
        f"Enhanced generator: {len(enhanced_candidates)} candidates in {enhanced_time:.4f} seconds"
    )

    # Test the advanced generator
    start_time = time.time()
    advanced_candidates = advanced_generator.generate_candidates(
        text, max_edit_distance=max_edit_distance
    )
    advanced_time = time.time() - start_time
    logger.info(
        f"Advanced generator: {len(advanced_candidates)} candidates in {advanced_time:.4f} seconds"
    )

    # Compare the top candidates
    logger.info("Top 5 candidates from enhanced generator:")
    for i, (candidate, score) in enumerate(enhanced_candidates[:5]):
        logger.info(f"  {i+1}. {candidate} ({score:.4f})")

    logger.info("Top 5 candidates from advanced generator:")
    for i, (candidate, score) in enumerate(advanced_candidates[:5]):
        logger.info(f"  {i+1}. {candidate} ({score:.4f})")


def main():
    """Main function to test the candidate generators."""
    # Load the lexicon
    lexicon_path = os.path.abspath("data/wordlist.txt")
    if not os.path.exists(lexicon_path):
        logger.error(f"Lexicon file not found: {lexicon_path}")
        return 1

    lexicon = load_lexicon(lexicon_path)
    if not lexicon:
        logger.error("Failed to load lexicon")
        return 1

    # Create the generators
    enhanced_generator = EnhancedCandidateGenerator(
        lexicon=lexicon,
        max_candidates=20,
        max_edits=5000,
        keyboard_boost=0.2,
        strict_filtering=True,
    )

    advanced_generator = AdvancedCandidateGenerator(
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
        "teh",  # Common transposition
        "wrld",  # Missing vowel
    ]

    for word in test_words:
        test_single_word(enhanced_generator, advanced_generator, word)
        print()

    # Test with multi-word text
    test_texts = [
        "helo wrld",  # Simple multi-word
    ]

    for text in test_texts:
        test_multi_word(enhanced_generator, advanced_generator, text)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
