#!/usr/bin/env python3
"""
Simple test script for context-aware correction.

This script tests the basic functionality of the context-aware correction
without running the full test suite.
"""

import os
import sys
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the corrector
from lib.corrector.corrector import NoisyChannelCorrector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    # Create a corrector
    corrector = NoisyChannelCorrector(
        max_candidates=5,
        context_window_size=2,
        context_weight=0.7,
    )

    # Load the models
    if os.path.exists("models/ppm_model.pkl"):
        corrector.load_ppm_model("models/ppm_model.pkl")
    else:
        logger.warning("PPM model file not found: models/ppm_model.pkl")

    if os.path.exists("models/confusion_matrix.json"):
        corrector.load_confusion_model("models/confusion_matrix.json")
    else:
        logger.warning("Confusion matrix file not found: models/confusion_matrix.json")

    if os.path.exists("models/word_ngram_model.pkl"):
        corrector.load_word_ngram_model("models/word_ngram_model.pkl")
    else:
        logger.warning("Word n-gram model file not found: models/word_ngram_model.pkl")

    if os.path.exists("data/wordlist.txt"):
        corrector.load_lexicon_from_file("data/wordlist.txt")
    else:
        logger.warning("Lexicon file not found: data/wordlist.txt")

    # Test cases
    test_cases = [
        {
            "noisy_input": "teh",
            "context": "I wrote",
            "description": "Simple case: 'teh' -> 'the'",
        },
        {
            "noisy_input": "wnat",
            "context": "I to go",
            "description": "Ambiguous case: 'wnat' -> 'want' (with context) or 'what' (without context)",
        },
    ]

    # Run the tests
    print("=== Testing Context-Aware Correction ===")

    for test_case in test_cases:
        print(f"\n--- {test_case['description']} ---")
        
        noisy_input = test_case["noisy_input"]
        context = test_case.get("context")
        
        print(f"Noisy input: '{noisy_input}'")
        if context:
            print(f"Context: '{context}'")
        else:
            print("No context provided")

        # Correct without context
        no_context_corrections = corrector.correct(noisy_input, context=None)

        # Correct with context
        with_context_corrections = corrector.correct(noisy_input, context=context)

        # Print the results
        print("\nCorrections without context:")
        for i, (correction, score) in enumerate(no_context_corrections[:3]):
            print(f"  {i+1}. {correction} (score: {score:.4f})")

        print("\nCorrections with context:")
        for i, (correction, score) in enumerate(with_context_corrections[:3]):
            print(f"  {i+1}. {correction} (score: {score:.4f})")


if __name__ == "__main__":
    main()
