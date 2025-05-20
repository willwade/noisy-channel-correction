#!/usr/bin/env python3
"""
Language-specific keyboard confusion matrix test for en-GB.

This script demonstrates how to use keyboard-specific confusion matrices
for different keyboard layouts (QWERTY, ABC, frequency-based) with en-GB specific data.
"""

import os
import sys
import logging
import random
from typing import List, Tuple, Dict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the keyboard confusion matrix
from lib.confusion_matrix.keyboard_confusion_matrix import KeyboardConfusionMatrix

# Import the keyboard noise model
from lib.noise_model.noise_model import KeyboardNoiseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_en_gb_lexicon(lexicon_path: str = None) -> List[str]:
    """
    Load the en-GB lexicon from a file.

    Args:
        lexicon_path: Path to the lexicon file (defaults to en-GB qwerty lexicon)

    Returns:
        List of words from the lexicon
    """
    if lexicon_path is None:
        # Use the default en-GB lexicon
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        lexicon_path = os.path.join(base_dir, "data", "keyboard_lexicons_en_gb", "qwerty_lexicon.txt")

    words = []
    try:
        with open(lexicon_path, "r") as f:
            for line in f:
                # Each line may have a word and a frequency, separated by whitespace
                parts = line.strip().split()
                if parts:
                    words.append(parts[0])
        
        logger.info(f"Loaded {len(words)} words from {lexicon_path}")
        return words
    except Exception as e:
        logger.error(f"Error loading lexicon from {lexicon_path}: {e}")
        return []


def generate_test_data(num_samples: int = 10, language: str = "en-GB") -> Dict[str, List[Tuple[str, str]]]:
    """
    Generate test data for different keyboard layouts using language-specific words.

    Args:
        num_samples: Number of samples to generate per layout
        language: Language code (e.g., "en-GB")

    Returns:
        Dictionary mapping layout names to lists of (clean, noisy) pairs
    """
    # Load language-specific words
    if language == "en-GB":
        words = load_en_gb_lexicon()
    else:
        # Fallback to common English words if language not supported
        words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"
        ]

    # Ensure we have words to work with
    if not words:
        logger.warning(f"No words found for language {language}. Using fallback words.")
        words = [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at"
        ]

    # Create keyboard noise models with layout-specific error patterns
    keyboard_models = {
        "qwerty": KeyboardNoiseModel(
            layout_name="en",
            error_rates={
                "proximity": 0.3,    # QWERTY has more proximity errors due to key arrangement
                "deletion": 0.15,
                "insertion": 0.15,
                "transposition": 0.1,
            },
        ),
        "abc": KeyboardNoiseModel(
            layout_name="abc",
            error_rates={
                "proximity": 0.2,    # ABC has fewer proximity errors but more insertions
                "deletion": 0.15,
                "insertion": 0.3,    # Higher insertion rate for ABC layout
                "transposition": 0.1,
            },
        ),
        "frequency": KeyboardNoiseModel(
            layout_name="frequency",
            error_rates={
                "proximity": 0.2,    # Frequency layout has fewer proximity errors
                "deletion": 0.25,    # But more deletions and transpositions
                "insertion": 0.15,
                "transposition": 0.2, # Higher transposition rate for frequency layout
            },
        ),
    }

    # Generate data for each layout
    data = {}
    for layout, model in keyboard_models.items():
        pairs = []
        while len(pairs) < num_samples:
            # Select a random word (prefer longer words for more interesting errors)
            # Filter for words with at least 4 characters for more interesting errors
            eligible_words = [w for w in words if len(w) >= 4]
            if not eligible_words:
                eligible_words = words  # Fallback if no long words
                
            clean = random.choice(eligible_words)

            # Generate a noisy version
            noisy = model.apply(clean)
            
            # Only add if there's an actual difference
            if clean != noisy:
                pairs.append((clean, noisy))
            # If we got an unchanged word, try again

        data[layout] = pairs

    return data


def test_keyboard_matrices(language: str = "en-GB"):
    """
    Test keyboard-specific confusion matrices with language-specific data.
    
    Args:
        language: Language code (e.g., "en-GB")
    """
    # Generate test data
    print(f"Generating test data for {language}...")
    test_data = generate_test_data(10, language)

    # Print the test data
    print("\nTest Data:")
    for layout, pairs in test_data.items():
        print(f"\n{layout.upper()} Layout:")
        for clean, noisy in pairs:
            print(f"  Clean: {clean} -> Noisy: {noisy}")

    # Load the en-GB specific keyboard confusion matrices if they exist
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    matrices_path = os.path.join(base_dir, "models", "keyboard_confusion_matrices_en_gb.json")
    
    if os.path.exists(matrices_path):
        print(f"\nLoading keyboard confusion matrices from {matrices_path}...")
        matrices = KeyboardConfusionMatrix.load(matrices_path)

        # Print statistics for each layout
        print("\nKeyboard Confusion Matrix Statistics:")
        for layout, stats in matrices.get_stats().items():
            print(f"\n{layout.upper()} Layout:")
            print(f"  Total pairs: {stats['total']}")
            if stats["total"] > 0:
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
    else:
        print(f"\nKeyboard confusion matrices file not found: {matrices_path}")
        print("Run build_keyboard_matrices.py first to create the matrices.")


def main():
    """Main function."""
    test_keyboard_matrices("en-GB")


if __name__ == "__main__":
    main()
