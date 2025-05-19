#!/usr/bin/env python3
"""
Simple test script for keyboard-specific confusion matrices.

This script demonstrates how to use keyboard-specific confusion matrices
for different keyboard layouts (QWERTY, ABC, frequency-based).
"""

import os
import sys
import logging
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
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "I",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
    ]
    
    # Create keyboard noise models for different layouts
    keyboard_models = {
        "qwerty": KeyboardNoiseModel(
            layout_name="en",
            error_rates={
                "proximity": 0.1,
                "deletion": 0.05,
                "insertion": 0.05,
                "transposition": 0.02
            }
        ),
        "abc": KeyboardNoiseModel(
            layout_name="en",  # Use "en" as fallback for "abc"
            error_rates={
                "proximity": 0.1,
                "deletion": 0.05,
                "insertion": 0.05,
                "transposition": 0.02
            }
        ),
        "frequency": KeyboardNoiseModel(
            layout_name="en",  # Use "en" as fallback for "frequency"
            error_rates={
                "proximity": 0.1,
                "deletion": 0.05,
                "insertion": 0.05,
                "transposition": 0.02
            }
        )
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


def test_keyboard_matrices():
    """Test keyboard-specific confusion matrices."""
    # Generate test data
    print("Generating test data...")
    test_data = generate_test_data(10)
    
    # Print the test data
    print("\nTest Data:")
    for layout, pairs in test_data.items():
        print(f"\n{layout.upper()} Layout:")
        for clean, noisy in pairs:
            print(f"  Clean: {clean} -> Noisy: {noisy}")
    
    # Load the keyboard confusion matrices if they exist
    matrices_path = "models/keyboard_confusion_matrices.json"
    if os.path.exists(matrices_path):
        print(f"\nLoading keyboard confusion matrices from {matrices_path}...")
        matrices = KeyboardConfusionMatrix.load(matrices_path)
        
        # Print statistics for each layout
        print("\nKeyboard Confusion Matrix Statistics:")
        for layout, stats in matrices.get_stats().items():
            print(f"\n{layout.upper()} Layout:")
            print(f"  Total pairs: {stats['total']}")
            if stats['total'] > 0:
                print(f"  Correct: {stats['correct']} ({stats['correct']/stats['total']*100:.2f}%)")
                print(f"  Substitutions: {stats['substitutions']} ({stats['substitutions']/stats['total']*100:.2f}%)")
                print(f"  Deletions: {stats['deletions']} ({stats['deletions']/stats['total']*100:.2f}%)")
                print(f"  Insertions: {stats['insertions']} ({stats['insertions']/stats['total']*100:.2f}%)")
                print(f"  Transpositions: {stats['transpositions']} ({stats['transpositions']/stats['total']*100:.2f}%)")
    else:
        print(f"\nKeyboard confusion matrices file not found: {matrices_path}")
        print("Run build_keyboard_matrices.py first to create the matrices.")


def main():
    """Main function."""
    test_keyboard_matrices()


if __name__ == "__main__":
    main()
