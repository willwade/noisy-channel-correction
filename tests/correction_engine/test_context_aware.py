#!/usr/bin/env python3
"""
Test script for context-aware correction.

This script demonstrates how to use the context-aware correction functionality
of the NoisyChannelCorrector class.
"""

import os
import sys
import logging
from typing import List, Tuple, Optional, Union

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the corrector
from lib.corrector.corrector import NoisyChannelCorrector
from lib.corrector.enhanced_ppm_predictor import EnhancedPPMPredictor
from lib.corrector.word_ngram_model import WordNGramModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def train_models(
    training_text_path: str, save_models: bool = True
) -> Tuple[EnhancedPPMPredictor, WordNGramModel]:
    """
    Train the PPM and word n-gram models on the given text.

    Args:
        training_text_path: Path to the training text file
        save_models: Whether to save the models after training

    Returns:
        Tuple of (ppm_model, word_ngram_model)
    """
    # Load the training text
    try:
        with open(training_text_path, "r", encoding="utf-8") as f:
            training_text = f.read()
    except Exception as e:
        logger.error(f"Error loading training text from {training_text_path}: {e}")
        return None, None

    # Train the PPM model
    ppm_model = EnhancedPPMPredictor()
    ppm_success = ppm_model.train_on_text(
        training_text, save_model=save_models, model_path="models/ppm_model.pkl"
    )

    if not ppm_success:
        logger.error("Failed to train PPM model")

    # Train the word n-gram model
    word_ngram_model = WordNGramModel()
    ngram_success = word_ngram_model.train(training_text)

    if ngram_success and save_models:
        word_ngram_model.save("models/word_ngram_model.pkl")
    elif not ngram_success:
        logger.error("Failed to train word n-gram model")

    return ppm_model, word_ngram_model


def test_context_aware_correction(
    corrector: NoisyChannelCorrector,
    noisy_input: str,
    context: Optional[Union[str, List[str]]] = None,
    max_edit_distance: int = 2,
) -> None:
    """
    Test context-aware correction.

    Args:
        corrector: The corrector to use
        noisy_input: The noisy input text
        context: Optional context for correction
        max_edit_distance: Maximum edit distance for candidate generation
    """
    print(f"\nNoisy input: '{noisy_input}'")

    if context:
        if isinstance(context, list):
            print(f"Context: {context}")
        else:
            print(f"Context: '{context}'")
    else:
        print("No context provided")

    # Correct without context
    no_context_corrections = corrector.correct(
        noisy_input, context=None, max_edit_distance=max_edit_distance
    )

    # Correct with context
    with_context_corrections = corrector.correct(
        noisy_input, context=context, max_edit_distance=max_edit_distance
    )

    # Print the results
    print("\nCorrections without context:")
    for i, (correction, score) in enumerate(no_context_corrections):
        print(f"  {i+1}. {correction} (score: {score:.4f})")

    print("\nCorrections with context:")
    for i, (correction, score) in enumerate(with_context_corrections):
        print(f"  {i+1}. {correction} (score: {score:.4f})")


def main():
    """Main function."""
    # Create the models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Check if models exist
    ppm_model_path = "models/ppm_model.pkl"
    word_ngram_model_path = "models/word_ngram_model.pkl"
    confusion_matrix_path = "models/confusion_matrix.json"
    lexicon_path = "data/wordlist.txt"

    # Initialize the models
    ppm_model = EnhancedPPMPredictor()
    word_ngram_model = WordNGramModel()

    # Load or train the models
    if os.path.exists(ppm_model_path) and os.path.exists(word_ngram_model_path):
        # Load the models
        ppm_model.load_model(ppm_model_path)
        word_ngram_model.load(word_ngram_model_path)
    else:
        # Train the models
        training_text_path = "data/training_text.txt"
        if os.path.exists(training_text_path):
            ppm_model, word_ngram_model = train_models(training_text_path)
        else:
            logger.error(f"Training text file not found: {training_text_path}")
            logger.info("Using sample text for training")
            sample_text = """
            The quick brown fox jumps over the lazy dog. This is a sample text for training
            the language models. We need to provide enough text to train both the PPM model
            and the word n-gram model. The more text we provide, the better the models will
            perform. Context-aware correction uses previous words to improve the accuracy of
            the correction. This is especially useful for correcting words that are ambiguous
            without context.
            """
            ppm_model, word_ngram_model = train_models(sample_text, save_models=False)

    # Create the corrector
    corrector = NoisyChannelCorrector(
        ppm_model=ppm_model,
        word_ngram_model=word_ngram_model,
        context_window_size=2,
        context_weight=0.7,
    )

    # Load the confusion matrix if it exists
    if os.path.exists(confusion_matrix_path):
        corrector.load_confusion_model(confusion_matrix_path)
    else:
        logger.warning(f"Confusion matrix file not found: {confusion_matrix_path}")

    # Load the lexicon if it exists
    if os.path.exists(lexicon_path):
        corrector.load_lexicon_from_file(lexicon_path)
    else:
        logger.warning(f"Lexicon file not found: {lexicon_path}")

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
        {
            "noisy_input": "thier",
            "context": "They left books at home",
            "description": "Context helps: 'thier' -> 'their'",
        },
        {
            "noisy_input": "rane",
            "context": ["The", "weather", "forecast", "predicted"],
            "description": "List context: 'rane' -> 'rain' (with weather context)",
        },
        {
            "noisy_input": "rane",
            "context": ["The", "train", "arrived", "on"],
            "description": "Different context: 'rane' -> 'time' (with train context)",
        },
    ]

    # Run the tests
    print("=== Testing Context-Aware Correction ===")

    for test_case in test_cases:
        print(f"\n--- {test_case['description']} ---")
        test_context_aware_correction(
            corrector,
            test_case["noisy_input"],
            test_case.get("context"),
        )


if __name__ == "__main__":
    main()
