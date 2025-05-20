#!/usr/bin/env python3
"""
Train the PPM model for the noisy channel correction project.

This script trains the PPM model and saves it to the models directory.
"""

import os
import sys
import argparse
import logging

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Add module4 and its subdirectories to the path
module4_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(module4_dir)


sys.path.append(os.path.join(module4_dir, "ngram"))

# Import the PPM predictor
try:
    from lib.corrector.enhanced_ppm_predictor import EnhancedPPMPredictor
    from lib.corrector.word_ngram_model import WordNGramModel
except ImportError:
    # Fallback imports if the lib structure is different
    try:
        from ppm.enhanced_ppm_predictor import EnhancedPPMPredictor
        from ngram.word_ngram_model import WordNGramModel
    except ImportError:
        print("Error: Could not import required modules. Make sure the library structure is correct.")
        sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_ppm_model(
    input_file: str,
    output_file: str,
    max_order: int = 5,
) -> bool:
    """
    Train a PPM model on the given text corpus.

    Args:
        input_file: Path to the input text file
        output_file: Path to save the model
        max_order: Maximum context length to consider

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the input text
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Create the PPM model
        ppm_model = EnhancedPPMPredictor(max_order=max_order)

        # Train the model without saving (to avoid recursion error)
        ppm_model.train_on_text(
            text=text,
            save_model=False,
        )

        # Manually save the model
        import pickle
        import sys

        # Increase recursion limit for complex models
        sys.setrecursionlimit(10000)

        # Create a simplified model state to avoid recursion issues
        model_state = {
            "vocab": ppm_model.vocab,
            "lm": None,  # We don't need the full language model
            "max_order": ppm_model.max_order,
            "word_frequencies": ppm_model.word_frequencies,
            "word_recency": ppm_model.word_recency,
            "word_contexts": ppm_model.word_contexts,
            "bigrams": ppm_model.bigrams,
            "trigrams": {},  # Empty trigrams to avoid recursion
        }

        # Save the model state
        try:
            with open(output_file, "wb") as f:
                pickle.dump(model_state, f, protocol=4)
            logger.info(f"PPM model trained and saved to {output_file}")
            return True
        except Exception as e:
            logger.error(f"Error saving PPM model: {e}")
            return False

    except Exception as e:
        logger.error(f"Error training PPM model: {e}")
        return False


def train_word_ngram_model(
    input_file: str,
    output_file: str,
    max_order: int = 3,
    smoothing_method: str = "kneser_ney",
) -> bool:
    """
    Train a word-level n-gram model on the given text corpus.

    Args:
        input_file: Path to the input text file
        output_file: Path to save the model
        max_order: Maximum n-gram order (e.g., 3 for trigrams)
        smoothing_method: Smoothing method to use

    Returns:
        True if successful, False otherwise
    """
    try:
        # Load the input text
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        # Create the word n-gram model
        word_ngram_model = WordNGramModel(
            max_order=max_order, smoothing_method=smoothing_method
        )

        # Train the model
        success = word_ngram_model.train(text)

        if not success:
            logger.error("Failed to train word n-gram model")
            return False

        # Save the model
        success = word_ngram_model.save(output_file)

        if not success:
            logger.error(f"Failed to save word n-gram model to {output_file}")
            return False

        logger.info(f"Word n-gram model trained and saved to {output_file}")
        return True

    except Exception as e:
        logger.error(f"Error training word n-gram model: {e}")
        return False


def generate_sample_text() -> str:
    """
    Generate a sample text for training if no input file is provided.

    Returns:
        Sample text
    """
    return """
    The quick brown fox jumps over the lazy dog.
    This is a sample text for training the PPM model.
    It contains common words and phrases that might be used in AAC systems.
    Hello, how are you today?
    I am fine, thank you.
    The weather is nice today.
    I would like to go for a walk.
    Please help me with this task.
    I need to send an email.
    Can you pass me the water?
    I am hungry and would like to eat something.
    I enjoy reading books and watching movies.
    What time is it?
    I need to go to the bathroom.
    Could you please help me?
    I'm feeling tired today.
    I want to talk to my friend.
    Can you call my doctor?
    I need to take my medication.
    I would like to listen to music.
    """


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train the PPM model and word n-gram model for the noisy channel correction project."
    )

    parser.add_argument("--input", "-i", type=str, help="Path to the input text file")

    parser.add_argument(
        "--ppm-output",
        type=str,
        default="models/ppm_model.pkl",
        help="Path to save the PPM model (default: models/ppm_model.pkl)",
    )

    parser.add_argument(
        "--ngram-output",
        type=str,
        default="models/word_ngram_model.pkl",
        help="Path to save the word n-gram model (default: models/word_ngram_model.pkl)",
    )

    parser.add_argument(
        "--ppm-order",
        type=int,
        default=5,
        help="Maximum context length for PPM model (default: 5)",
    )

    parser.add_argument(
        "--ngram-order",
        type=int,
        default=3,
        help="Maximum n-gram order for word n-gram model (default: 3)",
    )

    parser.add_argument(
        "--smoothing",
        type=str,
        default="kneser_ney",
        choices=["kneser_ney", "laplace", "witten_bell"],
        help="Smoothing method for word n-gram model (default: kneser_ney)",
    )

    parser.add_argument(
        "--copy-confusion-matrix",
        action="store_true",
        help="Copy the confusion matrix from data/ to models/",
    )

    args = parser.parse_args()

    # Create the models directory if it doesn't exist
    os.makedirs(os.path.dirname(args.ppm_output), exist_ok=True)
    os.makedirs(os.path.dirname(args.ngram_output), exist_ok=True)

    # Use sample text if no input file is provided
    input_text = args.input
    if not input_text:
        logger.info("No input file provided. Using sample text.")
        sample_text_path = "sample_training_text.txt"
        with open(sample_text_path, "w", encoding="utf-8") as f:
            f.write(generate_sample_text())
        input_text = sample_text_path

    # Train the PPM model
    ppm_success = train_ppm_model(input_text, args.ppm_output, args.ppm_order)

    # Train the word n-gram model
    ngram_success = train_word_ngram_model(
        input_text, args.ngram_output, args.ngram_order, args.smoothing
    )

    # Copy the confusion matrix if requested
    if args.copy_confusion_matrix:
        import shutil

        src = "data/confusion_matrix.json"
        dst = "models/confusion_matrix.json"
        if os.path.exists(src):
            shutil.copy(src, dst)
            logger.info(f"Copied confusion matrix from {src} to {dst}")
        else:
            logger.error(f"Confusion matrix file not found: {src}")

    # Print summary
    if ppm_success:
        print(f"PPM model trained and saved to {args.ppm_output}")
    else:
        print("Failed to train PPM model")

    if ngram_success:
        print(f"Word n-gram model trained and saved to {args.ngram_output}")
    else:
        print("Failed to train word n-gram model")


if __name__ == "__main__":
    main()
