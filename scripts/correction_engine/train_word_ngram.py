#!/usr/bin/env python3
"""
Train a word-level n-gram model for context-aware correction.

This script trains a word-level n-gram model on a given text corpus
and saves it to a file for later use.
"""

import os
import sys
import logging
import argparse

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the word n-gram model
from module4.ngram.word_ngram_model import WordNGramModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train a word-level n-gram model for context-aware correction."
    )
    
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to the input text file"
    )
    
    parser.add_argument(
        "--output", "-o", type=str, default="models/word_ngram_model.pkl",
        help="Path to save the model (default: models/word_ngram_model.pkl)"
    )
    
    parser.add_argument(
        "--max-order", "-m", type=int, default=3,
        help="Maximum n-gram order (e.g., 3 for trigrams) (default: 3)"
    )
    
    parser.add_argument(
        "--smoothing", "-s", type=str, default="kneser_ney",
        choices=["kneser_ney", "laplace", "witten_bell"],
        help="Smoothing method to use (default: kneser_ney)"
    )
    
    args = parser.parse_args()
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Train the model
    success = train_word_ngram_model(
        args.input, args.output, args.max_order, args.smoothing
    )
    
    if success:
        print(f"Word n-gram model trained and saved to {args.output}")
    else:
        print("Failed to train word n-gram model")
        sys.exit(1)


if __name__ == "__main__":
    main()
