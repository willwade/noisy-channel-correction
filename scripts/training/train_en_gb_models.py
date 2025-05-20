#!/usr/bin/env python3
"""
Script to train better language models for en-GB.

This script trains PPM and word n-gram models on a larger corpus of British English text.
"""

import os
import sys
import logging
import argparse
import requests
import tempfile
import zipfile
from typing import List

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import the PPM model
from lib.corrector.enhanced_ppm_predictor import EnhancedPPMPredictor

# Import the word n-gram model
from lib.corrector.word_ngram_model import WordNGramModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def download_corpus(url: str, output_dir: str) -> str:
    """
    Download a corpus from a URL.
    
    Args:
        url: URL to download the corpus from
        output_dir: Directory to save the corpus to
        
    Returns:
        Path to the downloaded corpus
    """
    try:
        # Create a temporary file to download to
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as temp_file:
            temp_path = temp_file.name
        
        # Download the corpus
        logger.info(f"Downloading corpus from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Save the corpus to the temporary file
        with open(temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract the corpus
        logger.info(f"Extracting corpus to {output_dir}")
        with zipfile.ZipFile(temp_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Remove the temporary file
        os.remove(temp_path)
        
        # Return the path to the extracted corpus
        return output_dir
    except Exception as e:
        logger.error(f"Error downloading corpus from {url}: {e}")
        return ""

def load_corpus(corpus_dir: str) -> List[str]:
    """
    Load a corpus from a directory.
    
    Args:
        corpus_dir: Directory containing the corpus
        
    Returns:
        List of sentences in the corpus
    """
    sentences = []
    
    try:
        # Walk through the directory
        for root, _, files in os.walk(corpus_dir):
            for file in files:
                if file.endswith(".txt"):
                    # Load the file
                    with open(os.path.join(root, file), "r", encoding="utf-8", errors="ignore") as f:
                        # Split the file into sentences
                        for line in f:
                            # Split the line into sentences
                            for sentence in line.strip().split("."):
                                if sentence.strip():
                                    sentences.append(sentence.strip())
        
        logger.info(f"Loaded {len(sentences)} sentences from {corpus_dir}")
        return sentences
    except Exception as e:
        logger.error(f"Error loading corpus from {corpus_dir}: {e}")
        return []

def train_ppm_model(sentences: List[str], output_path: str) -> bool:
    """
    Train a PPM model on a corpus.
    
    Args:
        sentences: List of sentences to train on
        output_path: Path to save the model to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize the PPM model
        ppm_model = EnhancedPPMPredictor()
        
        # Train the model
        logger.info(f"Training PPM model on {len(sentences)} sentences")
        ppm_model.train(sentences)
        
        # Save the model
        logger.info(f"Saving PPM model to {output_path}")
        ppm_model.save_model(output_path)
        
        logger.info(f"PPM model trained successfully. Vocabulary size: {len(ppm_model.vocabulary)}")
        return True
    except Exception as e:
        logger.error(f"Error training PPM model: {e}")
        return False

def train_word_ngram_model(sentences: List[str], output_path: str) -> bool:
    """
    Train a word n-gram model on a corpus.
    
    Args:
        sentences: List of sentences to train on
        output_path: Path to save the model to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Initialize the word n-gram model
        word_ngram_model = WordNGramModel()
        
        # Train the model
        logger.info(f"Training word n-gram model on {len(sentences)} sentences")
        word_ngram_model.train(sentences)
        
        # Save the model
        logger.info(f"Saving word n-gram model to {output_path}")
        word_ngram_model.save(output_path)
        
        logger.info(f"Word n-gram model trained successfully. Vocabulary size: {len(word_ngram_model.vocabulary)}")
        return True
    except Exception as e:
        logger.error(f"Error training word n-gram model: {e}")
        return False

def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train better language models for en-GB."
    )
    
    parser.add_argument(
        "--corpus-url",
        type=str,
        default="https://www.gutenberg.org/files/10/10-0.zip",
        help="URL to download the corpus from",
    )
    
    parser.add_argument(
        "--corpus-dir",
        type=str,
        default="data/corpus/en_gb",
        help="Directory to save the corpus to",
    )
    
    parser.add_argument(
        "--ppm-output",
        type=str,
        default="models/ppm_model_en_gb.pkl",
        help="Path to save the PPM model to",
    )
    
    parser.add_argument(
        "--word-ngram-output",
        type=str,
        default="models/word_ngram_model_en_gb.pkl",
        help="Path to save the word n-gram model to",
    )
    
    args = parser.parse_args()
    
    # Download the corpus
    corpus_dir = download_corpus(args.corpus_url, args.corpus_dir)
    
    if not corpus_dir:
        logger.error("Failed to download corpus")
        return
    
    # Load the corpus
    sentences = load_corpus(corpus_dir)
    
    if not sentences:
        logger.error("Failed to load corpus")
        return
    
    # Train the PPM model
    train_ppm_model(sentences, args.ppm_output)
    
    # Train the word n-gram model
    train_word_ngram_model(sentences, args.word_ngram_output)

if __name__ == "__main__":
    main()
