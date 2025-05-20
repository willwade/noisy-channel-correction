#!/usr/bin/env python3
"""
Script to enhance the en-GB lexicon by merging it with a standard English dictionary.

This script downloads a comprehensive English dictionary and merges it with the
existing en-GB AAC lexicon to create a more complete lexicon for correction.
"""

import argparse
import logging
import requests
from typing import Set, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def load_lexicon(file_path: str) -> Dict[str, int]:
    """
    Load a lexicon from a file.
    
    Args:
        file_path: Path to the lexicon file
        
    Returns:
        Dictionary mapping words to frequencies
    """
    lexicon = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word = parts[0].lower()
                    freq = int(parts[1])
                    lexicon[word] = freq
                elif len(parts) == 1:
                    word = parts[0].lower()
                    lexicon[word] = 1  # Default frequency
        
        logger.info(f"Loaded {len(lexicon)} words from {file_path}")
        return lexicon
    except Exception as e:
        logger.error(f"Error loading lexicon from {file_path}: {e}")
        return {}

def download_english_dictionary(url: str) -> Set[str]:
    """
    Download a comprehensive English dictionary.
    
    Args:
        url: URL to download the dictionary from
        
    Returns:
        Set of English words
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Parse the response
        words = set(word.lower() for word in response.text.splitlines() if word.strip())
        
        logger.info(f"Downloaded {len(words)} words from {url}")
        return words
    except Exception as e:
        logger.error(f"Error downloading dictionary from {url}: {e}")
        return set()

def merge_lexicons(aac_lexicon: Dict[str, int], english_dict: Set[str]) -> Dict[str, int]:
    """
    Merge the AAC lexicon with the English dictionary.
    
    Args:
        aac_lexicon: Dictionary mapping AAC words to frequencies
        english_dict: Set of English words
        
    Returns:
        Merged lexicon with frequencies
    """
    # Start with the AAC lexicon
    merged = aac_lexicon.copy()
    
    # Add words from the English dictionary that aren't in the AAC lexicon
    new_words = 0
    for word in english_dict:
        if word not in merged and len(word) >= 2:  # Skip single-letter words
            merged[word] = 1  # Default frequency
            new_words += 1
    
    logger.info(f"Added {new_words} new words from the English dictionary")
    logger.info(f"Merged lexicon contains {len(merged)} words")
    
    return merged

def save_lexicon(lexicon: Dict[str, int], output_path: str) -> None:
    """
    Save the lexicon to a file.
    
    Args:
        lexicon: Dictionary mapping words to frequencies
        output_path: Path to save the lexicon to
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for word, freq in sorted(lexicon.items()):
                f.write(f"{word}\t{freq}\n")
        
        logger.info(f"Saved {len(lexicon)} words to {output_path}")
    except Exception as e:
        logger.error(f"Error saving lexicon to {output_path}: {e}")

def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Enhance the en-GB lexicon by merging it with a standard English dictionary."
    )
    
    parser.add_argument(
        "--aac-lexicon",
        type=str,
        default="data/aac_lexicon_en_gb.txt",
        help="Path to the AAC lexicon file",
    )
    
    parser.add_argument(
        "--dictionary-url",
        type=str,
        default="https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt",
        help="URL to download the English dictionary from",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/enhanced_lexicon_en_gb.txt",
        help="Path to save the enhanced lexicon to",
    )
    
    args = parser.parse_args()
    
    # Load the AAC lexicon
    aac_lexicon = load_lexicon(args.aac_lexicon)
    
    # Download the English dictionary
    english_dict = download_english_dictionary(args.dictionary_url)
    
    # Merge the lexicons
    merged_lexicon = merge_lexicons(aac_lexicon, english_dict)
    
    # Save the merged lexicon
    save_lexicon(merged_lexicon, args.output)

if __name__ == "__main__":
    main()
