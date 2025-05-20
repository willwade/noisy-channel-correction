#!/usr/bin/env python3
"""
Create keyboard-specific lexicons for different keyboard layouts.

This script creates separate lexicons for different keyboard layouts
(qwerty, abc, frequency), optimizing each lexicon for the specific
keyboard layout and adding keyboard-specific word frequency information.
"""

import os
import sys
import argparse
import logging
from collections import Counter
from typing import Dict, Set, Tuple, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import utilities
from scripts.evaluation.utils import load_aac_conversations


def load_lexicon(file_path: str) -> Tuple[Set[str], Dict[str, int]]:
    """
    Load a lexicon from a file.

    Args:
        file_path: Path to the lexicon file

    Returns:
        Tuple of (set of words, dictionary mapping words to frequencies)
    """
    words = set()
    frequencies = {}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Check if the line contains frequency information
                parts = line.split("\t")
                if len(parts) == 2:
                    word, freq = parts
                    try:
                        freq = int(freq)
                        frequencies[word] = freq
                    except ValueError:
                        # If frequency is not a valid integer, just add the word
                        words.add(word)
                else:
                    words.add(line)

        logger.info(f"Loaded {len(words) + len(frequencies)} words from {file_path}")
        return words, frequencies
    except Exception as e:
        logger.error(f"Error loading lexicon from {file_path}: {e}")
        return set(), {}


def extract_keyboard_specific_frequencies(
    dataset: Any, lexicon: Set[str]
) -> Dict[str, Dict[str, int]]:
    """
    Extract keyboard-specific word frequencies from the dataset.

    Args:
        dataset: The dataset to extract frequencies from
        lexicon: Set of words to consider

    Returns:
        Dictionary mapping keyboard layouts to word frequency dictionaries
    """
    # Initialize frequency counters for each keyboard layout
    keyboard_frequencies = {
        "qwerty": Counter(),
        "abc": Counter(),
        "frequency": Counter(),
    }

    # Process each example
    for example in dataset:
        # Get the minimally corrected utterance
        minimally_corrected = example.get("minimally_corrected", "")
        if not minimally_corrected or not isinstance(minimally_corrected, str):
            continue

        # Split into words
        words = minimally_corrected.lower().split()

        # Filter words
        words = [word for word in words if word in lexicon]

        # Add words to all keyboard layouts with the same frequency
        # This is a simplified approach that doesn't rely on noisy utterances
        for word in words:
            for keyboard in keyboard_frequencies:
                keyboard_frequencies[keyboard][word] += 1

    return keyboard_frequencies


def create_keyboard_lexicons(
    input_path: str,
    output_dir: str,
    dataset: Any = None,
    min_frequency: int = 1,
    include_frequency: bool = True,
) -> bool:
    """
    Create keyboard-specific lexicons.

    Args:
        input_path: Path to the input lexicon file
        output_dir: Directory to save the keyboard-specific lexicons
        dataset: The dataset to extract frequencies from
        min_frequency: Minimum frequency for a word to be included
        include_frequency: Whether to include frequency information

    Returns:
        True if successful, False otherwise
    """
    # Load the input lexicon
    words, frequencies = load_lexicon(input_path)

    # Create keyboard-specific frequencies
    keyboard_layouts = ["qwerty", "abc", "frequency"]
    keyboard_frequencies = {}

    for keyboard in keyboard_layouts:
        # Start with the original frequencies
        keyboard_frequencies[keyboard] = dict(frequencies)

        # If we have a dataset, try to extract keyboard-specific frequencies
        if dataset is not None:
            try:
                # Extract keyboard-specific frequencies
                dataset_frequencies = extract_keyboard_specific_frequencies(
                    dataset, words
                )

                # Update with dataset-specific frequencies if available
                if keyboard in dataset_frequencies:
                    for word, freq in dataset_frequencies[keyboard].items():
                        if freq > 0:  # Only update if we have a positive frequency
                            keyboard_frequencies[keyboard][word] = freq
            except Exception as e:
                logger.warning(f"Error extracting frequencies for {keyboard}: {e}")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save each keyboard-specific lexicon
    success = True
    for keyboard, freq_dict in keyboard_frequencies.items():
        # Filter by frequency
        if min_frequency > 1:
            freq_dict = {
                word: freq for word, freq in freq_dict.items() if freq >= min_frequency
            }

        # Sort words by frequency (descending)
        sorted_words = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)

        # Save the lexicon
        output_path = os.path.join(output_dir, f"{keyboard}_lexicon.txt")
        try:
            with open(output_path, "w", encoding="utf-8") as f:
                if include_frequency:
                    for word, freq in sorted_words:
                        f.write(f"{word}\t{freq}\n")
                else:
                    for word, _ in sorted_words:
                        f.write(f"{word}\n")

            logger.info(
                f"Saved {keyboard} lexicon with {len(freq_dict)} words to {output_path}"
            )
        except Exception as e:
            logger.error(f"Error saving {keyboard} lexicon to {output_path}: {e}")
            success = False

    return success


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Create keyboard-specific lexicons for different keyboard layouts."
    )

    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the input lexicon file",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/keyboard_lexicons",
        help="Directory to save the keyboard-specific lexicons",
    )

    parser.add_argument(
        "--use-dataset",
        action="store_true",
        help="Use the AACConversations dataset to extract keyboard frequencies",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="willwade/AACConversations",
        help="Dataset to extract frequencies from",
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use",
    )

    parser.add_argument(
        "--min-frequency",
        type=int,
        default=1,
        help="Minimum frequency for a word to be included",
    )

    parser.add_argument(
        "--include-frequency",
        action="store_true",
        help="Include frequency information in the output",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache the dataset",
    )

    parser.add_argument(
        "--no-token",
        action="store_true",
        help="Don't use the Hugging Face auth token",
    )

    args = parser.parse_args()

    # Load the dataset if requested
    dataset = None
    if args.use_dataset:
        logger.info(f"Loading dataset {args.dataset} ({args.dataset_split} split)")
        dataset = load_aac_conversations(
            split=args.dataset_split,
            cache_dir=args.cache_dir,
            use_auth_token=not args.no_token,
        )

        if dataset is None:
            logger.error(
                "Failed to load dataset. Using input lexicon frequencies only."
            )

    # Create keyboard-specific lexicons
    logger.info(f"Creating keyboard-specific lexicons from {args.input}")
    success = create_keyboard_lexicons(
        args.input,
        args.output_dir,
        dataset=dataset,
        min_frequency=args.min_frequency,
        include_frequency=args.include_frequency,
    )

    if success:
        logger.info(
            f"Successfully created keyboard-specific lexicons in {args.output_dir}"
        )
    else:
        logger.error("Failed to create some keyboard-specific lexicons.")


if __name__ == "__main__":
    main()
