#!/usr/bin/env python3
"""
Merge multiple lexicons into a single enhanced lexicon.

This script merges multiple lexicon files, removing duplicates and
ensuring consistent formatting. It can also add frequency information
if available.
"""

import os
import sys
import argparse
import logging
from collections import Counter
from typing import Dict, List, Set, Tuple

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import utilities
from scripts.evaluation.utils import resolve_path


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


def merge_lexicons(
    input_files: List[str], min_frequency: int = 1
) -> Tuple[Set[str], Dict[str, int]]:
    """
    Merge multiple lexicons.

    Args:
        input_files: List of lexicon file paths
        min_frequency: Minimum frequency for a word to be included

    Returns:
        Tuple of (set of words, dictionary mapping words to frequencies)
    """
    all_words = set()
    all_frequencies = Counter()

    for file_path in input_files:
        words, frequencies = load_lexicon(file_path)
        all_words.update(words)
        all_frequencies.update(frequencies)

    # Add words without frequency information
    for word in all_words:
        if word not in all_frequencies:
            all_frequencies[word] = 1

    # Filter by frequency
    if min_frequency > 1:
        all_frequencies = Counter(
            {word: freq for word, freq in all_frequencies.items() if freq >= min_frequency}
        )

    return set(all_frequencies.keys()), all_frequencies


def save_lexicon(
    words: Set[str],
    frequencies: Dict[str, int],
    output_path: str,
    include_frequency: bool = False,
) -> bool:
    """
    Save the merged lexicon to a file.

    Args:
        words: Set of words
        frequencies: Dictionary mapping words to frequencies
        output_path: Path to save the lexicon
        include_frequency: Whether to include frequency information

    Returns:
        True if successful, False otherwise
    """
    try:
        # Resolve the path
        resolved_path = resolve_path(output_path)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(resolved_path)), exist_ok=True)

        # Sort words by frequency (descending)
        sorted_words = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)

        # Save the lexicon
        with open(resolved_path, "w", encoding="utf-8") as f:
            if include_frequency:
                for word, freq in sorted_words:
                    f.write(f"{word}\t{freq}\n")
            else:
                for word, _ in sorted_words:
                    f.write(f"{word}\n")

        logger.info(f"Saved merged lexicon with {len(frequencies)} words to {resolved_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving lexicon to {output_path}: {e}")
        return False


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Merge multiple lexicons into a single enhanced lexicon."
    )

    parser.add_argument(
        "--input",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the input lexicon files",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/enhanced_lexicon.txt",
        help="Path to save the merged lexicon",
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

    args = parser.parse_args()

    # Merge the lexicons
    logger.info(f"Merging {len(args.input)} lexicons")
    words, frequencies = merge_lexicons(
        args.input,
        min_frequency=args.min_frequency,
    )

    # Save the merged lexicon
    success = save_lexicon(
        words,
        frequencies,
        args.output,
        include_frequency=args.include_frequency,
    )

    if success:
        logger.info(f"Successfully merged {len(frequencies)} words.")
        logger.info(
            f"Top 20 words: {', '.join([word for word, _ in Counter(frequencies).most_common(20)])}"
        )
    else:
        logger.error("Failed to save merged lexicon.")


if __name__ == "__main__":
    main()
