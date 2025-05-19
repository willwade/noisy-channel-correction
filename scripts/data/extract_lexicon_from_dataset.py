#!/usr/bin/env python3
"""
Extract a domain-specific lexicon from the AACConversations dataset.

This script extracts unique words from the AACConversations dataset,
focusing on both the minimally_corrected and fully_corrected fields.
It filters out non-words, numbers, and special characters.
"""

import os
import sys
import argparse
import logging
import re
from collections import Counter
from typing import Set, Dict, List, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import utilities
from scripts.evaluation.utils import load_aac_conversations, resolve_path


def extract_words(text: str) -> List[str]:
    """
    Extract words from text.

    Args:
        text: The text to extract words from

    Returns:
        List of words
    """
    # Convert to lowercase
    text = text.lower()

    # Replace punctuation with spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Split into words
    words = text.split()

    # Filter out non-words, numbers, and special characters
    words = [word for word in words if re.match(r"^[a-z]+$", word)]

    return words


def extract_lexicon_from_dataset(
    dataset: Any,
    min_frequency: int = 1,
    include_speakers: bool = True,
    language_code: str = None,
) -> Dict[str, int]:
    """
    Extract a lexicon from the dataset.

    Args:
        dataset: The dataset to extract words from
        min_frequency: Minimum frequency for a word to be included
        include_speakers: Whether to include speaker utterances
        language_code: Language code to filter by (e.g., 'en-GB')

    Returns:
        Dictionary mapping words to their frequencies
    """
    word_counter = Counter()
    processed_examples = 0
    filtered_examples = 0

    # Process each example
    for example in dataset:
        # Get the fields
        minimally_corrected = example.get("minimally_corrected", "")
        fully_corrected = example.get("fully_corrected", "")
        intended = example.get("utterance_intended", "")
        speaker = example.get("speaker", "")
        lang_code = example.get("language_code", "")

        # Filter by language code if specified
        if language_code and lang_code and language_code.lower() != lang_code.lower():
            filtered_examples += 1
            continue

        # Skip non-AAC user utterances if specified
        if not include_speakers and "aac" not in speaker.lower():
            continue

        processed_examples += 1

        # Extract words from each field
        if minimally_corrected:
            words = extract_words(minimally_corrected)
            word_counter.update(words)

        if fully_corrected:
            words = extract_words(fully_corrected)
            word_counter.update(words)

        if intended and not minimally_corrected and not fully_corrected:
            words = extract_words(intended)
            word_counter.update(words)

    # Filter by frequency
    if min_frequency > 1:
        word_counter = Counter(
            {
                word: count
                for word, count in word_counter.items()
                if count >= min_frequency
            }
        )

    logger.info(
        f"Processed {processed_examples} examples, filtered out {filtered_examples} examples"
    )
    return word_counter


def save_lexicon(
    lexicon: Dict[str, int], output_path: str, include_frequency: bool = False
) -> bool:
    """
    Save the lexicon to a file.

    Args:
        lexicon: Dictionary mapping words to their frequencies
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
        sorted_words = sorted(lexicon.items(), key=lambda x: x[1], reverse=True)

        # Save the lexicon
        with open(resolved_path, "w", encoding="utf-8") as f:
            if include_frequency:
                for word, freq in sorted_words:
                    f.write(f"{word}\t{freq}\n")
            else:
                for word, _ in sorted_words:
                    f.write(f"{word}\n")

        logger.info(f"Saved lexicon with {len(lexicon)} words to {resolved_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving lexicon to {output_path}: {e}")
        return False


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Extract a lexicon from the AACConversations dataset."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="willwade/AACConversations",
        help="Dataset to extract words from",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data/aac_lexicon.txt",
        help="Path to save the lexicon",
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use",
    )

    parser.add_argument(
        "--language-code",
        type=str,
        default="en-GB",
        help="Language code to filter by (e.g., 'en-GB')",
    )

    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for a word to be included",
    )

    parser.add_argument(
        "--include-frequency",
        action="store_true",
        help="Include frequency information in the output",
    )

    parser.add_argument(
        "--include-speakers",
        action="store_true",
        help="Include words from all speakers (not just AAC users)",
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

    # Load the dataset
    dataset = load_aac_conversations(
        split=args.dataset_split,
        cache_dir=args.cache_dir,
        use_auth_token=not args.no_token,
    )

    if dataset is None:
        logger.error("Failed to load dataset. Exiting.")
        return

    # Extract the lexicon
    logger.info(f"Extracting lexicon from {args.dataset} ({args.dataset_split} split)")
    logger.info(f"Filtering by language code: {args.language_code}")

    lexicon = extract_lexicon_from_dataset(
        dataset,
        min_frequency=args.min_frequency,
        include_speakers=args.include_speakers,
        language_code=args.language_code,
    )

    # Save the lexicon
    output_path = args.output
    if args.language_code:
        # Add language code to output path
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{args.language_code.lower().replace('-', '_')}{ext}"

    success = save_lexicon(
        lexicon,
        output_path,
        include_frequency=args.include_frequency,
    )

    if success:
        logger.info(f"Successfully extracted {len(lexicon)} words.")
        logger.info(
            f"Top 20 words: {', '.join([word for word, _ in lexicon.most_common(20)])}"
        )
    else:
        logger.error("Failed to save lexicon.")


if __name__ == "__main__":
    main()
