#!/usr/bin/env python3
"""
Train models using the AACConversations dataset.

This script extracts text from the AACConversations dataset and uses it
to train PPM and word n-gram models for better prediction and correction.
"""

import os
import sys
import argparse
import logging
from typing import Any, Optional

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import the model training functions
from train_ppm_model import train_ppm_model, train_word_ngram_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import datasets
try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.error(
        "datasets library not available. Please install it with 'uv pip install datasets'"
    )

# Get Hugging Face token from environment variable
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)


def load_aac_conversations(
    split: str = "train", cache_dir: Optional[str] = None, use_auth_token: bool = True
) -> Any:
    """
    Load the AACConversations dataset from Hugging Face.

    Args:
        split: Dataset split to load ('train', 'validation', or 'test')
        cache_dir: Directory to cache the dataset
        use_auth_token: Whether to use the Hugging Face auth token

    Returns:
        The loaded dataset or None if loading failed
    """
    if not DATASETS_AVAILABLE:
        logger.error("Cannot load dataset: datasets library not available")
        return None

    try:
        # Load the dataset
        # Use the token from the environment variable
        dataset = load_dataset(
            "willwade/AACConversations",
            split=split,
            cache_dir=cache_dir,
            token=HUGGINGFACE_TOKEN if use_auth_token else None,
        )
        logger.info(
            f"Loaded AACConversations dataset ({split} split) with {len(dataset)} examples"
        )
        return dataset
    except Exception as e:
        logger.error(f"Error loading AACConversations dataset: {e}")
        return None


def filter_english_data(dataset: Any, language_code: str = "en-GB") -> Any:
    """
    Filter the dataset to include only data in the specified language.

    Args:
        dataset: The dataset to filter
        language_code: The language code to filter by

    Returns:
        The filtered dataset
    """
    if dataset is None:
        logger.error("Cannot filter dataset: dataset is None")
        return None

    try:
        # Filter by language code
        filtered = dataset.filter(
            lambda example: example.get("language_code") == language_code
        )
        logger.info(
            f"Filtered dataset to {len(filtered)} examples in language {language_code}"
        )
        return filtered
    except Exception as e:
        logger.error(f"Error filtering dataset by language: {e}")
        return dataset


def extract_text_from_dataset(dataset: Any, field: str = "fully_corrected") -> str:
    """
    Extract text from the dataset for training.

    Args:
        dataset: The dataset to extract text from
        field: The field to extract (e.g., 'fully_corrected', 'intended')

    Returns:
        Extracted text as a single string
    """
    if dataset is None:
        logger.error("Cannot extract text: dataset is None")
        return ""

    try:
        # Extract text from the specified field
        texts = []
        for example in dataset:
            if field in example and example[field]:
                texts.append(example[field])

        # Join the texts with newlines
        full_text = "\n".join(texts)
        logger.info(
            f"Extracted {len(texts)} text samples ({len(full_text)} characters)"
        )
        return full_text
    except Exception as e:
        logger.error(f"Error extracting text from dataset: {e}")
        return ""


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train models using the AACConversations dataset."
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        choices=["train", "test", "all"],
        help="Dataset split to use (all combines train and test)",
    )

    parser.add_argument(
        "--language-code",
        type=str,
        default="en-GB",
        help="Language code to filter by",
    )

    parser.add_argument(
        "--field",
        type=str,
        default="fully_corrected",
        choices=["fully_corrected", "intended", "noisy", "noisy_minimal"],
        help="Field to extract text from",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="models",
        help="Directory to save the models (default: models)",
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

    # Check if datasets library is available
    if not DATASETS_AVAILABLE:
        logger.error(
            "datasets library not available. Please install it with 'uv pip install datasets'"
        )
        return

    # Check if Hugging Face token is set
    if not HUGGINGFACE_TOKEN and not args.no_token:
        logger.warning(
            "HUGGINGFACE_TOKEN environment variable not set. Using public access."
        )

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define output file paths with language code suffix
    language_suffix = args.language_code.lower().replace("-", "_")
    ppm_output = os.path.join(args.output_dir, f"ppm_model_{language_suffix}.pkl")
    ngram_output = os.path.join(
        args.output_dir, f"word_ngram_model_{language_suffix}.pkl"
    )

    # Load the dataset
    if args.dataset_split == "all":
        # Load both train and test splits
        train_dataset = load_aac_conversations(
            split="train",
            cache_dir=args.cache_dir,
            use_auth_token=not args.no_token,
        )

        test_dataset = load_aac_conversations(
            split="test",
            cache_dir=args.cache_dir,
            use_auth_token=not args.no_token,
        )

        if train_dataset is None or test_dataset is None:
            logger.error("Failed to load datasets. Exiting.")
            return

        # Combine the datasets
        from datasets import concatenate_datasets

        dataset = concatenate_datasets([train_dataset, test_dataset])
        logger.info(f"Combined train and test datasets with {len(dataset)} examples")
    else:
        # Load a single split
        dataset = load_aac_conversations(
            split=args.dataset_split,
            cache_dir=args.cache_dir,
            use_auth_token=not args.no_token,
        )

        if dataset is None:
            logger.error("Failed to load dataset. Exiting.")
            return

    # Filter the dataset by language
    filtered_dataset = filter_english_data(dataset, args.language_code)

    if filtered_dataset is None or len(filtered_dataset) == 0:
        logger.error(f"No data found for language {args.language_code}. Exiting.")
        return

    # Extract text from the dataset
    training_text = extract_text_from_dataset(filtered_dataset, args.field)

    if not training_text:
        logger.error("Failed to extract text from dataset. Exiting.")
        return

    # Save the training text to a file
    training_text_path = os.path.join(
        args.output_dir, f"training_text_{language_suffix}.txt"
    )
    with open(training_text_path, "w", encoding="utf-8") as f:
        f.write(training_text)
    logger.info(f"Saved training text to {training_text_path}")

    # Train the PPM model
    logger.info(
        f"Training {args.language_code} PPM model with order {args.ppm_order}..."
    )
    ppm_success = train_ppm_model(training_text_path, ppm_output, args.ppm_order)

    # Train the word n-gram model
    logger.info(
        f"Training {args.language_code} word n-gram model with order {args.ngram_order}..."
    )
    ngram_success = train_word_ngram_model(
        training_text_path, ngram_output, args.ngram_order, args.smoothing
    )

    # Print summary
    print(f"\n=== {args.language_code} Model Training Summary ===")
    if ppm_success:
        print(f"✓ PPM model trained and saved to {ppm_output}")
    else:
        print("✗ Failed to train PPM model")

    if ngram_success:
        print(f"✓ Word n-gram model trained and saved to {ngram_output}")
    else:
        print("✗ Failed to train word n-gram model")

    print("\nTo use these models with the web demo, run:")
    print(
        f"uv run demo/web/app.py "
        f"--ppm-model {ppm_output} "
        f"--word-ngram-model {ngram_output}"
    )


if __name__ == "__main__":
    main()
