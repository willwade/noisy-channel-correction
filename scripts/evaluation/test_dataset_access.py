#!/usr/bin/env python3
"""
Test access to the AACConversations dataset.

This script tests if we can access the AACConversations dataset from Hugging Face
using the provided token.
"""

import os
import sys
import logging
from typing import Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check if datasets library is available
try:
    from datasets import load_dataset
    DATASETS_AVAILABLE = True
except ImportError:
    logger.warning("datasets library not available. Install with: pip install datasets")
    DATASETS_AVAILABLE = False

# Get the Hugging Face token from environment variable
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")
if not HUGGINGFACE_TOKEN:
    logger.warning("HUGGINGFACE_TOKEN environment variable not set")


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


def explore_dataset(dataset: Any) -> None:
    """
    Explore the dataset structure and content.

    Args:
        dataset: The dataset to explore
    """
    if dataset is None:
        logger.error("No dataset to explore")
        return

    # Print dataset info
    print("\n=== Dataset Info ===")
    print(f"Number of examples: {len(dataset)}")
    
    # Print features (columns)
    print("\n=== Dataset Features ===")
    features = dataset.features
    for feature_name, feature_type in features.items():
        print(f"  {feature_name}: {feature_type}")
    
    # Print a few examples
    print("\n=== Sample Examples ===")
    for i, example in enumerate(dataset.select(range(min(3, len(dataset))))):
        print(f"\nExample {i+1}:")
        for key, value in example.items():
            # Truncate long values
            if isinstance(value, str) and len(value) > 100:
                value = value[:100] + "..."
            print(f"  {key}: {value}")
    
    # Check for conversation structure
    print("\n=== Conversation Structure ===")
    conversation_ids = set()
    turn_numbers = set()
    speakers = set()
    
    # Sample up to 100 examples to analyze structure
    for example in dataset.select(range(min(100, len(dataset)))):
        if "conversation_id" in example:
            conversation_ids.add(example["conversation_id"])
        if "turn_number" in example:
            turn_numbers.add(example["turn_number"])
        if "speaker" in example:
            speakers.add(example["speaker"])
    
    print(f"Number of unique conversation IDs: {len(conversation_ids)}")
    print(f"Turn numbers found: {sorted(list(turn_numbers))[:10]}...")
    print(f"Speakers found: {speakers}")
    
    # Check for noise types and levels
    print("\n=== Noise Information ===")
    noise_types = set()
    noise_levels = set()
    
    for example in dataset.select(range(min(100, len(dataset)))):
        if "noise_type" in example:
            noise_types.add(example["noise_type"])
        if "noise_level" in example:
            noise_levels.add(example["noise_level"])
    
    print(f"Noise types found: {noise_types}")
    print(f"Noise levels found: {noise_levels}")
    
    # Check for intended and noisy utterances
    print("\n=== Utterance Information ===")
    has_intended = 0
    has_noisy = 0
    
    for example in dataset.select(range(min(100, len(dataset)))):
        if "utterance_intended" in example and example["utterance_intended"]:
            has_intended += 1
        if "noisy_utterance" in example and example["noisy_utterance"]:
            has_noisy += 1
    
    print(f"Examples with intended utterances: {has_intended}/100")
    print(f"Examples with noisy utterances: {has_noisy}/100")


def main():
    """Main function."""
    # Check if the Hugging Face token is set
    if not HUGGINGFACE_TOKEN:
        logger.error("HUGGINGFACE_TOKEN environment variable not set. Exiting.")
        return

    # Load the dataset
    for split in ["train", "validation", "test"]:
        dataset = load_aac_conversations(split=split)
        if dataset:
            print(f"\n=== Exploring {split} split ===")
            explore_dataset(dataset)
        else:
            logger.error(f"Failed to load {split} split")


if __name__ == "__main__":
    main()
