#!/usr/bin/env python3
"""
Filter the AACConversations dataset to include only English (en-GB) data.

This script loads the AACConversations dataset, filters it to include only
English data, and saves the filtered dataset for evaluation.
"""

import os
import sys
import logging
import json
import argparse
from typing import Any, Dict, List
from collections import defaultdict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the utils module
from scripts.evaluation.utils import load_aac_conversations, group_by_conversation

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
        filtered = dataset.filter(lambda example: example.get("language_code") == language_code)
        logger.info(f"Filtered dataset to {len(filtered)} examples in language {language_code}")
        return filtered
    except Exception as e:
        logger.error(f"Error filtering dataset by language: {e}")
        return dataset


def save_filtered_conversations(conversations: Dict[str, List[Dict[str, Any]]], output_path: str) -> bool:
    """
    Save filtered conversations to a file.
    
    Args:
        conversations: Dictionary mapping conversation IDs to lists of turns
        output_path: Path to save the filtered conversations
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Convert to a serializable format
        serializable_conversations = {}
        for conversation_id, turns in conversations.items():
            serializable_conversations[conversation_id] = [dict(turn) for turn in turns]
        
        # Save the conversations
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_conversations, f, indent=2)
        
        logger.info(f"Saved {len(conversations)} filtered conversations to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving filtered conversations: {e}")
        return False


def analyze_english_data(dataset: Any) -> None:
    """
    Analyze the English data in the dataset.
    
    Args:
        dataset: The filtered dataset
    """
    if dataset is None or len(dataset) == 0:
        logger.error("No data to analyze")
        return
    
    # Group by conversation
    conversations = group_by_conversation(dataset)
    
    # Analyze conversation lengths
    conversation_lengths = [len(turns) for turns in conversations.values()]
    avg_length = sum(conversation_lengths) / len(conversation_lengths) if conversation_lengths else 0
    
    print("\n=== English Data Analysis ===")
    print(f"Total examples: {len(dataset)}")
    print(f"Number of conversations: {len(conversations)}")
    print(f"Average conversation length: {avg_length:.2f} turns")
    print(f"Min conversation length: {min(conversation_lengths) if conversation_lengths else 0} turns")
    print(f"Max conversation length: {max(conversation_lengths) if conversation_lengths else 0} turns")
    
    # Analyze speakers
    speakers = defaultdict(int)
    for example in dataset:
        if "speaker" in example:
            speakers[example["speaker"]] += 1
    
    print("\nTop Speakers:")
    for speaker, count in sorted(speakers.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {speaker}: {count}")
    
    # Analyze noise types and levels
    noise_types = set()
    noise_levels = set()
    
    for example in dataset:
        # Check for noise types
        for key in example:
            if key.startswith("noisy_"):
                parts = key.split("_")
                if len(parts) >= 2:
                    noise_type = parts[1]
                    noise_types.add(noise_type)
                if len(parts) >= 3:
                    noise_level = parts[2]
                    noise_levels.add(noise_level)
    
    print("\nNoise Types:")
    for noise_type in sorted(noise_types):
        print(f"  {noise_type}")
    
    print("\nNoise Levels:")
    for noise_level in sorted(noise_levels):
        print(f"  {noise_level}")
    
    # Sample a few examples
    print("\nSample Examples:")
    for i, example in enumerate(dataset.select(range(min(5, len(dataset))))):
        print(f"\nExample {i+1}:")
        print(f"  Speaker: {example.get('speaker', 'Unknown')}")
        print(f"  Intended: {example.get('utterance_intended', '')}")
        
        # Print noisy utterances for different types and levels
        for noise_type in noise_types:
            for noise_level in noise_levels:
                key = f"noisy_{noise_type}_{noise_level}"
                if key in example and example[key]:
                    print(f"  {key}: {example[key]}")


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Filter the AACConversations dataset to include only English (en-GB) data."
    )
    
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use",
    )
    
    parser.add_argument(
        "--language-code",
        type=str,
        default="en-GB",
        help="Language code to filter by",
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="data/english_conversations.json",
        help="Path to save the filtered conversations",
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
    
    # Filter the dataset
    filtered_dataset = filter_english_data(dataset, args.language_code)
    
    if filtered_dataset is None or len(filtered_dataset) == 0:
        logger.error(f"No data found for language {args.language_code}. Exiting.")
        return
    
    # Analyze the filtered dataset
    analyze_english_data(filtered_dataset)
    
    # Group by conversation
    conversations = group_by_conversation(filtered_dataset)
    
    # Save the filtered conversations
    if args.output:
        save_filtered_conversations(conversations, args.output)


if __name__ == "__main__":
    main()
