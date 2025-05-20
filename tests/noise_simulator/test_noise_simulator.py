#!/usr/bin/env python3
"""
Test script for the noise model.

This script tests the basic functionality of the noise model.
"""

import os
import sys
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the noise models
from lib.noise_model.noise_model import (
    KeyboardNoiseModel,
    DwellNoiseModel,
    TranspositionNoiseModel,
    CompositeNoiseModel,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def add_noise_to_text(text: str, noise_level: float = 0.3) -> str:
    """
    Add noise to a text string using a composite noise model.

    Args:
        text: The input text
        noise_level: The level of noise to add (0.0-1.0)

    Returns:
        The noisy text
    """
    # Create a composite noise model with appropriate error rates
    keyboard_model = KeyboardNoiseModel(
        layout_name="qwerty",
        error_rates={"proximity": noise_level * 0.3, "deletion": 0.0, "insertion": 0.0},
        input_method="direct",
    )

    dwell_model = DwellNoiseModel(
        deletion_rate=noise_level * 0.15, insertion_rate=noise_level * 0.15
    )

    transposition_model = TranspositionNoiseModel(transposition_rate=noise_level * 0.1)

    # Combine the models
    noise_model = CompositeNoiseModel(
        models=[keyboard_model, dwell_model, transposition_model]
    )

    # Apply the noise model
    return noise_model.apply(text)


def add_noise_to_conversation(conversation, noise_level=0.3):
    """
    Add noise to a conversation using the noise model.

    Args:
        conversation: List of conversation turns
        noise_level: The level of noise to add (0.0-1.0)

    Returns:
        List of conversation turns with noise added
    """
    noisy_conversation = []

    for turn in conversation:
        speaker = turn["speaker"]
        utterance = turn["utterance"]

        # Only add noise to user utterances (not system or other speakers)
        if speaker.lower() in ["user", "aac user"]:
            # Add noise with the noise model
            noisy_utterance = add_noise_to_text(utterance, noise_level)
        else:
            # No noise for non-user utterances
            noisy_utterance = utterance

        noisy_conversation.append(
            {
                "speaker": speaker,
                "utterance": utterance,  # Original utterance
                "noisy_utterance": noisy_utterance,  # Noisy version
            }
        )

    return noisy_conversation


def main():
    """Main function."""
    # Test add_noise_to_text
    print("=== Testing add_noise_to_text ===")

    test_strings = [
        "hello",
        "this is a test",
        "The quick brown fox jumps over the lazy dog.",
    ]

    for text in test_strings:
        noisy_text = add_noise_to_text(text, noise_level=0.3)
        print(f"Original: {text}")
        print(f"Noisy:    {noisy_text}")
        print()

    # Test add_noise_to_conversation
    print("\n=== Testing add_noise_to_conversation ===")

    conversation = [
        {"speaker": "System", "utterance": "Hello! How can I help you today?"},
        {"speaker": "User", "utterance": "I would like to order a pizza please"},
        {"speaker": "System", "utterance": "Sure, what kind of pizza would you like?"},
        {"speaker": "User", "utterance": "I want a pepperoni pizza with extra cheese"},
    ]

    noisy_conversation = add_noise_to_conversation(conversation, noise_level=0.3)

    for i, turn in enumerate(noisy_conversation):
        speaker = turn["speaker"]
        original = turn["utterance"]
        noisy = turn["noisy_utterance"]

        print(f"Turn {i+1} - {speaker}:")
        print(f"  Original: {original}")
        print(f"  Noisy:    {noisy}")
        print()


if __name__ == "__main__":
    main()
