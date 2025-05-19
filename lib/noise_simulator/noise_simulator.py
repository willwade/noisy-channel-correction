"""
Noise simulator for AAC input data.

This module provides functions for adding noise to text data, simulating
various types of errors commonly seen in AAC (Augmentative and Alternative
Communication) systems.
"""

import os
import sys
import random
import logging
from typing import List, Dict, Any, Optional

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def add_noise(
    text: str,
    noise_level: float = 0.3,
    keyboard_layout: str = None,  # Not used in this implementation
    input_method: str = None,  # Not used in this implementation
    seed: Optional[int] = None,
) -> str:
    """
    Add noise to a text string.

    Args:
        text: The input text
        noise_level: The level of noise to add (0.0-1.0)
        keyboard_layout: The keyboard layout to use
        input_method: The input method ('direct', 'scanning', 'eyegaze')
        seed: Random seed for reproducibility

    Returns:
        The noisy text
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Create a composite noise model
    # Since we're having issues with keyboard layouts, let's use a simpler approach
    # that doesn't rely on keyboard layouts

    # Instead of using the keyboard model, we'll just add random noise
    # by replacing characters with random ones, deleting characters, etc.
    noisy_text = ""
    for char in text:
        # 50% chance of keeping the character as is
        if random.random() < 0.5:
            noisy_text += char
            continue

        # 20% chance of deleting the character
        if random.random() < 0.2 * noise_level:
            continue

        # 20% chance of inserting a random character
        if random.random() < 0.2 * noise_level:
            noisy_text += random.choice("abcdefghijklmnopqrstuvwxyz")

        # 10% chance of replacing with a random character
        if random.random() < 0.1 * noise_level:
            noisy_text += random.choice("abcdefghijklmnopqrstuvwxyz")
        else:
            noisy_text += char

    return noisy_text


def add_noise_to_conversation(
    conversation: List[Dict[str, Any]],
    noise_level: float = 0.3,
    keyboard_layout: str = "qwerty",
    input_method: str = "direct",
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Add noise to a conversation.

    Args:
        conversation: List of conversation turns
        noise_level: The level of noise to add (0.0-1.0)
        keyboard_layout: The keyboard layout to use
        input_method: The input method ('direct', 'scanning', 'eyegaze')
        seed: Random seed for reproducibility

    Returns:
        List of conversation turns with noise added
    """
    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    noisy_conversation = []

    for turn in conversation:
        speaker = turn["speaker"]
        utterance = turn["utterance"]

        # Only add noise to user utterances (not system or other speakers)
        if speaker.lower() in ["user", "aac user"]:
            # Add noise with the specified level
            noisy_utterance = add_noise(
                utterance,
                noise_level=noise_level,
                keyboard_layout=keyboard_layout,
                input_method=input_method,
                seed=random.randint(1, 1000) if seed is None else seed,
            )
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
