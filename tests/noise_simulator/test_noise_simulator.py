#!/usr/bin/env python3
"""
Test script for the noise simulator.

This script tests the basic functionality of the noise simulator.
"""

import os
import sys
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the noise simulator
from lib.noise_simulator.noise_simulator import add_noise, add_noise_to_conversation

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    # Test add_noise
    print("=== Testing add_noise ===")
    
    test_strings = [
        "hello",
        "this is a test",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    for text in test_strings:
        noisy_text = add_noise(text, noise_level=0.3, keyboard_layout="qwerty")
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
    
    noisy_conversation = add_noise_to_conversation(
        conversation, noise_level=0.3, keyboard_layout="qwerty"
    )
    
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
