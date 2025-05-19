#!/usr/bin/env python3
"""
Demo script for context-aware correction with conversation support.

This script demonstrates the enhanced context-aware correction functionality,
including conversation-level context and adaptive context weighting.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the corrector
from lib.corrector.corrector import NoisyChannelCorrector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sample_conversation(file_path: str) -> List[Dict[str, Any]]:
    """
    Load a sample conversation from a file.

    Args:
        file_path: Path to the conversation file

    Returns:
        List of conversation turns
    """
    conversation = []

    try:
        with open(file_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Parse the line (format: "speaker: utterance")
                parts = line.split(":", 1)
                if len(parts) == 2:
                    speaker = parts[0].strip()
                    utterance = parts[1].strip()
                    conversation.append({"speaker": speaker, "utterance": utterance})
                else:
                    # If no speaker specified, use "unknown"
                    conversation.append({"speaker": "unknown", "utterance": line})

        return conversation
    except Exception as e:
        logger.error(f"Error loading sample conversation: {e}")
        return []


def add_noise_to_conversation(
    conversation: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Add noise to a conversation.

    Args:
        conversation: List of conversation turns

    Returns:
        List of conversation turns with noise added
    """
    import random

    # Try to import from lib.noise_simulator first
    try:
        from lib.noise_simulator.noise_simulator import add_noise

        logger.info("Using lib.noise_simulator for noise generation")
    except ImportError:
        # Fallback to a simple noise generator if the module is not available
        logger.warning("lib.noise_simulator not found, using simple noise generator")

        def add_noise(text, noise_level=0.3, keyboard_layout=None, seed=None):
            """Simple fallback noise generator."""
            if seed is not None:
                random.seed(seed)

            chars = list(text)
            # Randomly modify some characters
            for i in range(len(chars)):
                if random.random() < noise_level:
                    # 33% chance to replace with adjacent character
                    if random.random() < 0.33:
                        chars[i] = random.choice("abcdefghijklmnopqrstuvwxyz")
                    # 33% chance to delete
                    elif random.random() < 0.5:
                        chars[i] = ""
                    # 33% chance to insert
                    else:
                        chars.insert(i, random.choice("abcdefghijklmnopqrstuvwxyz"))
            return "".join(chars)

    noisy_conversation = []

    for turn in conversation:
        speaker = turn["speaker"]
        utterance = turn["utterance"]

        # Only add noise to user utterances (not system or other speakers)
        if speaker.lower() in ["user", "aac user"]:
            # Add noise with a moderate level
            noisy_utterance = add_noise(
                utterance,
                noise_level=0.3,
                keyboard_layout="qwerty",
                seed=random.randint(1, 1000),
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


def process_conversation(
    conversation: List[Dict[str, Any]],
    corrector: NoisyChannelCorrector,
    use_conversation_context: bool = True,
    adaptive_context_weighting: bool = True,
) -> List[Dict[str, Any]]:
    """
    Process a conversation with the corrector.

    Args:
        conversation: List of conversation turns
        corrector: The corrector to use
        use_conversation_context: Whether to use conversation-level context
        adaptive_context_weighting: Whether to use adaptive context weighting

    Returns:
        List of processed conversation turns
    """
    processed_conversation = []
    context = []  # Simple context (list of previous utterances)

    for turn in conversation:
        speaker = turn["speaker"]
        utterance = turn["utterance"]  # Original utterance
        noisy_utterance = turn.get("noisy_utterance", utterance)  # Noisy utterance

        # Only correct user utterances
        if speaker.lower() in ["user", "aac user"]:
            # Correct with context
            corrections = corrector.correct(
                noisy_utterance, context=context, max_edit_distance=2
            )

            # Get the top correction
            top_correction = corrections[0][0] if corrections else noisy_utterance
            top_score = corrections[0][1] if corrections else 0.0

            # Add to processed conversation
            processed_turn = {
                "speaker": speaker,
                "original": utterance,
                "noisy": noisy_utterance,
                "corrected": top_correction,
                "score": top_score,
                "correct": top_correction.lower() == utterance.lower(),
                "corrections": [{"text": c, "score": s} for c, s in corrections[:5]],
                "context": list(context),  # Copy of current context
            }
        else:
            # Non-user utterances don't need correction
            processed_turn = {
                "speaker": speaker,
                "original": utterance,
                "noisy": noisy_utterance,
                "corrected": utterance,
                "score": 1.0,
                "correct": True,
                "corrections": [{"text": utterance, "score": 1.0}],
                "context": list(context),  # Copy of current context
            }

        processed_conversation.append(processed_turn)

        # Update context with the original utterance for non-user speakers
        # or the corrected utterance for user speakers
        if speaker.lower() in ["user", "aac user"]:
            context.append(top_correction)
        else:
            context.append(utterance)

        # Limit context size
        context = context[-5:]  # Keep last 5 utterances

        # Update conversation context if enabled
        if use_conversation_context:
            if speaker.lower() in ["user", "aac user"]:
                corrector.update_conversation_context(top_correction, speaker)
            else:
                corrector.update_conversation_context(utterance, speaker)

    return processed_conversation


def print_processed_conversation(processed_conversation: List[Dict[str, Any]]) -> None:
    """
    Print a processed conversation.

    Args:
        processed_conversation: List of processed conversation turns
    """
    print("\n=== Processed Conversation ===\n")

    for i, turn in enumerate(processed_conversation):
        speaker = turn["speaker"]
        original = turn["original"]
        noisy = turn["noisy"]
        corrected = turn["corrected"]
        score = turn["score"]
        correct = turn["correct"]

        print(f"Turn {i+1} - {speaker}:")

        if speaker.lower() in ["user", "aac user"]:
            print(f"  Original: {original}")
            print(f"  Noisy:    {noisy}")
            print(f"  Corrected: {corrected} (Score: {score:.4f})")
            print(f"  Correct:  {'✓' if correct else '✗'}")

            # Print alternative corrections
            if len(turn["corrections"]) > 1:
                print("  Alternatives:")
                for j, corr in enumerate(
                    turn["corrections"][1:4]
                ):  # Show top 3 alternatives
                    print(f"    {j+2}. {corr['text']} (Score: {corr['score']:.4f})")
        else:
            print(f"  {original}")

        print()


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Demo for context-aware correction with conversation support."
    )

    parser.add_argument(
        "--ppm-model",
        type=str,
        default="models/ppm_model.pkl",
        help="Path to the PPM model file",
    )

    parser.add_argument(
        "--confusion-matrix",
        type=str,
        default="models/confusion_matrix.json",
        help="Path to the confusion matrix file",
    )

    parser.add_argument(
        "--word-ngram-model",
        type=str,
        default="models/word_ngram_model.pkl",
        help="Path to the word n-gram model file",
    )

    parser.add_argument(
        "--lexicon",
        type=str,
        default="data/wordlist.txt",
        help="Path to the lexicon file",
    )

    parser.add_argument(
        "--conversation",
        type=str,
        default="data/sample_conversation.txt",
        help="Path to a sample conversation file",
    )

    parser.add_argument(
        "--no-conversation-context",
        action="store_true",
        help="Disable conversation-level context",
    )

    parser.add_argument(
        "--no-adaptive-weighting",
        action="store_true",
        help="Disable adaptive context weighting",
    )

    parser.add_argument(
        "--context-weight",
        type=float,
        default=0.7,
        help="Weight of context-based probability (0-1)",
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=3,
        help="Number of previous words to use as context",
    )

    args = parser.parse_args()

    # Create the corrector
    corrector = NoisyChannelCorrector(
        max_candidates=5,
        context_window_size=args.context_window,
        context_weight=args.context_weight,
        keyboard_layout="qwerty",
        use_conversation_context=not args.no_conversation_context,
        adaptive_context_weighting=not args.no_adaptive_weighting,
    )

    # Load the models
    if os.path.exists(args.ppm_model):
        corrector.load_ppm_model(args.ppm_model)
    else:
        logger.warning(f"PPM model file not found: {args.ppm_model}")

    if os.path.exists(args.confusion_matrix):
        corrector.load_confusion_model(args.confusion_matrix)
    else:
        logger.warning(f"Confusion matrix file not found: {args.confusion_matrix}")

    if os.path.exists(args.word_ngram_model):
        corrector.load_word_ngram_model(args.word_ngram_model)
    else:
        logger.warning(f"Word n-gram model file not found: {args.word_ngram_model}")

    if os.path.exists(args.lexicon):
        corrector.load_lexicon_from_file(args.lexicon)
    else:
        logger.warning(f"Lexicon file not found: {args.lexicon}")

    # Load the sample conversation
    if os.path.exists(args.conversation):
        conversation = load_sample_conversation(args.conversation)
    else:
        logger.warning(f"Conversation file not found: {args.conversation}")
        # Use a default conversation
        conversation = [
            {"speaker": "System", "utterance": "Hello! How can I help you today?"},
            {"speaker": "User", "utterance": "I would like to order a pizza please"},
            {
                "speaker": "System",
                "utterance": "Sure, what kind of pizza would you like?",
            },
            {
                "speaker": "User",
                "utterance": "I want a pepperoni pizza with extra cheese",
            },
            {
                "speaker": "System",
                "utterance": "Great choice! What size would you like?",
            },
            {"speaker": "User", "utterance": "Medium size is good for me"},
            {"speaker": "System", "utterance": "Would you like any drinks with that?"},
            {"speaker": "User", "utterance": "Yes a bottle of cola please"},
        ]

    # Add noise to the conversation
    noisy_conversation = add_noise_to_conversation(conversation)

    # Process the conversation
    processed_conversation = process_conversation(
        noisy_conversation,
        corrector,
        use_conversation_context=not args.no_conversation_context,
        adaptive_context_weighting=not args.no_adaptive_weighting,
    )

    # Print the processed conversation
    print_processed_conversation(processed_conversation)

    # Print summary
    correct_count = sum(
        1
        for turn in processed_conversation
        if turn["speaker"].lower() in ["user", "aac user"] and turn["correct"]
    )
    total_count = sum(
        1
        for turn in processed_conversation
        if turn["speaker"].lower() in ["user", "aac user"]
    )

    print("\n=== Summary ===\n")
    print(f"Conversation turns: {len(processed_conversation)}")
    print(f"User turns: {total_count}")
    print(
        f"Correctly corrected: {correct_count}/{total_count} ({correct_count/total_count*100:.1f}%)"
    )
    print(f"Using conversation context: {not args.no_conversation_context}")
    print(f"Using adaptive context weighting: {not args.no_adaptive_weighting}")


if __name__ == "__main__":
    main()
