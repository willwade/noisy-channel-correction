#!/usr/bin/env python3
"""
Demo script for the noisy channel corrector with context-aware correction.

This script provides a command-line interface for demonstrating the
noisy channel corrector with context-aware correction.
"""

import os
import sys
import argparse
import logging
from typing import List, Tuple

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the corrector
from lib.corrector.corrector import NoisyChannelCorrector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def interactive_mode(
    corrector: NoisyChannelCorrector, args: argparse.Namespace
) -> None:
    """
    Run the corrector in interactive mode.

    Args:
        corrector: The corrector to use
        args: Command-line arguments
    """
    print("\n=== Interactive Conversation Mode ===")
    print("Type 'q' or 'quit' to exit.")
    print("Type 'clear' to clear conversation history.")
    print("Type 'undo' to remove the last turn.")
    print("Type 'stats' to show correction statistics.")

    # Initialize conversation history
    conversation = []

    while True:
        # Show current context
        if conversation and args.show_context:
            print("\nConversation context:")
            for i, (text, corrected) in enumerate(conversation[-3:]):
                print(f"  [{i+1}] {corrected}")

        # Get user input
        user_input = input("\nEnter text (or command): ")

        # Check for commands
        if user_input.lower() in ["q", "quit", "exit"]:
            break
        elif user_input.lower() == "clear":
            conversation = []
            print("Conversation history cleared.")
            continue
        elif user_input.lower() == "undo" and conversation:
            conversation.pop()
            print("Last turn removed.")
            continue
        elif user_input.lower() == "stats":
            show_correction_stats(conversation)
            continue

        # Get context from conversation history
        context = [corrected for _, corrected in conversation[-args.context_window :]]

        # Apply noise if requested
        if args.apply_noise:
            original = user_input
            noisy = add_noise(original, args.noise_type, args.noise_level)
            print(f"Original: {original}")
            print(f"Noisy: {noisy}")
        else:
            noisy = user_input
            original = None

        # Correct with different methods if comparison is enabled
        if args.compare_methods:
            compare_correction_methods(noisy, context, corrector, args)
        else:
            # Correct the input with the selected method
            if args.correction_method == "context-aware" and context:
                print(f"Using context: {context}")
                corrections = corrector.correct(
                    noisy, context=context, max_edit_distance=args.max_edit_distance
                )
            else:
                corrections = corrector.correct(
                    noisy, context=None, max_edit_distance=args.max_edit_distance
                )

            # Print the corrections
            print("\nCorrections:")
            for i, (correction, score) in enumerate(corrections):
                print(f"  {i+1}. {correction} (score: {score:.4f})")

                # Highlight differences if original is available
                if original:
                    highlight_differences(original, correction)

        # Add to conversation history (using top correction)
        top_correction = corrections[0][0] if corrections else noisy
        conversation.append((noisy, top_correction))


def batch_mode(corrector: NoisyChannelCorrector, args: argparse.Namespace) -> None:
    """
    Run the corrector in batch mode.

    Args:
        corrector: The corrector to use
        args: Command-line arguments
    """
    print("\n=== Batch Mode ===")

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        return

    # Load the input file
    try:
        with open(args.input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except Exception as e:
        print(f"Error loading input file: {e}")
        return

    # Process each line
    results = []
    context = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Correct the line
        if args.correction_method == "context-aware" and context:
            corrections = corrector.correct(
                line, context=context, max_edit_distance=args.max_edit_distance
            )
        else:
            corrections = corrector.correct(
                line, context=None, max_edit_distance=args.max_edit_distance
            )

        # Add to results
        top_correction = corrections[0][0] if corrections else line
        results.append((line, top_correction))

        # Update context
        if args.correction_method == "context-aware":
            context.append(top_correction)
            context = context[-args.context_window :]

    # Print results
    print("\nResults:")
    for i, (original, corrected) in enumerate(results):
        print(f"{i+1}. Original: {original}")
        print(f"   Corrected: {corrected}")
        print()

    # Save results if output file is specified
    if args.output_file:
        try:
            with open(args.output_file, "w", encoding="utf-8") as f:
                for original, corrected in results:
                    f.write(f"Original: {original}\n")
                    f.write(f"Corrected: {corrected}\n\n")
            print(f"Results saved to {args.output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")


def conversation_mode(
    corrector: NoisyChannelCorrector, args: argparse.Namespace
) -> None:
    """
    Run the corrector in conversation mode.

    Args:
        corrector: The corrector to use
        args: Command-line arguments
    """
    print("\n=== Conversation Mode ===")

    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Input file not found: {args.input_file}")
        return

    # Load the conversation
    try:
        conversation = load_conversation(args.input_file)
    except Exception as e:
        print(f"Error loading conversation: {e}")
        return

    # Process the conversation
    results = process_conversation(conversation, corrector, args)

    # Print results
    print("\nResults:")
    for i, turn in enumerate(results["turns"]):
        print(f"Turn {i+1}:")
        print(f"  Speaker: {turn.get('speaker', 'Unknown')}")
        print(f"  Intended: {turn.get('intended', '')}")
        print(f"  Noisy: {turn.get('noisy', '')}")
        print(f"  Corrected: {turn['corrections'][0]['correction']}")
        print(f"  Correct at 1: {turn.get('correct_at_1', False)}")
        print()

    # Print metrics
    print("\nMetrics:")
    print(f"  Accuracy: {results['metrics']['accuracy']:.4f}")
    print(f"  Turn-level accuracy: {results['metrics']['turn_level_accuracy']}")

    # Save results if output file is specified
    if args.output_file:
        try:
            import json

            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")


def load_conversation(file_path: str) -> List[dict]:
    """
    Load a conversation from a file.

    Args:
        file_path: Path to the conversation file

    Returns:
        List of conversation turns
    """
    # Simple implementation for now
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        conversation = []
        current_turn = {}

        for line in lines:
            line = line.strip()
            if not line:
                if current_turn:
                    conversation.append(current_turn)
                    current_turn = {}
                continue

            if line.startswith("Speaker:"):
                current_turn["speaker"] = line[8:].strip()
            elif line.startswith("Utterance:"):
                current_turn["utterance_intended"] = line[10:].strip()
            elif line.startswith("Noisy:"):
                current_turn["noisy"] = line[6:].strip()

        if current_turn:
            conversation.append(current_turn)

        return conversation

    except Exception as e:
        logger.error(f"Error loading conversation from {file_path}: {e}")
        return []


def process_conversation(
    conversation: List[dict], corrector: NoisyChannelCorrector, args: argparse.Namespace
) -> dict:
    """
    Process a conversation.

    Args:
        conversation: List of conversation turns
        corrector: The corrector to use
        args: Command-line arguments

    Returns:
        Dictionary with results
    """
    # Initialize results
    results = {
        "turns": [],
        "metrics": {
            "accuracy": 0.0,
            "turn_level_accuracy": [],
        },
    }

    # Process each turn
    context = []

    for turn in conversation:
        # Get the noisy utterance
        noisy = turn.get("noisy", "")
        intended = turn.get("utterance_intended", "")

        # If no noisy utterance, use the intended utterance
        if not noisy:
            noisy = intended

        # Correct with context
        if args.correction_method == "context-aware" and context:
            corrections = corrector.correct(
                noisy, context=context, max_edit_distance=args.max_edit_distance
            )
        else:
            corrections = corrector.correct(
                noisy, context=None, max_edit_distance=args.max_edit_distance
            )

        # Update context with the corrected utterance
        top_correction = corrections[0][0] if corrections else noisy

        if args.use_gold_context:
            # Use the intended utterance for future context (oracle setting)
            context.append(intended)
        else:
            # Use the top correction for future context (realistic setting)
            context.append(top_correction)

        # Limit context size
        context = context[-args.context_window :]

        # Calculate turn-level metrics
        correct_at_1 = top_correction.lower() == intended.lower()
        correct_at_n = any(corr.lower() == intended.lower() for corr, _ in corrections)

        # Add to results
        results["turns"].append(
            {
                "turn_number": len(results["turns"]) + 1,
                "speaker": turn.get("speaker", "Unknown"),
                "intended": intended,
                "noisy": noisy,
                "corrections": [
                    {"correction": corr, "score": score} for corr, score in corrections
                ],
                "correct_at_1": correct_at_1,
                "correct_at_n": correct_at_n,
            }
        )

        results["metrics"]["turn_level_accuracy"].append(correct_at_1)

    # Calculate conversation-level metrics
    if results["metrics"]["turn_level_accuracy"]:
        results["metrics"]["accuracy"] = sum(
            results["metrics"]["turn_level_accuracy"]
        ) / len(results["metrics"]["turn_level_accuracy"])

    return results


def show_correction_stats(conversation: List[Tuple[str, str]]) -> None:
    """
    Show correction statistics.

    Args:
        conversation: List of (noisy, corrected) tuples
    """
    if not conversation:
        print("No conversation history to analyze.")
        return

    print("\nCorrection Statistics:")
    print(f"  Total turns: {len(conversation)}")

    # Count corrections
    corrections = sum(1 for noisy, corrected in conversation if noisy != corrected)
    print(
        f"  Total corrections: {corrections} ({corrections/len(conversation)*100:.1f}%)"
    )

    # TODO: Add more statistics


def add_noise(text: str, noise_type: str, noise_level: float) -> str:
    """
    Add noise to the input text.

    Args:
        text: The input text
        noise_type: Type of noise to add
        noise_level: Level of noise to add

    Returns:
        The noisy text
    """
    # Simple implementation for now
    import random

    if noise_type == "keyboard":
        # Keyboard adjacency noise
        keyboard_adjacency = {
            "a": "sqzw",
            "b": "vghn",
            "c": "xdfv",
            "d": "serfcx",
            "e": "wrsdf",
            "f": "drtgvc",
            "g": "ftyhbv",
            "h": "gyujnb",
            "i": "uojkl",
            "j": "huikmn",
            "k": "jiolm",
            "l": "kop;.",
            "m": "njk,",
            "n": "bhjm",
            "o": "iklp",
            "p": "ol;[",
            "q": "asw",
            "r": "edft",
            "s": "qazxdw",
            "t": "rfgy",
            "u": "yhji",
            "v": "cfgb",
            "w": "qase",
            "x": "zsdc",
            "y": "tghu",
            "z": "asx",
        }

        result = []
        for char in text:
            if char.lower() in keyboard_adjacency and random.random() < noise_level:
                # Replace with an adjacent key
                adjacent_keys = keyboard_adjacency[char.lower()]
                replacement = random.choice(adjacent_keys)
                if char.isupper():
                    replacement = replacement.upper()
                result.append(replacement)
            else:
                result.append(char)

        return "".join(result)

    elif noise_type == "swap":
        # Character swapping noise
        result = list(text)
        for i in range(len(result) - 1):
            if random.random() < noise_level:
                result[i], result[i + 1] = result[i + 1], result[i]

        return "".join(result)

    elif noise_type == "delete":
        # Character deletion noise
        result = []
        for char in text:
            if random.random() < noise_level:
                # Skip the character
                continue
            result.append(char)

        return "".join(result)

    else:
        # No noise
        return text


def highlight_differences(original: str, correction: str) -> None:
    """
    Highlight the differences between the original and corrected text.

    Args:
        original: The original text
        correction: The corrected text
    """
    # Simple implementation for now
    print("    Diff: ", end="")

    # Convert to lowercase for comparison
    orig_lower = original.lower()
    corr_lower = correction.lower()

    if orig_lower == corr_lower:
        print("(No differences)")
        return

    # Find the first differing character
    min_len = min(len(orig_lower), len(corr_lower))
    first_diff = min_len
    for i in range(min_len):
        if orig_lower[i] != corr_lower[i]:
            first_diff = i
            break

    # Print the differences
    if first_diff > 0:
        print(f"{correction[:first_diff]}", end="")

    print("[", end="")
    if first_diff < len(orig_lower):
        print(f"{original[first_diff:]}", end="")
    print(" -> ", end="")
    if first_diff < len(corr_lower):
        print(f"{correction[first_diff:]}", end="")
    print("]")


def compare_correction_methods(
    noisy_input: str,
    context: List[str],
    corrector: NoisyChannelCorrector,
    args: argparse.Namespace,
) -> None:
    """
    Compare different correction methods.

    Args:
        noisy_input: The noisy input text
        context: List of context words
        corrector: The corrector to use
        args: Command-line arguments
    """
    print("\nComparing correction methods:")

    # Baseline (no context)
    baseline_corrections = corrector.correct(
        noisy_input, context=None, max_edit_distance=args.max_edit_distance
    )

    # Context-aware
    context_corrections = corrector.correct(
        noisy_input, context=context, max_edit_distance=args.max_edit_distance
    )

    # Print the results
    print("\n1. Baseline (no context):")
    for i, (correction, score) in enumerate(baseline_corrections[:3]):
        print(f"  {i+1}. {correction} (score: {score:.4f})")

    print("\n2. Context-aware:")
    for i, (correction, score) in enumerate(context_corrections[:3]):
        print(f"  {i+1}. {correction} (score: {score:.4f})")


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Demo for the noisy channel corrector with context-aware correction."
    )

    parser.add_argument(
        "--mode",
        choices=["interactive", "batch", "conversation"],
        default="interactive",
        help="Demo mode: interactive for user input, batch for dataset processing, conversation for full conversations",
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
        "--language",
        type=str,
        default="en-GB",
        help="Language code (e.g., 'en-GB', 'en-US')",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=5,
        help="Maximum number of candidates to return",
    )

    parser.add_argument(
        "--max-edit-distance",
        type=int,
        default=2,
        help="Maximum edit distance to consider",
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Number of previous words to use as context",
    )

    parser.add_argument(
        "--context-weight",
        type=float,
        default=0.7,
        help="Weight of context-based probability (0-1)",
    )

    parser.add_argument(
        "--keyboard-layout",
        choices=["qwerty", "abc", "frequency"],
        default="qwerty",
        help="Keyboard layout to use for confusion matrix",
    )

    parser.add_argument(
        "--use-keyboard-matrices",
        action="store_true",
        help="Use keyboard-specific confusion matrices",
    )

    parser.add_argument(
        "--correction-method",
        choices=["baseline", "context-aware"],
        default="context-aware",
        help="Correction method to use",
    )

    parser.add_argument(
        "--compare-methods",
        action="store_true",
        help="Compare multiple correction methods side by side",
    )

    parser.add_argument(
        "--apply-noise", action="store_true", help="Apply noise to the input text"
    )

    parser.add_argument(
        "--noise-type",
        choices=["keyboard", "swap", "delete"],
        default="keyboard",
        help="Type of noise to apply",
    )

    parser.add_argument(
        "--noise-level", type=float, default=0.2, help="Level of noise to apply (0-1)"
    )

    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to the input file for batch and conversation modes",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to the output file for batch and conversation modes",
    )

    parser.add_argument(
        "--show-context", action="store_true", help="Show context in interactive mode"
    )

    parser.add_argument(
        "--use-gold-context",
        action="store_true",
        help="Use gold standard context in conversation mode",
    )

    args = parser.parse_args()

    # Create the corrector
    corrector = NoisyChannelCorrector(
        max_candidates=args.max_candidates,
        context_window_size=args.context_window,
        context_weight=args.context_weight,
    )

    # Try to load language-specific models first, then fall back to default models
    language_code = args.language.lower()
    language_path = language_code.replace('-', '_')
    
    # Load the PPM model
    # Check for language-specific PPM model first
    language_ppm_model = os.path.join(
        os.path.dirname(args.ppm_model),
        f"ppm_model_{language_path}.pkl"
    )
    if os.path.exists(language_ppm_model):
        corrector.load_ppm_model(language_ppm_model)
        logger.info(f"Loaded language-specific PPM model for {language_code}")
    elif os.path.exists(args.ppm_model):
        corrector.load_ppm_model(args.ppm_model)
        logger.info("Using default PPM model")
    else:
        logger.warning(f"PPM model file not found: {args.ppm_model}")

    # Load the confusion matrix
    # Check if we should use keyboard-specific matrices
    if args.use_keyboard_matrices:
        # Try to load language-specific keyboard matrices first
        language_keyboard_matrix = os.path.join(
            os.path.dirname(args.confusion_matrix),
            f"keyboard_confusion_matrices_{language_path}.json"
        )
        if os.path.exists(language_keyboard_matrix):
            corrector.load_confusion_model(
                language_keyboard_matrix, args.keyboard_layout
            )
            logger.info(
                f"Using language-specific keyboard matrix for {language_code} ({args.keyboard_layout} layout)"
            )
        else:
            # Try default keyboard matrices
            default_keyboard_matrix = os.path.join(
                os.path.dirname(args.confusion_matrix),
                "keyboard_confusion_matrices.json"
            )
            if os.path.exists(default_keyboard_matrix):
                corrector.load_confusion_model(
                    default_keyboard_matrix, args.keyboard_layout
                )
                logger.info(
                    f"Using default keyboard matrix for {args.keyboard_layout} layout"
                )
            elif os.path.exists(args.confusion_matrix):
                # Fall back to standard confusion matrix
                corrector.load_confusion_model(
                    args.confusion_matrix, args.keyboard_layout
                )
                logger.info("Using standard confusion matrix")
            else:
                logger.warning(f"Confusion matrix file not found: {args.confusion_matrix}")
    else:
        # Try to load language-specific standard confusion matrix
        language_confusion_matrix = os.path.join(
            os.path.dirname(args.confusion_matrix),
            f"confusion_matrix_{language_path}.json"
        )
        if os.path.exists(language_confusion_matrix):
            corrector.load_confusion_model(language_confusion_matrix, args.keyboard_layout)
            logger.info(f"Using language-specific confusion matrix for {language_code}")
        elif os.path.exists(args.confusion_matrix):
            corrector.load_confusion_model(args.confusion_matrix, args.keyboard_layout)
            logger.info("Using default confusion matrix")
        else:
            logger.warning(f"Confusion matrix file not found: {args.confusion_matrix}")

    # Load the word n-gram model
    # Check for language-specific n-gram model first
    language_ngram_model = os.path.join(
        os.path.dirname(args.word_ngram_model),
        f"word_ngram_model_{language_path}.pkl"
    )
    if os.path.exists(language_ngram_model):
        corrector.load_word_ngram_model(language_ngram_model)
        logger.info(f"Loaded language-specific word n-gram model for {language_code}")
    elif os.path.exists(args.word_ngram_model):
        corrector.load_word_ngram_model(args.word_ngram_model)
        logger.info("Using default word n-gram model")
    else:
        logger.warning(f"Word n-gram model file not found: {args.word_ngram_model}")

    # Load the lexicon
    # Check for language-specific lexicon first
    language_lexicon = os.path.join(
        os.path.dirname(args.lexicon),
        f"aac_lexicon_{language_path}.txt"
    )
    if os.path.exists(language_lexicon):
        corrector.load_lexicon_from_file(language_lexicon)
        logger.info(f"Loaded language-specific lexicon for {language_code}")
    elif os.path.exists(args.lexicon):
        corrector.load_lexicon_from_file(args.lexicon)
        logger.info("Using default lexicon")
    else:
        logger.warning(f"Lexicon file not found: {args.lexicon}")

    # Run the appropriate mode
    if args.mode == "interactive":
        interactive_mode(corrector, args)
    elif args.mode == "batch":
        batch_mode(corrector, args)
    elif args.mode == "conversation":
        conversation_mode(corrector, args)


if __name__ == "__main__":
    main()
