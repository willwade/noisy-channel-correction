#!/usr/bin/env python3
"""
CLI demo for the AAC Noisy Input Correction Engine.

This script provides a command-line interface for demonstrating the
noisy channel corrector on the AACConversations dataset or using module1's
noise simulator.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Tuple, Any, Optional
import json
import random
from pathlib import Path

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
# Also add module4 and its subdirectories to the path
module4_dir = os.path.join(parent_dir, "module4")
sys.path.append(module4_dir)
sys.path.append(os.path.join(module4_dir, "ppm"))
sys.path.append(os.path.join(module4_dir, "pylm"))

# Import the corrector
from module4.corrector import NoisyChannelCorrector
from module5.utils import (
    load_aac_conversations,
    get_noisy_utterance,
    get_random_examples,
    load_corrector,
    save_results,
    format_example_for_display,
    NOISE_TYPES,
    NOISE_LEVELS,
)

# Import module1 utilities
from module5.module1_utils import (
    add_noise,
    generate_noisy_pairs,
    load_wordlist,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CLI demo for the AAC Noisy Input Correction Engine."
    )

    # Source selection
    parser.add_argument(
        "--use-module1",
        action="store_true",
        help="Use module1's noise simulator instead of the Hugging Face dataset",
    )
    parser.add_argument(
        "--wordlist",
        type=str,
        default="../data/wordlist.txt",
        help="Path to the wordlist file for module1's noise simulator",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="Dataset split to use ('train', 'validation', or 'test')",
    )
    parser.add_argument("--cache-dir", type=str, help="Directory to cache the dataset")
    parser.add_argument(
        "--no-token",
        action="store_true",
        help="Don't use the Hugging Face auth token",
    )

    # Noise arguments
    parser.add_argument(
        "--noise-type",
        type=str,
        default="qwerty",
        choices=NOISE_TYPES,
        help="Type of noise to use",
    )
    parser.add_argument(
        "--noise-level",
        type=str,
        default="moderate",
        choices=NOISE_LEVELS,
        help="Level of noise to use",
    )

    # Sample arguments
    parser.add_argument(
        "--num-examples", type=int, default=5, help="Number of examples to sample"
    )
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument(
        "--example-id",
        type=int,
        help="Specific example ID to use (overrides random sampling)",
    )

    # Model arguments
    parser.add_argument("--ppm-model", type=str, help="Path to the PPM model file")
    parser.add_argument(
        "--confusion-matrix", type=str, help="Path to the confusion matrix file"
    )
    parser.add_argument("--lexicon", type=str, help="Path to the lexicon file")

    # Correction parameters
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
        "--no-keyboard-adjacency",
        action="store_true",
        help="Disable keyboard adjacency for candidate generation",
    )

    # Output arguments
    parser.add_argument("--output", type=str, help="Output file for correction results")

    # Interactive mode
    parser.add_argument(
        "--interactive", action="store_true", help="Run in interactive mode"
    )

    # Verbose output
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def interactive_mode(corrector: NoisyChannelCorrector, args):
    """Run the demo in interactive mode."""
    print("\n=== Interactive Mode ===")
    print("Type 'q' or 'quit' to exit.")
    print("Type 'help' for help.")

    while True:
        # Get user input
        user_input = input("\nEnter text to correct: ")

        # Check for exit command
        if user_input.lower() in ["q", "quit", "exit"]:
            break

        # Check for help command
        if user_input.lower() == "help":
            print("\nCommands:")
            print("  q, quit, exit: Exit interactive mode")
            print("  help: Show this help message")
            print("\nEnter any text to correct it using the noisy channel model.")
            continue

        # Correct the input
        corrections = corrector.correct(
            user_input,
            max_edit_distance=args.max_edit_distance,
            use_keyboard_adjacency=not args.no_keyboard_adjacency,
        )

        # Print the corrections
        print("\nCorrections:")
        for i, (correction, score) in enumerate(corrections):
            print(f"  {i+1}. {correction} (score: {score:.4f})")


def process_examples(
    dataset: Any, corrector: NoisyChannelCorrector, args
) -> List[Dict[str, Any]]:
    """Process examples from the dataset."""
    results = []

    # Get examples to process
    if args.example_id is not None:
        # Get a specific example
        try:
            examples = [dataset[args.example_id]]
            logger.info(f"Using example with ID {args.example_id}")
        except Exception as e:
            logger.error(f"Error getting example with ID {args.example_id}: {e}")
            return []
    else:
        # Get random examples
        examples = get_random_examples(dataset, args.num_examples, args.seed)
        logger.info(f"Sampled {len(examples)} random examples")

    # Process each example
    for example in examples:
        # Get the noisy utterance
        noisy = get_noisy_utterance(example, args.noise_type, args.noise_level)

        # Correct the noisy utterance
        corrections = corrector.correct(
            noisy,
            max_edit_distance=args.max_edit_distance,
            use_keyboard_adjacency=not args.no_keyboard_adjacency,
        )

        # Format the example for display
        formatted = format_example_for_display(
            example, args.noise_type, args.noise_level, corrections
        )

        # Print the formatted example
        print("\n" + "=" * 80)
        print(formatted)

        # Add to results
        results.append(
            {
                "conversation_id": example.get("conversation_id"),
                "turn_number": example.get("turn_number"),
                "scene": example.get("scene"),
                "speaker": example.get("speaker"),
                "utterance_intended": example.get("utterance_intended"),
                "noisy_utterance": noisy,
                "corrections": [
                    {"correction": corr, "score": score} for corr, score in corrections
                ],
                "noise_type": args.noise_type,
                "noise_level": args.noise_level,
            }
        )

    return results


def process_module1_examples(
    corrector: NoisyChannelCorrector, args
) -> List[Dict[str, Any]]:
    """Process examples using module1's noise simulator."""
    results = []

    # Load the wordlist
    words = load_wordlist(args.wordlist)
    if not words:
        logger.error(f"No words loaded from {args.wordlist}. Exiting.")
        return []

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Sample random words
    if len(words) > args.num_examples:
        sampled_words = random.sample(words, args.num_examples)
    else:
        sampled_words = words

    logger.info(f"Sampled {len(sampled_words)} words from {args.wordlist}")

    # Generate noisy pairs
    noisy_pairs = generate_noisy_pairs(
        sampled_words, args.noise_type, args.noise_level, num_variants=1
    )

    logger.info(f"Generated {len(noisy_pairs)} noisy pairs")

    # Process each pair
    for pair in noisy_pairs:
        intended = pair["intended"]
        noisy = pair["noisy"]

        # Skip if noisy is the same as intended (no noise)
        if noisy == intended:
            continue

        # Correct the noisy utterance
        corrections = corrector.correct(
            noisy,
            max_edit_distance=args.max_edit_distance,
            use_keyboard_adjacency=not args.no_keyboard_adjacency,
        )

        # Print the results
        print("\n" + "=" * 80)
        print(f"Original: {intended}")
        print(f"Noisy ({args.noise_type}, {args.noise_level}): {noisy}")
        print("\nCorrections:")
        for i, (correction, score) in enumerate(corrections):
            print(f"  {i+1}. {correction} (score: {score:.4f})")

        # Add to results
        results.append(
            {
                "intended": intended,
                "noisy": noisy,
                "corrections": [
                    {"correction": corr, "score": score} for corr, score in corrections
                ],
                "noise_type": args.noise_type,
                "noise_level": args.noise_level,
                "correct_at_1": corrections[0][0] == intended if corrections else False,
                "correct_at_n": any(corr == intended for corr, _ in corrections),
            }
        )

    return results


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load the corrector
    corrector = load_corrector(
        ppm_model_path=args.ppm_model,
        confusion_matrix_path=args.confusion_matrix,
        lexicon_path=args.lexicon,
        max_candidates=args.max_candidates,
    )

    # Run in interactive mode if requested
    if args.interactive:
        interactive_mode(corrector, args)
        return

    # Use module1's noise simulator if requested
    if args.use_module1:
        logger.info("Using module1's noise simulator")
        results = process_module1_examples(corrector, args)
    else:
        # Load the dataset
        dataset = load_aac_conversations(
            split=args.dataset_split,
            cache_dir=args.cache_dir,
            use_auth_token=not args.no_token,
        )

        if dataset is None:
            logger.error("Failed to load dataset. Exiting.")
            return

        # Process examples
        results = process_examples(dataset, corrector, args)

    # Save results if requested
    if args.output and results:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
