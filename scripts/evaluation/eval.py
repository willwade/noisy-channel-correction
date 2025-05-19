#!/usr/bin/env python3
"""
Evaluation script for the AAC Noisy Input Correction Engine.

This script evaluates the performance of the noisy channel corrector
on the AACConversations dataset or using the noise simulator.
"""

import os
import sys
import argparse
import logging
from typing import Dict, Any
from collections import defaultdict
import time
import random

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import the corrector
from lib.corrector.corrector import NoisyChannelCorrector
from scripts.evaluation.utils import (
    load_aac_conversations,
    get_noisy_utterance,
    load_corrector,
    save_results,
    NOISE_TYPES,
    NOISE_LEVELS,
)

# Import noise simulator utilities
from scripts.evaluation.noise_simulator_utils import (
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
        description="Evaluate the AAC Noisy Input Correction Engine."
    )

    # Source selection
    parser.add_argument(
        "--use-noise-simulator",
        action="store_true",
        help="Use the noise simulator instead of the Hugging Face dataset",
    )
    parser.add_argument(
        "--wordlist",
        type=str,
        default="../data/wordlist.txt",
        help="Path to the wordlist file for the noise simulator",
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        help="Dataset split to use ('train', 'validation', or 'test')",
    )
    parser.add_argument("--cache-dir", type=str, help="Directory to cache the dataset")
    parser.add_argument(
        "--no-token",
        action="store_true",
        help="Don't use the Hugging Face auth token",
    )

    # Evaluation arguments
    parser.add_argument(
        "--num-examples", type=int, default=100, help="Number of examples to evaluate"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--noise-types",
        type=str,
        nargs="+",
        default=NOISE_TYPES,
        choices=NOISE_TYPES,
        help="Noise types to evaluate",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        nargs="+",
        default=NOISE_LEVELS,
        choices=NOISE_LEVELS,
        help="Noise levels to evaluate",
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
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results.json",
        help="Output file for evaluation results",
    )

    # Verbose output
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    return parser.parse_args()


def evaluate_accuracy_dataset(
    dataset: Any, corrector: NoisyChannelCorrector, args
) -> Dict[str, Any]:
    """
    Evaluate the accuracy of the corrector on the Hugging Face dataset.

    Args:
        dataset: The dataset to evaluate on
        corrector: The corrector to evaluate
        args: Command-line arguments

    Returns:
        Dictionary of evaluation results
    """
    # Initialize results
    results = {
        "overall": {
            "accuracy@1": 0.0,
            "accuracy@n": 0.0,
            "total_examples": 0,
            "avg_time_ms": 0.0,
        },
        "by_noise_type": {
            noise_type: {
                "accuracy@1": 0.0,
                "accuracy@n": 0.0,
                "total_examples": 0,
                "avg_time_ms": 0.0,
            }
            for noise_type in args.noise_types
        },
        "by_noise_level": {
            noise_level: {
                "accuracy@1": 0.0,
                "accuracy@n": 0.0,
                "total_examples": 0,
                "avg_time_ms": 0.0,
            }
            for noise_level in args.noise_levels
        },
        "by_noise_type_and_level": {
            f"{noise_type}_{noise_level}": {
                "accuracy@1": 0.0,
                "accuracy@n": 0.0,
                "total_examples": 0,
                "avg_time_ms": 0.0,
            }
            for noise_type in args.noise_types
            for noise_level in args.noise_levels
        },
        "examples": [],
    }

    # Initialize counters
    correct_at_1 = defaultdict(int)
    correct_at_n = defaultdict(int)
    total = defaultdict(int)
    total_time = defaultdict(float)

    # Sample examples to evaluate
    random.seed(args.seed)
    indices = random.sample(range(len(dataset)), min(args.num_examples, len(dataset)))
    examples = [dataset[i] for i in indices]

    # Evaluate each example
    for example in examples:
        intended = example.get("utterance_intended", "")

        # Evaluate for each noise type and level
        for noise_type in args.noise_types:
            for noise_level in args.noise_levels:
                # Get the noisy utterance
                noisy = get_noisy_utterance(example, noise_type, noise_level)

                # Skip if noisy is the same as intended (no noise)
                if noisy == intended:
                    continue

                # Correct the noisy utterance and measure time
                start_time = time.time()
                corrections = corrector.correct(
                    noisy,
                    max_edit_distance=args.max_edit_distance,
                    use_keyboard_adjacency=not args.no_keyboard_adjacency,
                )
                end_time = time.time()
                correction_time_ms = (end_time - start_time) * 1000

                # Extract corrections
                correction_texts = [corr for corr, _ in corrections]

                # Check accuracy@1
                correct_1 = (
                    correction_texts[0].lower() == intended.lower()
                    if correction_texts
                    else False
                )

                # Check accuracy@N
                correct_n = any(
                    corr.lower() == intended.lower() for corr in correction_texts
                )

                # Update counters
                key_overall = "overall"
                key_type = noise_type
                key_level = noise_level
                key_type_level = f"{noise_type}_{noise_level}"

                for key in [key_overall, key_type, key_level, key_type_level]:
                    total[key] += 1
                    if correct_1:
                        correct_at_1[key] += 1
                    if correct_n:
                        correct_at_n[key] += 1
                    total_time[key] += correction_time_ms

                # Add example to results
                results["examples"].append(
                    {
                        "conversation_id": example.get("conversation_id"),
                        "turn_number": example.get("turn_number"),
                        "intended": intended,
                        "noisy": noisy,
                        "noise_type": noise_type,
                        "noise_level": noise_level,
                        "corrections": [
                            {"correction": corr, "score": score}
                            for corr, score in corrections
                        ],
                        "correct_at_1": correct_1,
                        "correct_at_n": correct_n,
                        "time_ms": correction_time_ms,
                    }
                )

    # Calculate accuracy metrics
    for key in total:
        if key == "overall":
            results["overall"]["total_examples"] = total[key]
            results["overall"]["accuracy@1"] = (
                correct_at_1[key] / total[key] if total[key] > 0 else 0.0
            )
            results["overall"]["accuracy@n"] = (
                correct_at_n[key] / total[key] if total[key] > 0 else 0.0
            )
            results["overall"]["avg_time_ms"] = (
                total_time[key] / total[key] if total[key] > 0 else 0.0
            )
        elif key in args.noise_types:
            results["by_noise_type"][key]["total_examples"] = total[key]
            results["by_noise_type"][key]["accuracy@1"] = (
                correct_at_1[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_type"][key]["accuracy@n"] = (
                correct_at_n[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_type"][key]["avg_time_ms"] = (
                total_time[key] / total[key] if total[key] > 0 else 0.0
            )
        elif key in args.noise_levels:
            results["by_noise_level"][key]["total_examples"] = total[key]
            results["by_noise_level"][key]["accuracy@1"] = (
                correct_at_1[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_level"][key]["accuracy@n"] = (
                correct_at_n[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_level"][key]["avg_time_ms"] = (
                total_time[key] / total[key] if total[key] > 0 else 0.0
            )
        else:
            results["by_noise_type_and_level"][key]["total_examples"] = total[key]
            results["by_noise_type_and_level"][key]["accuracy@1"] = (
                correct_at_1[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_type_and_level"][key]["accuracy@n"] = (
                correct_at_n[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_type_and_level"][key]["avg_time_ms"] = (
                total_time[key] / total[key] if total[key] > 0 else 0.0
            )

    return results


def evaluate_accuracy_noise_simulator(
    corrector: NoisyChannelCorrector, args
) -> Dict[str, Any]:
    """
    Evaluate the accuracy of the corrector using the noise simulator.

    Args:
        corrector: The corrector to evaluate
        args: Command-line arguments

    Returns:
        Dictionary of evaluation results
    """
    # Initialize results
    results = {
        "overall": {
            "accuracy@1": 0.0,
            "accuracy@n": 0.0,
            "total_examples": 0,
            "avg_time_ms": 0.0,
        },
        "by_noise_type": {
            noise_type: {
                "accuracy@1": 0.0,
                "accuracy@n": 0.0,
                "total_examples": 0,
                "avg_time_ms": 0.0,
            }
            for noise_type in args.noise_types
        },
        "by_noise_level": {
            noise_level: {
                "accuracy@1": 0.0,
                "accuracy@n": 0.0,
                "total_examples": 0,
                "avg_time_ms": 0.0,
            }
            for noise_level in args.noise_levels
        },
        "by_noise_type_and_level": {
            f"{noise_type}_{noise_level}": {
                "accuracy@1": 0.0,
                "accuracy@n": 0.0,
                "total_examples": 0,
                "avg_time_ms": 0.0,
            }
            for noise_type in args.noise_types
            for noise_level in args.noise_levels
        },
        "examples": [],
    }

    # Initialize counters
    correct_at_1 = defaultdict(int)
    correct_at_n = defaultdict(int)
    total = defaultdict(int)
    total_time = defaultdict(float)

    # Load the wordlist
    words = load_wordlist(args.wordlist)
    if not words:
        logger.error(f"No words loaded from {args.wordlist}. Exiting.")
        return results

    # Set random seed
    random.seed(args.seed)

    # Sample random words
    if len(words) > args.num_examples:
        sampled_words = random.sample(words, args.num_examples)
    else:
        sampled_words = words

    logger.info(f"Sampled {len(sampled_words)} words from {args.wordlist}")

    # Evaluate for each noise type and level
    for noise_type in args.noise_types:
        for noise_level in args.noise_levels:
            # Generate noisy pairs
            noisy_pairs = generate_noisy_pairs(
                sampled_words, noise_type, noise_level, num_variants=1
            )

            logger.info(
                f"Generated {len(noisy_pairs)} pairs for {noise_type}/{noise_level}"
            )

            # Process each pair
            for pair in noisy_pairs:
                intended = pair["intended"]
                noisy = pair["noisy"]

                # Skip if noisy is the same as intended (no noise)
                if noisy == intended:
                    continue

                # Correct the noisy utterance and measure time
                start_time = time.time()
                corrections = corrector.correct(
                    noisy,
                    max_edit_distance=args.max_edit_distance,
                    use_keyboard_adjacency=not args.no_keyboard_adjacency,
                )
                end_time = time.time()
                correction_time_ms = (end_time - start_time) * 1000

                # Extract corrections
                correction_texts = [corr for corr, _ in corrections]

                # Check accuracy@1
                correct_1 = (
                    correction_texts[0].lower() == intended.lower()
                    if correction_texts
                    else False
                )

                # Check accuracy@N
                correct_n = any(
                    corr.lower() == intended.lower() for corr in correction_texts
                )

                # Update counters
                key_overall = "overall"
                key_type = noise_type
                key_level = noise_level
                key_type_level = f"{noise_type}_{noise_level}"

                for key in [key_overall, key_type, key_level, key_type_level]:
                    total[key] += 1
                    if correct_1:
                        correct_at_1[key] += 1
                    if correct_n:
                        correct_at_n[key] += 1
                    total_time[key] += correction_time_ms

                # Add example to results
                results["examples"].append(
                    {
                        "intended": intended,
                        "noisy": noisy,
                        "noise_type": noise_type,
                        "noise_level": noise_level,
                        "corrections": [
                            {"correction": corr, "score": score}
                            for corr, score in corrections
                        ],
                        "correct_at_1": correct_1,
                        "correct_at_n": correct_n,
                        "time_ms": correction_time_ms,
                    }
                )

    # Calculate accuracy metrics
    for key in total:
        if key == "overall":
            results["overall"]["total_examples"] = total[key]
            results["overall"]["accuracy@1"] = (
                correct_at_1[key] / total[key] if total[key] > 0 else 0.0
            )
            results["overall"]["accuracy@n"] = (
                correct_at_n[key] / total[key] if total[key] > 0 else 0.0
            )
            results["overall"]["avg_time_ms"] = (
                total_time[key] / total[key] if total[key] > 0 else 0.0
            )
        elif key in args.noise_types:
            results["by_noise_type"][key]["total_examples"] = total[key]
            results["by_noise_type"][key]["accuracy@1"] = (
                correct_at_1[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_type"][key]["accuracy@n"] = (
                correct_at_n[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_type"][key]["avg_time_ms"] = (
                total_time[key] / total[key] if total[key] > 0 else 0.0
            )
        elif key in args.noise_levels:
            results["by_noise_level"][key]["total_examples"] = total[key]
            results["by_noise_level"][key]["accuracy@1"] = (
                correct_at_1[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_level"][key]["accuracy@n"] = (
                correct_at_n[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_level"][key]["avg_time_ms"] = (
                total_time[key] / total[key] if total[key] > 0 else 0.0
            )
        else:
            results["by_noise_type_and_level"][key]["total_examples"] = total[key]
            results["by_noise_type_and_level"][key]["accuracy@1"] = (
                correct_at_1[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_type_and_level"][key]["accuracy@n"] = (
                correct_at_n[key] / total[key] if total[key] > 0 else 0.0
            )
            results["by_noise_type_and_level"][key]["avg_time_ms"] = (
                total_time[key] / total[key] if total[key] > 0 else 0.0
            )

    return results


def print_results(results: Dict[str, Any]):
    """Print evaluation results in a readable format."""
    print("\n=== Evaluation Results ===\n")

    # Print overall results
    print("Overall:")
    print(f"  Total examples: {results['overall']['total_examples']}")
    print(f"  Accuracy@1: {results['overall']['accuracy@1']:.4f}")
    print(f"  Accuracy@N: {results['overall']['accuracy@n']:.4f}")
    print(f"  Average time: {results['overall']['avg_time_ms']:.2f} ms")

    # Print results by noise type
    print("\nBy Noise Type:")
    for noise_type, metrics in results["by_noise_type"].items():
        if metrics["total_examples"] > 0:
            print(f"  {noise_type}:")
            print(f"    Total examples: {metrics['total_examples']}")
            print(f"    Accuracy@1: {metrics['accuracy@1']:.4f}")
            print(f"    Accuracy@N: {metrics['accuracy@n']:.4f}")
            print(f"    Average time: {metrics['avg_time_ms']:.2f} ms")

    # Print results by noise level
    print("\nBy Noise Level:")
    for noise_level, metrics in results["by_noise_level"].items():
        if metrics["total_examples"] > 0:
            print(f"  {noise_level}:")
            print(f"    Total examples: {metrics['total_examples']}")
            print(f"    Accuracy@1: {metrics['accuracy@1']:.4f}")
            print(f"    Accuracy@N: {metrics['accuracy@n']:.4f}")
            print(f"    Average time: {metrics['avg_time_ms']:.2f} ms")


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

    # Use the noise simulator if requested
    if args.use_noise_simulator:
        logger.info("Using the noise simulator")
        results = evaluate_accuracy_noise_simulator(corrector, args)
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

        # Evaluate the corrector
        results = evaluate_accuracy_dataset(dataset, corrector, args)

    # Print the results
    print_results(results)

    # Save the results
    if args.output:
        save_results(results, args.output)
        logger.info(f"Saved evaluation results to {args.output}")


if __name__ == "__main__":
    main()
