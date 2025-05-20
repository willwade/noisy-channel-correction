#!/usr/bin/env python3
"""
Synthetic Conversation Evaluator

This script provides functionality for evaluating the correction engine on
synthetic conversations. It generates random conversations with configurable
parameters and evaluates the correction performance.
"""

import os
import sys
import argparse
import logging
import json

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the corrector
from lib.corrector.corrector import NoisyChannelCorrector
from scripts.evaluation.conversation_level_evaluation import ConversationLevelEvaluator
from scripts.evaluation.utils import resolve_path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate the correction engine on synthetic conversations."
    )

    parser.add_argument(
        "--ppm-model",
        type=str,
        default="models/ppm_model.pkl",
        help="Path to the PPM model",
    )

    parser.add_argument(
        "--word-ngram-model",
        type=str,
        default="models/word_ngram_model.pkl",
        help="Path to the word n-gram model",
    )

    parser.add_argument(
        "--confusion-matrix",
        type=str,
        default="models/confusion_matrix.json",
        help="Path to the confusion matrix",
    )

    parser.add_argument(
        "--lexicon",
        type=str,
        default="data/enhanced_lexicon.txt",
        help="Path to the lexicon",
    )

    parser.add_argument(
        "--num-conversations",
        type=int,
        default=5,
        help="Number of synthetic conversations to generate",
    )

    parser.add_argument(
        "--num-turns",
        type=int,
        default=10,
        help="Number of turns per synthetic conversation",
    )

    parser.add_argument(
        "--vocabulary-size",
        type=int,
        default=100,
        help="Size of the vocabulary for synthetic conversations",
    )

    parser.add_argument(
        "--error-rate",
        type=float,
        default=0.2,
        help="Error rate for synthetic conversations",
    )

    parser.add_argument(
        "--keyboard-layouts",
        type=str,
        nargs="+",
        default=["qwerty"],
        choices=["qwerty", "abc", "frequency"],
        help="Keyboard layouts to evaluate",
    )

    parser.add_argument(
        "--noise-levels",
        type=str,
        nargs="+",
        default=["minimal"],
        choices=["minimal", "light", "moderate", "severe"],
        help="Noise levels to evaluate",
    )

    parser.add_argument(
        "--context-window-size",
        type=int,
        default=5,
        help="Number of previous utterances to use as context",
    )

    parser.add_argument(
        "--use-gold-context",
        action="store_true",
        help="Use gold standard context instead of corrected utterances",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the evaluation results",
    )

    args = parser.parse_args()

    # Initialize the corrector
    logger.info("Initializing corrector")
    corrector = NoisyChannelCorrector(max_candidates=5)  # Reduced from default 10

    # Load models
    logger.info(f"Loading PPM model from {args.ppm_model}")
    corrector.load_ppm_model(resolve_path(args.ppm_model))

    logger.info(f"Loading word n-gram model from {args.word_ngram_model}")
    corrector.load_word_ngram_model(resolve_path(args.word_ngram_model))

    logger.info(f"Loading confusion matrix from {args.confusion_matrix}")
    corrector.load_confusion_model(resolve_path(args.confusion_matrix))

    logger.info(f"Loading lexicon from {args.lexicon}")
    corrector.load_lexicon_from_file(resolve_path(args.lexicon))

    # Update candidate generator settings to generate fewer candidates
    corrector.candidate_generator.max_edits = 1000  # Reduced from 20000
    corrector.candidate_generator.max_candidates = 100  # Reduced from default

    # Initialize the evaluator
    logger.info("Initializing evaluator")
    evaluator = ConversationLevelEvaluator(
        corrector=corrector,
        context_window_size=args.context_window_size,
        use_gold_context=args.use_gold_context,
        target_field="minimally_corrected",
    )

    # Generate and evaluate synthetic conversations
    logger.info("Generating and evaluating synthetic conversations")
    logger.info(f"Generating {args.num_conversations} conversations with {args.num_turns} turns")
    logger.info(f"Vocabulary size: {args.vocabulary_size}, Error rate: {args.error_rate}")

    results = evaluator.evaluate_synthetic_dataset(
        num_conversations=args.num_conversations,
        num_turns=args.num_turns,
        vocabulary_size=args.vocabulary_size,
        error_rate=args.error_rate,
        keyboard_layouts=args.keyboard_layouts,
        noise_levels=args.noise_levels,
    )

    # Print results
    print("\n=== Synthetic Conversation Evaluation Results ===")
    print(f"Total Conversations: {results['summary']['total_conversations']}")
    print(f"Total Turns: {results['summary']['total_turns']}")
    print("\nSummary by Keyboard Layout and Noise Level:")

    for keyboard in results["summary"]["keyboard_layouts"]:
        for noise in results["summary"]["noise_levels"]:
            print(f"\n{keyboard.upper()} - {noise.upper()} Noise:")

            no_context = results["summary"]["metrics"][keyboard][noise]["no_context"]
            with_context = results["summary"]["metrics"][keyboard][noise]["with_context"]

            print("  No Context:")
            print(
                f"    Accuracy@1: {no_context['accuracy_at_1']:.4f} "
                f"({no_context['correct_at_1']}/{no_context['total']})"
            )
            print(
                f"    Accuracy@N: {no_context['accuracy_at_n']:.4f} "
                f"({no_context['correct_at_n']}/{no_context['total']})"
            )

            print("  With Context:")
            print(
                f"    Accuracy@1: {with_context['accuracy_at_1']:.4f} "
                f"({with_context['correct_at_1']}/{with_context['total']})"
            )
            print(
                f"    Accuracy@N: {with_context['accuracy_at_n']:.4f} "
                f"({with_context['correct_at_n']}/{with_context['total']})"
            )

            # Calculate improvement
            if no_context["total"] > 0:
                improvement_at_1 = (
                    with_context["accuracy_at_1"] - no_context["accuracy_at_1"]
                ) * 100
                improvement_at_n = (
                    with_context["accuracy_at_n"] - no_context["accuracy_at_n"]
                ) * 100
                print("  Improvement with Context:")
                print(f"    Accuracy@1: {improvement_at_1:+.2f}%")
                print(f"    Accuracy@N: {improvement_at_n:+.2f}%")

    # Save results if output path is specified
    if args.output:
        output_path = resolve_path(args.output)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
