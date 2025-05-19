#!/usr/bin/env python3
"""
Debug script for the scoring mechanism in the noisy channel corrector.

This script helps diagnose and fix issues with the scoring mechanism
in the noisy channel corrector. It prints detailed information about
the scoring process for a given input.
"""

import os
import sys
import argparse
import logging
import math

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import the corrector
from lib.corrector.corrector import NoisyChannelCorrector

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Debug the scoring mechanism in the noisy channel corrector."
    )
    parser.add_argument(
        "--input", type=str, required=True, help="The noisy input text to correct"
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
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )
    return parser.parse_args()


def debug_scoring(corrector: NoisyChannelCorrector, noisy_input: str, max_edit_distance: int = 2):
    """
    Debug the scoring mechanism for a given input.

    Args:
        corrector: The noisy channel corrector
        noisy_input: The noisy input text
        max_edit_distance: Maximum edit distance to consider
    """
    print(f"\n=== Debugging Scoring for '{noisy_input}' ===\n")

    # Generate candidates
    print("Generating candidates...")
    candidates = corrector.candidate_generator.generate_candidates(
        noisy_input, max_edit_distance, True
    )
    print(f"Generated {len(candidates)} candidates")

    # Print the top 5 candidates
    print("\nTop 5 candidates (before scoring):")
    for i, (candidate, score) in enumerate(candidates[:5]):
        print(f"  {i+1}. {candidate} (initial score: {score:.4f})")

    # Debug the scoring process
    print("\nScoring process:")
    scored_candidates = []

    for candidate, _ in candidates[:5]:  # Only debug the top 5 candidates
        print(f"\nCandidate: '{candidate}'")

        # Debug P(intended) calculation
        p_intended = corrector._get_p_intended(candidate)
        print(f"  P(intended) = {p_intended:.8f}")

        # Debug word frequency
        if hasattr(corrector.ppm_model, "word_frequencies"):
            freq = corrector.ppm_model.word_frequencies.get(candidate.lower(), 0)
            total_words = sum(corrector.ppm_model.word_frequencies.values())
            print(f"  Word frequency: {freq} / {total_words}")

        # Debug P(noisy | intended) calculation
        p_noisy_given_intended = corrector._get_p_noisy_given_intended(noisy_input, candidate)
        print(f"  P(noisy | intended) = {p_noisy_given_intended:.8f}")

        # Debug alignment and character-level probabilities
        if corrector.confusion_model is not None:
            from lib.confusion_matrix.confusion_matrix import align_strings
            alignment = align_strings(candidate, noisy_input)
            print("  Character-level alignment:")
            for i_char, n_char, op in alignment:
                prob = 0.0
                if op == "match":
                    prob = corrector.confusion_model.get_probability(n_char, i_char)
                elif op == "substitution":
                    prob = corrector.confusion_model.get_probability(n_char, i_char)
                elif op == "deletion":
                    prob = corrector.confusion_model.get_probability("ε", i_char)
                elif op == "insertion":
                    prob = corrector.confusion_model.get_probability(n_char, "φ")
                print(f"    {i_char} → {n_char} ({op}): {prob:.8f}")

        # Calculate the final score
        log_p_intended = math.log(p_intended) if p_intended > 0 else -float("inf")
        log_p_noisy_given_intended = (
            math.log(p_noisy_given_intended) if p_noisy_given_intended > 0 else -float("inf")
        )
        score = log_p_intended + log_p_noisy_given_intended
        final_score = math.exp(score) if score != -float("inf") else 0.0
        print(f"  log(P(intended)) = {log_p_intended:.4f}")
        print(f"  log(P(noisy | intended)) = {log_p_noisy_given_intended:.4f}")
        print(f"  log(P(intended)) + log(P(noisy | intended)) = {score:.4f}")
        print(f"  Final score = exp({score:.4f}) = {final_score:.8f}")

        scored_candidates.append((candidate, final_score))

    # Sort by score (highest first)
    scored_candidates = sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    # Print the final scored candidates
    print("\nFinal scored candidates:")
    for i, (candidate, score) in enumerate(scored_candidates):
        print(f"  {i+1}. {candidate} (score: {score:.8f})")


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create a corrector
    corrector = NoisyChannelCorrector(max_candidates=args.max_candidates)

    # Load the PPM model
    if os.path.exists(args.ppm_model):
        print(f"Loading PPM model from {args.ppm_model}")
        success = corrector.load_ppm_model(args.ppm_model)
        if not success:
            logger.error(f"Failed to load PPM model from {args.ppm_model}")
            return
    else:
        logger.error(f"PPM model file not found: {args.ppm_model}")
        return

    # Load the confusion matrix
    if os.path.exists(args.confusion_matrix):
        print(f"Loading confusion matrix from {args.confusion_matrix}")
        success = corrector.load_confusion_model(args.confusion_matrix)
        if not success:
            logger.error(f"Failed to load confusion matrix from {args.confusion_matrix}")
            return
    else:
        logger.error(f"Confusion matrix file not found: {args.confusion_matrix}")
        return

    # Load the word n-gram model
    if os.path.exists(args.word_ngram_model):
        print(f"Loading word n-gram model from {args.word_ngram_model}")
        success = corrector.load_word_ngram_model(args.word_ngram_model)
        if not success:
            logger.error(f"Failed to load word n-gram model from {args.word_ngram_model}")
            # Continue anyway, as this is not critical
    else:
        logger.warning(f"Word n-gram model file not found: {args.word_ngram_model}")

    # Load the lexicon
    if os.path.exists(args.lexicon):
        print(f"Loading lexicon from {args.lexicon}")
        success = corrector.load_lexicon_from_file(args.lexicon)
        if not success:
            logger.error(f"Failed to load lexicon from {args.lexicon}")
            # Continue anyway, as this is not critical
    else:
        logger.warning(f"Lexicon file not found: {args.lexicon}")

    # Debug the scoring mechanism
    debug_scoring(corrector, args.input, args.max_edit_distance)


if __name__ == "__main__":
    main()
