#!/usr/bin/env python3
"""
Simulate errors in AAC input data.

This script generates noisy versions of clean input text by simulating
various types of errors commonly seen in AAC (Augmentative and Alternative
Communication) systems. It supports multiple error models and can be configured
for different error rates and patterns.

Usage:
    python simulate.py --input data/wordlist.txt --output data/noisy_pairs.json
    python simulate.py --input data/wordlist.txt --output data/noisy_pairs.json --config config.json
    python simulate.py --input data/wordlist.txt --output data/noisy_pairs.json --layout fr --method eyegaze

Options:
    --input: Path to the input file containing clean text (one per line)
    --output: Path to the output JSON file for noisy pairs
    --config: Path to a JSON configuration file for the noise model
    --layout: Keyboard layout language code (e.g., 'en', 'fr', 'de')
    --method: Input method ('direct', 'scanning', 'eyegaze')
    --variants: Number of noisy variants to generate for each input
    --proximity: Proximity error rate
    --deletion: Deletion error rate
    --insertion: Insertion error rate
    --transposition: Transposition error rate
"""

import argparse
import json
import logging
import os
import sys
from typing import List, Dict, Any

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import noise models
from module1.noise_model import (
    NoiseModel,
    KeyboardNoiseModel,
    DwellNoiseModel,
    TranspositionNoiseModel,
    CompositeNoiseModel,
    create_noise_model,
    load_noise_model_from_json,
    save_noisy_pairs,
)

# Import keyboard layouts
from module1.language_keyboards import LANGUAGE_NAMES, KEYBOARD_LAYOUTS


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Simulate errors in AAC input data")

    # Required arguments
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the input file containing clean text (one per line)",
    )
    parser.add_argument(
        "--output", required=True, help="Path to the output JSON file for noisy pairs"
    )

    # Configuration options
    parser.add_argument(
        "--config", help="Path to a JSON configuration file for the noise model"
    )
    parser.add_argument(
        "--layout",
        default="en",
        help="Keyboard layout language code (e.g., 'en', 'fr', 'de')",
    )
    parser.add_argument(
        "--method",
        default="direct",
        choices=["direct", "scanning", "eyegaze"],
        help="Input method ('direct', 'scanning', 'eyegaze')",
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Number of noisy variants to generate for each input",
    )

    # Error rates
    parser.add_argument(
        "--proximity", type=float, default=0.05, help="Proximity error rate"
    )
    parser.add_argument(
        "--deletion", type=float, default=0.03, help="Deletion error rate"
    )
    parser.add_argument(
        "--insertion", type=float, default=0.02, help="Insertion error rate"
    )
    parser.add_argument(
        "--transposition", type=float, default=0.01, help="Transposition error rate"
    )

    return parser.parse_args()


def load_input_texts(input_path: str) -> List[str]:
    """
    Load clean input texts from a file.

    Args:
        input_path: Path to the input file

    Returns:
        List of clean input texts
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]

        logger.info(f"Loaded {len(texts)} input texts from {input_path}")
        return texts

    except Exception as e:
        logger.error(f"Error loading input texts from {input_path}: {e}")
        return []


def create_noise_model_from_args(args) -> NoiseModel:
    """
    Create a noise model from command-line arguments.

    Args:
        args: Command-line arguments

    Returns:
        A configured noise model
    """
    # If a config file is provided, load it
    if args.config:
        return load_noise_model_from_json(args.config)

    # Otherwise, create a composite model from the command-line arguments
    keyboard_model = KeyboardNoiseModel(
        layout_name=args.layout,
        error_rates={
            "proximity": args.proximity,
            "deletion": 0.0,  # We'll use the DwellNoiseModel for deletions
            "insertion": 0.0,  # We'll use the DwellNoiseModel for insertions
            "transposition": 0.0,  # We'll use the TranspositionNoiseModel for transpositions
        },
        input_method=args.method,
    )

    dwell_model = DwellNoiseModel(
        deletion_rate=args.deletion, insertion_rate=args.insertion
    )

    transposition_model = TranspositionNoiseModel(transposition_rate=args.transposition)

    return CompositeNoiseModel(
        models=[keyboard_model, dwell_model, transposition_model]
    )


def main():
    """Main function."""
    # Parse command-line arguments
    args = parse_args()

    # Load input texts
    texts = load_input_texts(args.input)
    if not texts:
        logger.error("No input texts loaded. Exiting.")
        sys.exit(1)

    # Create noise model
    noise_model = create_noise_model_from_args(args)

    # Generate noisy pairs
    logger.info(f"Generating {args.variants} noisy variants for each input text...")
    pairs = noise_model.generate_noisy_pairs(texts, num_variants=args.variants)

    # Save noisy pairs
    success = save_noisy_pairs(pairs, args.output)
    if not success:
        logger.error("Failed to save noisy pairs. Exiting.")
        sys.exit(1)

    logger.info(f"Successfully generated {len(pairs)} noisy pairs.")
    logger.info(f"Output saved to {args.output}")


if __name__ == "__main__":
    main()
