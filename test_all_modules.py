#!/usr/bin/env python3
"""
Test script to verify that all modules work in isolation and together.
"""

import sys
import subprocess
import argparse


def run_command(command, description):
    """Run a command and print the result."""
    print(f"\n{'='*80}\n{description}\n{'='*80}")
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"SUCCESS: {description}")
        if result.stdout:
            print(f"Output:\n{result.stdout[:500]}...")
        return True
    else:
        print(f"FAILED: {description}")
        print(f"Error:\n{result.stderr}")
        return False


def test_noise_simulator():
    """Test Noise Simulator Module"""
    print("\n\n## Testing Noise Simulator Module ##\n")

    # Test the noise model
    success = run_command(
        "python -c \"import sys; sys.path.append('.'); from lib.noise_model.noise_model import NoiseModel; print('NoiseModel imported successfully')\"",
        "Testing noise model import",
    )

    if success:
        success = run_command(
            "python scripts/noise_simulator/simulate.py --input data/wordlist.txt --output data/test_noisy_pairs.json",
            "Testing noise simulation",
        )

    return success


def test_confusion_matrix_builder():
    """Test Confusion Matrix Builder Module"""
    print("\n\n## Testing Confusion Matrix Builder Module ##\n")

    # Test the confusion matrix
    success = run_command(
        "python -c \"import sys; sys.path.append('.'); from lib.confusion_matrix.confusion_matrix import ConfusionMatrix; print('ConfusionMatrix imported successfully')\"",
        "Testing confusion matrix import",
    )

    if success:
        success = run_command(
            "python scripts/confusion_matrix_builder/build_matrix.py --input data/noisy_pairs.json --output data/test_confusion_matrix.json",
            "Testing confusion matrix generation",
        )

    return success


def test_candidate_generator():
    """Test Candidate Generator Module"""
    print("\n\n## Testing Candidate Generator Module ##\n")

    # Test the candidate generator
    success = run_command(
        "python -c \"import sys; sys.path.append('.'); from lib.candidate_generator.candidate_generator import CandidateGenerator; print('CandidateGenerator imported successfully')\"",
        "Testing candidate generator import",
    )

    if success:
        success = run_command(
            "python scripts/candidate_generator/generate_candidates.py --input 'thes is a tst' --lexicon data/comprehensive_lexicon.txt",
            "Testing candidate generation",
        )

    return success


def test_correction_engine():
    """Test Correction Engine Module"""
    print("\n\n## Testing Correction Engine Module ##\n")

    # Test the corrector
    success = run_command(
        "python -c \"import sys; sys.path.append('.'); from lib.corrector.corrector import NoisyChannelCorrector; print('NoisyChannelCorrector imported successfully')\"",
        "Testing corrector import",
    )

    if success:
        success = run_command(
            "python scripts/correction_engine/correct.py --input 'thes is a tst' --output data/test_corrected.txt",
            "Testing correction engine",
        )

    return success


def test_evaluation():
    """Test Evaluation Module"""
    print("\n\n## Testing Evaluation Module ##\n")

    # Test the evaluation
    success = run_command(
        "python scripts/evaluation/eval.py --use-noise-simulator --wordlist data/wordlist.txt --output data/test_eval_results.json",
        "Testing evaluation",
    )

    return success


def test_integration():
    """Test all modules working together"""
    print("\n\n## Testing Integration ##\n")

    # Test the demo in non-interactive mode
    success = run_command(
        "python scripts/evaluation/demo.py --use-noise-simulator --wordlist data/wordlist.txt",
        "Testing integrated demo",
    )

    return success


def test_all_modules():
    """Test all modules in sequence."""
    results = {
        "Noise Simulator": test_noise_simulator(),
        "Confusion Matrix Builder": test_confusion_matrix_builder(),
        "Candidate Generator": test_candidate_generator(),
        "Correction Engine": test_correction_engine(),
        "Evaluation": test_evaluation(),
        "Integration": test_integration(),
    }

    print("\n\n## Summary ##\n")
    for module, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"{module}: {status}")

    return all(results.values())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test all modules in the noisy channel correction system."
    )
    parser.add_argument(
        "--module",
        type=str,
        choices=[
            "noise_simulator",
            "confusion_matrix_builder",
            "candidate_generator",
            "correction_engine",
            "evaluation",
        ],
        help="Test a specific module",
    )
    parser.add_argument(
        "--integration", action="store_true", help="Test integration of all modules"
    )
    args = parser.parse_args()

    if args.module:
        if args.module == "noise_simulator":
            success = test_noise_simulator()
        elif args.module == "confusion_matrix_builder":
            success = test_confusion_matrix_builder()
        elif args.module == "candidate_generator":
            success = test_candidate_generator()
        elif args.module == "correction_engine":
            success = test_correction_engine()
        elif args.module == "evaluation":
            success = test_evaluation()

        sys.exit(0 if success else 1)
    elif args.integration:
        success = test_integration()
        sys.exit(0 if success else 1)
    else:
        success = test_all_modules()
        sys.exit(0 if success else 1)
