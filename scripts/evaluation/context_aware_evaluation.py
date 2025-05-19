#!/usr/bin/env python3
"""
Evaluation script for context-aware correction using real AAC conversations.

This script evaluates the enhanced context-aware correction functionality
using real conversations from the AACConversations dataset or local JSON files.
It compares different context strategies and reports detailed metrics.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any
from collections import defaultdict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the corrector and utilities
from lib.corrector.corrector import NoisyChannelCorrector
from lib.corrector.enhanced_ppm_predictor import EnhancedPPMPredictor
from lib.corrector.word_ngram_model import WordNGramModel
from lib.confusion_matrix.confusion_matrix import ConfusionMatrix
from lib.confusion_matrix.keyboard_confusion_matrix import KeyboardConfusionMatrix
from lib.candidate_generator.improved_candidate_generator import (
    ImprovedCandidateGenerator,
)
from scripts.evaluation.utils import (
    load_aac_conversations,
    group_by_conversation,
    prepare_conversation_for_evaluation,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ContextAwareEvaluator:
    """
    Class for evaluating context-aware correction strategies on real AAC conversations.
    """

    def __init__(
        self,
        ppm_model_path: str,
        confusion_matrix_path: str,
        word_ngram_model_path: str,
        lexicon_path: str,
        max_candidates: int = 5,
        context_window_size: int = 3,
    ):
        """
        Initialize the evaluator with model paths and parameters.

        Args:
            ppm_model_path: Path to the PPM model file
            confusion_matrix_path: Path to the confusion matrix file
            word_ngram_model_path: Path to the word n-gram model file
            lexicon_path: Path to the lexicon file
            max_candidates: Maximum number of candidates to return
            context_window_size: Number of previous words to use as context
        """
        self.ppm_model_path = ppm_model_path
        self.confusion_matrix_path = confusion_matrix_path
        self.word_ngram_model_path = word_ngram_model_path
        self.lexicon_path = lexicon_path
        self.max_candidates = max_candidates
        self.context_window_size = context_window_size

        # Load models once
        self.ppm_model = None
        self.confusion_model = None
        self.word_ngram_model = None
        self.lexicon = set()

        # Load PPM model
        if os.path.exists(self.ppm_model_path):
            self.ppm_model = EnhancedPPMPredictor()
            self.ppm_model.load_model(self.ppm_model_path)
            logger.info(f"Loaded PPM model from {self.ppm_model_path}")
        else:
            logger.warning(f"PPM model file not found: {self.ppm_model_path}")

        # Load confusion matrix
        if os.path.exists(self.confusion_matrix_path):
            # Determine if it's a keyboard-specific matrix
            with open(self.confusion_matrix_path, "r") as f:
                import json

                data = json.load(f)
                if "keyboard_layouts" in data:
                    self.confusion_model = KeyboardConfusionMatrix.load(
                        self.confusion_matrix_path
                    )
                else:
                    self.confusion_model = ConfusionMatrix.load(
                        self.confusion_matrix_path
                    )
                logger.info(
                    f"Loaded confusion matrix from {self.confusion_matrix_path}"
                )
        else:
            logger.warning(
                f"Confusion matrix file not found: {self.confusion_matrix_path}"
            )

        # Load word n-gram model
        if os.path.exists(self.word_ngram_model_path):
            self.word_ngram_model = WordNGramModel()
            self.word_ngram_model.load(self.word_ngram_model_path)
            logger.info(f"Loaded word n-gram model from {self.word_ngram_model_path}")
        else:
            logger.warning(
                f"Word n-gram model file not found: {self.word_ngram_model_path}"
            )

        # Load lexicon
        if os.path.exists(self.lexicon_path):
            with open(self.lexicon_path, "r") as f:
                self.lexicon = set(line.strip() for line in f if line.strip())
            logger.info(
                f"Loaded lexicon with {len(self.lexicon)} words from {self.lexicon_path}"
            )
        else:
            logger.warning(f"Lexicon file not found: {self.lexicon_path}")

        # Create correctors with different strategies (but shared models)
        # Limit the number of candidates to a reasonable amount
        self.candidate_generator = ImprovedCandidateGenerator(
            lexicon=self.lexicon,
            max_candidates=10,  # Increase from 5 to 10
            max_edits=50,  # Drastically reduce from 20000 to 50
            keyboard_boost=0.5,  # Increase keyboard boost
            strict_filtering=False,  # Don't be too strict with filtering
            smart_filtering=True,
            use_frequency_info=True,
        )

        self.correctors = {
            "no_context": self._create_corrector(
                use_conversation_context=False,
                adaptive_context_weighting=False,
                context_weight=0.0,
            ),
            "simple_context": self._create_corrector(
                use_conversation_context=False,
                adaptive_context_weighting=False,
                context_weight=0.5,
            ),
            "conversation_context": self._create_corrector(
                use_conversation_context=True,
                adaptive_context_weighting=False,
                context_weight=0.5,
            ),
            "adaptive_context": self._create_corrector(
                use_conversation_context=True,
                adaptive_context_weighting=True,
                context_weight=0.5,
            ),
        }

    def _create_corrector(
        self,
        use_conversation_context: bool,
        adaptive_context_weighting: bool,
        context_weight: float,
    ) -> NoisyChannelCorrector:
        """
        Create a corrector with the specified strategy, using the already loaded models.

        Args:
            use_conversation_context: Whether to use conversation-level context
            adaptive_context_weighting: Whether to use adaptive context weighting
            context_weight: Weight of context-based probability (0-1)

        Returns:
            Configured NoisyChannelCorrector
        """
        corrector = NoisyChannelCorrector(
            ppm_model=self.ppm_model,
            confusion_model=self.confusion_model,
            word_ngram_model=self.word_ngram_model,
            lexicon=self.lexicon,
            max_candidates=self.max_candidates,
            context_window_size=self.context_window_size,
            context_weight=context_weight,
            keyboard_layout="qwerty",
            use_conversation_context=use_conversation_context,
            adaptive_context_weighting=adaptive_context_weighting,
        )

        # Use our shared candidate generator with limited candidates
        corrector.candidate_generator = self.candidate_generator

        return corrector

    def load_conversations_from_file(
        self, file_path: str
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Load conversations from a JSON file.

        Args:
            file_path: Path to the JSON file

        Returns:
            Dictionary mapping conversation IDs to lists of turns
        """
        try:
            with open(file_path, "r") as f:
                conversations = json.load(f)
            logger.info(f"Loaded {len(conversations)} conversations from {file_path}")
            return conversations
        except Exception as e:
            logger.error(f"Error loading conversations from {file_path}: {e}")
            return {}

    def evaluate_conversation(
        self,
        conversation: List[Dict[str, Any]],
        strategy: str,
        noise_type: str = "qwerty",
        noise_level: str = "moderate",
    ) -> Dict[str, Any]:
        """
        Evaluate a conversation using the specified correction strategy.

        Args:
            conversation: List of conversation turns
            strategy: Correction strategy to use
            noise_type: Type of noise ('qwerty', 'abc', or 'frequency')
            noise_level: Level of noise ('minimal', 'light', 'moderate', or 'severe')

        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Evaluating conversation with strategy: {strategy}")
        # Get the corrector for the specified strategy
        corrector = self.correctors.get(strategy)
        if not corrector:
            logger.error(f"Unknown correction strategy: {strategy}")
            return {}

        # Prepare the conversation for evaluation
        logger.info(
            f"Preparing conversation for evaluation with noise type: {noise_type}, level: {noise_level}"
        )
        prepared_conversation = prepare_conversation_for_evaluation(
            conversation, noise_type, noise_level
        )
        logger.info(f"Prepared conversation with {len(prepared_conversation)} turns")

        # Initialize results
        results = {
            "conversation_id": (
                prepared_conversation[0].get("conversation_id", "unknown")
                if prepared_conversation
                else "unknown"
            ),
            "strategy": strategy,
            "noise_type": noise_type,
            "noise_level": noise_level,
            "turns": [],
            "metrics": {
                "accuracy": 0.0,
                "turn_level_accuracy": [],
                "accuracy_by_position": defaultdict(list),
                "accuracy_by_speaker": defaultdict(list),
            },
        }

        # Initialize context
        context = []

        # Process each turn
        for i, turn in enumerate(prepared_conversation):
            # Get the noisy utterance and minimally corrected utterance (the target)
            noisy = turn.get("noisy_utterance", "")
            intended = turn.get("minimally_corrected", turn.get("utterance", ""))
            logger.info(f"Processing turn - Noisy: '{noisy}', Target: '{intended}'")
            speaker = turn.get("speaker", "Unknown")

            # Skip if no noisy utterance or intended utterance
            if not noisy or not intended:
                continue

            # Only process AAC user turns
            if "aac" not in speaker.lower() and "user" not in speaker.lower():
                # Add non-AAC user utterances to context
                context.append(intended)

                # Update conversation context if using it
                if strategy in ["conversation_context", "adaptive_context"]:
                    corrector.update_conversation_context(intended, speaker)

                continue

            # Correct with context
            logger.info(f"Correcting: '{noisy}' with context: {context}")
            corrections = corrector.correct(noisy, context=context)

            # Log all corrections
            for i, (correction, score) in enumerate(corrections[:5]):  # Show top 5
                logger.info(f"  Candidate {i+1}: '{correction}' (Score: {score:.4f})")

            # Get the top correction
            top_correction = corrections[0][0] if corrections else noisy
            logger.info(f"Top correction: '{top_correction}'")

            # Update context with the corrected utterance
            context.append(top_correction)

            # Update conversation context if using it
            if strategy in ["conversation_context", "adaptive_context"]:
                corrector.update_conversation_context(top_correction, speaker)

            # Limit context size
            context = context[-self.context_window_size :]

            # Calculate turn-level metrics
            correct_at_1 = top_correction.lower() == intended.lower()
            correct_at_n = any(
                corr.lower() == intended.lower() for corr, _ in corrections
            )

            # Find the rank of the correct answer
            correct_rank = -1
            for j, (corr, _) in enumerate(corrections):
                if corr.lower() == intended.lower():
                    correct_rank = j + 1
                    break

            # Add to results
            results["turns"].append(
                {
                    "turn_number": i + 1,
                    "speaker": speaker,
                    "intended": intended,
                    "noisy": noisy,
                    "corrections": [
                        {"correction": corr, "score": score}
                        for corr, score in corrections
                    ],
                    "correct_at_1": correct_at_1,
                    "correct_at_n": correct_at_n,
                    "correct_rank": correct_rank,
                    "context": list(context),  # Make a copy of the context
                }
            )

            # Update metrics
            results["metrics"]["turn_level_accuracy"].append(correct_at_1)
            results["metrics"]["accuracy_by_position"][i + 1].append(correct_at_1)
            results["metrics"]["accuracy_by_speaker"][speaker].append(correct_at_1)

        # Calculate conversation-level metrics
        if results["metrics"]["turn_level_accuracy"]:
            results["metrics"]["accuracy"] = sum(
                results["metrics"]["turn_level_accuracy"]
            ) / len(results["metrics"]["turn_level_accuracy"])

            # Calculate accuracy by position
            for position, accuracies in results["metrics"][
                "accuracy_by_position"
            ].items():
                if accuracies:
                    results["metrics"]["accuracy_by_position"][position] = sum(
                        accuracies
                    ) / len(accuracies)

            # Calculate accuracy by speaker
            for speaker, accuracies in results["metrics"][
                "accuracy_by_speaker"
            ].items():
                if accuracies:
                    results["metrics"]["accuracy_by_speaker"][speaker] = sum(
                        accuracies
                    ) / len(accuracies)

        return results

    def evaluate_conversations(
        self,
        conversations: Dict[str, List[Dict[str, Any]]],
        strategies: List[str] = None,
        noise_type: str = "qwerty",
        noise_level: str = "moderate",
    ) -> Dict[str, Any]:
        """
        Evaluate multiple conversations using different correction strategies.

        Args:
            conversations: Dictionary mapping conversation IDs to lists of turns
            strategies: List of correction strategies to evaluate
            noise_type: Type of noise ('qwerty', 'abc', or 'frequency')
            noise_level: Level of noise ('minimal', 'light', 'moderate', or 'severe')

        Returns:
            Dictionary with evaluation results
        """
        if strategies is None:
            strategies = list(self.correctors.keys())

        # Initialize results
        all_results = {
            "strategies": strategies,
            "noise_type": noise_type,
            "noise_level": noise_level,
            "conversations": [],
            "metrics": {
                "overall_accuracy": {},
                "conversation_accuracies": defaultdict(list),
            },
        }

        # Process each conversation with each strategy
        for _, conversation in conversations.items():
            conversation_results = []

            for strategy in strategies:
                # Evaluate the conversation with this strategy
                result = self.evaluate_conversation(
                    conversation, strategy, noise_type, noise_level
                )

                # Add to conversation results
                conversation_results.append(result)

                # Update overall metrics
                all_results["metrics"]["conversation_accuracies"][strategy].append(
                    result["metrics"]["accuracy"]
                )

            # Add to all results
            all_results["conversations"].append(conversation_results)

        # Calculate overall metrics for each strategy
        for strategy in strategies:
            accuracies = all_results["metrics"]["conversation_accuracies"][strategy]
            if accuracies:
                all_results["metrics"]["overall_accuracy"][strategy] = sum(
                    accuracies
                ) / len(accuracies)
            else:
                all_results["metrics"]["overall_accuracy"][strategy] = 0.0

        return all_results

    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of the evaluation results.

        Args:
            results: The evaluation results
        """
        print("\n=== Context-Aware Correction Evaluation Summary ===\n")

        # Print overall metrics
        print("Overall Accuracy by Strategy:")
        for strategy, accuracy in results["metrics"]["overall_accuracy"].items():
            print(f"  {strategy}: {accuracy:.4f}")

        # Print number of conversations and turns
        num_conversations = len(results["conversations"])
        num_turns = sum(
            len(conv[0]["turns"]) for conv in results["conversations"] if conv
        )
        print(f"\nNumber of Conversations: {num_conversations}")
        print(f"Total Turns: {num_turns}")

        # Print accuracy comparison
        print("\nAccuracy Comparison:")
        strategies = results["strategies"]

        # Calculate improvement over baseline (no_context)
        if "no_context" in results["metrics"]["overall_accuracy"]:
            baseline = results["metrics"]["overall_accuracy"]["no_context"]
            print(f"  Baseline (no_context): {baseline:.4f}")

            for strategy in strategies:
                if strategy != "no_context":
                    accuracy = results["metrics"]["overall_accuracy"][strategy]
                    improvement = accuracy - baseline
                    improvement_pct = (
                        (improvement / baseline) * 100 if baseline > 0 else 0
                    )
                    print(f"  {strategy}: {accuracy:.4f} ({improvement_pct:+.2f}%)")

        # Print example conversation results
        if results["conversations"]:
            print("\nExample Conversation Results:")

            # Get the first conversation
            conversation_results = results["conversations"][0]

            # Print results for each strategy
            for result in conversation_results:
                strategy = result["strategy"]
                accuracy = result["metrics"]["accuracy"]
                print(f"\n  Strategy: {strategy} (Accuracy: {accuracy:.4f})")

                # Print a few example turns
                for i, turn in enumerate(result["turns"][:3]):  # Show first 3 turns
                    print(f"    Turn {i+1} - {turn['speaker']}:")
                    print(f"      Intended: {turn['intended']}")
                    print(f"      Noisy:    {turn['noisy']}")
                    corr = turn["corrections"][0]["correction"]
                    score = turn["corrections"][0]["score"]
                    print(f"      Corrected: {corr} (Score: {score:.4f})")
                    print(f"      Correct:  {'✓' if turn['correct_at_1'] else '✗'}")
                    print()

                if len(result["turns"]) > 3:
                    print(f"    ... ({len(result['turns']) - 3} more turns)")


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate context-aware correction on real AAC conversations."
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
        "--conversations-file",
        type=str,
        default="data/english_conversations.json",
        help="Path to the conversations JSON file",
    )

    parser.add_argument(
        "--use-huggingface",
        action="store_true",
        help="Use HuggingFace dataset instead of local file",
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use (only with --use-huggingface)",
    )

    parser.add_argument(
        "--num-conversations",
        type=int,
        default=5,
        help="Number of conversations to evaluate",
    )

    parser.add_argument(
        "--noise-type",
        choices=["qwerty", "abc", "frequency"],
        default="qwerty",
        help="Type of noise to use from the dataset",
    )

    parser.add_argument(
        "--noise-level",
        choices=["minimal", "light", "moderate", "severe"],
        default="moderate",
        help="Level of noise to use from the dataset",
    )

    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=[
            "no_context",
            "simple_context",
            "conversation_context",
            "adaptive_context",
        ],
        default=["no_context"],
        help="Correction strategies to evaluate",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to save the evaluation results",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache the dataset (only with --use-huggingface)",
    )

    parser.add_argument(
        "--no-token",
        action="store_true",
        help="Don't use the Hugging Face auth token (only with --use-huggingface)",
    )

    args = parser.parse_args()

    # Create the evaluator
    evaluator = ContextAwareEvaluator(
        ppm_model_path=args.ppm_model,
        confusion_matrix_path=args.confusion_matrix,
        word_ngram_model_path=args.word_ngram_model,
        lexicon_path=args.lexicon,
        max_candidates=5,
        context_window_size=3,
    )

    # Load conversations
    if args.use_huggingface:
        # Load from HuggingFace
        dataset = load_aac_conversations(
            split=args.dataset_split,
            cache_dir=args.cache_dir,
            use_auth_token=not args.no_token,
        )

        if dataset is None:
            logger.error("Failed to load dataset. Exiting.")
            return

        # Group by conversation
        conversations = group_by_conversation(dataset, args.num_conversations)
    else:
        # Load from local file
        conversations = evaluator.load_conversations_from_file(args.conversations_file)

        # Limit the number of conversations if specified
        if args.num_conversations < len(conversations):
            conversation_ids = list(conversations.keys())[: args.num_conversations]
            conversations = {cid: conversations[cid] for cid in conversation_ids}

    # Evaluate conversations
    results = evaluator.evaluate_conversations(
        conversations,
        strategies=args.strategies,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
    )

    # Print the summary
    evaluator.print_summary(results)

    # Save the results
    if args.output:
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

            # Convert defaultdicts to regular dicts for JSON serialization
            serializable_results = json.loads(
                json.dumps(
                    results,
                    default=lambda x: dict(x) if isinstance(x, defaultdict) else x,
                )
            )

            # Save the results
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"Saved evaluation results to {args.output}")
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")


if __name__ == "__main__":
    main()
