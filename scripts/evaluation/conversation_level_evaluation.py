#!/usr/bin/env python3
"""
Comprehensive conversation-level evaluation script.

This script evaluates the performance of different correction strategies
on conversations from the AACConversations dataset or synthetic conversations.
It processes conversations turn-by-turn, tracks accuracy metrics, and compares
performance across different noise levels, types, and keyboard layouts.

Features:
- Support for multiple keyboard layouts (qwerty, abc, frequency)
- Support for multiple noise levels (minimal, light, moderate, severe)
- Support for language filtering (e.g., en-GB)
- Support for synthetic conversations
- Context-aware correction evaluation
- Detailed metrics and visualizations
"""

import os
import sys
import argparse
import logging
import json
import random
from collections import defaultdict
from typing import Dict, List, Any, Optional

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import utilities
from scripts.evaluation.utils import (
    load_aac_conversations,
    group_by_conversation,
    prepare_conversation_for_evaluation,
    resolve_path,
)

# Import correction engine
try:
    from lib.corrector.corrector import NoisyChannelCorrector
except ModuleNotFoundError:
    # Try alternative import paths
    from corrector.corrector import NoisyChannelCorrector


class ConversationLevelEvaluator:
    """Evaluator for conversation-level correction."""

    def __init__(
        self,
        corrector: NoisyChannelCorrector,
        context_window_size: int = 5,
        use_gold_context: bool = False,
        target_field: str = "minimally_corrected",
    ):
        """
        Initialize the evaluator.

        Args:
            corrector: The corrector to use
            context_window_size: Number of previous utterances to use as context
            use_gold_context: Whether to use gold standard context
            target_field: Field to use as the target for evaluation
        """
        self.corrector = corrector
        self.context_window_size = context_window_size
        self.use_gold_context = use_gold_context
        self.target_field = target_field

    def generate_synthetic_conversation(
        self,
        num_turns: int = 10,
        vocabulary_size: int = 100,
        error_rate: float = 0.2,
        speakers: List[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate a synthetic conversation for evaluation.

        Args:
            num_turns: Number of turns in the conversation
            vocabulary_size: Size of the vocabulary to use
            error_rate: Probability of introducing an error in each word
            speakers: List of speaker names to use (defaults to ["User", "AAC User"])

        Returns:
            List of conversation turns
        """
        if speakers is None:
            speakers = ["User", "AAC User"]

        # Create a simple vocabulary
        if hasattr(self.corrector, "lexicon") and self.corrector.lexicon:
            # Use words from the lexicon
            vocabulary = list(self.corrector.lexicon)
            if len(vocabulary) > vocabulary_size:
                vocabulary = random.sample(vocabulary, vocabulary_size)
        else:
            # Create a simple vocabulary with common words
            vocabulary = [
                "the",
                "be",
                "to",
                "of",
                "and",
                "a",
                "in",
                "that",
                "have",
                "I",
                "it",
                "for",
                "not",
                "on",
                "with",
                "he",
                "as",
                "you",
                "do",
                "at",
                "this",
                "but",
                "his",
                "by",
                "from",
                "they",
                "we",
                "say",
                "her",
                "she",
                "or",
                "an",
                "will",
                "my",
                "one",
                "all",
                "would",
                "there",
                "their",
                "what",
                "so",
                "up",
                "out",
                "if",
                "about",
                "who",
                "get",
                "which",
                "go",
                "me",
                "when",
                "make",
                "can",
                "like",
                "time",
                "no",
                "just",
                "him",
                "know",
                "take",
                "people",
                "into",
                "year",
                "your",
                "good",
                "some",
                "could",
                "them",
                "see",
                "other",
                "than",
                "then",
                "now",
                "look",
                "only",
                "come",
                "its",
                "over",
                "think",
                "also",
                "back",
                "after",
                "use",
                "two",
                "how",
                "our",
                "work",
                "first",
                "well",
                "way",
                "even",
                "new",
                "want",
                "because",
                "any",
                "these",
                "give",
                "day",
                "most",
                "us",
            ]

        # Generate the conversation
        conversation = []
        conversation_id = f"synthetic_{random.randint(1000, 9999)}"

        for i in range(num_turns):
            # Determine the speaker
            speaker = speakers[i % len(speakers)]

            # Generate a random utterance (3-8 words)
            num_words = random.randint(3, 8)
            intended_words = [random.choice(vocabulary) for _ in range(num_words)]
            intended = " ".join(intended_words)

            # Only introduce errors for AAC user
            if "aac" in speaker.lower() or "user" in speaker.lower():
                # Introduce errors
                noisy_words = []
                for word in intended_words:
                    if random.random() < error_rate and len(word) > 2:
                        # Introduce a random error
                        error_type = random.choice(
                            ["insertion", "deletion", "substitution", "transposition"]
                        )

                        if error_type == "insertion" and len(word) > 1:
                            # Insert a random character
                            pos = random.randint(0, len(word) - 1)
                            char = random.choice("abcdefghijklmnopqrstuvwxyz")
                            noisy_words.append(word[:pos] + char + word[pos:])

                        elif error_type == "deletion" and len(word) > 2:
                            # Delete a random character
                            pos = random.randint(0, len(word) - 1)
                            noisy_words.append(word[:pos] + word[pos + 1 :])

                        elif error_type == "substitution":
                            # Substitute a random character
                            pos = random.randint(0, len(word) - 1)
                            char = random.choice("abcdefghijklmnopqrstuvwxyz")
                            noisy_words.append(word[:pos] + char + word[pos + 1 :])

                        elif error_type == "transposition" and len(word) > 2:
                            # Transpose two adjacent characters
                            pos = random.randint(0, len(word) - 2)
                            noisy_words.append(
                                word[:pos] + word[pos + 1] + word[pos] + word[pos + 2 :]
                            )
                    else:
                        noisy_words.append(word)

                noisy = " ".join(noisy_words)
            else:
                noisy = intended

            # Create the turn
            turn = {
                "conversation_id": conversation_id,
                "turn_number": i,
                "speaker": speaker,
                "utterance": intended,
                "utterance_intended": intended,
                "noisy_utterance": noisy,
                "minimally_corrected": intended,
                "fully_corrected": intended,
            }

            conversation.append(turn)

        return conversation

    def process_conversation(
        self,
        conversation: List[Dict[str, Any]],
        keyboard_layout: str = "qwerty",
        noise_level: str = "minimal",
    ) -> Dict[str, Any]:
        """
        Process a conversation for compatibility with compare_correction_methods.py.

        Args:
            conversation: List of conversation turns
            keyboard_layout: Keyboard layout to use
            noise_level: Noise level to use

        Returns:
            Dictionary with evaluation results for compare_correction_methods.py
        """
        # First evaluate the conversation using our standard method
        results = self.evaluate_conversation(conversation, keyboard_layout, noise_level)

        # Convert to the format expected by compare_correction_methods.py
        processed_results = {
            "conversation_id": results["conversation_id"],
            "turns": results["turns"],
            "metrics": {
                "accuracy": results["metrics"]["with_context"]["accuracy_at_1"],
                "accuracy_by_position": {},
                "accuracy_by_speaker": {},
                "accuracy_trend": [],
            },
        }

        # Calculate accuracy by position
        position_counts = defaultdict(int)
        position_correct = defaultdict(int)

        # Calculate accuracy by speaker
        speaker_counts = defaultdict(int)
        speaker_correct = defaultdict(int)

        # Process each turn
        for turn in results["turns"]:
            position = turn["turn_number"]
            speaker = turn["speaker"]
            correct = turn["with_context_correct_at_1"]

            # Update position metrics
            position_counts[str(position)] += 1
            if correct:
                position_correct[str(position)] += 1

            # Update speaker metrics
            speaker_counts[speaker] += 1
            if correct:
                speaker_correct[speaker] += 1

            # Add to accuracy trend
            processed_results["metrics"]["accuracy_trend"].append(1 if correct else 0)

        # Calculate accuracy by position
        for position, count in position_counts.items():
            if count > 0:
                processed_results["metrics"]["accuracy_by_position"][position] = (
                    position_correct[position] / count
                )

        # Calculate accuracy by speaker
        for speaker, count in speaker_counts.items():
            if count > 0:
                processed_results["metrics"]["accuracy_by_speaker"][speaker] = (
                    speaker_correct[speaker] / count
                )

        return processed_results

    def evaluate_conversation(
        self,
        conversation: List[Dict[str, Any]],
        keyboard_layout: str = "qwerty",
        noise_level: str = "minimal",
    ) -> Dict[str, Any]:
        """
        Evaluate a conversation.

        Args:
            conversation: List of conversation turns
            keyboard_layout: Keyboard layout to use
            noise_level: Noise level to use

        Returns:
            Dictionary with evaluation results
        """
        # Prepare the conversation for evaluation
        prepared_conversation = prepare_conversation_for_evaluation(
            conversation, keyboard_layout, noise_level
        )

        # Initialize results
        results = {
            "conversation_id": conversation[0].get("conversation_id", "unknown"),
            "keyboard_layout": keyboard_layout,
            "noise_level": noise_level,
            "turns": [],
            "metrics": {
                "no_context": {
                    "correct_at_1": 0,
                    "correct_at_n": 0,
                    "total": 0,
                },
                "with_context": {
                    "correct_at_1": 0,
                    "correct_at_n": 0,
                    "total": 0,
                },
            },
        }

        # Initialize context
        context = []

        # Process each turn
        for i, turn in enumerate(prepared_conversation):
            # Get the noisy utterance and reference texts
            noisy = turn.get("noisy_utterance", "")
            intended = turn.get("utterance_intended", "")
            minimally_corrected = turn.get("minimally_corrected", "")
            fully_corrected = turn.get("fully_corrected", "")
            speaker = turn.get("speaker", "Unknown")

            # Determine the target for evaluation
            if self.target_field == "minimally_corrected" and minimally_corrected:
                target = minimally_corrected
            elif self.target_field == "fully_corrected" and fully_corrected:
                target = fully_corrected
            else:
                target = intended

            # Skip if no noisy utterance or target
            if not noisy or not target:
                continue

            # Only evaluate AAC user utterances
            if "aac" not in speaker.lower():
                # Add to context and continue
                if self.use_gold_context:
                    context.append(target)
                else:
                    context.append(noisy)
                context = context[-self.context_window_size :]
                continue

            # Correct without context
            no_context_corrections = self.corrector.correct(
                noisy, context=None, max_edit_distance=1  # Reduced from 2
            )

            # Correct with context
            with_context_corrections = self.corrector.correct(
                noisy, context=context, max_edit_distance=1  # Reduced from 2
            )

            # Get top corrections
            no_context_top = (
                no_context_corrections[0][0] if no_context_corrections else noisy
            )
            with_context_top = (
                with_context_corrections[0][0] if with_context_corrections else noisy
            )

            # Update context with the corrected utterance
            if self.use_gold_context:
                context.append(target)
            else:
                context.append(with_context_top)
            context = context[-self.context_window_size :]

            # Calculate metrics
            no_context_correct_at_1 = no_context_top.lower() == target.lower()
            no_context_correct_at_n = any(
                corr.lower() == target.lower() for corr, _ in no_context_corrections
            )

            with_context_correct_at_1 = with_context_top.lower() == target.lower()
            with_context_correct_at_n = any(
                corr.lower() == target.lower() for corr, _ in with_context_corrections
            )

            # Update results
            turn_result = {
                "turn_number": i,
                "speaker": speaker,
                "noisy": noisy,
                "target": target,
                "no_context_top": no_context_top,
                "with_context_top": with_context_top,
                "no_context_correct_at_1": no_context_correct_at_1,
                "no_context_correct_at_n": no_context_correct_at_n,
                "with_context_correct_at_1": with_context_correct_at_1,
                "with_context_correct_at_n": with_context_correct_at_n,
            }

            results["turns"].append(turn_result)

            # Update metrics
            results["metrics"]["no_context"]["total"] += 1
            results["metrics"]["with_context"]["total"] += 1

            if no_context_correct_at_1:
                results["metrics"]["no_context"]["correct_at_1"] += 1
            if no_context_correct_at_n:
                results["metrics"]["no_context"]["correct_at_n"] += 1

            if with_context_correct_at_1:
                results["metrics"]["with_context"]["correct_at_1"] += 1
            if with_context_correct_at_n:
                results["metrics"]["with_context"]["correct_at_n"] += 1

        # Calculate accuracy
        for context_type in ["no_context", "with_context"]:
            total = results["metrics"][context_type]["total"]
            if total > 0:
                results["metrics"][context_type]["accuracy_at_1"] = (
                    results["metrics"][context_type]["correct_at_1"] / total
                )
                results["metrics"][context_type]["accuracy_at_n"] = (
                    results["metrics"][context_type]["correct_at_n"] / total
                )
            else:
                results["metrics"][context_type]["accuracy_at_1"] = 0.0
                results["metrics"][context_type]["accuracy_at_n"] = 0.0

        return results

    def evaluate_synthetic_dataset(
        self,
        num_conversations: int = 5,
        num_turns: int = 10,
        vocabulary_size: int = 100,
        error_rate: float = 0.2,
        keyboard_layouts: List[str] = ["qwerty"],
        noise_levels: List[str] = ["minimal"],
    ) -> Dict[str, Any]:
        """
        Generate and evaluate synthetic conversations.

        Args:
            num_conversations: Number of conversations to generate
            num_turns: Number of turns per conversation
            vocabulary_size: Size of the vocabulary to use
            error_rate: Probability of introducing an error in each word
            keyboard_layouts: List of keyboard layouts to evaluate
            noise_levels: List of noise levels to evaluate

        Returns:
            Dictionary with evaluation results
        """
        # Generate synthetic conversations
        conversations = []
        for _ in range(num_conversations):
            conversation = self.generate_synthetic_conversation(
                num_turns=num_turns,
                vocabulary_size=vocabulary_size,
                error_rate=error_rate,
            )
            conversations.append(conversation)

        # Initialize results
        results = {
            "conversations": [],
            "summary": {
                "keyboard_layouts": keyboard_layouts,
                "noise_levels": noise_levels,
                "total_conversations": len(conversations),
                "total_turns": 0,
                "metrics": {},
            },
        }

        # Initialize summary metrics
        for keyboard in keyboard_layouts:
            results["summary"]["metrics"][keyboard] = {}
            for noise in noise_levels:
                results["summary"]["metrics"][keyboard][noise] = {
                    "no_context": {
                        "correct_at_1": 0,
                        "correct_at_n": 0,
                        "total": 0,
                        "accuracy_at_1": 0.0,
                        "accuracy_at_n": 0.0,
                    },
                    "with_context": {
                        "correct_at_1": 0,
                        "correct_at_n": 0,
                        "total": 0,
                        "accuracy_at_1": 0.0,
                        "accuracy_at_n": 0.0,
                    },
                }

        # Evaluate each conversation
        for i, conversation in enumerate(conversations):
            logger.info(f"Evaluating synthetic conversation {i+1}/{len(conversations)}")

            for keyboard in keyboard_layouts:
                for noise in noise_levels:
                    # Evaluate the conversation
                    conversation_result = self.evaluate_conversation(
                        conversation, keyboard, noise
                    )

                    # Add to results
                    results["conversations"].append(conversation_result)

                    # Update summary metrics
                    for context_type in ["no_context", "with_context"]:
                        results["summary"]["metrics"][keyboard][noise][context_type][
                            "correct_at_1"
                        ] += conversation_result["metrics"][context_type][
                            "correct_at_1"
                        ]
                        results["summary"]["metrics"][keyboard][noise][context_type][
                            "correct_at_n"
                        ] += conversation_result["metrics"][context_type][
                            "correct_at_n"
                        ]
                        results["summary"]["metrics"][keyboard][noise][context_type][
                            "total"
                        ] += conversation_result["metrics"][context_type]["total"]

        # Calculate summary accuracy
        for keyboard in keyboard_layouts:
            for noise in noise_levels:
                for context_type in ["no_context", "with_context"]:
                    total = results["summary"]["metrics"][keyboard][noise][
                        context_type
                    ]["total"]
                    results["summary"]["total_turns"] += total
                    if total > 0:
                        results["summary"]["metrics"][keyboard][noise][context_type][
                            "accuracy_at_1"
                        ] = (
                            results["summary"]["metrics"][keyboard][noise][
                                context_type
                            ]["correct_at_1"]
                            / total
                        )
                        results["summary"]["metrics"][keyboard][noise][context_type][
                            "accuracy_at_n"
                        ] = (
                            results["summary"]["metrics"][keyboard][noise][
                                context_type
                            ]["correct_at_n"]
                            / total
                        )

        return results

    def evaluate_dataset(
        self,
        dataset: List[Dict[str, Any]],
        keyboard_layouts: List[str] = ["qwerty"],
        noise_levels: List[str] = ["minimal"],
        max_conversations: Optional[int] = None,
        language_code: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate a dataset.

        Args:
            dataset: Dataset to evaluate
            keyboard_layouts: List of keyboard layouts to evaluate
            noise_levels: List of noise levels to evaluate
            max_conversations: Maximum number of conversations to evaluate
            language_code: Language code to filter by (e.g., 'en-GB')

        Returns:
            Dictionary with evaluation results
        """
        # Filter by language code if specified
        if language_code:
            logger.info(f"Filtering dataset by language code: {language_code}")
            filtered_dataset = [
                example
                for example in dataset
                if example.get("language_code", "").lower() == language_code.lower()
            ]
            logger.info(f"Filtered dataset: {len(filtered_dataset)} examples")
            dataset = filtered_dataset

        # Group by conversation
        conversation_dict = group_by_conversation(dataset)
        logger.info(f"Found {len(conversation_dict)} conversations")

        # Convert to list of conversations
        conversations = list(conversation_dict.values())

        # Limit the number of conversations if specified
        if max_conversations is not None and max_conversations > 0:
            conversations = conversations[:max_conversations]
            logger.info(f"Limited to {max_conversations} conversations")

        # Initialize results
        results = {
            "conversations": [],
            "summary": {
                "keyboard_layouts": keyboard_layouts,
                "noise_levels": noise_levels,
                "total_conversations": len(conversations),
                "total_turns": 0,
                "metrics": {},
            },
        }

        # Initialize summary metrics
        for keyboard in keyboard_layouts:
            results["summary"]["metrics"][keyboard] = {}
            for noise in noise_levels:
                results["summary"]["metrics"][keyboard][noise] = {
                    "no_context": {
                        "correct_at_1": 0,
                        "correct_at_n": 0,
                        "total": 0,
                        "accuracy_at_1": 0.0,
                        "accuracy_at_n": 0.0,
                    },
                    "with_context": {
                        "correct_at_1": 0,
                        "correct_at_n": 0,
                        "total": 0,
                        "accuracy_at_1": 0.0,
                        "accuracy_at_n": 0.0,
                    },
                }

        # Evaluate each conversation
        for i, conversation in enumerate(conversations):
            logger.info(f"Evaluating conversation {i+1}/{len(conversations)}")

            for keyboard in keyboard_layouts:
                for noise in noise_levels:
                    # Evaluate the conversation
                    conversation_result = self.evaluate_conversation(
                        conversation, keyboard, noise
                    )

                    # Add to results
                    results["conversations"].append(conversation_result)

                    # Update summary metrics
                    for context_type in ["no_context", "with_context"]:
                        results["summary"]["metrics"][keyboard][noise][context_type][
                            "correct_at_1"
                        ] += conversation_result["metrics"][context_type][
                            "correct_at_1"
                        ]
                        results["summary"]["metrics"][keyboard][noise][context_type][
                            "correct_at_n"
                        ] += conversation_result["metrics"][context_type][
                            "correct_at_n"
                        ]
                        results["summary"]["metrics"][keyboard][noise][context_type][
                            "total"
                        ] += conversation_result["metrics"][context_type]["total"]

        # Calculate summary accuracy
        for keyboard in keyboard_layouts:
            for noise in noise_levels:
                for context_type in ["no_context", "with_context"]:
                    total = results["summary"]["metrics"][keyboard][noise][
                        context_type
                    ]["total"]
                    results["summary"]["total_turns"] += total
                    if total > 0:
                        results["summary"]["metrics"][keyboard][noise][context_type][
                            "accuracy_at_1"
                        ] = (
                            results["summary"]["metrics"][keyboard][noise][
                                context_type
                            ]["correct_at_1"]
                            / total
                        )
                        results["summary"]["metrics"][keyboard][noise][context_type][
                            "accuracy_at_n"
                        ] = (
                            results["summary"]["metrics"][keyboard][noise][
                                context_type
                            ]["correct_at_n"]
                            / total
                        )

        return results


def print_results(results: Dict[str, Any]) -> None:
    """
    Print evaluation results.

    Args:
        results: Evaluation results
    """
    print("\n=== Conversation-Level Evaluation Results ===")
    print(f"Total Conversations: {results['summary']['total_conversations']}")
    print(f"Total Turns: {results['summary']['total_turns']}")
    print("\nSummary by Keyboard Layout and Noise Level:")

    for keyboard in results["summary"]["keyboard_layouts"]:
        for noise in results["summary"]["noise_levels"]:
            print(f"\n{keyboard.upper()} - {noise.upper()} Noise:")

            no_context = results["summary"]["metrics"][keyboard][noise]["no_context"]
            with_context = results["summary"]["metrics"][keyboard][noise][
                "with_context"
            ]

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


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate conversation-level correction performance."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="willwade/AACConversations",
        help="Dataset to evaluate",
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use",
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
        "--max-conversations",
        type=int,
        default=None,
        help="Maximum number of conversations to evaluate",
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
        "--target-field",
        type=str,
        default="minimally_corrected",
        choices=["minimally_corrected", "fully_corrected", "utterance_intended"],
        help="Field to use as the target for evaluation",
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
        "--output",
        type=str,
        default=None,
        help="Path to save the evaluation results",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache the dataset",
    )

    parser.add_argument(
        "--no-token",
        action="store_true",
        help="Don't use the Hugging Face auth token",
    )

    parser.add_argument(
        "--language-code",
        type=str,
        default="en-GB",
        help="Language code to filter by (e.g., 'en-GB')",
    )

    # Synthetic conversation options
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic conversations instead of the dataset",
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

    args = parser.parse_args()

    # Initialize the corrector with a smaller max_candidates value
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
        target_field=args.target_field,
    )

    # Use synthetic conversations or load the dataset
    if args.synthetic:
        logger.info("Using synthetic conversations")
        logger.info(f"Generating {args.num_conversations} conversations")
        logger.info(f"Turns per conversation: {args.num_turns}")
        logger.info(
            f"Vocabulary size: {args.vocabulary_size}, Error rate: {args.error_rate}"
        )

        results = evaluator.evaluate_synthetic_dataset(
            num_conversations=args.num_conversations,
            num_turns=args.num_turns,
            vocabulary_size=args.vocabulary_size,
            error_rate=args.error_rate,
            keyboard_layouts=args.keyboard_layouts,
            noise_levels=args.noise_levels,
        )
    else:
        # Load the dataset
        logger.info(f"Loading dataset {args.dataset} ({args.dataset_split} split)")
        dataset = load_aac_conversations(
            split=args.dataset_split,
            cache_dir=args.cache_dir,
            use_auth_token=not args.no_token,
        )

        if dataset is None:
            logger.error("Failed to load dataset. Exiting.")
            return

        # Evaluate the dataset
        logger.info("Evaluating dataset")
        results = evaluator.evaluate_dataset(
            dataset=dataset,
            keyboard_layouts=args.keyboard_layouts,
            noise_levels=args.noise_levels,
            max_conversations=args.max_conversations,
            language_code=args.language_code,
        )

    # Print results
    print_results(results)

    # Save results if output path is specified
    if args.output:
        output_path = resolve_path(args.output)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {output_path}")


if __name__ == "__main__":
    main()
