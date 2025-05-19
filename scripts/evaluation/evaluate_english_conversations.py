#!/usr/bin/env python3
"""
Evaluate English conversations from the AACConversations dataset.

This script loads the filtered English conversations and evaluates them
using the noisy channel corrector.
"""

import os
import sys
import logging
import json
import argparse
from typing import Any, Dict, List, Optional
from collections import defaultdict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the corrector and utilities
from lib.corrector.corrector import NoisyChannelCorrector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_english_conversations(file_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load filtered English conversations from a file.

    Args:
        file_path: Path to the filtered conversations file

    Returns:
        Dictionary mapping conversation IDs to lists of turns
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)

        logger.info(
            f"Loaded {len(conversations)} English conversations from {file_path}"
        )
        return conversations
    except Exception as e:
        logger.error(f"Error loading English conversations: {e}")
        return {}


class EnglishConversationEvaluator:
    """
    Class for evaluating the performance of the noisy channel corrector on English conversations.

    This class provides methods for processing English conversations from the
    AACConversations dataset, implementing turn-by-turn correction with context
    from previous turns, and tracking correction accuracy across conversation turns.
    """

    def __init__(
        self,
        corrector: NoisyChannelCorrector,
        use_gold_context: bool = False,
        context_window_size: int = 3,
        max_candidates: int = 5,
    ):
        """
        Initialize a conversation evaluator.

        Args:
            corrector: The noisy channel corrector to use
            use_gold_context: Whether to use gold standard context (intended utterances)
                             instead of corrected utterances for future context
            context_window_size: Number of previous turns to use as context
            max_candidates: Maximum number of candidates to return
        """
        self.corrector = corrector
        self.use_gold_context = use_gold_context
        self.context_window_size = context_window_size
        self.max_candidates = max_candidates

    def process_conversation(
        self,
        conversation: List[Dict[str, Any]],
        noise_type: str = "qwerty",
        noise_level: str = "moderate",
    ) -> Dict[str, Any]:
        """
        Process a conversation turn by turn with context.

        Args:
            conversation: List of conversation turns
            noise_type: Type of noise ('qwerty', 'abc', or 'frequency')
            noise_level: Level of noise ('minimal', 'light', 'moderate', or 'severe')

        Returns:
            Dictionary with results
        """
        # Initialize results
        results = {
            "conversation_id": (
                conversation[0].get("conversation_id", "unknown")
                if conversation
                else "unknown"
            ),
            "noise_type": noise_type,
            "noise_level": noise_level,
            "turns": [],
            "metrics": {
                "accuracy": 0.0,
                "turn_level_accuracy": [],
                "accuracy_by_position": defaultdict(list),
                "accuracy_by_speaker": defaultdict(list),
                "accuracy_trend": [],
            },
        }

        # Initialize context
        context = []

        # Process each turn
        for i, turn in enumerate(conversation):
            # Get the noisy utterance
            noisy_key = f"noisy_{noise_type}_{noise_level}"
            noisy = turn.get(noisy_key, "")
            intended = turn.get("utterance_intended", "")
            speaker = turn.get("speaker", "Unknown")

            # Skip non-AAC users (they don't have noisy utterances)
            if speaker != "AAC User":
                continue

            # If no noisy utterance, use the intended utterance
            if not noisy:
                noisy = intended

            # Correct with context
            try:
                # First check if the models are ready
                if not self.corrector.update_models_ready_status():
                    logger.warning("Models not ready. Skipping correction.")
                    corrections = [(noisy, 1.0)]
                else:
                    # Try to correct the input
                    corrections = self.corrector.correct(
                        noisy, context=context, max_edit_distance=2
                    )
            except Exception as e:
                logger.error(f"Error correcting '{noisy}': {e}")
                corrections = [(noisy, 1.0)]

            # Update context with the corrected utterance
            top_correction = corrections[0][0] if corrections else noisy

            if self.use_gold_context:
                # Use the intended utterance for future context (oracle setting)
                context.append(intended)
            else:
                # Use the top correction for future context (realistic setting)
                context.append(top_correction)

            # Limit context size
            context = context[-self.context_window_size :]

            # Calculate turn-level metrics
            # Use the fully_corrected field as the reference (ground truth)
            reference = turn.get("fully_corrected", "")

            # If no fully_corrected, try minimally_corrected
            if not reference:
                reference = turn.get("minimally_corrected", "")

            # If still no reference, use the utterance field
            if not reference:
                reference = turn.get("utterance", "")

            # If still no reference, use the noisy_minimal field
            if not reference:
                noisy_minimal_key = f"noisy_{noise_type}_minimal"
                reference = turn.get(noisy_minimal_key, "")

            # No debug print statements

            # Normalize strings for comparison (remove punctuation and convert to lowercase)
            def normalize_text(text):
                """Normalize text for comparison."""
                # Remove punctuation and convert to lowercase
                return "".join(c for c in text.lower() if c.isalnum() or c.isspace())

            # Check if the correction matches the reference
            normalized_top_correction = normalize_text(top_correction)
            normalized_reference = normalize_text(reference)
            correct_at_1 = normalized_top_correction == normalized_reference
            correct_at_n = any(
                normalize_text(corr) == normalized_reference for corr, _ in corrections
            )

            # Find the rank of the correct answer
            correct_rank = -1
            for j, (corr, _) in enumerate(corrections):
                if corr.lower() == reference.lower():
                    correct_rank = j + 1
                    break

            # Add to results
            results["turns"].append(
                {
                    "turn_number": i + 1,
                    "speaker": speaker,
                    "intended": intended,
                    "noisy": noisy,
                    "reference": reference,  # Add the reference
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

            # Track accuracy trend (moving average)
            if i >= 2:
                window_accuracy = (
                    sum(results["metrics"]["turn_level_accuracy"][-3:]) / 3
                )
                results["metrics"]["accuracy_trend"].append(window_accuracy)

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

    def process_conversations(
        self,
        conversations: Dict[str, List[Dict[str, Any]]],
        num_conversations: Optional[int] = None,
        noise_type: str = "qwerty",
        noise_level: str = "moderate",
    ) -> Dict[str, Any]:
        """
        Process multiple conversations.

        Args:
            conversations: Dictionary mapping conversation IDs to lists of turns
            num_conversations: Maximum number of conversations to process
            noise_type: Type of noise ('qwerty', 'abc', or 'frequency')
            noise_level: Level of noise ('minimal', 'light', 'moderate', or 'severe')

        Returns:
            Dictionary with results
        """
        # Limit the number of conversations if specified
        conversation_ids = list(conversations.keys())
        if num_conversations is not None and num_conversations < len(conversation_ids):
            conversation_ids = conversation_ids[:num_conversations]

        # Process each conversation
        all_results = {
            "conversations": [],
            "metrics": {
                "overall_accuracy": 0.0,
                "conversation_accuracies": [],
                "accuracy_by_position": defaultdict(list),
                "accuracy_by_speaker": defaultdict(list),
                "accuracy_trend": [],
            },
        }

        for conversation_id in conversation_ids:
            conversation = conversations[conversation_id]

            # Process the conversation
            results = self.process_conversation(
                conversation, noise_type=noise_type, noise_level=noise_level
            )

            # Add to all results
            all_results["conversations"].append(results)

            # Update overall metrics
            all_results["metrics"]["conversation_accuracies"].append(
                results["metrics"]["accuracy"]
            )

            # Update position-based metrics
            for position, accuracy in results["metrics"][
                "accuracy_by_position"
            ].items():
                all_results["metrics"]["accuracy_by_position"][position].append(
                    accuracy
                )

            # Update speaker-based metrics
            for speaker, accuracy in results["metrics"]["accuracy_by_speaker"].items():
                all_results["metrics"]["accuracy_by_speaker"][speaker].append(accuracy)

            # Update trend metrics
            all_results["metrics"]["accuracy_trend"].extend(
                results["metrics"]["accuracy_trend"]
            )

        # Calculate overall metrics
        if all_results["metrics"]["conversation_accuracies"]:
            all_results["metrics"]["overall_accuracy"] = sum(
                all_results["metrics"]["conversation_accuracies"]
            ) / len(all_results["metrics"]["conversation_accuracies"])

            # Calculate average accuracy by position
            for position, accuracies in all_results["metrics"][
                "accuracy_by_position"
            ].items():
                if accuracies:
                    all_results["metrics"]["accuracy_by_position"][position] = sum(
                        accuracies
                    ) / len(accuracies)

            # Calculate average accuracy by speaker
            for speaker, accuracies in all_results["metrics"][
                "accuracy_by_speaker"
            ].items():
                if accuracies:
                    all_results["metrics"]["accuracy_by_speaker"][speaker] = sum(
                        accuracies
                    ) / len(accuracies)

        return all_results

    def save_results(self, results: Dict[str, Any], output_path: str) -> bool:
        """
        Save evaluation results to a file.

        Args:
            results: The results to save
            output_path: Path to save the results

        Returns:
            True if successful, False otherwise
        """
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

            # Convert defaultdicts to regular dicts for JSON serialization
            serializable_results = json.loads(
                json.dumps(
                    results,
                    default=lambda x: dict(x) if isinstance(x, defaultdict) else x,
                )
            )

            # Save the results
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2)

            logger.info(f"Saved evaluation results to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
            return False

    def print_summary(self, results: Dict[str, Any]) -> None:
        """
        Print a summary of the evaluation results.

        Args:
            results: The evaluation results
        """
        print("\n=== English Conversation Evaluation Summary ===")

        # Print overall metrics
        print(f"\nOverall Accuracy: {results['metrics']['overall_accuracy']:.4f}")
        print(f"Number of Conversations: {len(results['conversations'])}")
        print(
            f"Total Turns: {sum(len(conv['turns']) for conv in results['conversations'])}"
        )

        # Print accuracy by position
        print("\nAccuracy by Position:")
        positions = sorted(results["metrics"]["accuracy_by_position"].keys())
        for position in positions:
            accuracy = results["metrics"]["accuracy_by_position"][position]
            print(f"  Turn {position}: {accuracy:.4f}")

        # Print accuracy by speaker
        print("\nAccuracy by Speaker:")
        for speaker, accuracy in sorted(
            results["metrics"]["accuracy_by_speaker"].items(),
            key=lambda x: x[1],
            reverse=True,
        )[:10]:
            print(f"  {speaker}: {accuracy:.4f}")

        # Print conversation-level metrics
        print("\nConversation-Level Metrics:")
        accuracies = results["metrics"]["conversation_accuracies"]
        print(f"  Min Accuracy: {min(accuracies):.4f}")
        print(f"  Max Accuracy: {max(accuracies):.4f}")
        print(f"  Median Accuracy: {sorted(accuracies)[len(accuracies)//2]:.4f}")

        # Print example conversation
        if results["conversations"]:
            print("\nExample Conversation:")
            conversation = results["conversations"][0]
            for turn in conversation["turns"][:5]:  # Show first 5 turns
                print(f"  Speaker: {turn['speaker']}")
                print(f"  Intended: {turn['intended']}")
                print(f"  Noisy: {turn['noisy']}")
                print(f"  Corrected: {turn['corrections'][0]['correction']}")

                # Get the reference directly from the turn data
                reference = turn.get("reference", "")

                print(f"  Reference: {reference}")
                print(f"  Correct: {turn['correct_at_1']}")
                print()

            if len(conversation["turns"]) > 5:
                print(f"  ... ({len(conversation['turns']) - 5} more turns)")


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate English conversations from the AACConversations dataset."
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
        default="data/confusion_matrix.json",
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
        "--english-conversations",
        type=str,
        default="data/english_conversations.json",
        help="Path to the filtered English conversations file",
    )

    parser.add_argument(
        "--num-conversations",
        type=int,
        default=10,
        help="Number of conversations to evaluate",
    )

    parser.add_argument(
        "--use-gold-context",
        action="store_true",
        help="Use gold standard context (intended utterances) instead of corrected utterances",
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=3,
        help="Number of previous turns to use as context",
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
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to save the evaluation results",
    )

    args = parser.parse_args()

    # Create the corrector
    corrector = NoisyChannelCorrector(
        max_candidates=5,
        context_window_size=args.context_window,
        context_weight=0.7,
        keyboard_layout=args.keyboard_layout,
    )

    # Load the PPM model if it exists
    if os.path.exists(args.ppm_model):
        corrector.load_ppm_model(args.ppm_model)
    else:
        logger.warning(f"PPM model file not found: {args.ppm_model}")

    # Load the confusion matrix if it exists
    if os.path.exists(args.confusion_matrix):
        # If using keyboard-specific matrices, check if the file exists
        if args.use_keyboard_matrices:
            keyboard_matrix_path = os.path.join(
                os.path.dirname(args.confusion_matrix),
                "keyboard_confusion_matrices.json",
            )
            if os.path.exists(keyboard_matrix_path):
                corrector.load_confusion_model(
                    keyboard_matrix_path, args.keyboard_layout
                )
            else:
                # Fall back to standard confusion matrix
                corrector.load_confusion_model(
                    args.confusion_matrix, args.keyboard_layout
                )
        else:
            # Use standard confusion matrix
            corrector.load_confusion_model(args.confusion_matrix, args.keyboard_layout)
    else:
        logger.warning(f"Confusion matrix file not found: {args.confusion_matrix}")

    # Load the word n-gram model if it exists
    if os.path.exists(args.word_ngram_model):
        corrector.load_word_ngram_model(args.word_ngram_model)
    else:
        logger.warning(f"Word n-gram model file not found: {args.word_ngram_model}")

    # Load the lexicon if it exists
    if os.path.exists(args.lexicon):
        corrector.load_lexicon_from_file(args.lexicon)
    else:
        logger.warning(f"Lexicon file not found: {args.lexicon}")

    # Update the models_ready flag using the dedicated method
    if corrector.update_models_ready_status():
        logger.info("Corrector models are ready")
    else:
        logger.warning(
            "Corrector models are not fully initialized. Evaluation results may be inaccurate."
        )
        logger.warning(
            "Make sure both PPM model and confusion matrix are properly loaded."
        )

    # Load the English conversations
    conversations = load_english_conversations(args.english_conversations)

    if not conversations:
        logger.error("Failed to load English conversations. Exiting.")
        return

    # Create the evaluator
    evaluator = EnglishConversationEvaluator(
        corrector=corrector,
        use_gold_context=args.use_gold_context,
        context_window_size=args.context_window,
        max_candidates=5,
    )

    # Process the conversations
    results = evaluator.process_conversations(
        conversations,
        args.num_conversations,
        noise_type=args.noise_type,
        noise_level=args.noise_level,
    )

    # Print the summary
    evaluator.print_summary(results)

    # Save the results
    if args.output:
        evaluator.save_results(results, args.output)


if __name__ == "__main__":
    main()
