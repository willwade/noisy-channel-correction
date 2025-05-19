#!/usr/bin/env python3
"""
Simple Conversation-Level Evaluation for AAC input correction.

This module provides a simplified version of the conversation evaluator
that doesn't rely on the PPM module. It uses a basic corrector that
only uses edit distance for correction.
"""

import os
import sys
import logging
import json
import argparse
import random
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleCorrector:
    """
    A simple corrector that uses edit distance for correction.
    
    This class provides a simplified version of the NoisyChannelCorrector
    that only uses edit distance for correction.
    """
    
    def __init__(self, lexicon_path: Optional[str] = None, max_candidates: int = 5):
        """
        Initialize a simple corrector.
        
        Args:
            lexicon_path: Path to a lexicon file
            max_candidates: Maximum number of candidates to return
        """
        self.lexicon = set()
        self.max_candidates = max_candidates
        
        # Load the lexicon if provided
        if lexicon_path and os.path.exists(lexicon_path):
            self.load_lexicon(lexicon_path)
    
    def load_lexicon(self, lexicon_path: str) -> bool:
        """
        Load a lexicon from a file.
        
        Args:
            lexicon_path: Path to the lexicon file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(lexicon_path, "r", encoding="utf-8") as f:
                self.lexicon = set(line.strip().lower() for line in f if line.strip())
            
            logger.info(f"Loaded {len(self.lexicon)} words from {lexicon_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading lexicon from {lexicon_path}: {e}")
            return False
    
    def edit_distance(self, s1: str, s2: str) -> int:
        """
        Calculate the Levenshtein edit distance between two strings.
        
        Args:
            s1: First string
            s2: Second string
            
        Returns:
            The edit distance between the strings
        """
        if len(s1) < len(s2):
            return self.edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def correct(self, text: str, context: Optional[List[str]] = None, max_edit_distance: int = 2) -> List[Tuple[str, float]]:
        """
        Correct potentially erroneous text using edit distance.
        
        Args:
            text: The potentially erroneous text to correct
            context: List of previous words or full previous sentence (not used in this simple implementation)
            max_edit_distance: Maximum edit distance to consider
            
        Returns:
            List of (correction, score) tuples, sorted by score
        """
        # If the lexicon is empty, return the original text
        if not self.lexicon:
            return [(text, 1.0)]
        
        # If the text is in the lexicon, return it
        if text.lower() in self.lexicon:
            return [(text, 1.0)]
        
        # Find candidates within the maximum edit distance
        candidates = []
        for word in self.lexicon:
            distance = self.edit_distance(text.lower(), word)
            if distance <= max_edit_distance:
                # Score is inversely proportional to the edit distance
                score = 1.0 / (distance + 1)
                candidates.append((word, score))
        
        # Sort by score and limit to max_candidates
        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates[:self.max_candidates] if candidates else [(text, 0.5)]


class SimpleConversationEvaluator:
    """
    A simplified conversation evaluator that uses the SimpleCorrector.
    
    This class provides methods for processing entire conversations,
    implementing turn-by-turn correction with context from previous turns,
    and tracking correction accuracy across conversation turns.
    """
    
    def __init__(
        self,
        corrector: SimpleCorrector,
        use_gold_context: bool = False,
        context_window_size: int = 3,
    ):
        """
        Initialize a conversation evaluator.
        
        Args:
            corrector: The corrector to use
            use_gold_context: Whether to use gold standard context (intended utterances)
                             instead of corrected utterances for future context
            context_window_size: Number of previous turns to use as context
        """
        self.corrector = corrector
        self.use_gold_context = use_gold_context
        self.context_window_size = context_window_size
    
    def process_conversation(
        self, conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process a conversation turn by turn with context.
        
        Args:
            conversation: List of conversation turns
            
        Returns:
            Dictionary with results
        """
        # Initialize results
        results = {
            "conversation_id": conversation[0].get("conversation_id", "unknown") if conversation else "unknown",
            "turns": [],
            "metrics": {
                "accuracy": 0.0,
                "turn_level_accuracy": [],
                "accuracy_by_position": defaultdict(list),
                "accuracy_by_speaker": defaultdict(list),
                "accuracy_trend": [],
            }
        }
        
        # Initialize context
        context = []
        
        # Process each turn
        for i, turn in enumerate(conversation):
            # Get the noisy utterance
            noisy = turn.get("noisy_utterance", turn.get("utterance", ""))
            intended = turn.get("utterance_intended", "")
            speaker = turn.get("speaker", "Unknown")
            
            # If no noisy utterance, use the intended utterance
            if not noisy:
                noisy = intended
            
            # Correct with context
            corrections = self.corrector.correct(
                noisy, context=context, max_edit_distance=2
            )
            
            # Update context with the corrected utterance
            top_correction = corrections[0][0] if corrections else noisy
            
            if self.use_gold_context:
                # Use the intended utterance for future context (oracle setting)
                context.append(intended)
            else:
                # Use the top correction for future context (realistic setting)
                context.append(top_correction)
            
            # Limit context size
            context = context[-self.context_window_size:]
            
            # Calculate turn-level metrics
            correct_at_1 = top_correction.lower() == intended.lower()
            correct_at_n = any(corr.lower() == intended.lower() for corr, _ in corrections)
            
            # Find the rank of the correct answer
            correct_rank = -1
            for j, (corr, _) in enumerate(corrections):
                if corr.lower() == intended.lower():
                    correct_rank = j + 1
                    break
            
            # Add to results
            results["turns"].append({
                "turn_number": i + 1,
                "speaker": speaker,
                "intended": intended,
                "noisy": noisy,
                "corrections": [{"correction": corr, "score": score} for corr, score in corrections],
                "correct_at_1": correct_at_1,
                "correct_at_n": correct_at_n,
                "correct_rank": correct_rank,
                "context": list(context),  # Make a copy of the context
            })
            
            # Update metrics
            results["metrics"]["turn_level_accuracy"].append(correct_at_1)
            results["metrics"]["accuracy_by_position"][i + 1].append(correct_at_1)
            results["metrics"]["accuracy_by_speaker"][speaker].append(correct_at_1)
            
            # Track accuracy trend (moving average)
            if i >= 2:
                window_accuracy = sum(results["metrics"]["turn_level_accuracy"][-3:]) / 3
                results["metrics"]["accuracy_trend"].append(window_accuracy)
        
        # Calculate conversation-level metrics
        if results["metrics"]["turn_level_accuracy"]:
            results["metrics"]["accuracy"] = sum(results["metrics"]["turn_level_accuracy"]) / len(results["metrics"]["turn_level_accuracy"])
            
            # Calculate accuracy by position
            for position, accuracies in results["metrics"]["accuracy_by_position"].items():
                if accuracies:
                    results["metrics"]["accuracy_by_position"][position] = sum(accuracies) / len(accuracies)
            
            # Calculate accuracy by speaker
            for speaker, accuracies in results["metrics"]["accuracy_by_speaker"].items():
                if accuracies:
                    results["metrics"]["accuracy_by_speaker"][speaker] = sum(accuracies) / len(accuracies)
        
        return results
    
    def generate_synthetic_conversation(
        self, num_turns: int = 10, vocabulary_size: int = 100, error_rate: float = 0.2
    ) -> List[Dict[str, Any]]:
        """
        Generate a synthetic conversation for testing.
        
        Args:
            num_turns: Number of turns in the conversation
            vocabulary_size: Size of the vocabulary to use
            error_rate: Probability of introducing an error in each word
            
        Returns:
            List of conversation turns
        """
        # Generate a simple vocabulary
        vocabulary = [f"word{i}" for i in range(vocabulary_size)]
        
        # Generate speakers
        speakers = ["User", "Assistant"]
        
        # Generate a conversation
        conversation = []
        for i in range(num_turns):
            # Generate an intended utterance
            num_words = random.randint(3, 8)
            intended = " ".join(random.choice(vocabulary) for _ in range(num_words))
            
            # Generate a noisy utterance by introducing errors
            noisy_words = []
            for word in intended.split():
                if random.random() < error_rate:
                    # Introduce an error
                    error_type = random.choice(["substitute", "delete", "insert", "transpose"])
                    if error_type == "substitute":
                        # Substitute a random character
                        pos = random.randint(0, len(word) - 1)
                        chars = list(word)
                        chars[pos] = random.choice("abcdefghijklmnopqrstuvwxyz")
                        noisy_words.append("".join(chars))
                    elif error_type == "delete":
                        # Delete a random character
                        pos = random.randint(0, len(word) - 1)
                        noisy_words.append(word[:pos] + word[pos+1:])
                    elif error_type == "insert":
                        # Insert a random character
                        pos = random.randint(0, len(word))
                        char = random.choice("abcdefghijklmnopqrstuvwxyz")
                        noisy_words.append(word[:pos] + char + word[pos:])
                    elif error_type == "transpose":
                        # Transpose two adjacent characters
                        if len(word) >= 2:
                            pos = random.randint(0, len(word) - 2)
                            noisy_words.append(word[:pos] + word[pos+1] + word[pos] + word[pos+2:])
                        else:
                            noisy_words.append(word)
                else:
                    # No error
                    noisy_words.append(word)
            
            noisy = " ".join(noisy_words)
            
            # Add to conversation
            conversation.append({
                "conversation_id": "synthetic",
                "turn_number": i + 1,
                "speaker": speakers[i % len(speakers)],
                "utterance_intended": intended,
                "noisy_utterance": noisy,
            })
        
        return conversation
    
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
            serializable_results = json.loads(json.dumps(results, default=lambda x: dict(x) if isinstance(x, defaultdict) else x))
            
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
        print("\n=== Conversation-Level Evaluation Summary ===")
        
        # Print overall metrics
        print(f"\nOverall Accuracy: {results['metrics']['accuracy']:.4f}")
        print(f"Number of Turns: {len(results['turns'])}")
        
        # Print accuracy by position
        print("\nAccuracy by Position:")
        positions = sorted(results["metrics"]["accuracy_by_position"].keys())
        for position in positions:
            accuracy = results["metrics"]["accuracy_by_position"][position]
            print(f"  Turn {position}: {accuracy:.4f}")
        
        # Print accuracy by speaker
        print("\nAccuracy by Speaker:")
        for speaker, accuracy in results["metrics"]["accuracy_by_speaker"].items():
            print(f"  {speaker}: {accuracy:.4f}")
        
        # Print example turns
        print("\nExample Turns:")
        for turn in results["turns"][:5]:  # Show first 5 turns
            print(f"  Speaker: {turn['speaker']}")
            print(f"  Intended: {turn['intended']}")
            print(f"  Noisy: {turn['noisy']}")
            print(f"  Corrected: {turn['corrections'][0]['correction']}")
            print(f"  Correct: {turn['correct_at_1']}")
            print()
        
        if len(results["turns"]) > 5:
            print(f"  ... ({len(results['turns']) - 5} more turns)")


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Simple Conversation-Level Evaluation for AAC input correction."
    )
    
    parser.add_argument(
        "--lexicon",
        type=str,
        default="data/wordlist.txt",
        help="Path to the lexicon file",
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
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Path to save the evaluation results",
    )
    
    args = parser.parse_args()
    
    # Create the corrector
    corrector = SimpleCorrector(lexicon_path=args.lexicon, max_candidates=5)
    
    # Create the evaluator
    evaluator = SimpleConversationEvaluator(
        corrector=corrector,
        use_gold_context=args.use_gold_context,
        context_window_size=args.context_window,
    )
    
    # Generate synthetic conversations
    all_results = {
        "conversations": [],
        "metrics": {
            "overall_accuracy": 0.0,
            "conversation_accuracies": [],
            "accuracy_by_position": defaultdict(list),
            "accuracy_by_speaker": defaultdict(list),
            "accuracy_trend": [],
        }
    }
    
    for i in range(args.num_conversations):
        logger.info(f"Processing conversation {i+1}/{args.num_conversations}")
        
        # Generate a synthetic conversation
        conversation = evaluator.generate_synthetic_conversation(
            num_turns=args.num_turns,
            vocabulary_size=args.vocabulary_size,
            error_rate=args.error_rate,
        )
        
        # Process the conversation
        results = evaluator.process_conversation(conversation)
        
        # Add to all results
        all_results["conversations"].append(results)
        
        # Update overall metrics
        all_results["metrics"]["conversation_accuracies"].append(results["metrics"]["accuracy"])
        
        # Update position-based metrics
        for position, accuracy in results["metrics"]["accuracy_by_position"].items():
            all_results["metrics"]["accuracy_by_position"][position].append(accuracy)
        
        # Update speaker-based metrics
        for speaker, accuracy in results["metrics"]["accuracy_by_speaker"].items():
            all_results["metrics"]["accuracy_by_speaker"][speaker].append(accuracy)
        
        # Update trend metrics
        all_results["metrics"]["accuracy_trend"].extend(results["metrics"]["accuracy_trend"])
    
    # Calculate overall metrics
    if all_results["metrics"]["conversation_accuracies"]:
        all_results["metrics"]["overall_accuracy"] = sum(all_results["metrics"]["conversation_accuracies"]) / len(all_results["metrics"]["conversation_accuracies"])
        
        # Calculate average accuracy by position
        for position, accuracies in all_results["metrics"]["accuracy_by_position"].items():
            if accuracies:
                all_results["metrics"]["accuracy_by_position"][position] = sum(accuracies) / len(accuracies)
        
        # Calculate average accuracy by speaker
        for speaker, accuracies in all_results["metrics"]["accuracy_by_speaker"].items():
            if accuracies:
                all_results["metrics"]["accuracy_by_speaker"][speaker] = sum(accuracies) / len(accuracies)
    
    # Print the summary
    evaluator.print_summary(all_results["conversations"][0])
    
    # Save the results
    if args.output:
        evaluator.save_results(all_results, args.output)


if __name__ == "__main__":
    main()
