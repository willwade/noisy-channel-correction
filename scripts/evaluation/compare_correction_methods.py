#!/usr/bin/env python3
"""
Compare different correction methods on the same conversations.

This script provides functionality for comparing the performance of different
correction methods (baseline, context-aware, keyboard-specific, combined)
on the same conversations from the AACConversations dataset.
"""

import os
import sys
import json
import argparse
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict
import time

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the corrector
from lib.corrector.corrector import NoisyChannelCorrector
from scripts.evaluation.utils import load_aac_conversations
from module5.conversation_evaluator import ConversationEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Visualizations will be limited.")
    MATPLOTLIB_AVAILABLE = False


def create_corrector(
    method: str,
    ppm_model_path: str,
    confusion_matrix_path: str,
    word_ngram_model_path: str,
    lexicon_path: str,
    keyboard_layout: str = "qwerty",
    use_keyboard_matrices: bool = False,
    context_window_size: int = 3,
    context_weight: float = 0.7,
) -> NoisyChannelCorrector:
    """
    Create a corrector with the specified method.
    
    Args:
        method: Correction method ('baseline', 'context-aware', 'keyboard-specific', 'combined')
        ppm_model_path: Path to the PPM model file
        confusion_matrix_path: Path to the confusion matrix file
        word_ngram_model_path: Path to the word n-gram model file
        lexicon_path: Path to the lexicon file
        keyboard_layout: Keyboard layout to use
        use_keyboard_matrices: Whether to use keyboard-specific confusion matrices
        context_window_size: Number of previous turns to use as context
        context_weight: Weight of context-based probability
        
    Returns:
        The configured corrector
    """
    # Create the corrector with appropriate settings
    if method == "baseline":
        # Baseline: No context, no keyboard-specific matrices
        corrector = NoisyChannelCorrector(
            max_candidates=5,
            context_window_size=0,  # No context
            context_weight=0.0,  # No context weight
            keyboard_layout="qwerty",  # Default layout
        )
    elif method == "context-aware":
        # Context-aware: Use context, no keyboard-specific matrices
        corrector = NoisyChannelCorrector(
            max_candidates=5,
            context_window_size=context_window_size,
            context_weight=context_weight,
            keyboard_layout="qwerty",  # Default layout
        )
    elif method == "keyboard-specific":
        # Keyboard-specific: No context, use keyboard-specific matrices
        corrector = NoisyChannelCorrector(
            max_candidates=5,
            context_window_size=0,  # No context
            context_weight=0.0,  # No context weight
            keyboard_layout=keyboard_layout,
        )
    elif method == "combined":
        # Combined: Use context and keyboard-specific matrices
        corrector = NoisyChannelCorrector(
            max_candidates=5,
            context_window_size=context_window_size,
            context_weight=context_weight,
            keyboard_layout=keyboard_layout,
        )
    else:
        # Default to combined
        logger.warning(f"Unknown method: {method}. Using combined method.")
        corrector = NoisyChannelCorrector(
            max_candidates=5,
            context_window_size=context_window_size,
            context_weight=context_weight,
            keyboard_layout=keyboard_layout,
        )
    
    # Load the PPM model if it exists
    if os.path.exists(ppm_model_path):
        corrector.load_ppm_model(ppm_model_path)
    else:
        logger.warning(f"PPM model file not found: {ppm_model_path}")
    
    # Load the confusion matrix if it exists
    if os.path.exists(confusion_matrix_path):
        # If using keyboard-specific matrices and method supports it
        if use_keyboard_matrices and method in ["keyboard-specific", "combined"]:
            keyboard_matrix_path = os.path.join(
                os.path.dirname(confusion_matrix_path),
                "keyboard_confusion_matrices.json"
            )
            if os.path.exists(keyboard_matrix_path):
                corrector.load_confusion_model(keyboard_matrix_path, keyboard_layout)
            else:
                # Fall back to standard confusion matrix
                corrector.load_confusion_model(confusion_matrix_path, keyboard_layout)
        else:
            # Use standard confusion matrix
            corrector.load_confusion_model(confusion_matrix_path, "qwerty")
    else:
        logger.warning(f"Confusion matrix file not found: {confusion_matrix_path}")
    
    # Load the word n-gram model if it exists and method uses context
    if method in ["context-aware", "combined"] and os.path.exists(word_ngram_model_path):
        corrector.load_word_ngram_model(word_ngram_model_path)
    elif method in ["context-aware", "combined"]:
        logger.warning(f"Word n-gram model file not found: {word_ngram_model_path}")
    
    # Load the lexicon if it exists
    if os.path.exists(lexicon_path):
        corrector.load_lexicon_from_file(lexicon_path)
    else:
        logger.warning(f"Lexicon file not found: {lexicon_path}")
    
    return corrector


def compare_methods(
    dataset: Any,
    methods: List[str],
    ppm_model_path: str,
    confusion_matrix_path: str,
    word_ngram_model_path: str,
    lexicon_path: str,
    num_conversations: int = 10,
    use_gold_context: bool = False,
    context_window_size: int = 3,
    context_weight: float = 0.7,
    keyboard_layout: str = "qwerty",
    use_keyboard_matrices: bool = False,
) -> Dict[str, Any]:
    """
    Compare different correction methods on the same conversations.
    
    Args:
        dataset: The dataset to process
        methods: List of correction methods to compare
        ppm_model_path: Path to the PPM model file
        confusion_matrix_path: Path to the confusion matrix file
        word_ngram_model_path: Path to the word n-gram model file
        lexicon_path: Path to the lexicon file
        num_conversations: Number of conversations to process
        use_gold_context: Whether to use gold standard context
        context_window_size: Number of previous turns to use as context
        context_weight: Weight of context-based probability
        keyboard_layout: Keyboard layout to use
        use_keyboard_matrices: Whether to use keyboard-specific confusion matrices
        
    Returns:
        Dictionary with comparison results
    """
    # Group examples by conversation_id
    conversations = defaultdict(list)
    
    for example in dataset:
        conversation_id = example.get("conversation_id", "unknown")
        conversations[conversation_id].append(example)
    
    # Sort each conversation by turn_number
    for conversation_id, turns in conversations.items():
        conversations[conversation_id] = sorted(
            turns, key=lambda x: x.get("turn_number", 0)
        )
    
    # Limit the number of conversations if specified
    conversation_ids = list(conversations.keys())
    if num_conversations is not None and num_conversations < len(conversation_ids):
        conversation_ids = conversation_ids[:num_conversations]
    
    # Process each method
    results = {
        "methods": methods,
        "method_results": {},
        "comparison": {
            "overall_accuracy": {},
            "accuracy_by_position": defaultdict(dict),
            "accuracy_by_speaker": defaultdict(dict),
            "processing_time": {},
        }
    }
    
    for method in methods:
        logger.info(f"Processing method: {method}")
        
        # Create the corrector for this method
        corrector = create_corrector(
            method=method,
            ppm_model_path=ppm_model_path,
            confusion_matrix_path=confusion_matrix_path,
            word_ngram_model_path=word_ngram_model_path,
            lexicon_path=lexicon_path,
            keyboard_layout=keyboard_layout,
            use_keyboard_matrices=use_keyboard_matrices,
            context_window_size=context_window_size,
            context_weight=context_weight,
        )
        
        # Create the evaluator
        evaluator = ConversationEvaluator(
            corrector=corrector,
            use_gold_context=use_gold_context,
            context_window_size=context_window_size,
            max_candidates=5,
        )
        
        # Process the conversations
        start_time = time.time()
        method_results = {
            "conversations": [],
            "metrics": {
                "overall_accuracy": 0.0,
                "conversation_accuracies": [],
                "accuracy_by_position": defaultdict(list),
                "accuracy_by_speaker": defaultdict(list),
                "accuracy_trend": [],
            }
        }
        
        for conversation_id in conversation_ids:
            conversation = conversations[conversation_id]
            
            # Process the conversation
            results_for_conversation = evaluator.process_conversation(conversation)
            
            # Add to method results
            method_results["conversations"].append(results_for_conversation)
            
            # Update overall metrics
            method_results["metrics"]["conversation_accuracies"].append(results_for_conversation["metrics"]["accuracy"])
            
            # Update position-based metrics
            for position, accuracy in results_for_conversation["metrics"]["accuracy_by_position"].items():
                method_results["metrics"]["accuracy_by_position"][position].append(accuracy)
            
            # Update speaker-based metrics
            for speaker, accuracy in results_for_conversation["metrics"]["accuracy_by_speaker"].items():
                method_results["metrics"]["accuracy_by_speaker"][speaker].append(accuracy)
            
            # Update trend metrics
            method_results["metrics"]["accuracy_trend"].extend(results_for_conversation["metrics"]["accuracy_trend"])
        
        # Calculate overall metrics
        if method_results["metrics"]["conversation_accuracies"]:
            method_results["metrics"]["overall_accuracy"] = sum(method_results["metrics"]["conversation_accuracies"]) / len(method_results["metrics"]["conversation_accuracies"])
            
            # Calculate average accuracy by position
            for position, accuracies in method_results["metrics"]["accuracy_by_position"].items():
                if accuracies:
                    method_results["metrics"]["accuracy_by_position"][position] = sum(accuracies) / len(accuracies)
            
            # Calculate average accuracy by speaker
            for speaker, accuracies in method_results["metrics"]["accuracy_by_speaker"].items():
                if accuracies:
                    method_results["metrics"]["accuracy_by_speaker"][speaker] = sum(accuracies) / len(accuracies)
        
        # Calculate processing time
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Add to results
        results["method_results"][method] = method_results
        results["comparison"]["overall_accuracy"][method] = method_results["metrics"]["overall_accuracy"]
        results["comparison"]["processing_time"][method] = processing_time
        
        # Add position-based metrics to comparison
        for position, accuracy in method_results["metrics"]["accuracy_by_position"].items():
            results["comparison"]["accuracy_by_position"][position][method] = accuracy
        
        # Add speaker-based metrics to comparison
        for speaker, accuracy in method_results["metrics"]["accuracy_by_speaker"].items():
            results["comparison"]["accuracy_by_speaker"][speaker][method] = accuracy
    
    return results


def visualize_method_comparison(results: Dict[str, Any], output_dir: Optional[str] = None) -> None:
    """
    Visualize the comparison of different correction methods.
    
    Args:
        results: The comparison results
        output_dir: Directory to save visualizations
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib not available. Cannot create visualizations.")
        return
    
    # Create the output directory if it doesn't exist
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Visualize overall accuracy comparison
    if "overall_accuracy" in results["comparison"]:
        methods = list(results["comparison"]["overall_accuracy"].keys())
        accuracies = [results["comparison"]["overall_accuracy"][method] for method in methods]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, accuracies, color='skyblue')
        plt.xlabel('Method')
        plt.ylabel('Overall Accuracy')
        plt.title('Comparison of Correction Methods')
        plt.ylim(0, 1.05)
        
        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.4f}', ha='center', va='bottom', rotation=0)
        
        # Save or show the plot
        if output_dir:
            plt.savefig(os.path.join(output_dir, "method_comparison.png"), dpi=300, bbox_inches='tight')
            logger.info(f"Saved method comparison visualization to {os.path.join(output_dir, 'method_comparison.png')}")
        else:
            plt.show()
        
        plt.close()
    
    # Visualize accuracy by position comparison
    if "accuracy_by_position" in results["comparison"]:
        positions = sorted(int(pos) for pos in results["comparison"]["accuracy_by_position"].keys())
        methods = results["methods"]
        
        plt.figure(figsize=(12, 6))
        
        for method in methods:
            accuracies = [results["comparison"]["accuracy_by_position"][str(pos)].get(method, 0) for pos in positions]
            plt.plot(positions, accuracies, marker='o', linestyle='-', label=method)
        
        plt.xlabel('Turn Position')
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Turn Position for Different Methods')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.xticks(positions)
        plt.ylim(0, 1.05)
        plt.legend()
        
        # Save or show the plot
        if output_dir:
            plt.savefig(os.path.join(output_dir, "position_comparison.png"), dpi=300, bbox_inches='tight')
            logger.info(f"Saved position comparison visualization to {os.path.join(output_dir, 'position_comparison.png')}")
        else:
            plt.show()
        
        plt.close()
    
    # Visualize processing time comparison
    if "processing_time" in results["comparison"]:
        methods = list(results["comparison"]["processing_time"].keys())
        times = [results["comparison"]["processing_time"][method] for method in methods]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(methods, times, color='lightgreen')
        plt.xlabel('Method')
        plt.ylabel('Processing Time (seconds)')
        plt.title('Processing Time Comparison')
        
        # Add value labels on top of the bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                     f'{height:.2f}s', ha='center', va='bottom', rotation=0)
        
        # Save or show the plot
        if output_dir:
            plt.savefig(os.path.join(output_dir, "time_comparison.png"), dpi=300, bbox_inches='tight')
            logger.info(f"Saved time comparison visualization to {os.path.join(output_dir, 'time_comparison.png')}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Compare different correction methods on the same conversations."
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
        "--dataset-split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to use",
    )
    
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=10,
        help="Number of conversations to evaluate",
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "context-aware", "keyboard-specific", "combined"],
        choices=["baseline", "context-aware", "keyboard-specific", "combined"],
        help="Correction methods to compare",
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
        "--context-weight",
        type=float,
        default=0.7,
        help="Weight of context-based probability (0-1)",
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
        default="method_comparison_results.json",
        help="Path to save the comparison results",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
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
        "--no-visualize",
        action="store_true",
        help="Don't create visualizations",
    )
    
    args = parser.parse_args()
    
    # Load the dataset
    dataset = load_aac_conversations(
        split=args.dataset_split,
        cache_dir=args.cache_dir,
        use_auth_token=not args.no_token,
    )
    
    if dataset is None:
        logger.error("Failed to load dataset. Exiting.")
        return
    
    # Compare methods
    results = compare_methods(
        dataset=dataset,
        methods=args.methods,
        ppm_model_path=args.ppm_model,
        confusion_matrix_path=args.confusion_matrix,
        word_ngram_model_path=args.word_ngram_model,
        lexicon_path=args.lexicon,
        num_conversations=args.num_conversations,
        use_gold_context=args.use_gold_context,
        context_window_size=args.context_window,
        context_weight=args.context_weight,
        keyboard_layout=args.keyboard_layout,
        use_keyboard_matrices=args.use_keyboard_matrices,
    )
    
    # Print summary
    print("\n=== Method Comparison Summary ===")
    print("\nOverall Accuracy:")
    for method, accuracy in results["comparison"]["overall_accuracy"].items():
        print(f"  {method}: {accuracy:.4f}")
    
    print("\nProcessing Time:")
    for method, time_taken in results["comparison"]["processing_time"].items():
        print(f"  {method}: {time_taken:.2f} seconds")
    
    # Save results
    if args.output:
        try:
            # Create the output directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
            
            # Convert defaultdicts to regular dicts for JSON serialization
            serializable_results = json.loads(json.dumps(results, default=lambda x: dict(x) if isinstance(x, defaultdict) else x))
            
            # Save the results
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(serializable_results, f, indent=2)
            
            logger.info(f"Saved comparison results to {args.output}")
        except Exception as e:
            logger.error(f"Error saving comparison results: {e}")
    
    # Create visualizations
    if not args.no_visualize:
        visualize_method_comparison(results, args.output_dir)


if __name__ == "__main__":
    main()
