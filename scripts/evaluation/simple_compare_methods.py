#!/usr/bin/env python3
"""
Simple comparison of different correction methods.

This script provides a simplified version of the method comparison
that uses the SimpleCorrector with different configurations.
"""

import os
import sys
import json
import argparse
import logging
import time
from typing import List, Dict, Any, Optional
from collections import defaultdict

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the simple conversation evaluator
from module5.simple_conversation_evaluator import SimpleCorrector, SimpleConversationEvaluator

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
    lexicon_path: str,
    max_edit_distance: int = 2,
) -> SimpleCorrector:
    """
    Create a corrector with the specified method.
    
    Args:
        method: Correction method ('baseline', 'strict', 'relaxed', 'combined')
        lexicon_path: Path to the lexicon file
        max_edit_distance: Maximum edit distance to consider
        
    Returns:
        The configured corrector
    """
    # Create the corrector with appropriate settings
    if method == "baseline":
        # Baseline: Standard edit distance
        corrector = SimpleCorrector(lexicon_path=lexicon_path, max_candidates=5)
    elif method == "strict":
        # Strict: Lower edit distance
        corrector = SimpleCorrector(lexicon_path=lexicon_path, max_candidates=5)
        corrector.max_edit_distance = 1  # More strict
    elif method == "relaxed":
        # Relaxed: Higher edit distance
        corrector = SimpleCorrector(lexicon_path=lexicon_path, max_candidates=5)
        corrector.max_edit_distance = 3  # More relaxed
    elif method == "combined":
        # Combined: Standard edit distance with more candidates
        corrector = SimpleCorrector(lexicon_path=lexicon_path, max_candidates=10)
    else:
        # Default to baseline
        logger.warning(f"Unknown method: {method}. Using baseline method.")
        corrector = SimpleCorrector(lexicon_path=lexicon_path, max_candidates=5)
    
    return corrector


def compare_methods(
    methods: List[str],
    lexicon_path: str,
    num_conversations: int = 5,
    num_turns: int = 10,
    vocabulary_size: int = 100,
    error_rate: float = 0.2,
    use_gold_context: bool = False,
    context_window_size: int = 3,
) -> Dict[str, Any]:
    """
    Compare different correction methods on the same conversations.
    
    Args:
        methods: List of correction methods to compare
        lexicon_path: Path to the lexicon file
        num_conversations: Number of conversations to generate
        num_turns: Number of turns per conversation
        vocabulary_size: Size of the vocabulary for synthetic conversations
        error_rate: Error rate for synthetic conversations
        use_gold_context: Whether to use gold standard context
        context_window_size: Number of previous turns to use as context
        
    Returns:
        Dictionary with comparison results
    """
    # Generate synthetic conversations
    conversations = []
    for i in range(num_conversations):
        # Create a temporary evaluator to generate a conversation
        temp_corrector = SimpleCorrector(lexicon_path=lexicon_path)
        temp_evaluator = SimpleConversationEvaluator(
            corrector=temp_corrector,
            use_gold_context=use_gold_context,
            context_window_size=context_window_size,
        )
        
        # Generate a synthetic conversation
        conversation = temp_evaluator.generate_synthetic_conversation(
            num_turns=num_turns,
            vocabulary_size=vocabulary_size,
            error_rate=error_rate,
        )
        
        conversations.append(conversation)
    
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
            lexicon_path=lexicon_path,
        )
        
        # Create the evaluator
        evaluator = SimpleConversationEvaluator(
            corrector=corrector,
            use_gold_context=use_gold_context,
            context_window_size=context_window_size,
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
        
        for conversation in conversations:
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
        description="Simple comparison of different correction methods."
    )
    
    parser.add_argument(
        "--lexicon",
        type=str,
        default="data/wordlist.txt",
        help="Path to the lexicon file",
    )
    
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "strict", "relaxed", "combined"],
        choices=["baseline", "strict", "relaxed", "combined"],
        help="Correction methods to compare",
    )
    
    parser.add_argument(
        "--num-conversations",
        type=int,
        default=5,
        help="Number of conversations to evaluate",
    )
    
    parser.add_argument(
        "--num-turns",
        type=int,
        default=10,
        help="Number of turns per conversation",
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
        "--no-visualize",
        action="store_true",
        help="Don't create visualizations",
    )
    
    args = parser.parse_args()
    
    # Compare methods
    results = compare_methods(
        methods=args.methods,
        lexicon_path=args.lexicon,
        num_conversations=args.num_conversations,
        num_turns=args.num_turns,
        vocabulary_size=args.vocabulary_size,
        error_rate=args.error_rate,
        use_gold_context=args.use_gold_context,
        context_window_size=args.context_window,
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
