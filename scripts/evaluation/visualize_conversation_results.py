#!/usr/bin/env python3
"""
Visualization tool for conversation-level evaluation results.

This script provides functionality for visualizing the results of
conversation-level evaluation, including accuracy trends, position-based
metrics, and speaker-based metrics.
"""

import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Optional

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Check if matplotlib is available
try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    logger.warning("Matplotlib not available. Visualizations will be limited.")
    MATPLOTLIB_AVAILABLE = False


def load_results(results_path: str) -> Dict[str, Any]:
    """
    Load evaluation results from a file.

    Args:
        results_path: Path to the results file

    Returns:
        The loaded results
    """
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            results = json.load(f)

        logger.info(f"Loaded evaluation results from {results_path}")
        return results
    except Exception as e:
        logger.error(f"Error loading evaluation results: {e}")
        return {}


def print_conversation_summary(
    results: Dict[str, Any], conversation_id: Optional[int] = None
) -> None:
    """
    Print a summary of a conversation.

    Args:
        results: The evaluation results
        conversation_id: ID of the conversation to summarize (0-based index)
    """
    if not results or "conversations" not in results or not results["conversations"]:
        logger.error("No conversations found in results")
        return

    # If conversation_id is not specified, use the first conversation
    if conversation_id is None:
        conversation_id = 0

    # Check if the conversation_id is valid
    if conversation_id < 0 or conversation_id >= len(results["conversations"]):
        logger.error(f"Invalid conversation ID: {conversation_id}")
        return

    # Get the conversation
    conversation = results["conversations"][conversation_id]

    print("\n=== Conversation Summary ===")
    print(f"Conversation ID: {conversation.get('conversation_id', 'unknown')}")
    print(f"Number of Turns: {len(conversation['turns'])}")
    print(f"Accuracy: {conversation['metrics']['accuracy']:.4f}")

    # Print turn-by-turn summary
    print("\nTurn-by-Turn Summary:")
    for turn in conversation["turns"]:
        print(f"\nTurn {turn['turn_number']}:")
        print(f"  Speaker: {turn['speaker']}")
        print(f"  Intended: {turn['intended']}")
        print(f"  Noisy: {turn['noisy']}")
        print(f"  Corrected: {turn['corrections'][0]['correction']}")
        print(f"  Correct: {turn['correct_at_1']}")

        # Print context if available
        if "context" in turn and turn["context"]:
            print(f"  Context: {turn['context']}")


def visualize_accuracy_by_position(
    results: Dict[str, Any], output_path: Optional[str] = None
) -> None:
    """
    Visualize accuracy by turn position.

    Args:
        results: The evaluation results
        output_path: Path to save the visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib not available. Cannot create visualization.")
        return

    if (
        not results
        or "metrics" not in results
        or "accuracy_by_position" not in results["metrics"]
    ):
        logger.error("No position-based metrics found in results")
        return

    # Get the position-based metrics
    accuracy_by_position = results["metrics"]["accuracy_by_position"]

    # Convert to lists for plotting
    positions = sorted(int(pos) for pos in accuracy_by_position.keys())
    accuracies = [accuracy_by_position[str(pos)] for pos in positions]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(positions, accuracies, marker="o", linestyle="-", color="blue")
    plt.xlabel("Turn Position")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Turn Position")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.xticks(positions)
    plt.ylim(0, 1.05)

    # Add a horizontal line for the overall accuracy
    if "overall_accuracy" in results["metrics"]:
        plt.axhline(
            y=results["metrics"]["overall_accuracy"],
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        plt.text(
            positions[-1],
            results["metrics"]["overall_accuracy"] + 0.02,
            f"Overall: {results['metrics']['overall_accuracy']:.4f}",
            color="red",
            ha="right",
        )

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved position-based visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_accuracy_by_speaker(
    results: Dict[str, Any], output_path: Optional[str] = None
) -> None:
    """
    Visualize accuracy by speaker.

    Args:
        results: The evaluation results
        output_path: Path to save the visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib not available. Cannot create visualization.")
        return

    if (
        not results
        or "metrics" not in results
        or "accuracy_by_speaker" not in results["metrics"]
    ):
        logger.error("No speaker-based metrics found in results")
        return

    # Get the speaker-based metrics
    accuracy_by_speaker = results["metrics"]["accuracy_by_speaker"]

    # Convert to lists for plotting
    speakers = list(accuracy_by_speaker.keys())
    accuracies = list(accuracy_by_speaker.values())

    # Sort by accuracy
    speakers, accuracies = zip(
        *sorted(zip(speakers, accuracies), key=lambda x: x[1], reverse=True)
    )

    # Create the plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(speakers, accuracies, color="skyblue")
    plt.xlabel("Speaker")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Speaker")
    plt.ylim(0, 1.05)

    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.02,
            f"{height:.4f}",
            ha="center",
            va="bottom",
            rotation=0,
        )

    # Add a horizontal line for the overall accuracy
    if "overall_accuracy" in results["metrics"]:
        plt.axhline(
            y=results["metrics"]["overall_accuracy"],
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        plt.text(
            len(speakers) - 1,
            results["metrics"]["overall_accuracy"] + 0.02,
            f"Overall: {results['metrics']['overall_accuracy']:.4f}",
            color="red",
            ha="right",
        )

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved speaker-based visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_accuracy_trend(
    results: Dict[str, Any], output_path: Optional[str] = None
) -> None:
    """
    Visualize accuracy trend across turns.

    Args:
        results: The evaluation results
        output_path: Path to save the visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib not available. Cannot create visualization.")
        return

    if (
        not results
        or "metrics" not in results
        or "accuracy_trend" not in results["metrics"]
    ):
        logger.error("No trend metrics found in results")
        return

    # Get the trend metrics
    accuracy_trend = results["metrics"]["accuracy_trend"]

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(1, len(accuracy_trend) + 1),
        accuracy_trend,
        marker="",
        linestyle="-",
        color="green",
    )
    plt.xlabel("Turn")
    plt.ylabel("Moving Average Accuracy (3 turns)")
    plt.title("Accuracy Trend Across Turns")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.ylim(0, 1.05)

    # Add a horizontal line for the overall accuracy
    if "overall_accuracy" in results["metrics"]:
        plt.axhline(
            y=results["metrics"]["overall_accuracy"],
            color="red",
            linestyle="--",
            alpha=0.7,
        )
        plt.text(
            len(accuracy_trend),
            results["metrics"]["overall_accuracy"] + 0.02,
            f"Overall: {results['metrics']['overall_accuracy']:.4f}",
            color="red",
            ha="right",
        )

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved trend visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_conversation_flow(
    results: Dict[str, Any], conversation_id: int = 0, output_path: Optional[str] = None
) -> None:
    """
    Visualize the flow of a conversation with corrections.

    Args:
        results: The evaluation results
        conversation_id: ID of the conversation to visualize (0-based index)
        output_path: Path to save the visualization
    """
    if not MATPLOTLIB_AVAILABLE:
        logger.error("Matplotlib not available. Cannot create visualization.")
        return

    if not results or "conversations" not in results or not results["conversations"]:
        logger.error("No conversations found in results")
        return

    # Check if the conversation_id is valid
    if conversation_id < 0 or conversation_id >= len(results["conversations"]):
        logger.error(f"Invalid conversation ID: {conversation_id}")
        return

    # Get the conversation
    conversation = results["conversations"][conversation_id]

    # Extract turn-by-turn data
    turns = conversation["turns"]
    turn_numbers = [turn["turn_number"] for turn in turns]
    speakers = [turn["speaker"] for turn in turns]
    correct_at_1 = [turn["correct_at_1"] for turn in turns]

    # Create a colormap for speakers
    unique_speakers = list(set(speakers))

    # Make sure we don't run out of colors
    if len(unique_speakers) > 10:
        # Use a continuous colormap if we have more than 10 speakers
        cmap = plt.cm.get_cmap("hsv", len(unique_speakers))
        speaker_colors = {speaker: cmap(i) for i, speaker in enumerate(unique_speakers)}
    else:
        # Use tab10 for 10 or fewer speakers
        speaker_colors = {
            speaker: color
            for speaker, color in zip(unique_speakers, plt.cm.tab10.colors)
        }

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot speaker turns
    for i, (turn_number, speaker, is_correct) in enumerate(
        zip(turn_numbers, speakers, correct_at_1)
    ):
        color = speaker_colors[speaker]
        marker = "o" if is_correct else "x"
        plt.scatter(turn_number, i % 2, color=color, marker=marker, s=100)

        # Add text for the turn
        intended = turns[i]["intended"]
        corrected = turns[i]["corrections"][0]["correction"]

        # Truncate long text
        if len(intended) > 30:
            intended = intended[:27] + "..."
        if len(corrected) > 30:
            corrected = corrected[:27] + "..."

        # Add text with different colors based on correctness
        text_color = "green" if is_correct else "red"
        plt.text(
            turn_number,
            (i % 2) + 0.1,
            f"{intended}",
            ha="center",
            va="bottom",
            color=text_color,
            fontsize=8,
        )

        if not is_correct:
            plt.text(
                turn_number,
                (i % 2) - 0.1,
                f"→ {corrected}",
                ha="center",
                va="top",
                color="blue",
                fontsize=8,
            )

    # Add a legend for speakers
    for speaker, color in speaker_colors.items():
        plt.scatter([], [], color=color, label=speaker)

    # Add a legend for correctness
    plt.scatter([], [], color="black", marker="o", label="Correct")
    plt.scatter([], [], color="black", marker="x", label="Incorrect")

    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.xlabel("Turn Number")
    plt.title(
        f'Conversation Flow (ID: {conversation.get("conversation_id", "unknown")})'
    )
    plt.yticks([])
    plt.grid(True, axis="x", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved conversation flow visualization to {output_path}")
    else:
        plt.show()

    plt.close()


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Visualization tool for conversation-level evaluation results."
    )

    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to the evaluation results file",
    )

    parser.add_argument(
        "--conversation",
        type=int,
        default=0,
        help="ID of the conversation to visualize (0-based index)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations",
        help="Directory to save visualizations",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualizations instead of saving them",
    )

    args = parser.parse_args()

    # Load the results
    results = load_results(args.results)

    if not results:
        logger.error("No results loaded. Exiting.")
        return

    # Create the output directory if saving visualizations
    if not args.show:
        os.makedirs(args.output_dir, exist_ok=True)

    # Print conversation summary
    print_conversation_summary(results, args.conversation)

    # Create visualizations
    if args.show:
        # Show visualizations
        visualize_accuracy_by_position(results)
        visualize_accuracy_by_speaker(results)
        visualize_accuracy_trend(results)
        visualize_conversation_flow(results, args.conversation)
    else:
        # Save visualizations
        visualize_accuracy_by_position(
            results, os.path.join(args.output_dir, "accuracy_by_position.png")
        )
        visualize_accuracy_by_speaker(
            results, os.path.join(args.output_dir, "accuracy_by_speaker.png")
        )
        visualize_accuracy_trend(
            results, os.path.join(args.output_dir, "accuracy_trend.png")
        )
        visualize_conversation_flow(
            results,
            args.conversation,
            os.path.join(args.output_dir, f"conversation_flow_{args.conversation}.png"),
        )


if __name__ == "__main__":
    main()
