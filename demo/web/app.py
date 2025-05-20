#!/usr/bin/env python3
"""
Gradio Web Demo for the AAC Noisy Input Correction Engine.

This script provides a web interface for demonstrating the
noisy channel corrector using Gradio.
"""

import os
import sys
import logging
import argparse
import random
from typing import List, Dict, Tuple, Optional, Any
import difflib

# Add the parent directory to the Python path
parent_dir = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
sys.path.append(parent_dir)

# Import Gradio
import gradio as gr

# Import the corrector
from lib.corrector.corrector import NoisyChannelCorrector

# Import dataset utilities
try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    print(
        "Warning: datasets library not available. Conversation simulation will be disabled."
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get Hugging Face token from environment variable
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", None)


def load_aac_conversations(
    split: str = "train", cache_dir: Optional[str] = None, use_auth_token: bool = True
) -> Any:
    """
    Load the AACConversations dataset from Hugging Face.

    Args:
        split: Dataset split to load ('train', 'validation', or 'test')
        cache_dir: Directory to cache the dataset
        use_auth_token: Whether to use the Hugging Face auth token

    Returns:
        The loaded dataset or None if loading failed
    """
    if not DATASETS_AVAILABLE:
        logger.error("Cannot load dataset: datasets library not available")
        return None

    try:
        # Load the dataset
        # Use the token from the environment variable
        dataset = load_dataset(
            "willwade/AACConversations",
            split=split,
            cache_dir=cache_dir,
            token=HUGGINGFACE_TOKEN if use_auth_token else None,
        )
        logger.info(
            f"Loaded AACConversations dataset ({split} split) with {len(dataset)} examples"
        )
        return dataset
    except Exception as e:
        logger.error(f"Error loading AACConversations dataset: {e}")
        return None


def filter_english_data(dataset: Any, language_code: str = "en-GB") -> Any:
    """
    Filter the dataset to include only data in the specified language.

    Args:
        dataset: The dataset to filter
        language_code: The language code to filter by

    Returns:
        The filtered dataset
    """
    if dataset is None:
        logger.error("Cannot filter dataset: dataset is None")
        return None

    try:
        # Filter by language code
        filtered = dataset.filter(
            lambda example: example.get("language_code") == language_code
        )
        logger.info(
            f"Filtered dataset to {len(filtered)} examples in language {language_code}"
        )
        return filtered
    except Exception as e:
        logger.error(f"Error filtering dataset by language: {e}")
        return dataset


def group_by_conversation(dataset: Any) -> Dict[int, List[Dict]]:
    """
    Group dataset examples by conversation ID.

    Args:
        dataset: The dataset to group

    Returns:
        Dictionary mapping conversation IDs to lists of examples
    """
    if dataset is None:
        logger.error("Cannot group dataset: dataset is None")
        return {}

    try:
        conversations = {}
        for example in dataset:
            conv_id = example.get("conversation_id", -1)
            if conv_id not in conversations:
                conversations[conv_id] = []
            conversations[conv_id].append(example)

        # Sort each conversation by turn number
        for conv_id in conversations:
            conversations[conv_id].sort(key=lambda x: x.get("turn_number", 0))

        logger.info(f"Grouped dataset into {len(conversations)} conversations")
        return conversations
    except Exception as e:
        logger.error(f"Error grouping dataset by conversation: {e}")
        return {}


def get_sample_conversations(
    dataset: Any,
    num_samples: int = 5,
    language_code: str = "en-GB",
    noise_level: str = "minimal",
    noise_type: str = "qwerty",
) -> List[Dict]:
    """
    Get a sample of conversations from the dataset.

    Args:
        dataset: The dataset to sample from
        num_samples: Number of conversations to sample
        language_code: Language code to filter by
        noise_level: Noise level to filter by (minimal, moderate, heavy)
        noise_type: Noise type to filter by (qwerty, cognitive, motor, etc.)

    Returns:
        List of sampled conversations
    """
    if dataset is None:
        logger.error("Cannot sample conversations: dataset is None")
        return []

    try:
        # Filter by language code
        filtered_dataset = filter_english_data(dataset, language_code)

        # Filter by noise level and type if specified
        if noise_level or noise_type:
            filtered_dataset = filtered_dataset.filter(
                lambda example: (
                    (not noise_level or example.get("noise_level") == noise_level)
                    and (not noise_type or example.get("noise_type") == noise_type)
                )
            )
            logger.info(
                f"Filtered dataset to examples with noise level '{noise_level}' and noise type '{noise_type}'"
            )

        # Group by conversation
        conversations = group_by_conversation(filtered_dataset)

        # Sample conversations with at least 2 turns and at least one AAC User turn
        valid_conv_ids = []
        for conv_id, turns in conversations.items():
            if len(turns) >= 2 and any(
                turn.get("speaker") == "AAC User" for turn in turns
            ):
                valid_conv_ids.append(conv_id)

        logger.info(f"Found {len(valid_conv_ids)} valid conversations")

        # Sample from valid conversations
        if len(valid_conv_ids) <= num_samples:
            sampled_ids = valid_conv_ids
        else:
            sampled_ids = random.sample(valid_conv_ids, num_samples)

        sampled_conversations = [conversations[conv_id] for conv_id in sampled_ids]
        logger.info(f"Sampled {len(sampled_conversations)} conversations")
        return sampled_conversations
    except Exception as e:
        logger.error(f"Error sampling conversations: {e}")
        return []


def highlight_differences(original: str, correction: str) -> str:
    """
    Highlight differences between original and corrected text.

    Args:
        original: Original text
        correction: Corrected text

    Returns:
        HTML-formatted string with differences highlighted
    """
    d = difflib.Differ()
    diff = list(d.compare(original.split(), correction.split()))

    result = []
    for word in diff:
        if word.startswith("+ "):
            result.append(f"<span style='color: green'>{word[2:]}</span>")
        elif word.startswith("- "):
            result.append(f"<span style='color: red'>{word[2:]}</span>")
        elif word.startswith("  "):
            result.append(word[2:])

    return " ".join(result)


class CorrectionDemo:
    """Class to manage the correction demo state and functionality."""

    def __init__(self, args):
        """Initialize the demo with command-line arguments."""
        self.args = args
        self.corrector = None
        self.dataset = None
        self.conversations = []
        self.current_conversation = []
        self.current_turn_index = 0
        self.conversation_context = []

        # Initialize the corrector
        self.initialize_corrector()

        # Load the dataset if available
        if DATASETS_AVAILABLE:
            self.load_dataset()

    def initialize_corrector(self):
        """Initialize the noisy channel corrector with models."""
        try:
            # Create the corrector
            self.corrector = NoisyChannelCorrector(
                max_candidates=self.args.max_candidates,
                context_window_size=self.args.context_window,
                context_weight=self.args.context_weight,
            )

            # Load the PPM model
            if os.path.exists(self.args.ppm_model):
                self.corrector.load_ppm_model(self.args.ppm_model)
                logger.info(f"Loaded PPM model from {self.args.ppm_model}")
            else:
                logger.warning(f"PPM model file not found: {self.args.ppm_model}")

            # Load the confusion matrix
            if os.path.exists(self.args.confusion_matrix):
                self.corrector.load_confusion_model(
                    self.args.confusion_matrix, self.args.keyboard_layout
                )
                logger.info(
                    f"Loaded confusion matrix from {self.args.confusion_matrix}"
                )
            else:
                logger.warning(
                    f"Confusion matrix file not found: {self.args.confusion_matrix}"
                )

            # Load the word n-gram model
            if os.path.exists(self.args.word_ngram_model):
                self.corrector.load_word_ngram_model(self.args.word_ngram_model)
                logger.info(
                    f"Loaded word n-gram model from {self.args.word_ngram_model}"
                )
            else:
                logger.warning(
                    f"Word n-gram model file not found: {self.args.word_ngram_model}"
                )

            # Load the lexicon
            if os.path.exists(self.args.lexicon):
                self.corrector.load_lexicon_from_file(self.args.lexicon)
                logger.info(f"Loaded lexicon from {self.args.lexicon}")
            else:
                logger.warning(f"Lexicon file not found: {self.args.lexicon}")

            logger.info("Corrector initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing corrector: {e}")

    def load_dataset(self):
        """Load the AACConversations dataset."""
        try:
            # Load the dataset
            self.dataset = load_aac_conversations(
                split=self.args.dataset_split,
                cache_dir=self.args.cache_dir,
                use_auth_token=not self.args.no_token,
            )

            if self.dataset is None:
                logger.error("Failed to load dataset")
                return

            # Try to get conversations with minimal noise and qwerty errors
            self.conversations = get_sample_conversations(
                self.dataset,
                num_samples=10,
                language_code="en-GB",
                noise_level="minimal",
                noise_type="qwerty",
            )

            # If no conversations found, try with just qwerty errors
            if not self.conversations:
                logger.info(
                    "No conversations with minimal noise and qwerty errors found. Trying with any noise level."
                )
                self.conversations = get_sample_conversations(
                    self.dataset,
                    num_samples=10,
                    language_code="en-GB",
                    noise_level=None,
                    noise_type="qwerty",
                )

            # If still no conversations, try with any noise type
            if not self.conversations:
                logger.info(
                    "No conversations with qwerty errors found. Trying with any noise type."
                )
                self.conversations = get_sample_conversations(
                    self.dataset,
                    num_samples=10,
                    language_code="en-GB",
                    noise_level=None,
                    noise_type=None,
                )

            if self.conversations:
                self.current_conversation = self.conversations[0]
                logger.info(f"Loaded {len(self.conversations)} sample conversations")
            else:
                logger.warning("No conversations loaded")
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")

    def correct_text(
        self, text: str, context: Optional[str] = None
    ) -> List[Tuple[str, float]]:
        """
        Correct text using the noisy channel corrector.

        Args:
            text: Text to correct
            context: Optional context for correction

        Returns:
            List of (correction, score) tuples
        """
        if not text:
            return []

        if self.corrector is None:
            logger.warning("Corrector not initialized")
            return [(text, 1.0)]

        try:
            # Correct the text
            corrections = self.corrector.correct(
                text,
                context=context,
                max_edit_distance=self.args.max_edit_distance,
            )
            return corrections
        except Exception as e:
            logger.error(f"Error correcting text: {e}")
            return [(text, 1.0)]

    def predict_next_words(self, text: str, num_predictions: int = 5) -> List[str]:
        """
        Predict next words using the PPM model.

        Args:
            text: Text to predict from
            num_predictions: Number of predictions to return

        Returns:
            List of predicted words
        """
        if not text:
            return []

        if self.corrector is None or self.corrector.ppm_model is None:
            logger.warning("PPM model not initialized")
            return []

        try:
            # Predict next words
            predictions = self.corrector.ppm_model.predict_next_words(
                text, num_predictions
            )
            return predictions
        except Exception as e:
            logger.error(f"Error predicting next words: {e}")
            return []

    def get_next_conversation_turn(self) -> Tuple[str, str, str, str]:
        """
        Get the next turn in the current conversation.

        Returns:
            Tuple of (speaker, noisy_text, intended_text, turn_info)
        """
        if not self.current_conversation:
            return ("", "", "", "No conversation loaded")

        if self.current_turn_index >= len(self.current_conversation):
            self.current_turn_index = 0
            return ("", "", "", "End of conversation")

        turn = self.current_conversation[self.current_turn_index]
        self.current_turn_index += 1

        speaker = turn.get("speaker", "")
        noisy_text = turn.get("noisy_minimal", turn.get("noisy", ""))
        intended_text = turn.get("fully_corrected", turn.get("intended", ""))

        # Format turn info
        turn_info = f"Turn {turn.get('turn_number', self.current_turn_index)}"
        if "noise_level" in turn:
            turn_info += f" | Noise Level: {turn.get('noise_level', '')}"
        if "noise_type" in turn:
            turn_info += f" | Noise Type: {turn.get('noise_type', '')}"

        return (speaker, noisy_text, intended_text, turn_info)

    def select_conversation(self, index: int) -> str:
        """
        Select a conversation by index.

        Args:
            index: Index of the conversation to select

        Returns:
            Information about the selected conversation
        """
        if not self.conversations or index < 0 or index >= len(self.conversations):
            return "Invalid conversation index"

        self.current_conversation = self.conversations[index]
        self.current_turn_index = 0
        self.conversation_context = []

        # Get conversation info
        if self.current_conversation:
            first_turn = self.current_conversation[0]
            conv_id = first_turn.get("conversation_id", "Unknown")
            language = first_turn.get("language_code", "Unknown")
            noise_level = first_turn.get("noise_level", "Unknown")
            noise_type = first_turn.get("noise_type", "Unknown")
            num_turns = len(self.current_conversation)

            return f"Selected conversation {conv_id} ({language}) with {num_turns} turns. Noise level: {noise_level}, Noise type: {noise_type}"
        else:
            return "Empty conversation"


def create_gradio_interface(demo: CorrectionDemo):
    """
    Create the Gradio interface for the correction demo.

    Args:
        demo: The CorrectionDemo instance

    Returns:
        The Gradio interface
    """
    # Define the interactive mode interface
    with gr.Blocks(title="AAC Noisy Input Correction Demo") as interface:
        gr.Markdown("# AAC Noisy Input Correction Demo")
        gr.Markdown(
            "This demo showcases the noisy channel correction engine for AAC users."
        )

        with gr.Tabs():
            # Interactive Mode Tab
            with gr.TabItem("Interactive Mode"):
                gr.Markdown("## Interactive Mode")
                gr.Markdown(
                    "Type text in the input box and see real-time corrections and predictions."
                )

                with gr.Row():
                    with gr.Column(scale=2):
                        input_text = gr.Textbox(
                            label="Input Text", placeholder="Type here...", lines=3
                        )

                        with gr.Row():
                            context_text = gr.Textbox(
                                label="Context (Optional)",
                                placeholder="Previous sentence or context...",
                                lines=2,
                            )

                            correct_btn = gr.Button("Correct")

                    with gr.Column(scale=2):
                        corrections_md = gr.Markdown("### Corrections will appear here")

                        with gr.Accordion("Word Predictions", open=False):
                            predictions_md = gr.Markdown(
                                "### Predictions will appear here"
                            )

                def on_text_input(text, context):
                    # Get corrections
                    corrections = demo.correct_text(text, context)

                    # Format corrections
                    corrections_html = "<div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>"
                    corrections_html += "<h3>Corrections:</h3>"

                    if corrections:
                        for i, (correction, score) in enumerate(corrections[:5]):
                            # Highlight differences
                            diff_html = highlight_differences(text, correction)

                            # Add to HTML
                            corrections_html += "<div style='margin: 5px 0; padding: 5px; border-bottom: 1px solid #ddd;'>"
                            corrections_html += f"<p><b>{i+1}.</b> {correction} <span style='color: #666; font-size: 0.9em;'>(score: {score:.4f})</span></p>"
                            corrections_html += f"<p><b>Diff:</b> {diff_html}</p>"
                            corrections_html += "</div>"
                    else:
                        corrections_html += "<p>No corrections found</p>"

                    corrections_html += "</div>"

                    # Get predictions
                    predictions = demo.predict_next_words(text)

                    # Format predictions
                    predictions_html = "<div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>"
                    predictions_html += "<h3>Next Word Predictions:</h3>"

                    if predictions:
                        predictions_html += (
                            "<div style='display: flex; flex-wrap: wrap;'>"
                        )
                        for word in predictions:
                            predictions_html += f"<div style='margin: 5px; padding: 5px 10px; background-color: #e0e0e0; border-radius: 15px;'>{word}</div>"
                        predictions_html += "</div>"
                    else:
                        predictions_html += "<p>No predictions available</p>"

                    predictions_html += "</div>"

                    return corrections_html, predictions_html

                # Connect the input to the correction function
                input_text.change(
                    on_text_input,
                    inputs=[input_text, context_text],
                    outputs=[corrections_md, predictions_md],
                )

                correct_btn.click(
                    on_text_input,
                    inputs=[input_text, context_text],
                    outputs=[corrections_md, predictions_md],
                )

            # Conversation Simulation Tab
            with gr.TabItem("Conversation Simulation"):
                gr.Markdown("## Conversation Simulation")
                gr.Markdown(
                    "Simulate a conversation from the AACConversations dataset to see how context affects correction."
                )

                with gr.Row():
                    with gr.Column(scale=1):
                        conv_dropdown = gr.Dropdown(
                            label="Select Conversation",
                            choices=[
                                f"Conversation {i+1}"
                                for i in range(len(demo.conversations))
                            ],
                            value="Conversation 1" if demo.conversations else None,
                        )

                        conv_info = gr.Markdown("### Conversation Information")

                        # Add conversation history display
                        conversation_history = gr.Markdown("### Conversation History")

                        next_turn_btn = gr.Button("Next Turn")

                    with gr.Column(scale=2):
                        speaker_label = gr.Markdown("### Speaker")

                        with gr.Row():
                            with gr.Column(scale=1):
                                noisy_text = gr.Textbox(
                                    label="Noisy Text (What was typed)",
                                    lines=3,
                                    interactive=False,
                                )

                            with gr.Column(scale=1):
                                intended_text = gr.Textbox(
                                    label="Intended Text (What was meant)",
                                    lines=3,
                                    interactive=False,
                                )

                        turn_info = gr.Markdown("### Turn Information")

                        # Add context information
                        context_info = gr.Markdown("### Context Used for Correction")

                        corrections_display = gr.Markdown("### Corrections")

                def on_conversation_select(conv_index_str):
                    # Extract the index from the string
                    try:
                        conv_index = int(conv_index_str.split(" ")[1]) - 1
                    except ValueError:
                        conv_index = 0

                    # Select the conversation
                    info = demo.select_conversation(conv_index)

                    # Reset the conversation history
                    history_html = "<div style='padding: 10px; background-color: #f8f8f8; border-radius: 5px;'>"
                    history_html += "<h3>Conversation History</h3>"
                    history_html += (
                        "<p>Click 'Next Turn' to start the conversation.</p>"
                    )
                    history_html += "</div>"

                    # Reset the display
                    return (
                        info,
                        "### Speaker",
                        "",
                        "",
                        "### Turn Information",
                        history_html,
                        "### Context Used for Correction",
                        "### Corrections",
                    )

                def on_next_turn():
                    # Get the next turn
                    speaker, noisy, intended, turn_info_text = (
                        demo.get_next_conversation_turn()
                    )

                    # Update the speaker label
                    speaker_html = f"### Speaker: {speaker}"

                    # Update the turn info
                    turn_info_html = f"### Turn Information: {turn_info_text}"

                    # Build conversation history display
                    history_html = "<div style='padding: 10px; background-color: #f8f8f8; border-radius: 5px;'>"
                    history_html += "<h3>Conversation History</h3>"

                    if not demo.conversation_context:
                        history_html += "<p>No conversation history yet.</p>"
                    else:
                        history_html += (
                            "<div style='max-height: 200px; overflow-y: auto;'>"
                        )
                        # Show the last 5 turns or all turns if less than 5
                        for i, text in enumerate(demo.conversation_context[-5:]):
                            turn_num = len(demo.conversation_context) - 5 + i + 1
                            # Determine if this was an AAC User or Communication Partner turn
                            if turn_num <= len(demo.current_conversation):
                                turn_speaker = demo.current_conversation[
                                    turn_num - 1
                                ].get("speaker", "Unknown")
                                history_html += f"<p><b>Turn {turn_num} ({turn_speaker}):</b> {text}</p>"
                            else:
                                history_html += f"<p><b>Turn {turn_num}:</b> {text}</p>"
                        history_html += "</div>"

                    history_html += "</div>"

                    # Get context information
                    context_html = "<div style='padding: 10px; background-color: #f8f8f8; border-radius: 5px;'>"
                    context_html += "<h3>Context Used for Correction</h3>"

                    # Get context from previous turns
                    context = None
                    if demo.conversation_context:
                        context = (
                            demo.conversation_context[-1]
                            if len(demo.conversation_context) > 0
                            else None
                        )
                        if context:
                            context_html += f"<p>{context}</p>"
                        else:
                            context_html += "<p>No context available.</p>"
                    else:
                        context_html += "<p>No context available.</p>"

                    context_html += "</div>"

                    # Get corrections if this is an AAC User turn
                    corrections_html = "<div style='padding: 10px; background-color: #f5f5f5; border-radius: 5px;'>"
                    corrections_html += "<h3>Corrections:</h3>"

                    if speaker == "AAC User" and noisy:
                        # Get corrections
                        corrections = demo.correct_text(noisy, context)

                        if corrections:
                            for i, (correction, score) in enumerate(corrections[:5]):
                                # Highlight differences
                                diff_html = highlight_differences(noisy, correction)

                                # Check if correction matches intended
                                match_indicator = ""
                                if correction.lower() == intended.lower():
                                    match_indicator = " ✓ Matches intended"

                                # Add to HTML
                                corrections_html += "<div style='margin: 5px 0; padding: 5px; border-bottom: 1px solid #ddd;'>"
                                corrections_html += f"<p><b>{i+1}.</b> {correction} <span style='color: #666; font-size: 0.9em;'>(score: {score:.4f}){match_indicator}</span></p>"
                                corrections_html += f"<p><b>Diff:</b> {diff_html}</p>"
                                corrections_html += "</div>"
                        else:
                            corrections_html += "<p>No corrections found</p>"
                    else:
                        corrections_html += "<p>No corrections needed for this turn</p>"

                    corrections_html += "</div>"

                    # Update the conversation context
                    if noisy:
                        demo.conversation_context.append(noisy)

                    return (
                        speaker_html,
                        noisy,
                        intended,
                        turn_info_html,
                        history_html,
                        context_html,
                        corrections_html,
                    )

                # Connect the conversation selection dropdown
                conv_dropdown.change(
                    on_conversation_select,
                    inputs=[conv_dropdown],
                    outputs=[
                        conv_info,
                        speaker_label,
                        noisy_text,
                        intended_text,
                        turn_info,
                        conversation_history,
                        context_info,
                        corrections_display,
                    ],
                )

                # Connect the next turn button
                next_turn_btn.click(
                    on_next_turn,
                    inputs=[],
                    outputs=[
                        speaker_label,
                        noisy_text,
                        intended_text,
                        turn_info,
                        conversation_history,
                        context_info,
                        corrections_display,
                    ],
                )

            # Settings Tab
            with gr.TabItem("Settings"):
                gr.Markdown("## Settings")
                gr.Markdown("Configure the correction engine parameters.")

                with gr.Row():
                    with gr.Column():
                        max_candidates = gr.Slider(
                            label="Max Candidates",
                            minimum=1,
                            maximum=100,
                            value=demo.args.max_candidates,
                            step=1,
                        )

                        context_window = gr.Slider(
                            label="Context Window Size",
                            minimum=1,
                            maximum=10,
                            value=demo.args.context_window,
                            step=1,
                        )

                        context_weight = gr.Slider(
                            label="Context Weight",
                            minimum=0.0,
                            maximum=1.0,
                            value=demo.args.context_weight,
                            step=0.1,
                        )

                    with gr.Column():
                        max_edit_distance = gr.Slider(
                            label="Max Edit Distance",
                            minimum=1,
                            maximum=5,
                            value=demo.args.max_edit_distance,
                            step=1,
                        )

                        keyboard_layout = gr.Dropdown(
                            label="Keyboard Layout",
                            choices=["qwerty", "azerty", "dvorak"],
                            value=demo.args.keyboard_layout,
                        )

                        apply_settings_btn = gr.Button("Apply Settings")

                def on_apply_settings(
                    max_cand, ctx_window, ctx_weight, max_edit, kb_layout
                ):
                    # Update the demo settings
                    demo.args.max_candidates = int(max_cand)
                    demo.args.context_window = int(ctx_window)
                    demo.args.context_weight = float(ctx_weight)
                    demo.args.max_edit_distance = int(max_edit)
                    demo.args.keyboard_layout = kb_layout

                    # Reinitialize the corrector with new settings
                    demo.initialize_corrector()

                    return "Settings applied successfully"

                # Connect the apply settings button
                apply_settings_btn.click(
                    on_apply_settings,
                    inputs=[
                        max_candidates,
                        context_window,
                        context_weight,
                        max_edit_distance,
                        keyboard_layout,
                    ],
                    outputs=[gr.Markdown(value="")],
                )

        # Add footer
        gr.Markdown("### About")
        gr.Markdown(
            "This demo showcases the AAC Noisy Input Correction Engine, "
            "which uses a noisy channel model to correct typing errors in AAC communication."
        )

    return interface


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Web demo for the AAC Noisy Input Correction Engine."
    )

    parser.add_argument(
        "--ppm-model",
        type=str,
        default="models/ppm_model_en_gb.pkl",
        help="Path to the PPM model file",
    )

    parser.add_argument(
        "--confusion-matrix",
        type=str,
        default="models/qwerty_confusion_matrix_en_gb.json",
        help="Path to the confusion matrix file",
    )

    parser.add_argument(
        "--word-ngram-model",
        type=str,
        default="models/word_ngram_model_en_gb.pkl",
        help="Path to the word n-gram model file",
    )

    parser.add_argument(
        "--lexicon",
        type=str,
        default="data/enhanced_lexicon_en_gb.txt",
        help="Path to the lexicon file",
    )

    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10,
        help="Maximum number of correction candidates to generate",
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help="Size of the context window for context-aware correction",
    )

    parser.add_argument(
        "--context-weight",
        type=float,
        default=0.5,
        help="Weight of the context in the correction score",
    )

    parser.add_argument(
        "--max-edit-distance",
        type=int,
        default=2,
        help="Maximum edit distance for candidate generation",
    )

    parser.add_argument(
        "--keyboard-layout",
        type=str,
        default="qwerty",
        choices=["qwerty", "azerty", "dvorak"],
        help="Keyboard layout for confusion matrix",
    )

    parser.add_argument(
        "--dataset-split",
        type=str,
        default="test",
        choices=["train", "test"],
        help="Dataset split to use",
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
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio server on",
    )

    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for the demo",
    )

    args = parser.parse_args()

    # Create the demo
    demo = CorrectionDemo(args)

    # Create the Gradio interface
    interface = create_gradio_interface(demo)

    # Launch the interface
    interface.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", args.port)),
        share=args.share,
    )


if __name__ == "__main__":
    main()
