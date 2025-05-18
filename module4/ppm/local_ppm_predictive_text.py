"""
Predictive text input component using local PPM for the AAC app.
"""

import gradio as gr
import pandas as pd
import logging
import os
from typing import List, Dict, Tuple, Optional, Any, Callable, Union

try:
    from local_ppm_predictor import LocalPPMPredictor
    from enhanced_ppm_predictor import EnhancedPPMPredictor
    from continuous_learning import ContinuousLearningManager

    PPMPredictor = Union[LocalPPMPredictor, EnhancedPPMPredictor]
except ImportError:
    # Fallback for when imports are from different directories
    from ppm.local_ppm_predictor import LocalPPMPredictor
    from ppm.enhanced_ppm_predictor import EnhancedPPMPredictor
    from ppm.continuous_learning import ContinuousLearningManager

    PPMPredictor = Union[LocalPPMPredictor, EnhancedPPMPredictor]

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LocalPPMPredictiveTextInput:
    """A predictive text input component for Gradio using local PPM."""

    def __init__(
        self,
        ppm_predictor: Optional[PPMPredictor] = None,
        training_text_path: str = "training_text.txt",
        model_path: str = "ppm_model.pkl",
        max_order: int = 5,
        debug: bool = False,
        save_model: bool = True,
        continuous_learning_manager: Optional[ContinuousLearningManager] = None,
    ):
        """Initialize the predictive text input component.

        Args:
            ppm_predictor: PPM predictor instance to use for predictions
            training_text_path: Path to the training text file
            model_path: Path to save/load the model
            max_order: Maximum context length to consider
            debug: Whether to print debug information
            save_model: Whether to save the model after training
            continuous_learning_manager: Optional continuous learning manager for personalized predictions
        """
        # Use EnhancedPPMPredictor by default if no predictor is provided
        self.ppm_predictor = ppm_predictor or EnhancedPPMPredictor(
            max_order=max_order, debug=debug
        )
        self.training_text_path = training_text_path
        self.model_path = model_path
        self.save_model = save_model
        self.model_ready = self.ppm_predictor.model_ready
        self.continuous_learning_manager = continuous_learning_manager
        self.current_person_id = None
        self.current_topic = None

        # Try to load the model if it exists
        if os.path.exists(model_path):
            logger.info(f"Loading PPM model from {model_path}")
            if self.ppm_predictor.load_model(model_path):
                self.model_ready = True
                logger.info("PPM model loaded successfully")
                return
            else:
                logger.warning(f"Failed to load PPM model from {model_path}")

        # Initialize the model if the training text exists
        if os.path.exists(training_text_path):
            self._initialize_model()

    def _initialize_model(self) -> None:
        """Initialize the PPM model with training text."""
        try:
            logger.info(
                f"Initializing PPM model with training text: {self.training_text_path}"
            )

            # Read the training text
            with open(self.training_text_path, "r") as f:
                training_text = f.read()

            # Train the model and save it if requested
            success = self.ppm_predictor.train_on_text(
                text=training_text,
                save_model=self.save_model,
                model_path=self.model_path,
            )

            if success:
                logger.info("PPM model initialized successfully")
                if self.save_model:
                    logger.info(f"PPM model saved to {self.model_path}")
                self.model_ready = True
            else:
                logger.warning("Failed to train PPM model")
                self.model_ready = False

        except Exception as e:
            logger.error(f"Error initializing PPM model: {e}")
            self.model_ready = False

    def update_model(self, text: str) -> bool:
        """Update the model with new text.

        Args:
            text: Text to update the model with

        Returns:
            True if successful, False otherwise
        """
        if not self.model_ready:
            logger.warning("PPM model not ready. Cannot update.")
            return False

        try:
            logger.info(f"Updating PPM model with text ({len(text)} characters)")

            # Update the model and save it if requested
            success = self.ppm_predictor.update_model(
                text=text, save_model=self.save_model, model_path=self.model_path
            )

            if success:
                logger.info("PPM model updated successfully")
                if self.save_model:
                    logger.info(f"PPM model saved to {self.model_path}")
                return True
            else:
                logger.warning("Failed to update PPM model")
                return False

        except Exception as e:
            logger.error(f"Error updating PPM model: {e}")
            return False

    def set_context(
        self, person_id: Optional[str] = None, topic: Optional[str] = None
    ) -> None:
        """Set the current person and topic for personalized predictions.

        Args:
            person_id: ID of the person in the conversation
            topic: Selected topic (if any)
        """
        self.current_person_id = person_id
        self.current_topic = topic
        logger.info(
            f"Set context for personalized predictions: person={person_id}, topic={topic}"
        )

    def create_interface(
        self, on_submit: Optional[Callable] = None, on_change: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Create the Gradio interface components.

        Args:
            on_submit: Function to call when the text is submitted
            on_change: Function to call when the text changes

        Returns:
            Dictionary of Gradio components
        """
        with gr.Column() as container:
            # Model status indicator
            model_status = gr.Markdown(
                value=f"PPM Model Status: {'Ready' if self.model_ready else 'Not Ready'}"
            )

            # Main text input
            text_input = gr.Textbox(
                label="My Response:",
                placeholder="Start typing or select a prediction...",
                lines=3,
                elem_id="predictive_text_input",
            )

            with gr.Row():
                # Next character predictions
                next_chars = gr.Dataframe(
                    headers=["Next Character"],
                    datatype=["str"],
                    col_count=1,
                    row_count=5,
                    interactive=False,
                    elem_id="next_chars",
                )

                # Word completion predictions
                word_completions = gr.Dataframe(
                    headers=["Word Completion"],
                    datatype=["str"],
                    col_count=1,
                    row_count=5,
                    interactive=False,
                    elem_id="word_completions",
                )

            # Next word predictions
            next_words = gr.Dataframe(
                headers=["Next Word"],
                datatype=["str"],
                col_count=1,
                row_count=5,
                interactive=False,
                elem_id="next_words",
            )

            # Create buttons for predictions
            with gr.Row():
                char_btn = gr.Button("Use Character", size="sm")
                completion_btn = gr.Button("Use Completion", size="sm")
                word_btn = gr.Button("Use Word", size="sm")

            # Submit button
            submit_btn = gr.Button("Use This Response", variant="primary")

            # Selected prediction indicators
            selected_char_idx = gr.State(0)
            selected_completion_idx = gr.State(0)
            selected_word_idx = gr.State(0)

            # Set up event handlers
            text_input.change(
                self._on_text_change,
                inputs=[text_input],
                outputs=[next_chars, word_completions, next_words, model_status],
            )

            # Handle clicking on character button
            char_btn.click(
                self._on_char_click,
                inputs=[text_input, next_chars, selected_char_idx],
                outputs=[
                    text_input,
                    next_chars,
                    word_completions,
                    next_words,
                    model_status,
                ],
            )

            # Handle clicking on completion button
            completion_btn.click(
                self._on_completion_click,
                inputs=[text_input, word_completions, selected_completion_idx],
                outputs=[
                    text_input,
                    next_chars,
                    word_completions,
                    next_words,
                    model_status,
                ],
            )

            # Handle clicking on word button
            word_btn.click(
                self._on_word_click,
                inputs=[text_input, next_words, selected_word_idx],
                outputs=[
                    text_input,
                    next_chars,
                    word_completions,
                    next_words,
                    model_status,
                ],
            )

            # Set up submit handler if provided
            if on_submit:
                submit_btn.click(
                    on_submit,
                    inputs=[text_input],
                    outputs=[],  # Outputs will be defined by the caller
                )

        # Return the components for external access
        return {
            "container": container,
            "text_input": text_input,
            "next_chars": next_chars,
            "word_completions": word_completions,
            "next_words": next_words,
            "model_status": model_status,
            "submit_btn": submit_btn,
        }

    def _on_text_change(
        self, text: str
    ) -> Tuple[List[List[str]], List[List[str]], List[List[str]], str]:
        """Handle text input changes and update predictions.

        Args:
            text: Current text in the input field

        Returns:
            Updated predictions for next characters, word completions, and next words, and model status
        """
        # Check if the model is ready
        if not self.model_ready:
            # Try to initialize the model if it's not ready
            if os.path.exists(self.training_text_path):
                self._initialize_model()

            # If still not ready, return empty predictions
            if not self.model_ready:
                status = "PPM Model Status: Not Ready (check logs for details)"
                return (
                    [[""] for _ in range(5)],
                    [[""] for _ in range(5)],
                    [[""] for _ in range(5)],
                    status,
                )

        # Get predictions from the PPM predictor
        try:
            # Get base predictions
            predictions = self.ppm_predictor.get_all_predictions(
                text, num_predictions=5
            )

            # If continuous learning manager is available, use personalized predictions
            if self.continuous_learning_manager and self.current_person_id:
                try:
                    # Get personalized next word predictions
                    personalized_words = (
                        self.continuous_learning_manager.get_personalized_predictions(
                            text=text,
                            person_id=self.current_person_id,
                            topic=self.current_topic,
                            num_predictions=5,
                        )
                    )

                    # Replace the next words with personalized predictions if available
                    if personalized_words:
                        predictions["next_words"] = personalized_words
                        logger.info(
                            f"Using personalized predictions for {self.current_person_id}: {personalized_words}"
                        )
                except Exception as e:
                    logger.error(f"Error getting personalized predictions: {e}")

            # Format for Gradio Dataframe
            char_preds = [[char] for char in predictions["next_chars"]]
            completion_preds = [[word] for word in predictions["word_completions"]]
            word_preds = [[word] for word in predictions["next_words"]]

            # Ensure we have exactly 5 rows for each prediction type
            while len(char_preds) < 5:
                char_preds.append([""])
            while len(completion_preds) < 5:
                completion_preds.append([""])
            while len(word_preds) < 5:
                word_preds.append([""])

            # Update status to show if personalized predictions are being used
            if self.continuous_learning_manager and self.current_person_id:
                status = f"PPM Model Status: Ready (Personalized for {self.current_person_id})"
            else:
                status = "PPM Model Status: Ready"

            return char_preds, completion_preds, word_preds, status

        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            status = f"PPM Model Status: Error ({str(e)[:50]}...)"
            return (
                [[""] for _ in range(5)],
                [[""] for _ in range(5)],
                [[""] for _ in range(5)],
                status,
            )

    def _on_char_click(
        self, text: str, char_df: List[List[str]], idx: int
    ) -> Tuple[str, List[List[str]], List[List[str]], List[List[str]], str]:
        """Handle clicking on a next character prediction.

        Args:
            text: Current text in the input field
            char_df: Next characters dataframe
            idx: Index of the selected character

        Returns:
            Updated text and predictions
        """
        # Get the selected character
        selected_char = ""
        try:
            # Check if DataFrame is not empty and has values
            if isinstance(char_df, list) and idx < len(char_df) and char_df[idx]:
                selected_char = char_df[idx][0]
                if pd.isna(selected_char):  # Check for NaN values
                    selected_char = ""
        except Exception as e:
            logger.error(f"Error extracting character: {e}")

        if not selected_char:
            return text, *self._on_text_change(text)

        # Append the character to the text
        new_text = text + selected_char

        # Get updated predictions
        char_preds, completion_preds, word_preds, status = self._on_text_change(
            new_text
        )

        return new_text, char_preds, completion_preds, word_preds, status

    def _on_completion_click(
        self, text: str, completion_df: List[List[str]], idx: int
    ) -> Tuple[str, List[List[str]], List[List[str]], List[List[str]], str]:
        """Handle clicking on a word completion prediction.

        Args:
            text: Current text in the input field
            completion_df: Word completions dataframe
            idx: Index of the selected completion

        Returns:
            Updated text and predictions
        """
        # Get the selected completion
        selected_completion = ""
        try:
            # Check if DataFrame is not empty and has values
            if (
                isinstance(completion_df, list)
                and idx < len(completion_df)
                and completion_df[idx]
            ):
                selected_completion = completion_df[idx][0]
                if pd.isna(selected_completion):  # Check for NaN values
                    selected_completion = ""
        except Exception as e:
            logger.error(f"Error extracting completion: {e}")

        if not selected_completion:
            return text, *self._on_text_change(text)

        # Replace the current partial word with the complete word
        words = text.split()
        if words:
            # Get the current partial word
            current_word = words[-1]
            # Replace it with the complete word
            words[-1] = selected_completion
            # Add a space after the word
            new_text = " ".join(words) + " "
        else:
            # If no text yet, just use the selected word
            new_text = selected_completion + " "

        # Get updated predictions
        char_preds, completion_preds, word_preds, status = self._on_text_change(
            new_text
        )

        return new_text, char_preds, completion_preds, word_preds, status

    def _on_word_click(
        self, text: str, word_df: List[List[str]], idx: int
    ) -> Tuple[str, List[List[str]], List[List[str]], List[List[str]], str]:
        """Handle clicking on a next word prediction.

        Args:
            text: Current text in the input field
            word_df: Next words dataframe
            idx: Index of the selected word

        Returns:
            Updated text and predictions
        """
        # Get the selected word
        selected_word = ""
        try:
            # Check if DataFrame is not empty and has values
            if isinstance(word_df, list) and idx < len(word_df) and word_df[idx]:
                selected_word = word_df[idx][0]
                if pd.isna(selected_word):  # Check for NaN values
                    selected_word = ""
        except Exception as e:
            logger.error(f"Error extracting word: {e}")

        if not selected_word:
            return text, *self._on_text_change(text)

        # Append the word to the text
        if text and not text.endswith(" "):
            new_text = text + " " + selected_word + " "
        else:
            new_text = text + selected_word + " "

        # Get updated predictions
        char_preds, completion_preds, word_preds, status = self._on_text_change(
            new_text
        )

        return new_text, char_preds, completion_preds, word_preds, status
