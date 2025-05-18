"""
Local PPM (Prediction by Partial Matching) implementation for word prediction.

This module uses the pylm library to provide word prediction functionality
without sending data to external servers.
"""

import os
import logging
import re
from typing import List, Dict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the PPM language model from pylm
try:
    from pylm.vocabulary import Vocabulary
    from pylm.ppm_language_model import PPMLanguageModel

    pylm_available = True
except ImportError:
    logger.warning("pylm library not available. Downloading from GitHub...")
    pylm_available = False

    # If pylm is not available, download it from GitHub
    import requests

    def download_file(url, local_path):
        """Download a file from a URL to a local path."""
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise an exception for HTTP errors

            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)

            # Write the content to the local file
            with open(local_path, "w") as f:
                f.write(response.text)

            logger.info(f"Downloaded {url} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            return False

    # Create a pylm directory
    os.makedirs("pylm", exist_ok=True)

    # Download the necessary files
    base_url = "https://raw.githubusercontent.com/willwade/pylm/main/"
    files_to_download = ["vocabulary.py", "ppm_language_model.py"]

    all_downloaded = True
    for file in files_to_download:
        url = base_url + file
        local_path = os.path.join("pylm", file)
        if not download_file(url, local_path):
            all_downloaded = False

    # Create __init__.py file (it doesn't exist in the repo)
    init_content = '''"""
pylm - Python Language Models

A collection of simple adaptive language models that are cheap enough
memory- and processor-wise to train on the fly.
"""

from .vocabulary import Vocabulary
from .ppm_language_model import PPMLanguageModel, Context, Node

__all__ = ['Vocabulary', 'PPMLanguageModel', 'Context', 'Node']
'''

    init_path = os.path.join("pylm", "__init__.py")
    try:
        with open(init_path, "w") as f:
            f.write(init_content)
        logger.info(f"Created {init_path}")
    except Exception as e:
        logger.error(f"Error creating {init_path}: {e}")
        all_downloaded = False

    if all_downloaded:
        logger.info("All pylm files downloaded successfully")
        # Now try to import again
        import sys

        sys.path.append(".")  # Add current directory to path

        try:
            from pylm.vocabulary import Vocabulary
            from pylm.ppm_language_model import PPMLanguageModel

            pylm_available = True
            logger.info("pylm library imported successfully")
        except ImportError as e:
            logger.error(f"Failed to import pylm after downloading: {e}")
            pylm_available = False
    else:
        logger.error("Failed to download all pylm files")


class LocalPPMPredictor:
    """A word predictor using the PPM algorithm locally."""

    def __init__(self, max_order: int = 5, debug: bool = False):
        """Initialize the PPM predictor.

        Args:
            max_order: Maximum context length to consider
            debug: Whether to print debug information
        """
        self.max_order = max_order
        self.debug = debug
        self.vocab = None
        self.lm = None
        self.context = None
        self.model_ready = pylm_available

        if self.model_ready:
            self._initialize_model()

    def _initialize_model(self):
        """Initialize the PPM language model."""
        try:
            logger.info("Initializing PPM language model")
            self.vocab = Vocabulary()

            # Add special tokens
            self.vocab.add_item(" ")  # Space
            self.vocab.add_item(".")  # Period
            self.vocab.add_item(",")  # Comma
            self.vocab.add_item("?")  # Question mark
            self.vocab.add_item("!")  # Exclamation mark

            # Initialize the language model
            self.lm = PPMLanguageModel(self.vocab, self.max_order, debug=self.debug)
            self.context = self.lm.create_context()

            logger.info("PPM language model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing PPM language model: {e}")
            self.model_ready = False

    def train_on_text(
        self, text: str, save_model: bool = False, model_path: str = "ppm_model.pkl"
    ):
        """Train the model on the given text.

        Args:
            text: Text to train on
            save_model: Whether to save the model after training
            model_path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        if not self.model_ready:
            logger.warning("PPM model not ready. Cannot train.")
            return False

        try:
            logger.info(f"Training PPM model on text ({len(text)} characters)")

            # Add all characters to vocabulary
            for char in set(text):
                self.vocab.add_item(char)

            # Train the model
            context = self.lm.create_context()
            for char in text:
                symbol_id = self.vocab.get_id_or_oov(char)
                self.lm.add_symbol_and_update(context, symbol_id)

            logger.info(
                f"PPM model trained successfully. Vocabulary size: {self.vocab.size()}"
            )

            # Save the model if requested
            if save_model:
                self._save_model(model_path)

            return True
        except Exception as e:
            logger.error(f"Error training PPM model: {e}")
            return False

    def _save_model(self, model_path: str) -> bool:
        """Save the model to a file.

        Args:
            model_path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        if not self.model_ready:
            logger.warning("PPM model not ready. Cannot save.")
            return False

        try:
            import pickle

            # Create a dictionary with the model state
            model_state = {
                "vocab": self.vocab,
                "lm": self.lm,
                "max_order": self.max_order,
            }

            # Save the model state to a file
            with open(model_path, "wb") as f:
                pickle.dump(model_state, f)

            logger.info(f"PPM model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving PPM model: {e}")
            return False

    def load_model(self, model_path: str) -> bool:
        """Load the model from a file.

        Args:
            model_path: Path to load the model from

        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle

            # Load the model state from a file
            with open(model_path, "rb") as f:
                model_state = pickle.load(f)

            # Restore the model state
            self.vocab = model_state["vocab"]
            self.lm = model_state["lm"]
            self.max_order = model_state["max_order"]
            self.model_ready = True

            logger.info(
                f"PPM model loaded from {model_path}. Vocabulary size: {self.vocab.size()}"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading PPM model: {e}")
            return False

    def update_model(
        self, text: str, save_model: bool = False, model_path: str = "ppm_model.pkl"
    ) -> bool:
        """Update the model with new text.

        Args:
            text: Text to update the model with
            save_model: Whether to save the model after updating
            model_path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        if not self.model_ready:
            logger.warning("PPM model not ready. Cannot update.")
            return False

        try:
            logger.info(f"Updating PPM model with text ({len(text)} characters)")

            # Add new characters to vocabulary
            for char in set(text):
                self.vocab.add_item(char)

            # Update the model
            context = self.lm.create_context()
            for char in text:
                symbol_id = self.vocab.get_id_or_oov(char)
                self.lm.add_symbol_and_update(context, symbol_id)

            logger.info(
                f"PPM model updated successfully. Vocabulary size: {self.vocab.size()}"
            )

            # Save the model if requested
            if save_model:
                self._save_model(model_path)

            return True
        except Exception as e:
            logger.error(f"Error updating PPM model: {e}")
            return False

    def train_on_words(
        self, text: str, save_model: bool = False, model_path: str = "ppm_model.pkl"
    ):
        """Train the model on words from the given text.

        Args:
            text: Text to extract words from and train on
            save_model: Whether to save the model after training
            model_path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        if not self.model_ready:
            logger.warning("PPM model not ready. Cannot train.")
            return False

        try:
            logger.info(
                f"Training PPM model on words from text ({len(text)} characters)"
            )

            # Tokenize the text into words
            words = self._tokenize(text)

            # Add all words to vocabulary
            for word in set(words):
                self.vocab.add_item(word)

            # Train the model
            context = self.lm.create_context()
            for word in words:
                word_id = self.vocab.get_id_or_oov(word)
                self.lm.add_symbol_and_update(context, word_id)

            logger.info(
                f"PPM model trained successfully on {len(words)} words. Vocabulary size: {self.vocab.size()}"
            )

            # Save the model if requested
            if save_model:
                self._save_model(model_path)

            return True
        except Exception as e:
            logger.error(f"Error training PPM model on words: {e}")
            return False

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of words
        """
        # Replace punctuation with spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Condense all whitespace to a single space
        text = re.sub(r"\s+", " ", text)

        # Trim leading and trailing whitespaces
        text = text.strip()

        # Split the text into tokens
        tokens = text.split()

        return tokens

    def predict_next_chars(self, text: str, num_predictions: int = 5) -> List[str]:
        """Predict the next characters based on the current text.

        Args:
            text: The text typed so far
            num_predictions: Number of predictions to return

        Returns:
            List of predicted next characters
        """
        if not self.model_ready:
            logger.warning("PPM model not ready. Cannot predict.")
            return []

        try:
            # Create a new context
            context = self.lm.create_context()

            # Add each character to the context and update the model
            for char in text:
                char_id = self.vocab.get_id_or_oov(char)
                # Use add_symbol_and_update instead of add_symbol_to_context
                self.lm.add_symbol_and_update(context, char_id)

            # Get predictions
            predicted_ids = self.lm.predict_next_ids(context, num_predictions)

            # Convert IDs to characters
            predicted_chars = []
            for index, prob in predicted_ids:
                char = self.vocab.get_item_by_id(index)
                if char:  # Only add valid characters
                    predicted_chars.append(char)

            # Add some fallback predictions if we don't have enough
            if len(predicted_chars) < num_predictions:
                common_chars = [" ", "e", "a", "t", "o", "i", "n", "s", "h", "r"]
                for char in common_chars:
                    if (
                        char not in predicted_chars
                        and len(predicted_chars) < num_predictions
                    ):
                        predicted_chars.append(char)

            return predicted_chars
        except Exception as e:
            logger.error(f"Error predicting next characters: {e}")
            return []

    def predict_next_words(self, text: str, num_predictions: int = 5) -> List[str]:
        """Predict the next words based on the current text.

        Args:
            text: The text typed so far
            num_predictions: Number of predictions to return

        Returns:
            List of predicted next words
        """
        if not self.model_ready:
            logger.warning("PPM model not ready. Cannot predict.")
            return []

        try:
            # Tokenize the text
            words = self._tokenize(text)

            # Create a new context
            context = self.lm.create_context()

            # Add each word to the context and update the model
            for word in words:
                word_id = self.vocab.get_id_or_oov(word)
                # Use add_symbol_and_update instead of add_symbol_to_context
                self.lm.add_symbol_and_update(context, word_id)

            # Get predictions
            predicted_ids = self.lm.predict_next_ids(context, num_predictions)

            # Convert IDs to words
            predicted_words = []
            for index, prob in predicted_ids:
                word = self.vocab.get_item_by_id(index)
                if word:  # Only add valid words
                    predicted_words.append(word)

            # Add some fallback predictions if we don't have enough
            if len(predicted_words) < num_predictions:
                common_words = [
                    "the",
                    "and",
                    "to",
                    "of",
                    "a",
                    "in",
                    "is",
                    "it",
                    "you",
                    "that",
                ]
                for word in common_words:
                    if (
                        word not in predicted_words
                        and len(predicted_words) < num_predictions
                    ):
                        predicted_words.append(word)

            return predicted_words
        except Exception as e:
            logger.error(f"Error predicting next words: {e}")
            return []

    def predict_word_completions(
        self, partial_word: str, num_predictions: int = 5
    ) -> List[str]:
        """Predict word completions based on the partial word.

        Args:
            partial_word: The partial word typed so far
            num_predictions: Number of predictions to return

        Returns:
            List of predicted word completions
        """
        if not self.model_ready or not partial_word:
            return []

        try:
            # Find all words in vocabulary that start with the partial word
            completions = []

            # Get all items from vocabulary
            all_items = self._get_all_vocabulary_items()

            # Find completions
            for word in all_items:
                if word and word.startswith(partial_word) and word != partial_word:
                    completions.append(word)

            # Sort by length (shorter completions first)
            completions.sort(key=len)

            return completions[:num_predictions]
        except Exception as e:
            logger.error(f"Error predicting word completions: {e}")
            return []

    def _get_all_vocabulary_items(self) -> List[str]:
        """Get all items from the vocabulary.

        Returns:
            List of all items in the vocabulary
        """
        if not self.model_ready:
            return []

        try:
            # Check if the vocabulary has the get_all_items method
            if hasattr(self.vocab, "get_all_items"):
                return self.vocab.get_all_items()
            else:
                # Fallback to iterating through the vocabulary
                items = []
                for i in range(self.vocab.size()):
                    item = self.vocab.get_item_by_id(i)
                    if item:  # Only add valid items
                        items.append(item)
                return items
        except Exception as e:
            logger.error(f"Error getting vocabulary items: {e}")
            return []

    def get_all_predictions(
        self, text: str, num_predictions: int = 5
    ) -> Dict[str, List[str]]:
        """Get all types of predictions for the current text.

        Args:
            text: The text typed so far
            num_predictions: Number of predictions to return

        Returns:
            Dictionary with different types of predictions
        """
        # Get the current word (characters after the last space)
        words = text.split()
        current_word = words[-1] if words else ""

        if not self.model_ready:
            # Return fallback predictions if the model is not ready
            return self._get_fallback_predictions(text, current_word, num_predictions)

        # Get predictions
        next_chars = self.predict_next_chars(text, num_predictions)
        word_completions = self.predict_word_completions(current_word, num_predictions)
        next_words = self.predict_next_words(text, num_predictions)

        # If any of the predictions are empty, use fallback predictions
        if not next_chars or not word_completions or not next_words:
            fallback_predictions = self._get_fallback_predictions(
                text, current_word, num_predictions
            )

            if not next_chars:
                next_chars = fallback_predictions["next_chars"]

            if not word_completions:
                word_completions = fallback_predictions["word_completions"]

            if not next_words:
                next_words = fallback_predictions["next_words"]

        return {
            "next_chars": next_chars,
            "word_completions": word_completions,
            "next_words": next_words,
        }

    def _get_fallback_predictions(
        self, text: str, current_word: str, num_predictions: int = 5
    ) -> Dict[str, List[str]]:
        """Get fallback predictions when the model is not ready.

        Args:
            text: The text typed so far
            current_word: The current word being typed
            num_predictions: Number of predictions to return

        Returns:
            Dictionary with different types of predictions
        """
        # Common characters for next character prediction
        common_chars = [" ", "e", "a", "t", "o", "i", "n", "s", "h", "r"]
        next_chars = common_chars[:num_predictions]

        # Common word completions based on the current word
        word_completions = []
        if current_word:
            common_words = [
                "the",
                "and",
                "to",
                "of",
                "a",
                "in",
                "is",
                "it",
                "you",
                "that",
                "he",
                "was",
                "for",
                "on",
                "are",
                "with",
                "as",
                "his",
                "they",
                "at",
                "be",
                "this",
                "have",
                "from",
                "or",
                "one",
                "had",
                "by",
                "word",
                "but",
                "not",
                "what",
                "all",
                "were",
                "we",
                "when",
                "your",
                "can",
                "said",
                "there",
                "use",
                "an",
                "each",
                "which",
                "she",
                "do",
                "how",
                "their",
                "if",
                "will",
            ]

            for word in common_words:
                if word.startswith(current_word) and word != current_word:
                    word_completions.append(word)
                    if len(word_completions) >= num_predictions:
                        break

        # Common next words
        common_next_words = [
            "the",
            "and",
            "to",
            "of",
            "a",
            "in",
            "is",
            "it",
            "you",
            "that",
            "he",
            "was",
            "for",
            "on",
            "are",
            "with",
            "as",
            "his",
            "they",
            "at",
        ]
        next_words = common_next_words[:num_predictions]

        return {
            "next_chars": next_chars,
            "word_completions": word_completions,
            "next_words": next_words,
        }
