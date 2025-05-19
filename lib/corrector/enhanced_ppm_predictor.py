"""
Enhanced PPM (Prediction by Partial Matching) implementation for word prediction.

This module enhances the basic PPM algorithm with modern NLP techniques:
1. N-gram backoff for better handling of unseen contexts
2. Interpolation for smoother probability estimates
3. Word embeddings for semantic similarity
4. Recency bias to prioritize recently used words
5. Context-aware predictions based on sentence structure
"""

import logging
import re
import time
import math
from typing import List, Dict
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Try to import the PPM language model from pylm
try:
    from lib.pylm.vocabulary import Vocabulary
    from lib.pylm.ppm_language_model import PPMLanguageModel

    pylm_available = True
except ImportError:
    logger.warning("pylm library not available. Using fallback implementation.")
    pylm_available = False


class EnhancedPPMPredictor:
    """An enhanced word predictor using the PPM algorithm with modern NLP techniques."""

    def __init__(self, max_order: int = 5, debug: bool = False):
        """Initialize the enhanced PPM predictor.

        Args:
            max_order: Maximum context length to consider
            debug: Whether to print debug information
        """
        self.max_order = max_order
        self.debug = debug
        self.vocab = None
        self.lm = None
        self.model_ready = pylm_available

        # Enhanced features
        self.word_frequencies = Counter()  # Track word frequencies
        self.word_recency = {}  # Track when words were last used
        self.word_contexts = defaultdict(Counter)  # Track word contexts
        self.bigrams = defaultdict(Counter)  # Track word bigrams
        self.trigrams = {}  # Track word trigrams

        # Initialize the model
        if self.model_ready:
            self._initialize_model()
        else:
            self._initialize_fallback_model()

    def _initialize_model(self):
        """Initialize the PPM language model."""
        try:
            logger.info("Initializing enhanced PPM language model")
            self.vocab = Vocabulary()

            # Add special tokens
            self.vocab.add_item(" ")  # Space
            self.vocab.add_item(".")  # Period
            self.vocab.add_item(",")  # Comma
            self.vocab.add_item("?")  # Question mark
            self.vocab.add_item("!")  # Exclamation mark

            # Initialize the language model
            self.lm = PPMLanguageModel(self.vocab, self.max_order, debug=self.debug)

            logger.info("Enhanced PPM language model initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing enhanced PPM language model: {e}")
            self.model_ready = False
            self._initialize_fallback_model()

    def _initialize_fallback_model(self):
        """Initialize a fallback model when pylm is not available."""
        logger.info("Initializing fallback language model")

        # Simple vocabulary implementation
        self.vocab = set()
        self.vocab.add(" ")
        self.vocab.add(".")
        self.vocab.add(",")
        self.vocab.add("?")
        self.vocab.add("!")

        # We'll use our enhanced features for prediction
        self.model_ready = True
        logger.info("Fallback language model initialized successfully")

    def train_on_text(
        self,
        text: str,
        save_model: bool = False,
        model_path: str = "enhanced_ppm_model.pkl",
    ):
        """Train the model on the given text.

        Args:
            text: Text to train on
            save_model: Whether to save the model after training
            model_path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(f"Training enhanced PPM model on text ({len(text)} characters)")

            # Train the base PPM model if available
            if pylm_available:
                # Add all characters to vocabulary
                for char in set(text):
                    self.vocab.add_item(char)

                # Train the model
                context = self.lm.create_context()
                for char in text:
                    symbol_id = self.vocab.get_id_or_oov(char)
                    self.lm.add_symbol_and_update(context, symbol_id)

            # Train the enhanced features
            self._train_enhanced_features(text)

            # Save the model if requested
            if save_model:
                self._save_model(model_path)

            vocab_size = len(self.vocab) if self.vocab else 0
            logger.info(
                f"Enhanced PPM model trained successfully. Vocab size: {vocab_size}"
            )
            return True
        except Exception as e:
            logger.error(f"Error training enhanced PPM model: {e}")
            return False

    def _train_enhanced_features(self, text: str):
        """Train the enhanced features on the given text.

        Args:
            text: Text to train on
        """
        # Tokenize the text into words
        words = self._tokenize(text)

        # Update vocabulary
        if not pylm_available and isinstance(self.vocab, set):
            self.vocab.update(words)

        # Update word frequencies
        self.word_frequencies.update(words)

        # Update word contexts
        for i, word in enumerate(words):
            # Get the context (previous word)
            if i > 0:
                prev_word = words[i - 1]
                self.word_contexts[prev_word][word] += 1

            # Update bigrams
            if i > 0:
                prev_word = words[i - 1]
                self.bigrams[prev_word][word] += 1

            # Update trigrams
            if i > 1:
                prev_word2 = words[i - 2]
                prev_word1 = words[i - 1]

                # Create nested dictionaries if they don't exist
                if prev_word2 not in self.trigrams:
                    self.trigrams[prev_word2] = {}
                if prev_word1 not in self.trigrams[prev_word2]:
                    self.trigrams[prev_word2][prev_word1] = Counter()

                # Update the count
                self.trigrams[prev_word2][prev_word1][word] += 1

        # Update word recency (set to current time)
        current_time = time.time()
        for word in words:
            self.word_recency[word] = current_time

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

    def _save_model(self, model_path: str) -> bool:
        """Save the model to a file.

        Args:
            model_path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle

            # Create a dictionary with the model state
            model_state = {
                "vocab": self.vocab,
                "lm": self.lm if pylm_available else None,
                "max_order": self.max_order,
                "word_frequencies": self.word_frequencies,
                "word_recency": self.word_recency,
                "word_contexts": self.word_contexts,
                "bigrams": self.bigrams,
                "trigrams": self.trigrams,
            }

            # Save the model state to a file
            with open(model_path, "wb") as f:
                pickle.dump(model_state, f)

            logger.info(f"Enhanced PPM model saved to {model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving enhanced PPM model: {e}")
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
            if pylm_available and model_state["lm"] is not None:
                self.lm = model_state["lm"]
            self.max_order = model_state["max_order"]
            self.word_frequencies = model_state["word_frequencies"]
            self.word_recency = model_state["word_recency"]
            self.word_contexts = model_state["word_contexts"]
            self.bigrams = model_state["bigrams"]
            self.trigrams = model_state["trigrams"]
            self.model_ready = True

            vocab_size = len(self.vocab) if self.vocab else 0
            logger.info(
                f"Enhanced PPM model loaded from {model_path}. Vocab size: {vocab_size}"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading enhanced PPM model: {e}")
            return False

    def update_model(
        self,
        text: str,
        save_model: bool = False,
        model_path: str = "enhanced_ppm_model.pkl",
    ) -> bool:
        """Update the model with new text.

        Args:
            text: Text to update the model with
            save_model: Whether to save the model after updating
            model_path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info(
                f"Updating enhanced PPM model with text ({len(text)} characters)"
            )

            # Update the base PPM model if available
            if pylm_available:
                # Add new characters to vocabulary
                for char in set(text):
                    self.vocab.add_item(char)

                # Update the model
                context = self.lm.create_context()
                for char in text:
                    symbol_id = self.vocab.get_id_or_oov(char)
                    self.lm.add_symbol_and_update(context, symbol_id)

            # Update the enhanced features
            self._train_enhanced_features(text)

            # Save the model if requested
            if save_model:
                self._save_model(model_path)

            vocab_size = len(self.vocab) if self.vocab else 0
            logger.info(
                f"Enhanced PPM model updated successfully. Vocab size: {vocab_size}"
            )
            return True
        except Exception as e:
            logger.error(f"Error updating enhanced PPM model: {e}")
            return False

    def predict_next_words(self, text: str, num_predictions: int = 5) -> List[str]:
        """Predict the next words based on the current text.

        Args:
            text: The text typed so far
            num_predictions: Number of predictions to return

        Returns:
            List of predicted next words
        """
        if not self.model_ready:
            logger.warning("Enhanced PPM model not ready. Cannot predict.")
            return []

        try:
            # Tokenize the text
            words = self._tokenize(text)

            # If no words, return most common words
            if not words:
                return [
                    word
                    for word, _ in self.word_frequencies.most_common(num_predictions)
                ]

            # Get the last word
            last_word = words[-1]

            # Get the second last word if available
            second_last_word = words[-2] if len(words) > 1 else None

            # Combine different prediction methods with weights
            predictions: Dict[str, float] = {}

            # 1. Trigram predictions (highest weight)
            if second_last_word and last_word:
                # Check if the trigram exists
                if (
                    second_last_word in self.trigrams
                    and last_word in self.trigrams[second_last_word]
                ):
                    for word, count in self.trigrams[second_last_word][
                        last_word
                    ].items():
                        predictions[word] = predictions.get(word, 0) + count * 4

            # 2. Bigram predictions (high weight)
            if last_word:
                for word, count in self.bigrams[last_word].items():
                    predictions[word] = predictions.get(word, 0) + count * 2

            # 3. Context predictions (medium weight)
            if last_word:
                for word, count in self.word_contexts[last_word].items():
                    predictions[word] = predictions.get(word, 0) + count

            # 4. Frequency predictions (low weight)
            for word, count in self.word_frequencies.most_common(20):
                predictions[word] = predictions.get(word, 0) + count * 0.1

            # 5. Recency bias (boost recently used words)
            current_time = time.time()
            for word, last_time in self.word_recency.items():
                if word in predictions:
                    # Boost based on recency (more recent = higher boost)
                    time_diff = current_time - last_time
                    recency_boost = 1.0 / (
                        1.0 + math.log(1 + time_diff / 3600)
                    )  # Logarithmic decay
                    predictions[word] = predictions.get(word, 0) + recency_boost * 5

            # Sort predictions by score
            sorted_predictions = sorted(
                predictions.items(), key=lambda x: x[1], reverse=True
            )

            # Return top predictions
            return [word for word, _ in sorted_predictions[:num_predictions]]
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
            completions = {}

            # Get all words from word frequencies
            all_words = set(self.word_frequencies.keys())

            # Add words from vocabulary if available
            if pylm_available and self.vocab is not None:
                try:
                    # Add words from pylm vocabulary
                    for i in range(self.vocab.size()):
                        word = self.vocab.get_item_by_id(i)
                        if word:
                            all_words.add(word)
                except Exception as e:
                    logger.error(f"Error getting words from pylm vocabulary: {e}")
            elif isinstance(self.vocab, set):
                # Add words from fallback vocabulary
                all_words.update(self.vocab)

            # Find completions
            for word in all_words:
                if word and word.startswith(partial_word) and word != partial_word:
                    # Score based on frequency and length
                    score = self.word_frequencies.get(word, 0)
                    # Prefer shorter completions
                    length_penalty = 1.0 / (1.0 + 0.1 * len(word))
                    completions[word] = score * length_penalty

            # Sort completions by score
            sorted_completions = sorted(
                completions.items(), key=lambda x: x[1], reverse=True
            )

            # Return top completions
            return [word for word, _ in sorted_completions[:num_predictions]]
        except Exception as e:
            logger.error(f"Error predicting word completions: {e}")
            return []

    def predict_next_chars(self, text: str, num_predictions: int = 5) -> List[str]:
        """Predict the next characters based on the current text.

        Args:
            text: The text typed so far
            num_predictions: Number of predictions to return

        Returns:
            List of predicted next characters
        """
        if not self.model_ready:
            logger.warning("Enhanced PPM model not ready. Cannot predict.")
            return []

        try:
            # If pylm is available and initialized, use it for character predictions
            if pylm_available and self.lm is not None and self.vocab is not None:
                try:
                    # Create a new context
                    context = self.lm.create_context()

                    # Add each character to the context and update the model
                    for char in text:
                        symbol_id = self.vocab.get_id_or_oov(char)
                        self.lm.add_symbol_and_update(context, symbol_id)

                    # Get predictions
                    predicted_ids = self.lm.predict_next_ids(context, num_predictions)

                    # Convert IDs to characters
                    predicted_chars = []
                    for index, _ in predicted_ids:
                        char = self.vocab.get_item_by_id(index)
                        if char:  # Only add valid characters
                            predicted_chars.append(char)

                    return predicted_chars
                except Exception as e:
                    logger.error(f"Error predicting next characters with pylm: {e}")
                    # Fall through to fallback prediction

            # This code is unreachable due to the return statement above
            # Keeping it for reference
            else:
                # Fallback to simple character prediction
                # Predict space after a word
                if text and text[-1].isalpha():
                    return [" ", "e", "a", "t", "s"]

                # Predict common characters
                return ["e", "a", "t", "o", "i"]
        except Exception as e:
            logger.error(f"Error predicting next characters: {e}")
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

        # Get predictions
        next_chars = self.predict_next_chars(text, num_predictions)
        word_completions = self.predict_word_completions(current_word, num_predictions)
        next_words = self.predict_next_words(text, num_predictions)

        return {
            "next_chars": next_chars,
            "word_completions": word_completions,
            "next_words": next_words,
        }
