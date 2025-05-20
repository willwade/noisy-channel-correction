"""
Correction Engine (Noisy Channel Model) for AAC input correction.

This module implements a noisy channel model for correcting noisy AAC input.
It combines a PPM language model for P(intended) and a confusion matrix for
P(noisy | intended) to rank candidate corrections.

The main components are:
1. NoisyChannelCorrector: Class for correcting noisy input
2. correct: Function to correct a noisy input

This module has been enhanced to support:
1. Context-aware correction, which uses previous words/sentences to improve accuracy
2. Keyboard-specific confusion matrices for different keyboard layouts
"""

import os
import sys
import logging
import math
from typing import List, Tuple, Set, Optional, Union

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the PPM model
from lib.corrector.enhanced_ppm_predictor import EnhancedPPMPredictor

# Import the word n-gram model
from lib.corrector.word_ngram_model import WordNGramModel

# Import the conversation context manager
from lib.corrector.conversation_context import ConversationContext

# Import the confusion matrices
from lib.confusion_matrix.confusion_matrix import ConfusionMatrix, get_error_probability
from lib.confusion_matrix.keyboard_confusion_matrix import (
    KeyboardConfusionMatrix,
    get_keyboard_error_probability,
)

# Import the candidate generator
from lib.candidate_generator.improved_candidate_generator import (
    ImprovedCandidateGenerator,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NoisyChannelCorrector:
    """
    Class for correcting noisy input using a noisy channel model.

    This class combines a PPM language model for P(intended) and a confusion matrix
    for P(noisy | intended) to rank candidate corrections.

    It has been enhanced to support:
    1. Context-aware correction, which uses previous words/sentences to improve accuracy
    2. Keyboard-specific confusion matrices for different keyboard layouts
    """

    def __init__(
        self,
        ppm_model: Optional[EnhancedPPMPredictor] = None,
        confusion_model: Optional[
            Union[ConfusionMatrix, KeyboardConfusionMatrix]
        ] = None,
        word_ngram_model: Optional[WordNGramModel] = None,
        lexicon: Optional[Set[str]] = None,
        max_candidates: int = 10,
        context_window_size: int = 2,
        context_weight: float = 0.5,
        keyboard_layout: str = "qwerty",
        use_conversation_context: bool = True,
        adaptive_context_weighting: bool = True,
    ):
        """
        Initialize a noisy channel corrector.

        Args:
            ppm_model: PPM language model for P(intended)
            confusion_model: Confusion matrix for P(noisy | intended)
            word_ngram_model: Word-level n-gram model for context-aware correction
            lexicon: Set of valid words to use for filtering candidates
            max_candidates: Maximum number of candidates to return
            context_window_size: Number of previous words to use as context
            context_weight: Weight of context-based probability (0-1)
            keyboard_layout: Keyboard layout to use ('qwerty', 'abc', 'frequency')
            use_conversation_context: Whether to use conversation-level context
            adaptive_context_weighting: Whether to dynamically adjust context weight
        """
        # Initialize the PPM model
        self.ppm_model = ppm_model if ppm_model is not None else EnhancedPPMPredictor()

        # Initialize the confusion matrix
        self.confusion_model = confusion_model

        # Initialize the word n-gram model
        self.word_ngram_model = (
            word_ngram_model if word_ngram_model is not None else WordNGramModel()
        )

        # Initialize the lexicon
        self.lexicon = lexicon if lexicon is not None else set()

        # Initialize the candidate generator
        self.candidate_generator = ImprovedCandidateGenerator(
            lexicon=self.lexicon,
            max_candidates=max_candidates,
            max_edits=200,  # Reduced from 20000 to 200
            keyboard_boost=0.5,  # Increased from 0.3 to 0.5
            strict_filtering=True,  # Use strict filtering
            smart_filtering=True,  # Enable smart filtering
            use_frequency_info=True,  # Use word frequency information
        )

        # Maximum number of candidates to return
        self.max_candidates = max_candidates

        # Context parameters
        self.context_window_size = context_window_size
        self.context_weight = context_weight
        self.use_conversation_context = use_conversation_context
        self.adaptive_context_weighting = adaptive_context_weighting

        # Initialize conversation context manager if enabled
        self.conversation_context = (
            ConversationContext(
                max_history=10,
                recency_weight=0.8,
                speaker_specific=True,
                topic_aware=True,
            )
            if use_conversation_context
            else None
        )

        # Keyboard layout
        self.keyboard_layout = keyboard_layout

        # Flag to indicate if we're using keyboard-specific confusion matrices
        self.using_keyboard_matrices = isinstance(
            self.confusion_model, KeyboardConfusionMatrix
        )

        # Check if the models are ready
        self.update_models_ready_status()

        if not self.models_ready:
            logger.warning(
                "Models not fully initialized. Some functionality may be limited."
            )

        if not self.context_aware:
            logger.warning(
                "Word n-gram model not initialized. Context-aware correction limited."
            )

    def update_models_ready_status(self) -> bool:
        """
        Update the models_ready flag based on the current state of the models.

        This method checks if the PPM model and confusion matrix are initialized
        and updates the models_ready and context_aware flags accordingly.

        Returns:
            True if models are ready, False otherwise
        """
        # Check if the PPM model is ready
        ppm_ready = self.ppm_model is not None and self.ppm_model.model_ready

        # Check if the confusion matrix is ready
        confusion_ready = self.confusion_model is not None

        # Update the models_ready flag
        self.models_ready = ppm_ready and confusion_ready

        # Update the context_aware flag
        self.context_aware = (
            self.word_ngram_model is not None and self.word_ngram_model.model_ready
        )

        return self.models_ready

    def load_ppm_model(self, model_path: str) -> bool:
        """
        Load a PPM model from a file.

        Args:
            model_path: Path to the PPM model file

        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.ppm_model.load_model(model_path)
            if success:
                logger.info(f"Loaded PPM model from {model_path}")
            else:
                logger.error(f"Failed to load PPM model from {model_path}")

            # Update the models_ready flag
            self.update_models_ready_status()
            return success
        except Exception as e:
            logger.error(f"Error loading PPM model from {model_path}: {e}")
            return False

    def load_confusion_model(
        self, model_path: str, keyboard_layout: str = "qwerty"
    ) -> bool:
        """
        Load a confusion matrix from a file.

        Args:
            model_path: Path to the confusion matrix file
            keyboard_layout: Keyboard layout to use ('qwerty', 'abc', 'frequency')

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if the file is a keyboard-specific confusion matrix
            if model_path.endswith("keyboard_confusion_matrices.json"):
                # Load as keyboard-specific confusion matrix
                self.confusion_model = KeyboardConfusionMatrix.load(model_path)
                self.using_keyboard_matrices = True
                self.keyboard_layout = keyboard_layout
                logger.info(
                    f"Loaded keyboard-specific confusion matrices from {model_path}"
                )
            else:
                # Load as regular confusion matrix
                self.confusion_model = ConfusionMatrix.load(model_path)
                self.using_keyboard_matrices = False
                logger.info(f"Loaded standard confusion matrix from {model_path}")

            # Update the models_ready flag
            self.update_models_ready_status()
            return True
        except Exception as e:
            logger.error(f"Error loading confusion matrix from {model_path}: {e}")
            return False

    def load_word_ngram_model(self, model_path: str) -> bool:
        """
        Load a word n-gram model from a file.

        Args:
            model_path: Path to the word n-gram model file

        Returns:
            True if successful, False otherwise
        """
        try:
            success = self.word_ngram_model.load(model_path)
            if success:
                logger.info(f"Loaded word n-gram model from {model_path}")
            else:
                logger.error(f"Failed to load word n-gram model from {model_path}")

            # Update the models_ready and context_aware flags
            self.update_models_ready_status()
            return success
        except Exception as e:
            logger.error(f"Error loading word n-gram model from {model_path}: {e}")
            return False

    def load_lexicon_from_file(self, file_path: str) -> bool:
        """
        Load a lexicon from a file.

        Args:
            file_path: Path to the lexicon file (one word per line)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if an enhanced lexicon exists
            enhanced_path = file_path.replace("aac_lexicon", "enhanced_lexicon")
            if os.path.exists(enhanced_path):
                logger.info(f"Found enhanced lexicon at {enhanced_path}")
                file_path = enhanced_path

            # Load the lexicon
            with open(file_path, "r") as f:
                self.lexicon = set(
                    line.strip().split()[0].lower() for line in f if line.strip()
                )

            # Update the candidate generator's lexicon
            self.candidate_generator.lexicon = self.lexicon

            # Check if a word frequencies file exists
            word_freq_path = os.path.join(
                os.path.dirname(file_path), "word_frequencies_en_gb.txt"
            )
            if os.path.exists(word_freq_path):
                logger.info(f"Found word frequencies file at {word_freq_path}")
                self._load_word_frequencies(word_freq_path)

            logger.info(
                f"Loaded lexicon with {len(self.lexicon)} words from {file_path}"
            )
            return True
        except Exception as e:
            logger.error(f"Error loading lexicon from {file_path}: {e}")
            return False

    def _load_word_frequencies(self, file_path: str) -> bool:
        """
        Load word frequencies from a file.

        Args:
            file_path: Path to the word frequencies file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Load the word frequencies into the candidate generator
            if self.candidate_generator.load_word_frequencies(file_path):
                logger.info(f"Loaded word frequencies from {file_path}")
                return True
            else:
                logger.warning(f"Failed to load word frequencies from {file_path}")
                return False
        except Exception as e:
            logger.error(f"Error loading word frequencies: {e}")
            return False

    def correct(
        self,
        noisy_input: str,
        context: Optional[Union[str, List[str]]] = None,
        max_edit_distance: int = 2,
        use_keyboard_adjacency: bool = True,
    ) -> List[Tuple[str, float]]:
        """
        Correct a noisy input using the noisy channel model.

        Args:
            noisy_input: The noisy input text
            context: Optional context for correction (previous words or sentence)
            max_edit_distance: Maximum edit distance to consider
            use_keyboard_adjacency: Whether to use keyboard adjacency

        Returns:
            List of (correction, score) tuples, sorted by score (highest first)
        """
        # Update the models_ready flag to ensure it's current
        self.update_models_ready_status()

        # Check if the models are ready
        if not self.models_ready:
            logger.warning(
                "Models not fully initialized. Using fallback correction. "
                "Make sure both PPM model and confusion matrix are loaded."
            )
            return [(noisy_input, 1.0)]  # Return the input as is

        try:
            # Process context if provided
            processed_context = self._process_context(context)

            # Generate candidates
            candidates = self.candidate_generator.generate_candidates(
                noisy_input, max_edit_distance, use_keyboard_adjacency
            )

            # If no candidates were found, return the input as is
            if not candidates:
                logger.warning(
                    f"No candidates found for '{noisy_input}'. Returning input as is."
                )
                return [(noisy_input, 1.0)]

            # Score candidates using the noisy channel model
            scored_candidates = self._score_candidates(
                noisy_input, candidates, processed_context
            )

            # Return the top N candidates
            return scored_candidates[: self.max_candidates]
        except Exception as e:
            logger.error(f"Error correcting '{noisy_input}': {e}")
            return [(noisy_input, 1.0)]  # Return the input as is in case of error

    def _process_context(self, context: Optional[Union[str, List[str]]]) -> List[str]:
        """
        Process the context into a list of words.

        Args:
            context: Context as a string or list of words

        Returns:
            List of context words
        """
        if context is None:
            # If conversation context is enabled and has content, use it
            if self.use_conversation_context and self.conversation_context:
                return self.conversation_context.get_context_for_correction(
                    max_words=self.context_window_size * 3
                )
            return []

        # If context is a string, split it into words
        if isinstance(context, str):
            # Simple tokenization by splitting on whitespace
            processed_context = [
                word.strip(".,;:!?\"'()[]{}")
                for word in context.lower().split()
                if word.strip(".,;:!?\"'()[]{}")
            ]
        else:
            # If context is already a list, ensure all items are strings
            processed_context = [str(word).lower() for word in context]

        # If conversation context is enabled, enhance with topic keywords
        if self.use_conversation_context and self.conversation_context:
            # Get topic keywords
            topic_keywords = self.conversation_context.get_topic_keywords(
                max_keywords=5
            )

            # Add topic keywords to the context if they're not already there
            for keyword in topic_keywords:
                if keyword not in processed_context:
                    processed_context.append(keyword)

        return processed_context

    def _score_candidates(
        self,
        noisy_input: str,
        candidates: List[Tuple[str, float]],
        context: List[str] = [],
    ) -> List[Tuple[str, float]]:
        """
        Score candidates using the noisy channel model.

        Args:
            noisy_input: The noisy input text
            candidates: List of (candidate, initial_score) tuples
            context: List of context words

        Returns:
            List of (candidate, score) tuples, sorted by score (highest first)
        """
        scored_candidates = []

        for candidate, _ in candidates:
            # Calculate P(intended) using the language models
            p_intended = self._get_p_intended(candidate, context)

            # Calculate P(noisy | intended) using the confusion matrix
            p_noisy_given_intended = self._get_p_noisy_given_intended(
                noisy_input, candidate
            )

            # Calculate the final score using the noisy channel model formula
            # score = log(P(intended)) + log(P(noisy | intended))
            log_p_intended = math.log(p_intended) if p_intended > 0 else -float("inf")
            log_p_noisy_given_intended = (
                math.log(p_noisy_given_intended)
                if p_noisy_given_intended > 0
                else -float("inf")
            )

            # Combine the scores
            score = log_p_intended + log_p_noisy_given_intended

            # For display purposes, we'll use a normalized score between 0 and 1
            # We need to map the log score to a more readable range
            # This avoids the extremely small numbers that are hard to display

            # Improved normalization function:
            # 1. Shift the log score to a more reasonable range (-20 to 0)
            # 2. Apply a sigmoid function to map to (0,1)
            # 3. Scale to ensure better distribution of values

            if score == -float("inf"):
                normalized_score = 0.0
            else:
                # Shift and scale the log score
                # Most log scores will be between -30 and 0
                # Shift by +15 to center around -15
                shifted_score = score + 15

                # Apply sigmoid function: 1 / (1 + e^-x)
                # This maps any real number to (0,1)
                sigmoid = 1.0 / (1.0 + math.exp(-shifted_score))

                # Scale to spread out the values more (0.01 to 0.99)
                # This helps avoid all scores looking like 0.0000
                normalized_score = (sigmoid - 0.01) * 0.98 / 0.98

                # Ensure the score is between 0 and 1
                normalized_score = max(0.0, min(1.0, normalized_score))

            # Add to the list of scored candidates
            scored_candidates.append((candidate, normalized_score))

        # Sort by score (highest first)
        return sorted(scored_candidates, key=lambda x: x[1], reverse=True)

    def _get_p_intended(self, candidate: str, context: List[str] = []) -> float:
        """
        Calculate P(intended) using the language models.

        Args:
            candidate: The candidate correction
            context: List of context words

        Returns:
            The probability P(intended)
        """
        # If no models are ready, use a simple frequency-based approach
        if not self.ppm_model.model_ready and not self.context_aware:
            # Return a small probability for all candidates
            return 0.01

        # Calculate character-level probability using the PPM model
        char_prob = self._language_probability(candidate)

        # Initialize context probability and confidence
        context_prob = 0.0
        context_confidence = 0.0

        # If context is available and the word n-gram model is ready, use it
        if context and self.context_aware:
            # Get probability and confidence from the word n-gram model
            ngram_prob, ngram_confidence = (
                self.word_ngram_model.probability_with_confidence(candidate, context)
            )

            # Calculate context-based probability using the PPM model as well
            ppm_context_prob = self.ppm_model.calculate_word_probability(
                candidate, context
            )

            # Determine confidence for PPM context prediction based on multiple factors
            # 1. Word presence in vocabulary
            vocab_factor = (
                0.7 if candidate.lower() in self.ppm_model.word_frequencies else 0.3
            )

            # 2. Context relevance - how many context words are in the PPM model's vocabulary
            context_words_in_vocab = sum(
                1 for word in context if word.lower() in self.ppm_model.word_frequencies
            )
            context_relevance = min(context_words_in_vocab / max(1, len(context)), 1.0)

            # 3. Word frequency - more frequent words get higher confidence
            word_freq = self.ppm_model.word_frequencies.get(candidate.lower(), 0)
            max_freq = (
                max(self.ppm_model.word_frequencies.values())
                if self.ppm_model.word_frequencies
                else 1
            )
            freq_factor = min(word_freq / max(1, max_freq), 1.0)

            # Combine factors for overall PPM confidence
            ppm_confidence = (
                0.4 * vocab_factor + 0.3 * context_relevance + 0.3 * freq_factor
            )

            # Combine the context probabilities from both models
            # Weight by their respective confidences
            if ngram_confidence + ppm_confidence > 0:
                # Enhanced integration: use a weighted combination of both models
                # with weights determined by their confidence scores
                context_prob = (
                    (ngram_prob * ngram_confidence)
                    + (ppm_context_prob * ppm_confidence)
                ) / (ngram_confidence + ppm_confidence)

                # Calculate a combined confidence score
                # We use a weighted average rather than just the maximum
                context_confidence = (ngram_confidence * 0.6) + (ppm_confidence * 0.4)
            else:
                context_prob = 0.0
                context_confidence = 0.0

            # If using adaptive context weighting, adjust based on multiple factors
            if self.adaptive_context_weighting:
                # 1. Context length factor - longer context is more reliable
                context_length_factor = min(len(context) / 5.0, 1.0)

                # 2. Context coherence - are the context words related?
                # This is approximated by checking if context words appear together in training data
                coherence_score = 0.5  # Default medium coherence
                if len(context) > 1:
                    # Check if adjacent context words appear together in bigrams
                    bigram_matches = 0
                    for i in range(len(context) - 1):
                        if (
                            context[i] in self.ppm_model.bigrams
                            and context[i + 1] in self.ppm_model.bigrams[context[i]]
                        ):
                            bigram_matches += 1

                    coherence_score = min(
                        bigram_matches / max(1, len(context) - 1), 1.0
                    )

                # 3. Candidate quality - is this a common word or rare?
                candidate_quality = min(
                    self.ppm_model.word_frequencies.get(candidate.lower(), 0)
                    / max(1, max(self.ppm_model.word_frequencies.values()) / 100),
                    1.0,
                )

                # Combine all factors for the final weight adjustment
                adjusted_weight = self.context_weight * (
                    0.4 * context_confidence
                    + 0.3 * context_length_factor
                    + 0.2 * coherence_score
                    + 0.1 * candidate_quality
                )

                # Ensure weight is within bounds
                adjusted_weight = max(0.1, min(adjusted_weight, 0.9))

                # Combine the probabilities using the adjusted weight
                return (adjusted_weight * context_prob) + (
                    (1 - adjusted_weight) * char_prob
                )
            else:
                # For non-adaptive weighting, still use both models but with fixed weights
                # This is an improvement over the original which only used the n-gram model
                combined_context_prob = (0.7 * context_prob) + (0.3 * ppm_context_prob)

                # Combine the probabilities using the fixed context weight
                return (self.context_weight * combined_context_prob) + (
                    (1 - self.context_weight) * char_prob
                )

        # If no context or model not ready, use only character-level probability
        return char_prob

    def _language_probability(self, word: str) -> float:
        """
        Calculate the character-level probability of a word.

        Args:
            word: The word to calculate probability for

        Returns:
            Character-level probability of the word
        """
        # If the PPM model is not ready, use a simple frequency-based approach
        if not self.ppm_model.model_ready:
            # Return a small probability for all words
            return 0.01

        # Use the PPM model to get the probability of the word
        # For simplicity, we'll use the word frequency as a proxy for P(intended)

        # Get the word frequency from the PPM model
        freq = self.ppm_model.word_frequencies.get(word.lower(), 0)

        # Normalize by the total number of words
        total_words = sum(self.ppm_model.word_frequencies.values())

        # Calculate the probability (with smoothing to avoid zero probabilities)
        p_intended = (freq + 1) / (total_words + len(self.ppm_model.word_frequencies))

        return p_intended

    def _language_probability_with_context(
        self, word: str, context: List[str]
    ) -> float:
        """
        Calculate the probability of a word given its context.

        Args:
            word: The word to calculate probability for
            context: List of previous words

        Returns:
            Probability of the word given the context
        """
        # If word n-gram model not ready, fall back to character-level probability
        if not self.context_aware:
            return self._language_probability(word)

        # Limit context to the context window size
        context = context[-self.context_window_size :]

        # Calculate the probability using the word n-gram model
        return self.word_ngram_model.probability(word, context)

    def update_conversation_context(
        self, utterance: str, speaker: str = "user", timestamp: Optional[float] = None
    ) -> bool:
        """
        Update the conversation context with a new utterance.

        Args:
            utterance: The utterance text
            speaker: The speaker identifier
            timestamp: Optional timestamp (defaults to current time)

        Returns:
            True if successful, False otherwise
        """
        if not self.use_conversation_context or self.conversation_context is None:
            return False

        try:
            # Add the utterance to the conversation context
            self.conversation_context.add_utterance(utterance, speaker, timestamp)
            return True
        except Exception as e:
            logger.error(f"Error updating conversation context: {e}")
            return False

    def _get_p_noisy_given_intended(self, noisy: str, intended: str) -> float:
        """
        Calculate P(noisy | intended) using the confusion matrix.

        Args:
            noisy: The noisy input text
            intended: The candidate correction

        Returns:
            The probability P(noisy | intended)
        """
        # If the confusion matrix is not available, use a simple edit distance approach
        if self.confusion_model is None:
            # Calculate edit distance
            distance = self.candidate_generator._levenshtein_distance(noisy, intended)

            # Convert distance to a probability (higher for smaller distances)
            max_len = max(len(noisy), len(intended))
            p_noisy_given_intended = 1.0 - (distance / max_len if max_len > 0 else 0)

            return p_noisy_given_intended

        # Check if we're using keyboard-specific confusion matrices
        if self.using_keyboard_matrices:
            # Use the keyboard-specific confusion matrix for the current layout
            return get_keyboard_error_probability(
                noisy, intended, self.confusion_model, self.keyboard_layout
            )
        else:
            # Use the standard confusion matrix
            return get_error_probability(noisy, intended, self.confusion_model)


def correct(
    noisy_input: str,
    context: Optional[Union[str, List[str]]] = None,
    ppm_model_path: Optional[str] = None,
    confusion_matrix_path: Optional[str] = None,
    word_ngram_model_path: Optional[str] = None,
    lexicon_path: Optional[str] = None,
    max_candidates: int = 5,
    context_window_size: int = 2,
    context_weight: float = 0.5,
    keyboard_layout: str = "qwerty",
    use_keyboard_specific_matrices: bool = True,
    use_conversation_context: bool = True,
    adaptive_context_weighting: bool = True,
    speaker: str = "user",
    update_context: bool = True,
) -> List[Tuple[str, float]]:
    """
    Correct a noisy input using the noisy channel model.

    This is a convenience function that creates a NoisyChannelCorrector and uses it
    to correct a noisy input.

    Args:
        noisy_input: The noisy input text
        context: Optional context for correction (previous words or sentence)
        ppm_model_path: Path to the PPM model file
        confusion_matrix_path: Path to the confusion matrix file
        word_ngram_model_path: Path to the word n-gram model file
        lexicon_path: Path to the lexicon file
        max_candidates: Maximum number of candidates to return
        context_window_size: Number of previous words to use as context
        context_weight: Weight of context-based probability (0-1)
        keyboard_layout: Keyboard layout to use for confusion matrix
        use_keyboard_specific_matrices: Whether to use keyboard-specific matrices
        use_conversation_context: Whether to use conversation-level context
        adaptive_context_weighting: Whether to dynamically adjust context weight
        speaker: Speaker identifier for conversation context
        update_context: Whether to update conversation context with correction

    Returns:
        List of (correction, score) tuples, sorted by score (highest first)
    """
    # Create a corrector
    corrector = NoisyChannelCorrector(
        max_candidates=max_candidates,
        context_window_size=context_window_size,
        context_weight=context_weight,
        keyboard_layout=keyboard_layout,
        use_conversation_context=use_conversation_context,
        adaptive_context_weighting=adaptive_context_weighting,
    )

    # Load the PPM model if provided
    if ppm_model_path:
        corrector.load_ppm_model(ppm_model_path)

    # Load the confusion matrix if provided
    if confusion_matrix_path:
        # If using keyboard-specific matrices, check if the file exists
        if use_keyboard_specific_matrices:
            keyboard_matrix_path = os.path.join(
                os.path.dirname(confusion_matrix_path),
                "keyboard_confusion_matrices.json",
            )
            if os.path.exists(keyboard_matrix_path):
                corrector.load_confusion_model(keyboard_matrix_path, keyboard_layout)
            else:
                # Fall back to standard confusion matrix
                corrector.load_confusion_model(confusion_matrix_path, keyboard_layout)
        else:
            # Use standard confusion matrix
            corrector.load_confusion_model(confusion_matrix_path, keyboard_layout)

    # Load the word n-gram model if provided
    if word_ngram_model_path:
        corrector.load_word_ngram_model(word_ngram_model_path)

    # Load the lexicon if provided
    if lexicon_path:
        corrector.load_lexicon_from_file(lexicon_path)

    # Correct the input
    corrections = corrector.correct(noisy_input, context=context)

    # Update conversation context if enabled
    if update_context and use_conversation_context and corrections:
        # Use the top correction for the context update
        top_correction = corrections[0][0]
        corrector.update_conversation_context(top_correction, speaker)

    return corrections
