"""
Utility functions for the AAC Noisy Input Correction Engine demo.

This module provides utility functions for accessing and processing the
AACConversations dataset from Hugging Face, as well as other helper functions
for the CLI demo and evaluation.
"""

import os
import sys
import logging
from typing import List, Dict, Tuple, Any, Optional
import json
import random
import re

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Import the common configuration
from lib.config import (
    DEFAULT_PPM_MODEL_PATH,
    DEFAULT_WORD_NGRAM_MODEL_PATH,
    DEFAULT_CONFUSION_MATRIX_PATH,
    DEFAULT_LEXICON_PATH,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_MAX_EDIT_DISTANCE,
    DEFAULT_NOISE_TYPE,
    DEFAULT_NOISE_LEVEL,
    NOISE_TYPES,
    NOISE_LEVELS,
    resolve_path,
)

# Import the corrector
from lib.corrector.corrector import NoisyChannelCorrector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Get Hugging Face token from environment variable
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACE_TOKEN")

# Try to import the datasets library
try:
    from datasets import load_dataset

    DATASETS_AVAILABLE = True
except ImportError:
    logger.warning("datasets library not available. Install with: pip install datasets")
    DATASETS_AVAILABLE = False


# Noise types and levels are now imported from lib.config


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison by removing punctuation and lowercasing.

    Args:
        text: The text to normalize

    Returns:
        Normalized text
    """
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = " ".join(text.split())

    return text


def texts_essentially_equal(text1: str, text2: str) -> bool:
    """
    Check if two texts are essentially equal after normalization.

    Args:
        text1: First text
        text2: Second text

    Returns:
        True if texts are essentially equal, False otherwise
    """
    return normalize_text(text1) == normalize_text(text2)


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


def get_noisy_utterance(
    example: Dict[str, Any], noise_type: str = "qwerty", noise_level: str = "moderate"
) -> str:
    """
    Get the noisy utterance from an example based on noise type and level.
    If the noisy utterance is essentially the same as the intended utterance,
    generate a noisy version using the noise simulator.

    Args:
        example: Dataset example
        noise_type: Type of noise ('qwerty', 'abc', or 'frequency')
        noise_level: Level of noise ('minimal', 'light', 'moderate', or 'severe')

    Returns:
        The noisy utterance
    """
    # Validate noise type and level
    if noise_type not in NOISE_TYPES:
        logger.warning(f"Invalid noise type: {noise_type}. Using 'qwerty'.")
        noise_type = "qwerty"

    if noise_level not in NOISE_LEVELS:
        logger.warning(f"Invalid noise level: {noise_level}. Using 'moderate'.")
        noise_level = "moderate"

    # Get the noisy utterance
    key = f"noisy_{noise_type}_{noise_level}"
    noisy = example.get(key, example.get("utterance", ""))

    # Get the intended utterance
    intended = example.get("utterance_intended", example.get("utterance", ""))
    minimally_corrected = example.get("minimally_corrected", intended)

    # Check if the noisy utterance is essentially the same as the intended utterance
    # or minimally corrected utterance (after normalization)
    if texts_essentially_equal(noisy, intended) or texts_essentially_equal(
        noisy, minimally_corrected
    ):
        # Try to import the noise simulator utilities
        try:
            from scripts.evaluation.noise_simulator_utils import generate_noisy_text

            # Generate a noisy version of the intended utterance
            logger.info(
                f"Generating noisy version of '{intended}' with {noise_type} {noise_level} noise"
            )
            noisy = generate_noisy_text(intended, noise_type, noise_level)
            logger.info(f"Generated noisy text: '{noisy}'")
        except ImportError:
            logger.warning(
                "Could not import noise simulator utilities. Using original text."
            )

    return noisy


def get_random_examples(
    dataset: Any, num_examples: int = 10, seed: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Get random examples from the dataset.

    Args:
        dataset: The dataset to sample from
        num_examples: Number of examples to sample
        seed: Random seed for reproducibility

    Returns:
        List of sampled examples
    """
    if dataset is None:
        logger.error("Cannot get random examples: dataset is None")
        return []

    # Set random seed if provided
    if seed is not None:
        random.seed(seed)

    # Sample random examples
    indices = random.sample(range(len(dataset)), min(num_examples, len(dataset)))
    return [dataset[i] for i in indices]


def group_by_conversation(
    dataset: Any, max_conversations: Optional[int] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Group dataset examples by conversation ID.

    Args:
        dataset: The dataset to process
        max_conversations: Maximum number of conversations to include

    Returns:
        Dictionary mapping conversation IDs to lists of turns
    """
    if dataset is None:
        logger.error("Cannot group by conversation: dataset is None")
        return {}

    # Group examples by conversation_id
    conversations = {}

    for example in dataset:
        conversation_id = str(example.get("conversation_id", "unknown"))
        if conversation_id not in conversations:
            conversations[conversation_id] = []
        conversations[conversation_id].append(dict(example))

    # Sort each conversation by turn_number
    for conversation_id, turns in conversations.items():
        conversations[conversation_id] = sorted(
            turns, key=lambda x: x.get("turn_number", 0)
        )

    # Limit the number of conversations if specified
    if max_conversations is not None and max_conversations < len(conversations):
        conversation_ids = list(conversations.keys())[:max_conversations]
        conversations = {cid: conversations[cid] for cid in conversation_ids}

    return conversations


def prepare_conversation_for_evaluation(
    conversation: List[Dict[str, Any]],
    noise_type: str = "qwerty",
    noise_level: str = "moderate",
) -> List[Dict[str, Any]]:
    """
    Prepare a conversation for evaluation by extracting relevant fields.

    Args:
        conversation: List of conversation turns
        noise_type: The noise type to use
        noise_level: The noise level to use

    Returns:
        List of prepared conversation turns
    """
    prepared_conversation = []

    for turn in conversation:
        # Get the intended, minimally corrected, fully corrected, and noisy utterances
        intended = turn.get("utterance_intended", turn.get("utterance", ""))
        minimally_corrected = turn.get("minimally_corrected", turn.get("utterance", ""))
        fully_corrected = turn.get("fully_corrected", turn.get("utterance", ""))
        noisy = get_noisy_utterance(turn, noise_type, noise_level)

        # Prepare the turn
        prepared_turn = {
            "conversation_id": str(turn.get("conversation_id", "unknown")),
            "turn_number": turn.get("turn_number", 0),
            "speaker": turn.get("speaker", "Unknown"),
            "utterance_intended": intended,
            "minimally_corrected": minimally_corrected,
            "fully_corrected": fully_corrected,
            "noisy_utterance": noisy,
            "language_code": turn.get("language_code", "en"),
            "context_speakers": turn.get("context_speakers", []),
            "context_utterances": turn.get("context_utterances", []),
        }

        prepared_conversation.append(prepared_turn)

    return prepared_conversation


def load_corrector(
    ppm_model_path: Optional[str] = DEFAULT_PPM_MODEL_PATH,
    confusion_matrix_path: Optional[str] = DEFAULT_CONFUSION_MATRIX_PATH,
    word_ngram_model_path: Optional[str] = DEFAULT_WORD_NGRAM_MODEL_PATH,
    lexicon_path: Optional[str] = DEFAULT_LEXICON_PATH,
    max_candidates: int = DEFAULT_MAX_CANDIDATES,
) -> NoisyChannelCorrector:
    """
    Load the noisy channel corrector with the specified models.

    Args:
        ppm_model_path: Path to the PPM model file
        confusion_matrix_path: Path to the confusion matrix file
        word_ngram_model_path: Path to the word n-gram model file
        lexicon_path: Path to the lexicon file
        max_candidates: Maximum number of candidates to return

    Returns:
        The loaded corrector
    """
    # Create a corrector
    corrector = NoisyChannelCorrector(max_candidates=max_candidates)

    # Load the PPM model if provided
    if ppm_model_path:
        resolved_path = resolve_path(ppm_model_path)
        logger.info(f"Loading PPM model from {resolved_path}")
        success = corrector.load_ppm_model(resolved_path)
        if not success:
            logger.warning(f"Failed to load PPM model from {resolved_path}")

    # Load the confusion matrix if provided
    if confusion_matrix_path:
        resolved_path = resolve_path(confusion_matrix_path)
        logger.info(f"Loading confusion matrix from {resolved_path}")
        success = corrector.load_confusion_model(resolved_path)
        if not success:
            logger.warning(f"Failed to load confusion matrix from {resolved_path}")

    # Load the word n-gram model if provided
    if word_ngram_model_path:
        resolved_path = resolve_path(word_ngram_model_path)
        logger.info(f"Loading word n-gram model from {resolved_path}")
        success = corrector.load_word_ngram_model(resolved_path)
        if not success:
            logger.warning(f"Failed to load word n-gram model from {resolved_path}")

    # Load the lexicon if provided
    if lexicon_path:
        resolved_path = resolve_path(lexicon_path)
        logger.info(f"Loading lexicon from {resolved_path}")
        success = corrector.load_lexicon_from_file(resolved_path)
        if not success:
            logger.warning(f"Failed to load lexicon from {resolved_path}")

    # Update the models_ready flag
    corrector.models_ready = (
        corrector.ppm_model.model_ready and corrector.confusion_model is not None
    )
    if corrector.models_ready:
        logger.info("Corrector models are ready")
    else:
        logger.warning("Corrector models are not fully initialized")

    return corrector


def save_results(results: List[Dict[str, Any]], output_path: str) -> bool:
    """
    Save correction results to a file.

    Args:
        results: List of correction results
        output_path: Path to save the results (can be relative to project root)

    Returns:
        True if successful, False otherwise
    """
    try:
        # Resolve the path
        resolved_path = resolve_path(output_path)

        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(resolved_path)), exist_ok=True)

        # Save the results
        with open(resolved_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Saved results to {resolved_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving results to {resolved_path}: {e}")
        return False


def format_example_for_display(
    example: Dict[str, Any],
    noise_type: str = DEFAULT_NOISE_TYPE,
    noise_level: str = DEFAULT_NOISE_LEVEL,
    corrections: Optional[List[Tuple[str, float]]] = None,
) -> str:
    """
    Format an example for display in the CLI.

    Args:
        example: Dataset example
        noise_type: Type of noise
        noise_level: Level of noise
        corrections: List of (correction, score) tuples

    Returns:
        Formatted string for display
    """
    # Get the noisy utterance
    noisy = get_noisy_utterance(example, noise_type, noise_level)

    # Format the example
    formatted = f"Conversation ID: {example.get('conversation_id', 'N/A')}\n"
    formatted += f"Turn Number: {example.get('turn_number', 'N/A')}\n"
    formatted += f"Scene: {example.get('scene', 'N/A')}\n"
    formatted += f"Speaker: {example.get('speaker', 'N/A')}\n\n"

    # Add context if available
    if example.get("context_speakers") and example.get("context_utterances"):
        formatted += "Context:\n"
        for speaker, utterance in zip(
            example["context_speakers"], example["context_utterances"]
        ):
            formatted += f"  {speaker}: {utterance}\n"
        formatted += "\n"

    # Add the utterances
    formatted += f"Original (Intended): {example.get('utterance_intended', 'N/A')}\n"
    formatted += (
        f"Original (Minimally Corrected): {example.get('minimally_corrected', 'N/A')}\n"
    )
    formatted += (
        f"Original (Fully Corrected): {example.get('fully_corrected', 'N/A')}\n"
    )
    formatted += f"Noisy ({noise_type}, {noise_level}): {noisy}\n"

    # Add the corrections if provided
    if corrections:
        formatted += "\nCorrections:\n"
        for i, (correction, score) in enumerate(corrections):
            formatted += f"  {i+1}. {correction} (score: {score:.4f})\n"

    return formatted
