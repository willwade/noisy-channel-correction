"""
Common configuration for the AAC Noisy Input Correction Engine.

This module provides common configuration settings for the correction engine,
including default paths for model files, dataset locations, and other settings.
"""

import os

# Base directories
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODELS_DIR = os.path.join(ROOT_DIR, "models")

# Model paths
DEFAULT_PPM_MODEL_PATH = os.path.join(MODELS_DIR, "ppm_model_en_gb.pkl")
DEFAULT_WORD_NGRAM_MODEL_PATH = os.path.join(MODELS_DIR, "word_ngram_model_en_gb.pkl")
DEFAULT_CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix_en_gb.json")
DEFAULT_KEYBOARD_CONFUSION_MATRIX_PATH = os.path.join(
    MODELS_DIR, "keyboard_confusion_matrices_en_gb.json"
)

# Lexicon paths
DEFAULT_LEXICON_PATH = os.path.join(DATA_DIR, "enhanced_lexicon_en_gb.txt")
DEFAULT_COMPREHENSIVE_LEXICON_PATH = os.path.join(
    DATA_DIR, "comprehensive_lexicon_en_gb.txt"
)

# Dataset paths
DEFAULT_ENGLISH_CONVERSATIONS_PATH = os.path.join(
    DATA_DIR, "english_conversations.json"
)
DEFAULT_NOISY_PAIRS_PATH = os.path.join(DATA_DIR, "noisy_pairs.json")

# Evaluation settings
DEFAULT_NOISE_TYPE = "qwerty"
DEFAULT_NOISE_LEVEL = "moderate"
DEFAULT_MAX_CANDIDATES = 5
DEFAULT_MAX_EDIT_DISTANCE = 2
DEFAULT_CONTEXT_WINDOW_SIZE = 3
DEFAULT_CONTEXT_WEIGHT = 0.7

# Keyboard layouts
DEFAULT_KEYBOARD_LAYOUT = "qwerty"
SUPPORTED_KEYBOARD_LAYOUTS = ["qwerty", "abc", "frequency"]

# Noise levels
NOISE_LEVELS = ["minimal", "light", "moderate", "severe"]

# Noise types
NOISE_TYPES = ["qwerty", "abc", "frequency"]

# Hugging Face dataset
DEFAULT_DATASET_NAME = "willwade/AACConversations"
DEFAULT_DATASET_SPLIT = "train"


# Path utility functions
def get_path(relative_path):
    """
    Get the absolute path for a path relative to the project root.

    Args:
        relative_path: Path relative to the project root

    Returns:
        Absolute path
    """
    return os.path.join(ROOT_DIR, relative_path)


def get_data_path(relative_path):
    """
    Get the absolute path for a path relative to the data directory.

    Args:
        relative_path: Path relative to the data directory

    Returns:
        Absolute path
    """
    return os.path.join(DATA_DIR, relative_path)


def get_models_path(relative_path):
    """
    Get the absolute path for a path relative to the models directory.

    Args:
        relative_path: Path relative to the models directory

    Returns:
        Absolute path
    """
    return os.path.join(MODELS_DIR, relative_path)


def resolve_path(path):
    """
    Resolve a path that might be relative to different base directories.

    This function handles various path formats:
    - Absolute paths are returned as-is
    - Paths starting with 'data/' are resolved relative to the data directory
    - Paths starting with 'models/' are resolved relative to the models directory
    - Other relative paths are resolved relative to the project root

    Args:
        path: Path to resolve

    Returns:
        Resolved absolute path
    """
    if os.path.isabs(path):
        return path

    if path.startswith("data/"):
        return get_data_path(path[5:])  # Remove 'data/' prefix

    if path.startswith("models/"):
        return get_models_path(path[7:])  # Remove 'models/' prefix

    # Default to project root
    return get_path(path)
