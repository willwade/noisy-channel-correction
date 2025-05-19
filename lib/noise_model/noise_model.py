"""
Noise model for AAC data augmentation.

This module provides a framework for generating noisy data by simulating
various types of input errors commonly seen in AAC (Augmentative and Alternative
Communication) systems. It supports multiple error models and can be configured
for different error rates and patterns.

The main components are:
1. NoiseModel: Base class for all noise models
2. KeyboardNoiseModel: Simulates typing errors based on keyboard layouts
3. DwellNoiseModel: Simulates dwell-time errors (missed or doubled letters)
4. TranspositionNoiseModel: Simulates transposition errors (swapped characters)
5. CompositeNoiseModel: Combines multiple noise models
"""

import random
import json
from typing import List, Dict, Tuple, Any
import os
import sys
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the keyboard error model
from lib.noise_model.keyboard_error_model import KeyboardErrorModel

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NoiseModel:
    """
    Base class for all noise models.

    This abstract class defines the interface for noise models that can
    introduce errors into clean text data.
    """

    def __init__(self, error_rate: float = 0.1):
        """
        Initialize a noise model.

        Args:
            error_rate: The overall probability of introducing an error
        """
        self.error_rate = error_rate

    def apply(self, text: str) -> str:
        """
        Apply noise to the input text.

        Args:
            text: The clean input text

        Returns:
            The noisy output text
        """
        raise NotImplementedError("Subclasses must implement this method")

    def generate_noisy_pairs(
        self, texts: List[str], num_variants: int = 1
    ) -> List[Tuple[str, str]]:
        """
        Generate pairs of (clean, noisy) text.

        Args:
            texts: List of clean input texts
            num_variants: Number of noisy variants to generate for each input

        Returns:
            List of (clean, noisy) text pairs
        """
        pairs = []
        for text in texts:
            for _ in range(num_variants):
                noisy = self.apply(text)
                pairs.append((text, noisy))
        return pairs


class KeyboardNoiseModel(NoiseModel):
    """
    Noise model that simulates typing errors based on keyboard layouts.

    This model uses the KeyboardErrorModel to introduce realistic typing errors
    based on the physical properties of keyboard layouts.
    """

    def __init__(
        self,
        layout_name: str = "en",
        error_rates: Dict[str, float] = None,
        rows: int = 3,
        cols: int = 10,
        input_method: str = "direct",
    ):
        """
        Initialize a keyboard noise model.

        Args:
            layout_name: Language code for the keyboard layout (e.g., 'en', 'fr', 'de')
            error_rates: Dictionary of error rates for different error types
            rows: Number of rows in the keyboard grid
            cols: Number of columns in the keyboard grid
            input_method: Input method ('direct', 'scanning', 'eyegaze')
        """
        # Calculate overall error rate from individual error rates
        overall_error_rate = 0.0
        if error_rates:
            overall_error_rate = sum(error_rates.values())

        super().__init__(error_rate=overall_error_rate)

        # Initialize the keyboard error model
        self.keyboard_model = KeyboardErrorModel(
            layout_name=layout_name,
            error_rates=error_rates,
            rows=rows,
            cols=cols,
            input_method=input_method,
        )

    def apply(self, text: str) -> str:
        """
        Apply keyboard-based noise to the input text.

        Args:
            text: The clean input text

        Returns:
            The noisy output text with simulated keyboard errors
        """
        return self.keyboard_model.apply_errors(text)


class DwellNoiseModel(NoiseModel):
    """
    Noise model that simulates dwell-time errors.

    This model simulates errors that occur due to dwell-time issues in AAC systems,
    such as missing letters (deletions) or doubled letters (insertions).
    """

    def __init__(self, deletion_rate: float = 0.03, insertion_rate: float = 0.02):
        """
        Initialize a dwell noise model.

        Args:
            deletion_rate: Probability of deleting a character
            insertion_rate: Probability of inserting a duplicate character
        """
        overall_error_rate = deletion_rate + insertion_rate
        super().__init__(error_rate=overall_error_rate)

        self.deletion_rate = deletion_rate
        self.insertion_rate = insertion_rate

    def apply(self, text: str) -> str:
        """
        Apply dwell-time noise to the input text.

        Args:
            text: The clean input text

        Returns:
            The noisy output text with simulated dwell-time errors
        """
        if not text:
            return text

        result = []
        i = 0

        while i < len(text):
            char = text[i]

            # Deletion error: skip the character
            if random.random() < self.deletion_rate:
                i += 1
                continue

            # Add the character
            result.append(char)

            # Insertion error: duplicate the character
            if random.random() < self.insertion_rate:
                result.append(char)

            i += 1

        return "".join(result)


class TranspositionNoiseModel(NoiseModel):
    """
    Noise model that simulates transposition errors.

    This model simulates errors where adjacent characters are swapped,
    which is common in fast typing or when using prediction systems.
    """

    def __init__(self, transposition_rate: float = 0.01):
        """
        Initialize a transposition noise model.

        Args:
            transposition_rate: Probability of swapping adjacent characters
        """
        super().__init__(error_rate=transposition_rate)
        self.transposition_rate = transposition_rate

    def apply(self, text: str) -> str:
        """
        Apply transposition noise to the input text.

        Args:
            text: The clean input text

        Returns:
            The noisy output text with simulated transposition errors
        """
        if not text or len(text) < 2:
            return text

        result = list(text)
        i = 0

        while i < len(result) - 1:
            # Transposition error: swap with the next character
            if random.random() < self.transposition_rate:
                result[i], result[i + 1] = result[i + 1], result[i]
                i += 2  # Skip over both transposed characters
            else:
                i += 1

        return "".join(result)


class CompositeNoiseModel(NoiseModel):
    """
    Composite noise model that combines multiple noise models.

    This model applies multiple noise models in sequence, allowing for
    more complex and realistic error patterns.
    """

    def __init__(self, models: List[NoiseModel]):
        """
        Initialize a composite noise model.

        Args:
            models: List of noise models to apply
        """
        # Calculate overall error rate as the sum of individual model rates
        overall_error_rate = sum(model.error_rate for model in models)
        super().__init__(error_rate=overall_error_rate)

        self.models = models

    def apply(self, text: str) -> str:
        """
        Apply multiple noise models to the input text.

        Args:
            text: The clean input text

        Returns:
            The noisy output text with errors from all models
        """
        result = text
        for model in self.models:
            result = model.apply(result)
        return result


def create_noise_model(config: Dict[str, Any]) -> NoiseModel:
    """
    Create a noise model from a configuration dictionary.

    Args:
        config: Configuration dictionary with model type and parameters

    Returns:
        A configured noise model
    """
    model_type = config.get("type", "keyboard")

    if model_type == "keyboard":
        return KeyboardNoiseModel(
            layout_name=config.get("layout_name", "en"),
            error_rates=config.get("error_rates"),
            rows=config.get("rows", 3),
            cols=config.get("cols", 10),
            input_method=config.get("input_method", "direct"),
        )

    elif model_type == "dwell":
        return DwellNoiseModel(
            deletion_rate=config.get("deletion_rate", 0.03),
            insertion_rate=config.get("insertion_rate", 0.02),
        )

    elif model_type == "transposition":
        return TranspositionNoiseModel(
            transposition_rate=config.get("transposition_rate", 0.01)
        )

    elif model_type == "composite":
        # Recursively create sub-models
        sub_models = [
            create_noise_model(sub_config) for sub_config in config.get("models", [])
        ]
        return CompositeNoiseModel(models=sub_models)

    else:
        raise ValueError(f"Unknown noise model type: {model_type}")


def load_noise_model_from_json(json_path: str) -> NoiseModel:
    """
    Load a noise model configuration from a JSON file.

    Args:
        json_path: Path to the JSON configuration file

    Returns:
        A configured noise model
    """
    try:
        with open(json_path, "r") as f:
            config = json.load(f)
        return create_noise_model(config)

    except Exception as e:
        logger.error(f"Error loading noise model from {json_path}: {e}")
        # Return a default model
        return KeyboardNoiseModel()


def save_noisy_pairs(pairs: List[Tuple[str, str]], output_path: str) -> bool:
    """
    Save pairs of (clean, noisy) text to a JSON file.

    Args:
        pairs: List of (clean, noisy) text pairs
        output_path: Path to the output JSON file

    Returns:
        True if successful, False otherwise
    """
    try:
        # Convert pairs to a list of dictionaries
        data = [{"clean": clean, "noisy": noisy} for clean, noisy in pairs]

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        # Write to JSON file
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(pairs)} noisy pairs to {output_path}")
        return True

    except Exception as e:
        logger.error(f"Error saving noisy pairs to {output_path}: {e}")
        return False
