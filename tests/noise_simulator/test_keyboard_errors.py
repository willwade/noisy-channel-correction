#!/usr/bin/env python3
"""
Test script for keyboard error models.

This script demonstrates the functionality of the keyboard error models
by generating noisy versions of sample texts using different keyboard layouts
and input methods.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the noise models
from lib.noise_model.noise_model import (
    KeyboardNoiseModel,
    DwellNoiseModel,
    TranspositionNoiseModel,
    CompositeNoiseModel
)

# Import keyboard layouts
from lib.noise_model.language_keyboards import LANGUAGE_NAMES


def test_keyboard_model(text: str, layout_name: str, input_method: str) -> str:
    """
    Test the keyboard noise model with a specific layout and input method.
    
    Args:
        text: The clean input text
        layout_name: Language code for the keyboard layout
        input_method: Input method ('direct', 'scanning', 'eyegaze')
        
    Returns:
        The noisy output text
    """
    # Create a keyboard noise model
    model = KeyboardNoiseModel(
        layout_name=layout_name,
        error_rates={
            "proximity": 0.05,
            "deletion": 0.03,
            "insertion": 0.02,
            "transposition": 0.01
        },
        input_method=input_method
    )
    
    # Apply the model to the input text
    noisy_text = model.apply(text)
    
    return noisy_text


def test_composite_model(text: str, layout_name: str, input_method: str) -> str:
    """
    Test a composite noise model with a specific layout and input method.
    
    Args:
        text: The clean input text
        layout_name: Language code for the keyboard layout
        input_method: Input method ('direct', 'scanning', 'eyegaze')
        
    Returns:
        The noisy output text
    """
    # Create individual models
    keyboard_model = KeyboardNoiseModel(
        layout_name=layout_name,
        error_rates={
            "proximity": 0.05,
            "deletion": 0.0,
            "insertion": 0.0,
            "transposition": 0.0
        },
        input_method=input_method
    )
    
    dwell_model = DwellNoiseModel(
        deletion_rate=0.03,
        insertion_rate=0.02
    )
    
    transposition_model = TranspositionNoiseModel(
        transposition_rate=0.01
    )
    
    # Create a composite model
    composite_model = CompositeNoiseModel(
        models=[keyboard_model, dwell_model, transposition_model]
    )
    
    # Apply the model to the input text
    noisy_text = composite_model.apply(text)
    
    return noisy_text


def main():
    """Main function."""
    # Sample texts in different languages
    sample_texts = {
        "en": "The quick brown fox jumps over the lazy dog",
        "fr": "Portez ce vieux whisky au juge blond qui fume",
        "de": "Victor jagt zwölf Boxkämpfer quer über den Sylter Deich",
        "es": "El veloz murciélago hindú comía feliz cardillo y kiwi",
        "it": "Pranzo d'acqua fa volti sghembi",
        "nl": "Lynx c.q. vos prikt bh: dag zwemjuf!"
    }
    
    # Input methods
    input_methods = ["direct", "scanning", "eyegaze"]
    
    # Test each language and input method
    for lang_code, text in sample_texts.items():
        print(f"\n=== Testing {LANGUAGE_NAMES.get(lang_code, lang_code)} ===")
        print(f"Original: {text}")
        
        for method in input_methods:
            print(f"\n--- Input Method: {method} ---")
            
            # Test keyboard model
            keyboard_noisy = test_keyboard_model(text, lang_code, method)
            print(f"Keyboard Model: {keyboard_noisy}")
            
            # Test composite model
            composite_noisy = test_composite_model(text, lang_code, method)
            print(f"Composite Model: {composite_noisy}")


if __name__ == "__main__":
    main()
