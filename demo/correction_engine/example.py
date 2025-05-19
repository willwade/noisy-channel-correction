#!/usr/bin/env python3
"""
Example usage of the noisy channel corrector.

This script demonstrates how to use the NoisyChannelCorrector class
to correct noisy input.
"""

import os
import sys
import logging

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the corrector
from lib.corrector.corrector import NoisyChannelCorrector
from module4.ppm.enhanced_ppm_predictor import EnhancedPPMPredictor
from lib.confusion_matrix.confusion_matrix import build_confusion_matrix

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Main function."""
    # Create a PPM model
    logger.info("Creating PPM model...")
    ppm_model = EnhancedPPMPredictor()
    
    # Train the model on a sample text
    sample_text = """
    The quick brown fox jumps over the lazy dog.
    This is a sample text for training the PPM model.
    It contains common words and phrases that might be used in AAC systems.
    Hello, how are you today?
    I am fine, thank you.
    The weather is nice today.
    I would like to go for a walk.
    Please help me with this task.
    I need to send an email.
    Can you pass me the water?
    I am hungry and would like to eat something.
    """
    
    logger.info("Training PPM model...")
    ppm_model.train_on_text(sample_text)
    
    # Create a confusion matrix
    logger.info("Creating confusion matrix...")
    pairs = [
        ("hello", "helo"),  # Deletion
        ("world", "worlld"),  # Insertion
        ("test", "tset"),  # Transposition
        ("cat", "kat"),  # Substitution
        ("dog", "dog"),  # Exact match
        ("the", "teh"),  # Transposition
        ("quick", "quikc"),  # Transposition
        ("brown", "brwon"),  # Transposition
        ("fox", "fxo"),  # Transposition
        ("jumps", "jumsp"),  # Transposition
        ("over", "ovre"),  # Transposition
        ("lazy", "lasy"),  # Substitution
        ("today", "todya"),  # Transposition
        ("weather", "weatehr"),  # Transposition
        ("nice", "niec"),  # Transposition
        ("would", "wuold"),  # Transposition
        ("like", "liek"),  # Transposition
        ("walk", "wlak"),  # Transposition
        ("please", "plase"),  # Deletion
        ("help", "hlep"),  # Transposition
        ("with", "wiht"),  # Transposition
        ("this", "tihs"),  # Transposition
        ("task", "taks"),  # Transposition
        ("need", "nede"),  # Transposition
        ("send", "sedn"),  # Transposition
        ("email", "emial"),  # Transposition
        ("pass", "pss"),  # Deletion
        ("water", "watre"),  # Transposition
        ("hungry", "hungyr"),  # Transposition
        ("eat", "aet"),  # Transposition
        ("something", "somethign"),  # Transposition
    ]
    
    confusion_matrix = build_confusion_matrix(pairs)
    
    # Create a lexicon
    logger.info("Creating lexicon...")
    lexicon = set(
        word.lower()
        for word in """
        the quick brown fox jumps over lazy dog this is a sample text for
        training ppm model it contains common words and phrases that might
        be used in aac systems hello how are you today i am fine thank weather
        nice would like to go walk please help with task need send email can
        pass me water hungry eat something
        """.split()
    )
    
    # Create a corrector
    logger.info("Creating corrector...")
    corrector = NoisyChannelCorrector(
        ppm_model=ppm_model,
        confusion_model=confusion_matrix,
        lexicon=lexicon,
        max_candidates=5,
    )
    
    # Test the corrector with some noisy inputs
    test_inputs = [
        "thsi",  # Transposition of "this"
        "teh",  # Transposition of "the"
        "quikc",  # Substitution in "quick"
        "brwon",  # Transposition of "brown"
        "fxo",  # Transposition of "fox"
        "jumsp",  # Transposition of "jumps"
        "ovre",  # Transposition of "over"
        "lasy",  # Substitution in "lazy"
        "dgo",  # Transposition of "dog"
        "hlelo",  # Transposition of "hello"
        "todya",  # Transposition of "today"
        "weatehr",  # Transposition of "weather"
        "niec",  # Transposition of "nice"
        "wuold",  # Transposition of "would"
        "liek",  # Transposition of "like"
        "wlak",  # Transposition of "walk"
        "plase",  # Deletion in "please"
        "hlep",  # Transposition of "help"
        "wiht",  # Transposition of "with"
        "tihs",  # Transposition of "this"
        "taks",  # Transposition of "task"
        "nede",  # Transposition of "need"
        "sedn",  # Transposition of "send"
        "emial",  # Transposition of "email"
        "pss",  # Deletion in "pass"
        "watre",  # Transposition of "water"
        "hungyr",  # Transposition of "hungry"
        "aet",  # Transposition of "eat"
        "somethign",  # Transposition of "something"
    ]
    
    # Test each input
    for noisy_input in test_inputs:
        logger.info(f"\nCorrecting: {noisy_input}")
        corrections = corrector.correct(noisy_input)
        
        # Print the corrections
        for i, (correction, score) in enumerate(corrections):
            logger.info(f"{i+1}. {correction} (score: {score:.4f})")


if __name__ == "__main__":
    main()
