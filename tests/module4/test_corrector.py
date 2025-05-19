#!/usr/bin/env python3
"""
Unit tests for the noisy channel corrector.

This module contains tests for the NoisyChannelCorrector class and related functions.
"""

import unittest
import os
import sys
import tempfile

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from lib.corrector.corrector import NoisyChannelCorrector, correct
from module4.ppm.enhanced_ppm_predictor import EnhancedPPMPredictor
from lib.confusion_matrix.confusion_matrix import build_confusion_matrix


class TestNoisyChannelCorrector(unittest.TestCase):
    """Tests for the NoisyChannelCorrector class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple PPM model
        self.ppm_model = EnhancedPPMPredictor()
        self.ppm_model.train_on_text(
            "this is a test of the noisy channel model for aac input correction"
        )

        # Create a simple confusion matrix
        pairs = [
            ("hello", "helo"),  # Deletion
            ("world", "worlld"),  # Insertion
            ("test", "tset"),  # Transposition
            ("cat", "kat"),  # Substitution
            ("dog", "dog"),  # Exact match
        ]
        self.confusion_matrix = build_confusion_matrix(pairs)

        # Create a simple lexicon
        self.lexicon = {
            "this",
            "is",
            "a",
            "test",
            "of",
            "the",
            "noisy",
            "channel",
            "model",
            "for",
            "aac",
            "input",
            "correction",
            "hello",
            "world",
            "cat",
            "dog",
            "these",
            "that",
            "those",
            "them",
            "they",
            "their",
        }

        # Create a corrector
        self.corrector = NoisyChannelCorrector(
            ppm_model=self.ppm_model,
            confusion_model=self.confusion_matrix,
            lexicon=self.lexicon,
            max_candidates=5,
        )

    def test_initialization(self):
        """Test initialization of the corrector."""
        self.assertIsNotNone(self.corrector)
        self.assertIsNotNone(self.corrector.ppm_model)
        self.assertIsNotNone(self.corrector.confusion_model)
        self.assertIsNotNone(self.corrector.lexicon)
        self.assertIsNotNone(self.corrector.candidate_generator)
        self.assertTrue(self.corrector.models_ready)

    def test_correct_exact_match(self):
        """Test correction of an exact match."""
        corrections = self.corrector.correct("test")
        self.assertEqual(len(corrections), 1)
        self.assertEqual(corrections[0][0], "test")
        # Don't check exact score value as it depends on implementation details
        self.assertGreater(corrections[0][1], 0.0)

    def test_correct_simple_error(self):
        """Test correction of a simple error."""
        corrections = self.corrector.correct("tset")  # Transposition of "test"
        self.assertGreater(len(corrections), 0)
        self.assertEqual(corrections[0][0], "test")

    def test_correct_with_context(self):
        """Test correction with context."""
        corrections = self.corrector.correct("tset", context="this is a")
        self.assertGreater(len(corrections), 0)
        self.assertEqual(corrections[0][0], "test")

    def test_correct_unknown_word(self):
        """Test correction of an unknown word."""
        # For unknown words with no close matches, the corrector might return the input as is
        # or attempt to find the closest word in the lexicon
        corrections = self.corrector.correct("xyz")
        self.assertGreater(len(corrections), 0)
        # We don't assert that the correction is in the lexicon, as the fallback might return the input

    def test_correct_multiple_candidates(self):
        """Test correction with multiple candidates."""
        corrections = self.corrector.correct("thes")  # Could be "these", "this", etc.
        self.assertGreater(len(corrections), 1)
        # The top correction should be a valid word
        self.assertIn(corrections[0][0], self.lexicon)

    def test_convenience_function(self):
        """Test the convenience function."""
        # Create temporary files for the models
        with tempfile.NamedTemporaryFile(
            suffix=".pkl"
        ) as ppm_file, tempfile.NamedTemporaryFile(
            suffix=".json"
        ) as confusion_file, tempfile.NamedTemporaryFile(
            suffix=".txt"
        ) as lexicon_file:

            # Save the PPM model
            self.ppm_model._save_model = lambda model_path: True  # Mock the save method

            # Save the confusion matrix
            self.confusion_matrix.save(confusion_file.name)

            # Save the lexicon
            with open(lexicon_file.name, "w") as f:
                for word in self.lexicon:
                    f.write(f"{word}\n")

            # Test the convenience function
            corrections = correct(
                "tset",
                ppm_model_path=ppm_file.name,
                confusion_matrix_path=confusion_file.name,
                lexicon_path=lexicon_file.name,
                max_candidates=5,
            )

            # The convenience function should return a list of tuples
            self.assertIsInstance(corrections, list)
            self.assertGreater(len(corrections), 0)
            self.assertIsInstance(corrections[0], tuple)
            self.assertEqual(len(corrections[0]), 2)


if __name__ == "__main__":
    unittest.main()
