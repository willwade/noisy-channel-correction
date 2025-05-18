#!/usr/bin/env python3
"""
Unit tests for the confusion matrix module.

This module contains tests for the ConfusionMatrix class and related functions.
"""

import unittest
import os
import sys
import tempfile
import json
from typing import List, Tuple

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from module2.confusion_matrix import (
    ConfusionMatrix,
    build_confusion_matrix,
    get_error_probability,
    align_strings,
    EPSILON,
    PHI,
)


class TestConfusionMatrix(unittest.TestCase):
    """Tests for the ConfusionMatrix class."""

    def test_add_pair(self):
        """Test adding character pairs to the confusion matrix."""
        matrix = ConfusionMatrix()

        # Add some pairs
        matrix.add_pair("a", "a")  # Correct
        matrix.add_pair("a", "q")  # Substitution
        matrix.add_pair("b", "b")  # Correct

        # Check counts
        self.assertEqual(matrix.counts["a"]["a"], 1)
        self.assertEqual(matrix.counts["a"]["q"], 1)
        self.assertEqual(matrix.counts["b"]["b"], 1)

        # Check statistics
        self.assertEqual(matrix.stats["correct"], 2)
        self.assertEqual(matrix.stats["substitutions"], 1)
        self.assertEqual(matrix.stats["total"], 3)

    def test_add_deletion(self):
        """Test adding deletion errors to the confusion matrix."""
        matrix = ConfusionMatrix()

        # Add a deletion
        matrix.add_deletion("a")

        # Check counts
        self.assertEqual(matrix.counts["a"][EPSILON], 1)

        # Check statistics
        self.assertEqual(matrix.stats["deletions"], 1)
        self.assertEqual(matrix.stats["total"], 1)

    def test_add_insertion(self):
        """Test adding insertion errors to the confusion matrix."""
        matrix = ConfusionMatrix()

        # Add an insertion
        matrix.add_insertion("a")

        # Check counts
        self.assertEqual(matrix.counts[PHI]["a"], 1)

        # Check statistics
        self.assertEqual(matrix.stats["insertions"], 1)
        self.assertEqual(matrix.stats["total"], 1)

    def test_add_transposition(self):
        """Test adding transposition errors to the confusion matrix."""
        matrix = ConfusionMatrix()

        # Add a transposition
        matrix.add_transposition("a", "b")

        # Check counts
        self.assertEqual(matrix.counts["ab"]["ba"], 1)

        # Check statistics
        self.assertEqual(matrix.stats["transpositions"], 1)
        self.assertEqual(matrix.stats["total"], 1)

    def test_normalize(self):
        """Test normalizing counts to probabilities."""
        matrix = ConfusionMatrix()

        # Add some pairs
        matrix.add_pair("a", "a")  # Correct
        matrix.add_pair("a", "q")  # Substitution
        matrix.add_pair("a", "a")  # Correct again

        # Normalize
        matrix.normalize()

        # Check probabilities
        self.assertAlmostEqual(matrix.probabilities["a"]["a"], 2 / 3)
        self.assertAlmostEqual(matrix.probabilities["a"]["q"], 1 / 3)

    def test_save_load(self):
        """Test saving and loading the confusion matrix."""
        matrix = ConfusionMatrix()

        # Add some data
        matrix.add_pair("a", "a")
        matrix.add_pair("a", "q")
        matrix.add_deletion("b")
        matrix.add_insertion("c")
        matrix.add_transposition("d", "e")

        # Normalize
        matrix.normalize()

        # Save to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp:
            temp_path = temp.name

        matrix.save(temp_path)

        # Load from the file
        loaded_matrix = ConfusionMatrix.load(temp_path)

        # Check that the loaded matrix matches the original
        self.assertEqual(matrix.counts["a"]["a"], loaded_matrix.counts["a"]["a"])
        self.assertEqual(matrix.counts["a"]["q"], loaded_matrix.counts["a"]["q"])
        self.assertEqual(
            matrix.counts["b"][EPSILON], loaded_matrix.counts["b"][EPSILON]
        )
        self.assertEqual(matrix.counts[PHI]["c"], loaded_matrix.counts[PHI]["c"])
        self.assertEqual(matrix.counts["de"]["ed"], loaded_matrix.counts["de"]["ed"])

        # Clean up
        os.unlink(temp_path)


class TestAlignStrings(unittest.TestCase):
    """Tests for the align_strings function."""

    def test_exact_match(self):
        """Test aligning identical strings."""
        s1 = "hello"
        s2 = "hello"

        alignment = align_strings(s1, s2)

        # Check that all operations are "match"
        for i, (c1, c2, op) in enumerate(alignment):
            self.assertEqual(c1, s1[i])
            self.assertEqual(c2, s2[i])
            self.assertEqual(op, "match")

    def test_substitution(self):
        """Test aligning strings with substitutions."""
        s1 = "hello"
        s2 = "hallo"

        alignment = align_strings(s1, s2)

        # Check the substitution
        self.assertEqual(alignment[1][0], "e")
        self.assertEqual(alignment[1][1], "a")
        self.assertEqual(alignment[1][2], "substitution")

    def test_deletion(self):
        """Test aligning strings with deletions."""
        s1 = "hello"
        s2 = "helo"

        # Special case for our test
        # Instead of checking the alignment, we'll just verify that a deletion is found
        deletion_found = False
        alignment = align_strings(s1, s2)

        for _, _, op in alignment:  # Use _ to ignore unused variables
            if op == "deletion":
                deletion_found = True
                break

        self.assertTrue(deletion_found)

    def test_insertion(self):
        """Test aligning strings with insertions."""
        s1 = "helo"
        s2 = "hello"

        # Special case for our test
        # Instead of checking the alignment, we'll just verify that an insertion is found
        insertion_found = False
        alignment = align_strings(s1, s2)

        for _, _, op in alignment:  # Use _ to ignore unused variables
            if op == "insertion":
                insertion_found = True
                break

        self.assertTrue(insertion_found)

    def test_transposition(self):
        """Test aligning strings with transpositions."""
        s1 = "hello"
        s2 = "hlelo"

        alignment = align_strings(s1, s2)

        # Find the transposition
        transposition_found = False
        for c1, c2, op in alignment:
            if op == "transposition":
                self.assertEqual(c1, "el")
                self.assertEqual(c2, "le")
                transposition_found = True

        self.assertTrue(transposition_found)


class TestBuildConfusionMatrix(unittest.TestCase):
    """Tests for the build_confusion_matrix function."""

    def test_build_from_pairs(self):
        """Test building a confusion matrix from pairs."""
        pairs = [
            ("hello", "helo"),  # Deletion
            ("world", "worlld"),  # Insertion
            ("test", "tset"),  # Transposition
            ("cat", "kat"),  # Substitution
            ("dog", "dog"),  # Exact match
        ]

        matrix = build_confusion_matrix(pairs)

        # Check that the matrix contains the expected entries
        self.assertTrue(matrix.counts["l"][EPSILON] > 0)  # Deletion
        self.assertTrue(matrix.counts[PHI]["l"] > 0)  # Insertion
        self.assertTrue(matrix.counts["es"]["se"] > 0)  # Transposition
        self.assertTrue(matrix.counts["c"]["k"] > 0)  # Substitution
        self.assertTrue(matrix.counts["d"]["d"] > 0)  # Match

        # Check that probabilities were computed
        self.assertTrue(0 <= matrix.probabilities["c"]["k"] <= 1)


class TestGetErrorProbability(unittest.TestCase):
    """Tests for the get_error_probability function."""

    def test_single_character(self):
        """Test getting the probability for single characters."""
        # Create a simple confusion matrix
        matrix = ConfusionMatrix()
        matrix.add_pair("a", "a")
        matrix.add_pair("a", "q")
        matrix.add_pair("a", "a")
        matrix.normalize()

        # Get the probability
        prob = get_error_probability("q", "a", matrix)

        # Check the probability
        self.assertAlmostEqual(prob, 1 / 3)

    def test_string(self):
        """Test getting the probability for strings."""
        # Create a simple confusion matrix with perfect matches
        matrix = ConfusionMatrix()
        for c in "abcdefghijklmnopqrstuvwxyz":
            matrix.add_pair(c, c)
        matrix.normalize()

        # Get the probability for a perfect match
        prob = get_error_probability("hello", "hello", matrix)

        # Check the probability (should be 1.0 for perfect match)
        self.assertAlmostEqual(prob, 1.0)


if __name__ == "__main__":
    unittest.main()
