"""
Unit tests for the candidate generator module.
"""

import os
import sys
import unittest

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the module to test
from lib.candidate_generator.candidate_generator import CandidateGenerator, generate_candidates


class TestCandidateGenerator(unittest.TestCase):
    """Test cases for the CandidateGenerator class."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a small test lexicon
        self.test_lexicon = {
            "hello",
            "world",
            "test",
            "testing",
            "cat",
            "dog",
            "bat",
            "hat",
            "rat",
            "sat",
            "mat",
            "fat",
            "computer",
            "keyboard",
            "language",
        }

        # Create a candidate generator with the test lexicon
        self.generator = CandidateGenerator(lexicon=self.test_lexicon)

    def test_initialization(self):
        """Test initialization of the CandidateGenerator."""
        self.assertEqual(len(self.generator.lexicon), len(self.test_lexicon))
        self.assertEqual(self.generator.max_candidates, 10)

    def test_load_lexicon_from_file(self):
        """Test loading a lexicon from a file."""
        # Create a temporary lexicon file
        temp_file = "temp_lexicon.txt"
        with open(temp_file, "w") as f:
            f.write("\n".join(self.test_lexicon))

        # Create a new generator and load the lexicon
        generator = CandidateGenerator()
        result = generator.load_lexicon_from_file(temp_file)

        # Check that the lexicon was loaded correctly
        self.assertTrue(result)
        self.assertEqual(len(generator.lexicon), len(self.test_lexicon))

        # Clean up
        os.remove(temp_file)

    def test_edit_distance_1(self):
        """Test generating candidates with edit distance 1."""
        # Test with a simple word
        candidates = self.generator._edit_distance_1("cat")

        # Check that some expected candidates are in the result
        self.assertIn("at", candidates)  # Deletion
        self.assertIn("act", candidates)  # Insertion
        self.assertIn("bat", candidates)  # Replacement

        # Check that we have a reasonable number of candidates
        # The exact count may vary based on implementation details
        self.assertTrue(len(candidates) > 100)

    def test_get_edit_distance_candidates(self):
        """Test generating candidates within a certain edit distance."""
        # Test with edit distance 1
        candidates_1 = self.generator._get_edit_distance_candidates("cat", 1)
        self.assertIn("bat", candidates_1)

        # Test with edit distance 2
        candidates_2 = self.generator._get_edit_distance_candidates("cat", 2)
        self.assertIn("bat", candidates_2)
        self.assertIn("hat", candidates_2)  # Edit distance 2 from "cat"

    def test_get_keyboard_adjacency_candidates(self):
        """Test generating candidates based on keyboard adjacency."""
        # Test with a simple word
        candidates = self.generator._get_keyboard_adjacency_candidates("cat")

        # Check that some expected candidates are in the result
        # 'c' is adjacent to 'v', 'x', 'd', 'f' on a QWERTY keyboard
        self.assertIn("vat", candidates)
        self.assertIn("xat", candidates)

        # 'a' is adjacent to 'q', 'w', 's', 'z' on a QWERTY keyboard
        self.assertIn("cqt", candidates)
        self.assertIn("cwt", candidates)

        # 't' is adjacent to 'r', 'y', 'g', 'f' on a QWERTY keyboard
        self.assertIn("car", candidates)
        self.assertIn("cay", candidates)

    def test_filter_candidates(self):
        """Test filtering candidates to only include valid words."""
        # Create a set of candidates, some in the lexicon and some not
        candidates = {"cat", "dog", "bat", "xyz", "abc"}

        # Filter the candidates
        filtered = self.generator._filter_candidates(candidates)

        # Check that only the valid words are in the result
        self.assertIn("cat", filtered)
        self.assertIn("dog", filtered)
        self.assertIn("bat", filtered)
        self.assertNotIn("xyz", filtered)
        self.assertNotIn("abc", filtered)

    def test_rank_candidates(self):
        """Test ranking candidates by likelihood."""
        # Create a set of candidates
        candidates = {"cat", "bat", "hat", "rat", "sat", "mat", "fat"}

        # Rank the candidates for the noisy input "kat"
        ranked = self.generator._rank_candidates(candidates, "kat")

        # Check that all candidates are included and sorted by score
        self.assertEqual(len(ranked), len(candidates))

        # Check that all scores are between 0 and 1
        for _, score in ranked:
            self.assertTrue(
                0 <= score <= 1.5
            )  # Allow for boost from keyboard adjacency

        # Check that the top candidates have high scores
        self.assertTrue(ranked[0][1] > 0.5)

    def test_generate_candidates(self):
        """Test the main generate_candidates method."""
        # Test with a noisy input that is already in the lexicon
        candidates = self.generator.generate_candidates("cat")
        self.assertEqual(candidates[0][0], "cat")
        self.assertEqual(candidates[0][1], 1.0)  # Perfect score

        # Test with a noisy input that has multiple close matches
        candidates = self.generator.generate_candidates("hat")
        self.assertEqual(candidates[0][0], "hat")  # Exact match

        # Test with a noisy input that has no close matches
        candidates = self.generator.generate_candidates("xyz")
        self.assertLessEqual(len(candidates), self.generator.max_candidates)

        # Test with a noisy input that has close matches
        candidates = self.generator.generate_candidates("kat")
        # Check that we get some candidates
        self.assertTrue(len(candidates) > 0)
        # Check that all candidates are in the lexicon
        for candidate, _ in candidates:
            self.assertIn(candidate, self.test_lexicon)

    def test_convenience_function(self):
        """Test the convenience function generate_candidates."""
        # Test with a noisy input and a lexicon
        candidates = generate_candidates("kat", lexicon=self.test_lexicon)

        # Check that we get some candidates
        self.assertTrue(len(candidates) > 0)

        # Check that all candidates are in the lexicon
        for candidate, _ in candidates:
            self.assertIn(candidate, self.test_lexicon)


if __name__ == "__main__":
    unittest.main()
