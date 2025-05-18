# Module 3: Candidate Generator

This module provides functionality to generate plausible candidates for a given noisy input. It uses edit distance, keyboard adjacency, and lexicon-based filtering to generate and rank candidates.

## Overview

The candidate generator is a key component of the noisy channel correction system. It takes a noisy input (e.g., a misspelled word) and generates a list of plausible candidates that might be the intended input. These candidates are then ranked by likelihood and passed to the correction engine for final selection.

## Features

- **Edit Distance Calculation**: Generates candidates within a specified edit distance (Levenshtein distance)
- **Keyboard Adjacency**: Considers keyboard layout when generating candidates
- **Lexicon Filtering**: Filters candidates to only include valid words from a lexicon
- **Candidate Ranking**: Ranks candidates by likelihood based on similarity to the noisy input

## Usage

### Basic Usage

```python
from module3.candidate_generator import CandidateGenerator

# Create a candidate generator with a lexicon
generator = CandidateGenerator(lexicon={"hello", "world", "test"})

# Generate candidates for a noisy input
candidates = generator.generate_candidates("helo")
# Returns: [("hello", 0.8), ...]
```

### Loading a Lexicon from a File

```python
from module3.candidate_generator import CandidateGenerator

# Create a candidate generator
generator = CandidateGenerator()

# Load a lexicon from a file
generator.load_lexicon_from_file("data/wordlist.txt")

# Generate candidates for a noisy input
candidates = generator.generate_candidates("helo")
```

### Using the Convenience Function

```python
from module3.candidate_generator import generate_candidates

# Generate candidates for a noisy input
candidates = generate_candidates("helo", lexicon={"hello", "world", "test"})
```

## API Reference

### `CandidateGenerator` Class

#### `__init__(lexicon=None, max_candidates=10)`

Initialize a candidate generator.

- `lexicon`: Set of valid words to use for filtering candidates
- `max_candidates`: Maximum number of candidates to return

#### `load_lexicon_from_file(file_path)`

Load a lexicon from a file.

- `file_path`: Path to the lexicon file (one word per line)
- Returns: `True` if successful, `False` otherwise

#### `generate_candidates(noisy_input, max_edit_distance=2, use_keyboard_adjacency=True)`

Generate candidates for a noisy input.

- `noisy_input`: The noisy input text
- `max_edit_distance`: Maximum edit distance to consider
- `use_keyboard_adjacency`: Whether to use keyboard adjacency for candidate generation
- Returns: List of `(candidate, score)` tuples, sorted by score (highest first)

### `generate_candidates` Function

#### `generate_candidates(noisy_input, lexicon=None, max_edit_distance=2)`

Generate candidates for a noisy input.

- `noisy_input`: The noisy input text
- `lexicon`: Set of valid words to use for filtering candidates
- `max_edit_distance`: Maximum edit distance to consider
- Returns: List of `(candidate, score)` tuples, sorted by score (highest first)

## Implementation Details

### Edit Distance Calculation

The module uses the Levenshtein distance algorithm to calculate the edit distance between two strings. It considers four types of operations:

1. **Deletions**: Remove one character
2. **Insertions**: Insert one character
3. **Substitutions**: Replace one character with another
4. **Transpositions**: Swap adjacent characters

### Keyboard Adjacency

The module uses the keyboard layout model from Module 1 to determine which keys are adjacent on the keyboard. This allows it to generate candidates that are likely to be typing errors.

### Candidate Ranking

Candidates are ranked by their similarity to the noisy input. The similarity is calculated as:

```
similarity = 1.0 - (edit_distance / max_length)
```

where `max_length` is the maximum length of the noisy input and the candidate.

## Testing

The module includes a comprehensive test suite that verifies the functionality of the candidate generator. To run the tests:

```bash
python -m unittest module3.test_candidate_generator
```

## Integration with Other Modules

This module is designed to work with the other modules in the noisy channel correction system:

- It uses the keyboard layout model from Module 1 to determine keyboard adjacency
- It will be used by Module 4 (Correction Engine) to generate candidates for correction
