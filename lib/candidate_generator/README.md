# Candidate Generator

This module provides functionality to generate plausible candidates for a given noisy input. It uses edit distance, keyboard adjacency, and lexicon-based filtering to generate and rank candidates.

The main implementation is the `ImprovedCandidateGenerator` class, which combines the best features from all previous versions.

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
from lib.candidate_generator.improved_candidate_generator import ImprovedCandidateGenerator

# Create a candidate generator
generator = ImprovedCandidateGenerator()

# Load a lexicon from a file
with open("data/wordlist.txt", "r") as f:
    lexicon = set(line.strip().lower() for line in f if line.strip())
generator.lexicon = lexicon

# Generate candidates for a noisy input
candidates = generator.generate_candidates("helo")

# Print the top candidates
for candidate, score in candidates[:5]:
    print(f"{candidate}: {score:.4f}")
```

### Using the Base Class

```python
from lib.candidate_generator.candidate_generator import CandidateGenerator

# Create a basic candidate generator
generator = CandidateGenerator(lexicon={"hello", "help", "world", "test"})

# Generate candidates for a noisy input
candidates = generator.generate_candidates("helo")
```

## API Reference

### `ImprovedCandidateGenerator` Class

#### `__init__(lexicon=None, max_candidates=30, max_edits=20000, keyboard_boost=0.3, strict_filtering=True, smart_filtering=True, use_frequency_info=True)`

Initialize an improved candidate generator.

- `lexicon`: Set of valid words to use for filtering candidates
- `max_candidates`: Maximum number of candidates to return
- `max_edits`: Maximum number of edit candidates to generate
- `keyboard_boost`: Boost factor for keyboard-adjacent substitutions
- `strict_filtering`: Whether to strictly filter candidates by lexicon
- `smart_filtering`: Whether to use smart filtering for candidates
- `use_frequency_info`: Whether to use word frequency information

### `CandidateGenerator` Class (Base Class)

#### `__init__(lexicon=None, max_candidates=10)`

Initialize a basic candidate generator.

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
