# Module 2: Confusion Matrix Builder

This module builds a confusion matrix from pairs of (intended, noisy) text data. The confusion matrix represents the probability of observing a noisy character given an intended character, which is a key component of the noisy channel model for text correction.

## Overview

The confusion matrix captures different types of errors:
- **Substitutions**: One character replaced by another (e.g., 'a' → 'q')
- **Deletions**: A character is omitted (e.g., 'hello' → 'helo')
- **Insertions**: An extra character is added (e.g., 'hello' → 'helllo')
- **Transpositions**: Adjacent characters are swapped (e.g., 'hello' → 'hlelo')

## Usage

```python
from confusion_matrix.confusion_matrix import build_confusion_matrix, get_error_probability

# Load pairs of (intended, noisy) text
pairs = [
    ("hello", "helo"),
    ("world", "worlld"),
    ("testing", "testting"),
    # ...
]

# Build the confusion matrix
confusion_matrix = build_confusion_matrix(pairs)

# Get the probability of a specific error
prob = get_error_probability("a", "q", confusion_matrix)
print(f"P(noisy='q' | intended='a') = {prob}")

# Save the confusion matrix for later use
confusion_matrix.save("data/confusion_matrix.json")
```

## Functions

- `build_confusion_matrix(pairs)`: Builds a confusion matrix from pairs of (intended, noisy) text
- `get_error_probability(noisy, intended, matrix)`: Gets the probability of observing a noisy character given an intended character

## Implementation Details

The confusion matrix is implemented as a nested dictionary where:
- The outer key is the intended character
- The inner key is the noisy character
- The value is the count or probability of that error

Special characters are used to represent insertions and deletions:
- `ε` (epsilon) represents a deletion (intended character → nothing)
- `φ` (phi) represents an insertion (nothing → noisy character)

## Extensions

The module can be extended to handle:
- Bigram-level confusion matrices
- Word-level confusion matrices
- Context-dependent error probabilities
