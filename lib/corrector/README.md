# Module 4: Correction Engine (Noisy Channel Model)

This module implements a noisy channel model for correcting noisy AAC input. It combines a PPM language model for P(intended) and a confusion matrix for P(noisy | intended) to rank candidate corrections.

## Overview

The noisy channel model is a probabilistic framework for text correction that models the process of generating noisy text as a communication channel. The model assumes that a user intends to type a correct word (the "intended" text), but due to noise in the input process (e.g., typing errors, motor control issues), what actually appears is a noisy version of that text.

The correction process involves finding the most likely intended text given the observed noisy text. This is formulated using Bayes' rule:

P(intended | noisy) ∝ P(intended) × P(noisy | intended)

Where:
- P(intended) is the prior probability of the intended text, estimated using a language model (in our case, a PPM model)
- P(noisy | intended) is the likelihood of observing the noisy text given the intended text, estimated using a confusion matrix

The correction engine ranks candidate corrections by computing:

score = log(P(intended)) + log(P(noisy | intended))

And returns the top-N candidates with the highest scores.

## Components

- `corrector.py`: Implements the `NoisyChannelCorrector` class and the `correct` function
- `correct.py`: Command-line interface for the corrector
- `ppm/`: Directory containing the PPM language model implementation

## Usage

### Python API

```python
from module4.corrector import NoisyChannelCorrector

# Create a corrector
corrector = NoisyChannelCorrector()

# Load models and lexicon
corrector.load_ppm_model("path/to/ppm_model.pkl")
corrector.load_confusion_model("path/to/confusion_matrix.json")
corrector.load_lexicon_from_file("path/to/lexicon.txt")

# Correct a noisy input
corrections = corrector.correct("thes is a tst")
for correction, score in corrections:
    print(f"{correction} (score: {score:.4f})")
```

### Command-Line Interface

```bash
# Basic usage
python module4/correct.py --input "thes is a tst"

# Specify models and lexicon
python module4/correct.py --input "thes is a tst" --ppm-model path/to/ppm_model.pkl --confusion-matrix path/to/confusion_matrix.json --lexicon path/to/lexicon.txt

# Save the top correction to a file
python module4/correct.py --input "thes is a tst" --output corrected.txt

# Customize correction parameters
python module4/correct.py --input "thes is a tst" --max-candidates 10 --max-edit-distance 3 --no-keyboard-adjacency

# Provide context for the PPM model
python module4/correct.py --input "thes is a tst" --context "I think that"
```

## Implementation Details

### NoisyChannelCorrector Class

The `NoisyChannelCorrector` class is the main component of the correction engine. It combines:

1. A PPM language model for estimating P(intended)
2. A confusion matrix for estimating P(noisy | intended)
3. A candidate generator for generating plausible corrections

The correction process involves:

1. Generating candidates for the noisy input using the candidate generator
2. Scoring each candidate using the noisy channel model formula
3. Returning the top-N candidates with the highest scores

### PPM Language Model

The PPM (Prediction by Partial Matching) language model is used to estimate P(intended). It is implemented in the `module4/ppm/enhanced_ppm_predictor.py` file. The model uses a combination of character-level prediction, word completion, and word prediction to estimate the probability of a given text.

### Confusion Matrix

The confusion matrix is used to estimate P(noisy | intended). It is implemented in the `module2/confusion_matrix.py` file. The matrix represents the probability of observing a noisy character given an intended character, and supports various types of errors (substitutions, deletions, insertions, transpositions).

### Candidate Generator

The candidate generator is used to generate plausible corrections for a noisy input. It is implemented in the `module3/candidate_generator.py` file. The generator uses edit distance, keyboard adjacency, and lexicon-based filtering to generate and rank candidates.

## Integration with Other Modules

The correction engine integrates with:

1. Module 2 (Confusion Matrix Builder) for the error model
2. Module 3 (Candidate Generator) for generating plausible corrections
3. Module 4/ppm (PPM Language Model) for estimating the probability of intended text

## Future Enhancements

- Context-aware correction using multi-token context in the PPM model
- Beam search for longer utterance corrections
- Dynamic access model switching based on user profile
- Word-level confusion matrices for handling word-level errors
- Integration with AAC systems or TTS pipelines
