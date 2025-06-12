# AAC Noisy Input Correction Engine

A lightweight, interpretable correction system for noisy AAC (Augmentative and Alternative Communication) input. This system uses a **Noisy Channel Model** framework to correct user-generated text errors commonly seen in switch-access, eyegaze, dwell typing, and other AAC modalities.

It leverages a  **PPM language model** to score intended inputs (`P(intended)`) and a configurable **error model** to estimate `P(noisy | intended)` based on simulated input errors.

---


https://github.com/user-attachments/assets/b2bb8bff-9a08-401e-8167-0fd296291481


## 💡 Use Case

AAC users often make input errors due to physical or interface constraints. This project helps correct such errors in real time or during post-processing, improving intelligibility and user experience without requiring a neural network.

It also aims to be a best-in-class prediction system. 

Note: AAC users have typically short utterances and high repeatability. If we knew the context of both sides of the conversation can we help improve prediction and correction? Thats what we are trying to understand with this code

---

## 🔧 Architecture

The system is modular :

```
noisy-channel-correction/
├── lib/                      # Shared library code
│   ├── noise_model/          # Noise simulation
│   ├── confusion_matrix/     # Confusion matrix generation
│   ├── candidate_generator/  # Candidate generation
│   ├── corrector/            # Correction engine
│   ├── pylm/                 # Language model code
│   └── models/               # Model files
├── data/                     # Data files
│   ├── conversational_corpora/  # Training corpus data
│   ├── evaluation_results/      # Evaluation results
│   └── visualizations/          # Visualization outputs
├── scripts/                  # Utility scripts
│   ├── noise_simulator/         # Noise simulator scripts
│   ├── confusion_matrix_builder/# Confusion matrix builder scripts
│   ├── candidate_generator/     # Candidate generator scripts
│   ├── correction_engine/       # Correction engine scripts
│   └── evaluation/              # Evaluation scripts
├── tests/                    # Test files
└── demo/                     # Demo applications
    ├── noise_simulator/         # Noise simulator demos
    ├── confusion_matrix_builder/# Confusion matrix builder demos
    ├── candidate_generator/     # Candidate generator demos
    ├── correction_engine/       # Correction engine demos
    └── evaluation/              # Evaluation demos
```

Each module is standalone and reusable in other AAC or correction projects.

---

## 📦 Features

- **Modular Error Simulation**: Simulate access-method-specific errors (missed hits, double hits, proximity errors).
- **Character-Level Confusion Matrix**: Tuned from realistic simulated noise, supports insertion, deletion, substitution, and transposition.
- **Flexible Candidate Generation**: Efficiently generate and filter plausible corrections using edit distance, keyboard adjacency, and lexicon constraints.
- **Probabilistic Correction Engine**: Ranks candidates using:

score = log P(intended) + log P(noisy | intended)

Powered by your enhanced PPM model.

- **No Neural Dependencies**: Designed for low-resource, interpretable deployment.

---

## 🛠 Setup

Requirements:
- Python 3.8+
- Dependencies listed in `requirements.txt` (e.g. Levenshtein, numpy)

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Using pip

```bash
# Create a virtual environment
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🚀 Usage

### Model Files

The system uses several model files that should be placed in the `models/` directory:

- `ppm_model.pkl`: The PPM (Prediction by Partial Matching) language model
- `word_ngram_model.pkl`: The word n-gram language model for context-aware correction
- `confusion_matrix.json`: The character confusion matrix

These files are used by default in all scripts. If you need to specify different paths, you can use the appropriate command-line arguments.

### 1. Simulate Errors

```bash
python scripts/noise_simulator/simulate.py --input data/wordlist.txt --output data/noisy_pairs.json
```

### 2. Build Confusion Matrix

```bash
python scripts/confusion_matrix_builder/build_matrix.py --input data/noisy_pairs.json --output models/confusion_matrix.json
```

### 3. Generate Candidates

```bash
python scripts/candidate_generator/generate_candidates.py --input "thes is a tst" --lexicon data/wordlist.txt
```

### 4. Run Correction

```bash
python scripts/correction_engine/demo.py --mode interactive
```

This will load the models from the standard paths:
- PPM model: `models/ppm_model.pkl`
- Confusion matrix: `models/confusion_matrix.json`
- Word n-gram model: `models/word_ngram_model.pkl`
- Lexicon: `data/wordlist.txt`

You can specify different paths if needed:

```bash
python scripts/correction_engine/demo.py --mode interactive \
    --ppm-model models/custom_ppm_model.pkl \
    --confusion-matrix models/custom_confusion_matrix.json \
    --word-ngram-model models/custom_word_ngram_model.pkl \
    --lexicon data/custom_wordlist.txt
```

### 5. Run Evaluation

```bash
python scripts/evaluation/demo.py --use-noise-simulator --wordlist data/wordlist.txt
```

This will also load the models from the standard paths. You can specify different paths if needed:

```bash
python scripts/evaluation/demo.py --use-noise-simulator --wordlist data/wordlist.txt \
    --ppm-model models/custom_ppm_model.pkl \
    --confusion-matrix models/custom_confusion_matrix.json \
    --word-ngram-model models/custom_word_ngram_model.pkl
```

---

## 🧪 Evaluation

Use `scripts/evaluation/demo.py` to test accuracy of correction across datasets with varying noise profiles. Includes:
- Accuracy@1
- Accuracy@N
- Context-aware reranking (optional)

### Advanced Features

#### Context-Aware Correction

The system supports context-aware correction using a word n-gram model:

```bash
python scripts/correction_engine/demo.py --mode conversation --input data/sample_conversations.txt
```

#### Keyboard-Specific Confusion Matrices

The system can use different confusion matrices for different keyboard layouts:

```bash
python scripts/correction_engine/demo.py --keyboard-layout qwerty --use-keyboard-matrices
```

---

## 📚 Future Enhancements

- Access-specific confusion models (eyegaze vs touch vs switch)
- Word-level confusion matrices
- Beam search decoding
- Integration with AAC systems

---

## 👥 Credits

- Developed by Will Wade
- Inspired by AAC research and real-world interaction data
- PPM model by willwade (pylm repo and in turn the jslm repo from google/Brian Roark)
