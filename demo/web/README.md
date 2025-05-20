# AAC Noisy Input Correction Web Demo

This directory contains a web-based demo for the AAC Noisy Input Correction Engine using Gradio.

## Features

- **Interactive Mode**: Type text and see real-time corrections and word predictions
- **Conversation Simulation**: Simulate conversations from the AACConversations dataset
- **Settings**: Configure correction parameters

## Requirements

- Python 3.8+
- Gradio
- Datasets (for conversation simulation)

## Installation

```bash
# Install dependencies
uv pip install gradio datasets
```

## Usage

```bash
# Run the demo with default settings
uv run demo/web/app.py

# Run with specific model paths
uv run demo/web/app.py --ppm-model models/ppm_model_en_gb.pkl --confusion-matrix models/qwerty_confusion_matrix_en_gb.json

# Use a different keyboard layout
uv run demo/web/app.py --keyboard-layout qwerty

# Create a public link
uv run demo/web/app.py --share
```

## Command-line Arguments

- `--ppm-model`: Path to the PPM model file (default: `models/ppm_model_en_gb.pkl`)
- `--confusion-matrix`: Path to the confusion matrix file (default: `models/qwerty_confusion_matrix_en_gb.json`)
- `--word-ngram-model`: Path to the word n-gram model file (default: `models/word_ngram_model_en_gb.pkl`)
- `--lexicon`: Path to the lexicon file (default: `data/enhanced_lexicon_en_gb.txt`)
- `--max-candidates`: Maximum number of correction candidates to generate (default: 10)
- `--context-window`: Size of the context window for context-aware correction (default: 2)
- `--context-weight`: Weight of the context in the correction score (default: 0.5)
- `--max-edit-distance`: Maximum edit distance for candidate generation (default: 2)
- `--keyboard-layout`: Keyboard layout for confusion matrix (default: qwerty)
- `--dataset-split`: Dataset split to use (default: test)
- `--cache-dir`: Directory to cache the dataset
- `--no-token`: Don't use the Hugging Face auth token
- `--port`: Port to run the Gradio server on (default: 7860)
- `--share`: Create a public link for the demo

## Modes

### Interactive Mode

In this mode, you can type text in the input box and see real-time corrections and word predictions. You can also provide context for context-aware correction.

### Conversation Simulation

This mode simulates conversations from the AACConversations dataset. You can select a conversation and step through it turn by turn, seeing the noisy input, intended text, and corrections.

### Settings

In this tab, you can configure the correction engine parameters, such as the maximum number of candidates, context window size, and context weight.

## Notes

- The demo requires the models to be trained and available at the specified paths.
- For conversation simulation, you need to have access to the AACConversations dataset on Hugging Face.
- The demo focuses on English (en-GB) data.

## Keyboard-Specific Confusion Matrices

The demo supports keyboard-specific confusion matrices, which model typing errors based on the physical layout of the keyboard. This is particularly important for AAC users who may make different types of errors depending on the keyboard layout they use.

The system supports three keyboard layouts:
- `qwerty`: Standard QWERTY keyboard layout
- `azerty`: AZERTY keyboard layout (common in French-speaking countries)
- `dvorak`: Dvorak keyboard layout

To use a keyboard-specific confusion matrix:

1. First, extract the layout-specific matrix from the keyboard confusion matrices file:
   ```bash
   uv run scripts/confusion_matrix_builder/extract_qwerty_matrix.py --input models/keyboard_confusion_matrices_en_gb.json --output models/qwerty_confusion_matrix_en_gb.json
   ```

2. The demo is already configured to use the QWERTY confusion matrix by default:
   ```bash
   uv run demo/web/app.py
   ```

   You can also explicitly specify the keyboard layout:
   ```bash
   uv run demo/web/app.py --keyboard-layout qwerty
   ```

This ensures that the correction engine takes into account the physical layout of the keyboard when correcting typing errors.
