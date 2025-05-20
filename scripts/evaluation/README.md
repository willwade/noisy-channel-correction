# Evaluation Tools for AAC Noisy Input Correction Engine

This directory provides evaluation tools for the AAC Noisy Input Correction Engine. It uses the AACConversations dataset from Hugging Face to demonstrate and evaluate the correction capabilities of the system.

## Directory Structure

The evaluation tools are organized into the following categories:

### Core Evaluation Scripts

- **demo.py**: Interactive CLI demo for testing the correction engine
- **eval.py**: Basic evaluation script for measuring correction accuracy
- **conversation_level_evaluation.py**: Comprehensive conversation-level evaluation with context-aware correction
- **compare_correction_methods.py**: Compare different correction methods on the same conversations
- **synthetic_conversation_evaluator.py**: Evaluate correction on synthetic conversations

### Utility Scripts

- **utils.py**: Common utilities for evaluation
- **noise_simulator_utils.py**: Utilities for noise simulation
- **filter_english_data.py**: Filters dataset for English data
- **visualize_conversation_results.py**: Visualizes evaluation results

## Getting Started

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Authentication

The AACConversations dataset is gated and requires authentication. Make sure you have access to the dataset and are logged in to Hugging Face:

```bash
pip install huggingface_hub
huggingface-cli login
```

## Usage

### Interactive Demo

Run the interactive demo to test the correction engine:

```bash
python demo.py --interactive --ppm-model models/ppm_model.pkl --confusion-matrix models/confusion_matrix.json --word-ngram-model models/word_ngram_model.pkl --lexicon data/wordlist.txt
```

### Basic Evaluation

Evaluate the correction engine on random examples from the dataset:

```bash
python eval.py --ppm-model models/ppm_model.pkl --confusion-matrix models/confusion_matrix.json --lexicon data/wordlist.txt --output eval_results.json
```

### Conversation-Level Evaluation

Evaluate the correction engine on entire conversations:

```bash
python conversation_level_evaluation.py --ppm-model models/ppm_model.pkl --confusion-matrix models/confusion_matrix.json --word-ngram-model models/word_ngram_model.pkl --lexicon data/wordlist.txt --keyboard-layouts qwerty --noise-levels minimal
```

#### Using Synthetic Conversations

You can also evaluate using synthetic conversations instead of the AACConversations dataset:

```bash
python conversation_level_evaluation.py --synthetic --ppm-model models/ppm_model.pkl --confusion-matrix models/confusion_matrix.json --word-ngram-model models/word_ngram_model.pkl --lexicon data/wordlist.txt --num-conversations 5 --num-turns 10 --error-rate 0.2
```

Alternatively, you can use the dedicated synthetic conversation evaluator:

```bash
python synthetic_conversation_evaluator.py --ppm-model models/ppm_model.pkl --confusion-matrix models/confusion_matrix.json --word-ngram-model models/word_ngram_model.pkl --lexicon data/wordlist.txt --num-conversations 5 --num-turns 10 --error-rate 0.2
```

### Method Comparison

Compare different correction methods on the same conversations:

```bash
python compare_correction_methods.py --ppm-model models/ppm_model.pkl --confusion-matrix models/confusion_matrix.json --word-ngram-model models/word_ngram_model.pkl --lexicon data/wordlist.txt --methods baseline context-aware keyboard-specific combined
```

## Key Parameters

### Common Parameters
- **--ppm-model**: Path to the PPM model file
- **--confusion-matrix**: Path to the confusion matrix file
- **--word-ngram-model**: Path to the word n-gram model file
- **--lexicon**: Path to the lexicon file
- **--keyboard-layouts**: Keyboard layouts to evaluate (qwerty, abc, frequency)
- **--noise-levels**: Noise levels to evaluate (minimal, light, moderate, severe)

### Dataset Parameters
- **--language-code**: Language code to filter by (e.g., en-GB)
- **--max-conversations**: Maximum number of conversations to evaluate
- **--target-field**: Field to use as the target for evaluation (minimally_corrected, fully_corrected, utterance_intended)

### Synthetic Conversation Parameters
- **--synthetic**: Use synthetic conversations instead of the dataset
- **--num-conversations**: Number of synthetic conversations to generate
- **--num-turns**: Number of turns per synthetic conversation
- **--vocabulary-size**: Size of the vocabulary for synthetic conversations
- **--error-rate**: Error rate for synthetic conversations

## Dataset

The AACConversations dataset contains conversations with AAC users, including:

- Original intended utterances. note: these will be quite different from the utterances in the dataset as they are imagined utterances
- Noisy versions with different types and levels of noise
- Conversation context
- Scene information

Each example in the dataset has the following structure:

```json
{
  "conversation_id": 42,
  "turn_number": 2,
  "language_code": "en-GB",
  "scene": "At a doctor's appointment",
  "context_speakers": ["Doctor", "Patient (AAC)"],
  "context_utterances": ["How have you been feeling lately?", "Not great"],
  "speaker": "Patient (AAC)",
  "utterance": "I've been having trouble sleeping",
  "utterance_intended": "I've been having trouble sleeping",
  "next_turn_speaker": "Doctor",
  "next_turn_utterance": "How long has this been going on?",
  "model": "gpt-4o-mini",
  "provider": "openai",
  "batch_id": "batch_682477c828bc81909f580a018af3a06c",
  "batch_number": 3,
  "noisy_qwerty_minimal": "I've been having troubke sleeping",
  "noisy_qwerty_light": "I've been havng troble sleepng",
  "noisy_qwerty_moderate": "I've ben havin troble sleping",
  "noisy_qwerty_severe": "Ive ben havin trble slping",
  "noisy_abc_minimal": "I've been having troubke sleeping",
  "noisy_abc_light": "I've been havng troble sleepng",
  "noisy_abc_moderate": "I've ben havin troble sleping",
  "noisy_abc_severe": "Ive ben havin trble slping",
  "noisy_frequency_minimal": "I've been having troubke sleeping",
  "noisy_frequency_light": "I've been havng troble sleepng",
  "noisy_frequency_moderate": "I've ben havin troble sleping",
  "noisy_frequency_severe": "Ive ben havin trble slping",
  "minimally_corrected": "I've been having trouble sleeping.",
  "fully_corrected": "I've been having trouble sleeping."
}
```

## Evaluation Metrics

The evaluation scripts calculate the following metrics:

- **Accuracy@1**: Percentage of examples where the top correction matches the intended utterance
- **Accuracy@N**: Percentage of examples where any of the top N corrections matches the intended utterance
- **Average Correction Time**: Average time taken to correct an utterance (in milliseconds)
- **Context Improvement**: Improvement in accuracy when using context-aware correction
- **Keyboard-Specific Accuracy**: Accuracy for different keyboard layouts

Results are broken down by noise type and level for detailed analysis.

## Deprecated Scripts

The following scripts have been moved to the deprecated_scripts directory:

- **conversation_evaluator.py**: Use conversation_level_evaluation.py instead
- **evaluate_english_conversations.py**: Use conversation_level_evaluation.py with --language-code en-GB instead
- **simple_conversation_evaluator.py**: Use conversation_level_evaluation.py instead
- **simple_compare_methods.py**: Use compare_correction_methods.py instead
- **context_aware_evaluation.py**: Use conversation_level_evaluation.py with appropriate context parameters instead
