# Module 5: Interface and Evaluation

This module provides a CLI demo and evaluation tools for the AAC Noisy Input Correction Engine. It uses the AACConversations dataset from Hugging Face to demonstrate and evaluate the correction capabilities of the system.

## Features

- **CLI Demo**: Interactive command-line interface for testing the correction engine
- **Evaluation Tools**: Scripts for measuring accuracy and performance
- **Dataset Integration**: Seamless integration with the AACConversations dataset
- **Configurable Noise**: Support for different noise types and levels

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

### CLI Demo

Run the CLI demo to test the correction engine on random examples from the dataset:

```bash
python demo.py --ppm-model ../data/ppm_model.json --confusion-matrix ../data/confusion_matrix.json --lexicon ../data/lexicon.txt
```

#### Options

- `--noise-type`: Type of noise to use (`qwerty`, `abc`, or `frequency`)
- `--noise-level`: Level of noise to use (`minimal`, `light`, `moderate`, or `severe`)
- `--num-examples`: Number of examples to sample
- `--interactive`: Run in interactive mode for custom input testing

### Evaluation

Evaluate the correction engine on the dataset:

```bash
python eval.py --ppm-model ../data/ppm_model.json --confusion-matrix ../data/confusion_matrix.json --lexicon ../data/lexicon.txt --output eval_results.json
```

#### Options

- `--noise-types`: Noise types to evaluate
- `--noise-levels`: Noise levels to evaluate
- `--num-examples`: Number of examples to evaluate
- `--seed`: Random seed for reproducibility

## Dataset

The AACConversations dataset contains conversations with AAC users, including:

- Original intended utterances
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

The evaluation script calculates the following metrics:

- **Accuracy@1**: Percentage of examples where the top correction matches the intended utterance
- **Accuracy@N**: Percentage of examples where any of the top N corrections matches the intended utterance
- **Average Correction Time**: Average time taken to correct an utterance (in milliseconds)

Results are broken down by noise type and level for detailed analysis.
