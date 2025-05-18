# Enhanced PPM (Prediction by Partial Matching) for AAC

This directory contains an enhanced implementation of the PPM (Prediction by Partial Matching) algorithm for word prediction in Augmentative and Alternative Communication (AAC) systems.

## Features

- **Character-level prediction**: Predict the next character based on context
- **Word completion**: Complete partial words based on context
- **Word prediction**: Predict the next word based on context
- **Enhanced with modern NLP techniques**:
  - N-gram backoff for better handling of unseen contexts
  - Interpolation for smoother probability estimates
  - Recency bias to prioritize recently used words
  - Context-aware predictions based on sentence structure
- **Continuous learning**: Update the model with new text
- **Model persistence**: Save and load models

## Directory Structure

- `ppm/`: Core PPM implementation
  - `enhanced_ppm_predictor.py`: Enhanced PPM implementation with modern NLP techniques
  - `local_ppm_predictor.py`: Basic PPM implementation using pylm
  - `local_ppm_predictive_text.py`: Predictive text input component using local PPM
  - `app_with_local_ppm.py`: Gradio app using local PPM for word prediction
- `ppm/training/`: Training scripts and utilities
  - `download_conversational_corpus.py`: Download conversational text for training
  - `download_training_corpus.py`: Download books for training
  - `generate_training_text_from_social_graph.py`: Generate training text from social graph
  - `regenerate_training_text.py`: Regenerate training text and restart app
  - `train_with_conversational_corpus.sh`: One-click script to train with conversational corpus

## Usage

### Basic Usage

```python
from ppm.enhanced_ppm_predictor import EnhancedPPMPredictor

# Create a predictor
predictor = EnhancedPPMPredictor()

# Train the model
with open("training_text.txt", "r") as f:
    training_text = f.read()
predictor.train_on_text(training_text)

# Get predictions
predictions = predictor.get_all_predictions("Hello, how are you")
print(predictions)
```

### Training with Conversational Corpus

```bash
cd ppm/training
./train_with_conversational_corpus.sh
```

### Using the Gradio App

```bash
python ppm/app_with_local_ppm.py
```

## Dependencies

- `pylm`: PPM language model implementation
- `gradio`: Web interface
- `requests`: HTTP requests for downloading corpora

## License

This code is licensed under the MIT License. See the LICENSE file for details.
