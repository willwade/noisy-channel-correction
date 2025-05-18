#!/bin/bash
# Train the PPM model with conversational corpus

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
REPO_ROOT="$( cd "$SCRIPT_DIR/../.." &> /dev/null && pwd )"

# Create directories if they don't exist
mkdir -p "$REPO_ROOT/conversational_corpora"

# Download and process all conversational corpora
echo "Downloading all conversational corpora..."
python "$SCRIPT_DIR/download_conversational_corpus.py" --corpus all

# Regenerate the training text with the combined conversational corpus
echo "Regenerating training text with conversational corpus..."
python "$SCRIPT_DIR/regenerate_training_text.py" \
  --text all \
  --text-output "$REPO_ROOT/conversational_corpora/combined_conversational.txt" \
  --social-graph "$REPO_ROOT/social_graph.json" \
  --training-text "$REPO_ROOT/training_text.txt" \
  --repeat 3

# Restart the app with the new training text
echo "Starting the app with the new training text..."
python "$REPO_ROOT/ppm/app_with_local_ppm.py"
