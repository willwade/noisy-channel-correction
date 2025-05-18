# PPM Training Scripts

This directory contains scripts for training the PPM (Prediction by Partial Matching) model with various text sources.

## Scripts

### `download_conversational_corpus.py`

Downloads modern conversational text from various sources for training the PPM model.

```bash
python download_conversational_corpus.py --corpus <corpus_id> --output <output_file>
```

Available corpora:
- `reddit_casual`: Reddit conversations dataset
- `subtitles`: OpenSubtitles sample (movie/TV dialogue)
- `daily_dialog`: Daily Dialog dataset (everyday conversations)
- `common_phrases`: Common phrases and expressions
- `twitter`: Twitter sample dataset
- `movie_dialogs`: Cornell Movie Dialogs sample
- `all`: Download and combine all available corpora

### `download_training_corpus.py`

Downloads books from Project Gutenberg for training the PPM model.

```bash
python download_training_corpus.py --book <book_id> --output <output_file>
```

Available books:
- `alice`: Alice's Adventures in Wonderland
- `sherlock`: The Adventures of Sherlock Holmes
- `frankenstein`: Frankenstein
- `pride`: Pride and Prejudice
- `jekyll`: Dr. Jekyll and Mr. Hyde
- `dracula`: Dracula
- `time_machine`: The Time Machine
- `war_worlds`: The War of the Worlds
- `dorian_gray`: The Picture of Dorian Gray
- `moby_dick`: Moby Dick

### `generate_training_text_from_social_graph.py`

Generates training text from a social graph JSON file.

```bash
python generate_training_text_from_social_graph.py --social-graph <social_graph_file> --output <output_file>
```

### `regenerate_training_text.py`

Regenerates the training text and restarts the app.

```bash
python regenerate_training_text.py --text <text_id> --social-graph <social_graph_file>
```

### `train_with_conversational_corpus.sh`

One-click script to train the PPM model with conversational corpus.

```bash
./train_with_conversational_corpus.sh
```

## Examples

### Train with Twitter data

```bash
python download_conversational_corpus.py --corpus twitter --output twitter.txt
python regenerate_training_text.py --text twitter --social-graph social_graph.json
```

### Train with all conversational corpora

```bash
python download_conversational_corpus.py --corpus all
python regenerate_training_text.py --text all --text-output conversational_corpora/combined_conversational.txt --social-graph social_graph.json
```

### Train with a book

```bash
python regenerate_training_text.py --text alice --use-books --social-graph social_graph.json
```
