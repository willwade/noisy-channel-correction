# Scripts for Noisy Channel Correction

This directory contains scripts for various tasks related to the noisy channel correction project.

## Data Scripts

### `data/enhance_en_gb_lexicon.py`

This script enhances the en-GB lexicon by merging it with a standard English dictionary. It downloads a comprehensive English dictionary and merges it with the existing en-GB AAC lexicon to create a more complete lexicon for correction.

Usage:
```bash
python scripts/data/enhance_en_gb_lexicon.py
```

Options:
- `--aac-lexicon`: Path to the AAC lexicon file (default: `data/aac_lexicon_en_gb.txt`)
- `--dictionary-url`: URL to download the English dictionary from (default: `https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt`)
- `--output`: Path to save the enhanced lexicon to (default: `data/enhanced_lexicon_en_gb.txt`)

## Candidate Generator Scripts

### `candidate_generator/optimize_settings.py`

This script creates a configuration file with optimized settings for the ImprovedCandidateGenerator to improve performance and accuracy.

Usage:
```bash
python scripts/candidate_generator/optimize_settings.py
```

Options:
- `--output`: Path to save the configuration file to (default: `config/candidate_generator_settings.json`)

### `candidate_generator/apply_optimized_settings.py`

This script modifies the ImprovedCandidateGenerator class to use the optimized settings from the configuration file.

Usage:
```bash
python scripts/candidate_generator/apply_optimized_settings.py
```

Options:
- `--config`: Path to the configuration file (default: `config/candidate_generator_settings.json`)
- `--file`: Path to the ImprovedCandidateGenerator file (default: `lib/candidate_generator/improved_candidate_generator.py`)

## Corrector Scripts

### `corrector/update_corrector.py`

This script modifies the NoisyChannelCorrector class to use the enhanced lexicon and optimized candidate generator settings.

Usage:
```bash
python scripts/corrector/update_corrector.py
```

Options:
- `--file`: Path to the NoisyChannelCorrector file (default: `lib/corrector/corrector.py`)

## Training Scripts

### `training/train_en_gb_models.py`

This script trains PPM and word n-gram models on a larger corpus of British English text.

Usage:
```bash
python scripts/training/train_en_gb_models.py
```

Options:
- `--corpus-url`: URL to download the corpus from (default: `https://www.gutenberg.org/files/10/10-0.zip`)
- `--corpus-dir`: Directory to save the corpus to (default: `data/corpus/en_gb`)
- `--ppm-output`: Path to save the PPM model to (default: `models/ppm_model_en_gb.pkl`)
- `--word-ngram-output`: Path to save the word n-gram model to (default: `models/word_ngram_model_en_gb.pkl`)

## Improvements Made

1. **Enhanced Lexicon**: The en-GB lexicon has been enhanced by merging it with a comprehensive English dictionary, increasing the vocabulary from 2,180 words to over 370,000 words.

2. **Optimized Candidate Generator**: The ImprovedCandidateGenerator has been optimized with better settings to improve performance and accuracy:
   - Reduced the maximum number of edits from 20,000 to 200
   - Increased the keyboard boost from 0.3 to 0.5
   - Enabled strict filtering and smart filtering
   - Reduced the timeout from 5.0 to 2.0 seconds
   - Added early stopping to avoid generating too many candidates

3. **Updated Corrector**: The NoisyChannelCorrector has been updated to use the enhanced lexicon and optimized candidate generator settings.

4. **Better Language Models**: Scripts have been provided to train better language models on a larger corpus of British English text.

These improvements have significantly enhanced the correction accuracy and performance of the system, particularly for en-GB language support.
