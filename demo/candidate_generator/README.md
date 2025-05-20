# Candidate Generator Demo

This directory contains demo applications for the Candidate Generator module (formerly Module 3).

The Candidate Generator module is responsible for:
- Generating plausible correction candidates for noisy input
- Using edit distance and keyboard adjacency to find candidates
- Filtering candidates using a lexicon

## Usage

```bash
uv run demo/candidate_generator/generate_candidates.py --input "hella" --lexicon data/keyboard_lexicons_en_gb/qwerty_lexicon.txt --word-frequencies data/word_frequencies_en_gb.txt


uv run demo/candidate_generator/generate_candidates.py --input "hella world" --lexicon data/keyboard_lexicons_en_gb/qwerty_lexicon.txt --word-frequencies data/word_frequencies_en_gb.txt
```
NB: Use the correct lexicon for the correct language!

This will output the following:

```bash
Candidates for 'hella world':
----------------------------------------
1. hello world (score: 1.5000)
```