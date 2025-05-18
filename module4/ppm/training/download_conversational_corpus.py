"""
Download conversational text corpora for training the PPM model.

This script downloads modern conversational text from various sources
and processes it for use as training data for the PPM model.
"""

import os
import re
import argparse
import requests
import zipfile
import io
import json
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

# Sources of conversational text
CONVERSATIONAL_CORPORA = {
    # Reddit conversations dataset (small sample)
    "reddit_casual": "https://raw.githubusercontent.com/microsoft/botframework-cli/main/packages/qnamaker/test/fixtures/testfiles/QnA/reddit_casual_dataset.json",
    
    # OpenSubtitles sample (movie/TV dialogue)
    "subtitles": "https://raw.githubusercontent.com/jiweil/Neural-Dialogue-Generation/master/data/OpenSubData/test.txt",
    
    # Daily Dialog dataset (everyday conversations)
    "daily_dialog": "https://raw.githubusercontent.com/ictnlp/DialoFlow/main/data/dailydialog/test.txt",
    
    # Common phrases and expressions
    "common_phrases": "https://raw.githubusercontent.com/first20hours/google-10000-english/master/google-10000-english-usa-no-swears.txt",
    
    # Twitter sample dataset
    "twitter": "https://raw.githubusercontent.com/Phylliida/Dialogue-Datasets/master/TwitterLowerAsciiCorpus.txt",
    
    # Cornell Movie Dialogs sample
    "movie_dialogs": "https://raw.githubusercontent.com/Phylliida/Dialogue-Datasets/master/CornellMovieDialogueCorpusLowerAscii.txt",
}

def download_corpus(corpus_id: str, output_file: str) -> bool:
    """Download a conversational corpus.
    
    Args:
        corpus_id: ID of the corpus to download
        output_file: Path to save the corpus
        
    Returns:
        True if successful, False otherwise
    """
    if corpus_id not in CONVERSATIONAL_CORPORA:
        print(f"Error: Corpus ID '{corpus_id}' not found. Available corpora: {', '.join(CONVERSATIONAL_CORPORA.keys())}")
        return False
    
    url = CONVERSATIONAL_CORPORA[corpus_id]
    
    try:
        print(f"Downloading {corpus_id} from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the raw text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Corpus downloaded successfully to {output_file}")
        return True
    except Exception as e:
        print(f"Error downloading corpus: {e}")
        return False

def process_corpus(corpus_id: str, input_file: str, output_file: str, max_lines: Optional[int] = None) -> bool:
    """Process a corpus for use as training data.
    
    Args:
        corpus_id: ID of the corpus
        input_file: Path to the input file
        output_file: Path to save the processed corpus
        max_lines: Maximum number of lines to include
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Processing {input_file}...")
        
        # Read the corpus
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        sentences = []
        
        # Process based on corpus type
        if corpus_id == "reddit_casual":
            # Parse JSON and extract conversations
            data = json.loads(text)
            for item in data:
                if "answer" in item:
                    sentences.append(item["answer"])
                if "questions" in item:
                    for question in item["questions"]:
                        sentences.append(question)
        
        elif corpus_id == "subtitles" or corpus_id == "daily_dialog":
            # Simple line-by-line text
            sentences = [line.strip() for line in text.split('\n') if line.strip()]
        
        elif corpus_id == "common_phrases":
            # Word list - convert to simple sentences
            words = [line.strip() for line in text.split('\n') if line.strip()]
            # Create simple sentences from common words
            common_phrases = []
            for word in words[:1000]:  # Use top 1000 words
                common_phrases.append(f"I need {word}.")
                common_phrases.append(f"Can you get me {word}?")
                common_phrases.append(f"Where is the {word}?")
                common_phrases.append(f"I like {word}.")
            sentences = common_phrases
        
        elif corpus_id == "twitter" or corpus_id == "movie_dialogs":
            # Simple line-by-line dialogue
            sentences = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Filter out very short or very long sentences
        sentences = [s for s in sentences if 3 <= len(s) <= 200]
        
        # Remove duplicates
        sentences = list(set(sentences))
        
        # Shuffle to mix different types of sentences
        random.shuffle(sentences)
        
        # Limit the number of sentences if requested
        if max_lines and len(sentences) > max_lines:
            print(f"Limiting to {max_lines} sentences (out of {len(sentences)})")
            sentences = sentences[:max_lines]
        
        # Save the processed text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sentences))
        
        print(f"Corpus processed successfully: {len(sentences)} sentences saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error processing corpus: {e}")
        return False

def download_and_process_corpus(corpus_id: str, output_file: str, max_lines: Optional[int] = None) -> bool:
    """Download and process a conversational corpus.
    
    Args:
        corpus_id: ID of the corpus to download
        output_file: Path to save the processed corpus
        max_lines: Maximum number of lines to include
        
    Returns:
        True if successful, False otherwise
    """
    # Create a temporary file for the raw corpus
    raw_file = f"{output_file}.raw"
    
    # Download the corpus
    if not download_corpus(corpus_id, raw_file):
        return False
    
    # Process the corpus
    success = process_corpus(corpus_id, raw_file, output_file, max_lines)
    
    # Remove the temporary file
    try:
        os.remove(raw_file)
    except Exception:
        pass
    
    return success

def download_all_corpora(output_dir: str, max_lines_per_corpus: Optional[int] = 1000) -> bool:
    """Download and process all available corpora.
    
    Args:
        output_dir: Directory to save the processed corpora
        max_lines_per_corpus: Maximum number of lines to include per corpus
        
    Returns:
        True if all downloads were successful, False otherwise
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    all_successful = True
    combined_sentences = []
    
    # Download and process each corpus
    for corpus_id in CONVERSATIONAL_CORPORA:
        output_file = os.path.join(output_dir, f"{corpus_id}.txt")
        success = download_and_process_corpus(corpus_id, output_file, max_lines_per_corpus)
        
        if success:
            # Read the processed corpus
            with open(output_file, 'r', encoding='utf-8') as f:
                sentences = f.read().split('\n')
                combined_sentences.extend(sentences)
        else:
            all_successful = False
    
    # Create a combined corpus
    combined_file = os.path.join(output_dir, "combined_conversational.txt")
    with open(combined_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(combined_sentences))
    
    print(f"Combined corpus created with {len(combined_sentences)} sentences: {combined_file}")
    
    return all_successful

def list_available_corpora() -> None:
    """List all available conversational corpora."""
    print("Available conversational corpora:")
    for corpus_id, url in CONVERSATIONAL_CORPORA.items():
        print(f"  {corpus_id}: {url}")
    print("\nSpecial options:")
    print("  all: Download and combine all available corpora")

def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Download conversational text for training the PPM model")
    parser.add_argument("--corpus", help="Corpus ID to download or 'all' for all corpora")
    parser.add_argument("--output", default="conversational.txt", help="Path to save the processed corpus")
    parser.add_argument("--max-lines", type=int, default=1000, help="Maximum number of lines to include per corpus")
    parser.add_argument("--list", action="store_true", help="List available corpora")
    args = parser.parse_args()
    
    if args.list:
        list_available_corpora()
        return
    
    if not args.corpus:
        parser.error("Please specify a corpus ID with --corpus")
    
    if args.corpus == "all":
        # Create an output directory
        output_dir = "conversational_corpora"
        download_all_corpora(output_dir, args.max_lines)
    else:
        download_and_process_corpus(args.corpus, args.output, args.max_lines)

if __name__ == "__main__":
    main()
