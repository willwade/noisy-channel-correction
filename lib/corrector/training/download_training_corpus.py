"""
Download a small book or corpus for training the PPM model.

This script downloads a small book or corpus from Project Gutenberg
and processes it for use as training data for the PPM model.
"""

import os
import re
import argparse
import requests
from typing import Optional

# Project Gutenberg URLs for some small books
GUTENBERG_BOOKS = {
    "alice": "https://www.gutenberg.org/files/11/11-0.txt",  # Alice's Adventures in Wonderland
    "sherlock": "https://www.gutenberg.org/files/1661/1661-0.txt",  # The Adventures of Sherlock Holmes
    "frankenstein": "https://www.gutenberg.org/files/84/84-0.txt",  # Frankenstein
    "pride": "https://www.gutenberg.org/files/1342/1342-0.txt",  # Pride and Prejudice
    "jekyll": "https://www.gutenberg.org/files/43/43-0.txt",  # Dr. Jekyll and Mr. Hyde
    "dracula": "https://www.gutenberg.org/files/345/345-0.txt",  # Dracula
    "time_machine": "https://www.gutenberg.org/files/35/35-0.txt",  # The Time Machine
    "war_worlds": "https://www.gutenberg.org/files/36/36-0.txt",  # The War of the Worlds
    "dorian_gray": "https://www.gutenberg.org/files/174/174-0.txt",  # The Picture of Dorian Gray
    "moby_dick": "https://www.gutenberg.org/files/2701/2701-0.txt",  # Moby Dick
}

def download_book(book_id: str, output_file: str) -> bool:
    """Download a book from Project Gutenberg.
    
    Args:
        book_id: ID of the book to download
        output_file: Path to save the book
        
    Returns:
        True if successful, False otherwise
    """
    if book_id not in GUTENBERG_BOOKS:
        print(f"Error: Book ID '{book_id}' not found. Available books: {', '.join(GUTENBERG_BOOKS.keys())}")
        return False
    
    url = GUTENBERG_BOOKS[book_id]
    
    try:
        print(f"Downloading {book_id} from {url}...")
        response = requests.get(url)
        response.raise_for_status()
        
        # Save the raw text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(response.text)
        
        print(f"Book downloaded successfully to {output_file}")
        return True
    except Exception as e:
        print(f"Error downloading book: {e}")
        return False

def process_book(input_file: str, output_file: str, max_lines: Optional[int] = None) -> bool:
    """Process a book for use as training data.
    
    Args:
        input_file: Path to the input file
        output_file: Path to save the processed book
        max_lines: Maximum number of lines to include
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"Processing {input_file}...")
        
        # Read the book
        with open(input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Remove Project Gutenberg header and footer
        text = re.sub(r'(?s)^.*?\*\*\* START OF (THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*', '', text)
        text = re.sub(r'(?s)\*\*\* END OF (THIS|THE) PROJECT GUTENBERG EBOOK.*$', '', text)
        
        # Split into sentences
        sentences = []
        for paragraph in text.split('\n\n'):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Split paragraph into sentences
            for sentence in re.split(r'(?<=[.!?])\s+', paragraph):
                sentence = sentence.strip()
                if len(sentence) > 10:  # Only include sentences of reasonable length
                    sentences.append(sentence)
        
        # Limit the number of sentences if requested
        if max_lines and len(sentences) > max_lines:
            print(f"Limiting to {max_lines} sentences (out of {len(sentences)})")
            sentences = sentences[:max_lines]
        
        # Save the processed text
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sentences))
        
        print(f"Book processed successfully: {len(sentences)} sentences saved to {output_file}")
        return True
    except Exception as e:
        print(f"Error processing book: {e}")
        return False

def download_and_process_book(book_id: str, output_file: str, max_lines: Optional[int] = None) -> bool:
    """Download and process a book from Project Gutenberg.
    
    Args:
        book_id: ID of the book to download
        output_file: Path to save the processed book
        max_lines: Maximum number of lines to include
        
    Returns:
        True if successful, False otherwise
    """
    # Create a temporary file for the raw book
    raw_file = f"{output_file}.raw"
    
    # Download the book
    if not download_book(book_id, raw_file):
        return False
    
    # Process the book
    success = process_book(raw_file, output_file, max_lines)
    
    # Remove the temporary file
    try:
        os.remove(raw_file)
    except Exception:
        pass
    
    return success

def list_available_books() -> None:
    """List all available books."""
    print("Available books:")
    for book_id, url in GUTENBERG_BOOKS.items():
        print(f"  {book_id}: {url}")

def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(description="Download a book for training the PPM model")
    parser.add_argument("--book", choices=list(GUTENBERG_BOOKS.keys()), help="Book ID to download")
    parser.add_argument("--output", default="book.txt", help="Path to save the processed book")
    parser.add_argument("--max-lines", type=int, help="Maximum number of lines to include")
    parser.add_argument("--list", action="store_true", help="List available books")
    args = parser.parse_args()
    
    if args.list:
        list_available_books()
        return
    
    if not args.book:
        parser.error("Please specify a book ID with --book")
    
    download_and_process_book(args.book, args.output, args.max_lines)

if __name__ == "__main__":
    main()
