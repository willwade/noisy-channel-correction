"""
Regenerate the training text and restart the app.

This script downloads a book, regenerates the training text,
and restarts the app with the new training text.
"""

import os
import argparse
import subprocess
import time
from typing import List, Optional


def download_training_text(
    text_id: str,
    output_file: str,
    max_lines: Optional[int] = None,
    use_conversational: bool = True,
) -> bool:
    """Download text for training.

    Args:
        text_id: ID of the text to download (book ID or corpus ID)
        output_file: Path to save the processed text
        max_lines: Maximum number of lines to include
        use_conversational: Whether to use conversational corpus (True) or books (False)

    Returns:
        True if successful, False otherwise
    """
    try:
        if use_conversational:
            cmd = [
                "python",
                "download_conversational_corpus.py",
                "--corpus",
                text_id,
                "--output",
                output_file,
            ]
        else:
            cmd = [
                "python",
                "download_training_corpus.py",
                "--book",
                text_id,
                "--output",
                output_file,
            ]

        if max_lines:
            cmd.extend(["--max-lines", str(max_lines)])

        print(
            f"Downloading {'conversational corpus' if use_conversational else 'book'} {text_id}..."
        )
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error downloading training text: {e}")
        print(e.stdout)
        print(e.stderr)
        return False


def regenerate_training_text(
    social_graph_file: str,
    output_file: str,
    additional_text_files: Optional[List[str]] = None,
    repeat_factor: int = 3,
    include_common_phrases: bool = True,
) -> bool:
    """Regenerate the training text.

    Args:
        social_graph_file: Path to the social graph file
        output_file: Path to save the training text
        additional_text_files: Additional text files to include
        repeat_factor: Repeat factor for important phrases
        include_common_phrases: Whether to include common English phrases

    Returns:
        True if successful, False otherwise
    """
    try:
        cmd = [
            "python",
            "generate_training_text_from_social_graph.py",
            "--social-graph",
            social_graph_file,
            "--output",
            output_file,
            "--repeat",
            str(repeat_factor),
        ]

        if additional_text_files:
            cmd.extend(["--additional-text"] + additional_text_files)

        if not include_common_phrases:
            cmd.append("--no-common-phrases")

        print("Regenerating training text...")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error regenerating training text: {e}")
        print(e.stdout)
        print(e.stderr)
        return False


def restart_app(app_script: str) -> None:
    """Restart the app.

    Args:
        app_script: Path to the app script
    """
    try:
        # Kill any running instances of the app
        subprocess.run(["pkill", "-f", app_script], check=False)

        # Wait for the app to shut down
        time.sleep(2)

        # Start the app
        print(f"Starting {app_script}...")
        subprocess.Popen(["python", app_script, "--regenerate-training"])
        print(f"App {app_script} started")
    except Exception as e:
        print(f"Error restarting app: {e}")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Regenerate training text and restart app"
    )
    parser.add_argument("--text", help="Text ID to download (book ID or corpus ID)")
    parser.add_argument(
        "--use-books",
        action="store_true",
        help="Use books instead of conversational corpus",
    )
    parser.add_argument(
        "--text-output",
        default="training_corpus.txt",
        help="Path to save the processed text",
    )
    parser.add_argument(
        "--max-lines", type=int, help="Maximum number of lines to include from the text"
    )
    parser.add_argument(
        "--social-graph",
        default="social_graph.json",
        help="Path to the social graph file",
    )
    parser.add_argument(
        "--training-text",
        default="training_text.txt",
        help="Path to save the training text",
    )
    parser.add_argument(
        "--additional-text", nargs="+", help="Additional text files to include"
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="Repeat factor for important phrases"
    )
    parser.add_argument(
        "--no-common-phrases",
        action="store_true",
        help="Don't include common English phrases",
    )
    parser.add_argument(
        "--app", default="app_with_local_ppm.py", help="Path to the app script"
    )
    parser.add_argument(
        "--no-restart", action="store_true", help="Don't restart the app"
    )

    # For backward compatibility
    parser.add_argument(
        "--book", help="Book ID to download (deprecated, use --text instead)"
    )
    parser.add_argument(
        "--book-output",
        default="book.txt",
        help="Path to save the processed book (deprecated)",
    )

    args = parser.parse_args()

    # Handle deprecated book argument
    if args.book and not args.text:
        args.text = args.book
        args.text_output = args.book_output
        args.use_books = True

    # Download training text if requested
    if args.text:
        if not download_training_text(
            args.text, args.text_output, args.max_lines, not args.use_books
        ):
            print("Failed to download training text. Exiting.")
            return

        # Add the text to the additional text files
        if args.additional_text:
            args.additional_text.append(args.text_output)
        else:
            args.additional_text = [args.text_output]

    # Regenerate the training text
    if not regenerate_training_text(
        args.social_graph,
        args.training_text,
        args.additional_text,
        args.repeat,
        not args.no_common_phrases,
    ):
        print("Failed to regenerate training text. Exiting.")
        return

    # Restart the app
    if not args.no_restart:
        restart_app(args.app)


if __name__ == "__main__":
    main()
