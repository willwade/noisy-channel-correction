#!/usr/bin/env python3
"""
Train en-GB specific models for the noisy channel correction project.

This script trains en-GB specific PPM and word n-gram models and sets up
the necessary files for en-GB specific correction.
"""

import os
import sys
import argparse
import logging
import shutil

# Add the parent directory to the Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, "confusion_matrix_builder"))

# Import the model training functions and conversion function
from train_ppm_model import train_ppm_model, train_word_ngram_model  # noqa: E402
from convert_keyboard_to_standard_matrix import convert_keyboard_to_standard  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_en_gb_training_text():
    """
    Generate a sample en-GB specific training text.

    This function creates a sample text with British English spelling
    and vocabulary for training the models.

    Returns:
        Sample text with British English content
    """
    return """
    Hello, I'm using British English for this text. Let me tell you about my day.
    I went to the centre of town this morning to buy some groceries. The colour of the sky 
    was grey, and I thought it might rain, so I brought my umbrella.
    I stopped by the petrol station to fill up my car, then went to the shopping centre. 
    I bought some biscuits, crisps, and fizzy drinks for the weekend.
    When I got home, I realised I had forgotten to buy some courgettes and aubergines for 
    the dinner I was planning to cook.
    I decided to make a cup of tea and watch some telly instead. There was a programme 
    about travelling through Europe that I quite fancied.
    Later, I went to the pub with my mates for a pint. We had a lovely evening chatting 
    about football and other sports.
    The next day, I need to go to the GP for a check-up, then to the chemist to pick up 
    my prescription.
    I also need to post some letters at the post office and buy some new trousers.
    It's been quite a busy week, but I'm looking forward to the weekend when I can have 
    a lie-in and maybe go for a ramble in the countryside.
    The weather forecast says it will be sunny, which would be marvellous.
    I hope you've enjoyed this little glimpse into my day using proper British English 
    spelling and vocabulary.
    Cheers!

    I want to go to the theatre tonight to see a play. The dialogue is supposed to be brilliant.
    My flat is on the ground floor of a block of flats near the city centre.
    I need to buy some plasters and paracetamol from the chemist.
    The lorry was parked on the pavement, blocking the zebra crossing.
    I'm going on holiday to Spain next month. I've already packed my swimming costume.
    Would you like a biscuit with your tea? I've got some lovely chocolate digestives.
    The lift in our building is broken, so we have to use the stairs.
    I need to put some petrol in my car before we set off.
    The queue at the post office was very long this morning.
    I've got a maths exam tomorrow, so I need to revise tonight.
    The rubbish bin is full, so I need to take the rubbish out.
    I'm going to watch the football match on the telly tonight.
    I need to buy some washing-up liquid and kitchen roll from the supermarket.
    The children are playing in the garden with their new toys.
    I'm going to have a bath and then go to bed.
    """


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train en-GB specific models for the noisy channel correction project."
    )

    parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Path to the input text file with en-GB specific content",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="models",
        help="Directory to save the models (default: models)",
    )

    parser.add_argument(
        "--ppm-order",
        type=int,
        default=5,
        help="Maximum context length for PPM model (default: 5)",
    )

    parser.add_argument(
        "--ngram-order",
        type=int,
        default=3,
        help="Maximum n-gram order for word n-gram model (default: 3)",
    )

    parser.add_argument(
        "--smoothing",
        type=str,
        default="kneser_ney",
        choices=["kneser_ney", "laplace", "witten_bell"],
        help="Smoothing method for word n-gram model (default: kneser_ney)",
    )

    args = parser.parse_args()

    # Create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Define output file paths with en_gb suffix
    ppm_output = os.path.join(args.output_dir, "ppm_model_en_gb.pkl")
    ngram_output = os.path.join(args.output_dir, "word_ngram_model_en_gb.pkl")
    confusion_matrix_output = os.path.join(args.output_dir, "confusion_matrix_en_gb.json")

    # Use sample text if no input file is provided
    input_text = args.input
    if not input_text:
        logger.info("No input file provided. Using sample en-GB text.")
        sample_text_path = os.path.join(args.output_dir, "sample_en_gb_training_text.txt")
        with open(sample_text_path, "w", encoding="utf-8") as f:
            f.write(generate_en_gb_training_text())
        input_text = sample_text_path

    # Train the PPM model
    logger.info(f"Training en-GB PPM model with order {args.ppm_order}...")
    ppm_success = train_ppm_model(input_text, ppm_output, args.ppm_order)

    # Train the word n-gram model
    logger.info(f"Training en-GB word n-gram model with order {args.ngram_order}...")
    ngram_success = train_word_ngram_model(
        input_text, ngram_output, args.ngram_order, args.smoothing
    )

    # Convert the en-GB specific keyboard confusion matrix if it exists
    keyboard_matrix_src = os.path.join(
        args.output_dir, "keyboard_confusion_matrices_en_gb.json"
    )
    if os.path.exists(keyboard_matrix_src):
        # Create a standard confusion matrix from the keyboard matrix
        logger.info(f"Using en-GB keyboard matrices from {keyboard_matrix_src}")
        # Convert the keyboard matrix to a standard confusion matrix
        conversion_success = convert_keyboard_to_standard(
            keyboard_matrix_src, confusion_matrix_output, "qwerty"
        )
        if conversion_success:
            logger.info(f"Converted en-GB matrix to {confusion_matrix_output}")
        else:
            logger.error("Failed to convert en-GB confusion matrix")
    else:
        logger.warning(
            f"en-GB keyboard matrices not found: {keyboard_matrix_src}"
        )
        logger.warning("Using default confusion matrix")
        # Copy the default confusion matrix
        default_matrix = os.path.join(args.output_dir, "confusion_matrix.json")
        if os.path.exists(default_matrix):
            shutil.copy(default_matrix, confusion_matrix_output)
            logger.info(f"Copied default matrix to {confusion_matrix_output}")
        else:
            logger.error(f"Default matrix not found: {default_matrix}")

    # Print summary
    print("\n=== en-GB Model Training Summary ===")
    if ppm_success:
        print(f"✓ PPM model trained and saved to {ppm_output}")
    else:
        print("✗ Failed to train PPM model")

    if ngram_success:
        print(f"✓ Word n-gram model trained and saved to {ngram_output}")
    else:
        print("✗ Failed to train word n-gram model")

    if os.path.exists(confusion_matrix_output):
        print(f"✓ Confusion matrix saved to {confusion_matrix_output}")
    else:
        print("✗ Failed to create confusion matrix")

    print("\nTo use these models with the correction engine, run:")
    print(
        f"uv run demo/correction_engine/demo.py --language en-GB "
        f"--ppm-model {ppm_output} "
        f"--word-ngram-model {ngram_output} "
        f"--confusion-matrix {confusion_matrix_output}"
    )


if __name__ == "__main__":
    main()
