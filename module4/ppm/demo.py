"""
Demo script for the enhanced PPM predictor.
"""

import os
import argparse
import logging
import gradio as gr
from enhanced_ppm_predictor import EnhancedPPMPredictor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Enhanced PPM Predictor Demo")
    parser.add_argument("--training-text", default="../training_text.txt", help="Path to the training text file")
    parser.add_argument("--model-path", default="enhanced_ppm_model.pkl", help="Path to save/load the model")
    parser.add_argument("--max-order", type=int, default=5, help="Maximum context length to consider")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    args = parser.parse_args()
    
    # Create the predictor
    predictor = EnhancedPPMPredictor(max_order=args.max_order, debug=args.debug)
    
    # Load the model if it exists
    if os.path.exists(args.model_path):
        logger.info(f"Loading model from {args.model_path}")
        predictor.load_model(args.model_path)
    # Otherwise, train the model if the training text exists
    elif os.path.exists(args.training_text):
        logger.info(f"Training model with {args.training_text}")
        with open(args.training_text, "r") as f:
            training_text = f.read()
        predictor.train_on_text(training_text, save_model=True, model_path=args.model_path)
    else:
        logger.warning(f"No model or training text found. Using empty model.")
    
    # Create the Gradio interface
    with gr.Blocks(title="Enhanced PPM Predictor Demo") as demo:
        gr.Markdown("# Enhanced PPM Predictor Demo")
        gr.Markdown("Type text in the input box and see the predictions update in real-time.")
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Type here...",
                    lines=3
                )
                
                update_btn = gr.Button("Update Model")
                
                feedback = gr.Textbox(
                    label="Feedback",
                    interactive=False
                )
            
            with gr.Column():
                with gr.Row():
                    next_chars = gr.Dataframe(
                        headers=["Next Character"],
                        datatype=["str"],
                        col_count=1,
                        row_count=5,
                        label="Next Characters"
                    )
                    
                    word_completions = gr.Dataframe(
                        headers=["Word Completion"],
                        datatype=["str"],
                        col_count=1,
                        row_count=5,
                        label="Word Completions"
                    )
                
                next_words = gr.Dataframe(
                    headers=["Next Word"],
                    datatype=["str"],
                    col_count=1,
                    row_count=5,
                    label="Next Words"
                )
        
        # Define the update function
        def update_predictions(text):
            if not text:
                return [[""] for _ in range(5)], [[""] for _ in range(5)], [[""] for _ in range(5)]
            
            predictions = predictor.get_all_predictions(text)
            
            # Format for Gradio Dataframe
            char_preds = [[char] for char in predictions["next_chars"]]
            completion_preds = [[word] for word in predictions["word_completions"]]
            word_preds = [[word] for word in predictions["next_words"]]
            
            # Ensure we have exactly 5 rows for each prediction type
            while len(char_preds) < 5:
                char_preds.append([""])
            while len(completion_preds) < 5:
                completion_preds.append([""])
            while len(word_preds) < 5:
                word_preds.append([""])
            
            return char_preds, completion_preds, word_preds
        
        # Define the update model function
        def update_model(text):
            if not text:
                return "No text provided. Model not updated."
            
            predictor.update_model(text, save_model=True, model_path=args.model_path)
            return f"Model updated with {len(text)} characters and saved to {args.model_path}"
        
        # Set up event handlers
        input_text.change(
            update_predictions,
            inputs=[input_text],
            outputs=[next_chars, word_completions, next_words]
        )
        
        update_btn.click(
            update_model,
            inputs=[input_text],
            outputs=[feedback]
        )
    
    # Launch the demo
    demo.launch()

if __name__ == "__main__":
    main()
