"""
Example usage of the enhanced PPM predictor.
"""

import os
from enhanced_ppm_predictor import EnhancedPPMPredictor

def main():
    """Main function."""
    # Create the predictor
    predictor = EnhancedPPMPredictor(max_order=5)
    
    # Training text
    training_text = """
    Hello, how are you today? I'm doing well, thank you for asking.
    The weather is nice today. I think I'll go for a walk in the park.
    I like to read books and watch movies in my free time.
    What are your hobbies? Do you like to travel?
    I've been to many countries, but I still have many places I want to visit.
    """
    
    # Train the model
    print("Training the model...")
    predictor.train_on_text(training_text)
    
    # Get predictions
    text = "Hello, how are"
    print(f"\nPredictions for: '{text}'")
    
    # Next characters
    next_chars = predictor.predict_next_chars(text)
    print(f"Next characters: {next_chars}")
    
    # Word completions (for the last word)
    current_word = text.split()[-1] if text.split() else ""
    word_completions = predictor.predict_word_completions(current_word)
    print(f"Word completions for '{current_word}': {word_completions}")
    
    # Next words
    next_words = predictor.predict_next_words(text)
    print(f"Next words: {next_words}")
    
    # Update the model with new text
    new_text = "I'm feeling great today. The sun is shining and the birds are singing."
    print(f"\nUpdating the model with new text: '{new_text}'")
    predictor.update_model(new_text)
    
    # Get predictions again
    print(f"\nPredictions after update for: '{text}'")
    
    # Next characters
    next_chars = predictor.predict_next_chars(text)
    print(f"Next characters: {next_chars}")
    
    # Word completions (for the last word)
    word_completions = predictor.predict_word_completions(current_word)
    print(f"Word completions for '{current_word}': {word_completions}")
    
    # Next words
    next_words = predictor.predict_next_words(text)
    print(f"Next words: {next_words}")
    
    # Save the model
    model_path = "example_model.pkl"
    print(f"\nSaving the model to {model_path}")
    predictor.train_on_text(training_text + new_text, save_model=True, model_path=model_path)
    
    # Load the model
    print(f"\nLoading the model from {model_path}")
    new_predictor = EnhancedPPMPredictor()
    new_predictor.load_model(model_path)
    
    # Get predictions with the loaded model
    print(f"\nPredictions with loaded model for: '{text}'")
    
    # Next characters
    next_chars = new_predictor.predict_next_chars(text)
    print(f"Next characters: {next_chars}")
    
    # Word completions (for the last word)
    word_completions = new_predictor.predict_word_completions(current_word)
    print(f"Word completions for '{current_word}': {word_completions}")
    
    # Next words
    next_words = new_predictor.predict_next_words(text)
    print(f"Next words: {next_words}")
    
    # Clean up
    if os.path.exists(model_path):
        os.remove(model_path)

if __name__ == "__main__":
    main()
