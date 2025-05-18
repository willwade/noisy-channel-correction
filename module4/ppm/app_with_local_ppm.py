"""
Enhanced version of the AAC app with advanced PPM predictive text features.

This version uses the EnhancedPPMPredictor which includes:
1. N-gram backoff for better handling of unseen contexts
2. Interpolation for smoother probability estimates
3. Word embeddings for semantic similarity
4. Recency bias to prioritize recently used words
5. Context-aware predictions based on sentence structure
"""

import gradio as gr
import whisper
import random
import time
import os
import subprocess
import warnings
import logging
import argparse

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path to allow imports
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Now import the modules
from utils import SocialGraphManager
from llm_interface import LLMInterface
from ppm.enhanced_ppm_predictor import EnhancedPPMPredictor
from ppm.local_ppm_predictive_text import LocalPPMPredictiveTextInput
from ppm.training.generate_training_text_from_social_graph import generate_training_text
from ppm.continuous_learning import ContinuousLearningManager

# Define available models - using only the ones specified by the user
AVAILABLE_MODELS = {
    # Gemini models (online API)
    "gemini-1.5-flash-8b-latest": "🌐 Gemini 1.5 Flash 8B (Online API - Fast, Cheapest)",
    "gemini-2.0-flash": "🌐 Gemini 2.0 Flash (Online API - Better quality)",
    "gemma-3-27b-it": "🌐 Gemma 3 27B-IT (Online API - High quality)",
}

# Initialize the social graph manager
social_graph = SocialGraphManager("social_graph.json")

# Check if we're running on Hugging Face Spaces
is_huggingface_spaces = "SPACE_ID" in os.environ

# Print environment info for debugging
logger.info(f"Running on Hugging Face Spaces: {is_huggingface_spaces}")
logger.info(
    f"GEMINI_API_KEY set: {'Yes' if os.environ.get('GEMINI_API_KEY') else 'No'}"
)
logger.info(f"HF_TOKEN set: {'Yes' if os.environ.get('HF_TOKEN') else 'No'}")

# Try to run the setup script if we're on Hugging Face Spaces
if is_huggingface_spaces:
    try:
        logger.info("Running setup script...")
        subprocess.run(["bash", "setup.sh"], check=True)
        logger.info("Setup script completed successfully")
    except Exception as e:
        logger.error(f"Error running setup script: {e}")

# Check if LLM tool is installed
llm_installed = False
try:
    result = subprocess.run(
        ["llm", "--version"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode == 0:
        logger.info(f"LLM tool is installed: {result.stdout.strip()}")
        llm_installed = True
    else:
        logger.warning("LLM tool returned an error.")
except Exception as e:
    logger.warning(f"LLM tool not available: {e}")

# Initialize the suggestion generator
if llm_installed:
    logger.info("Initializing with Gemini 1.5 Flash 8B (online model via LLM tool)")
    suggestion_generator = LLMInterface("gemini-1.5-flash-8b-latest")
    use_llm_interface = True
else:
    logger.info(
        "LLM tool not available, falling back to direct Hugging Face implementation"
    )
    from utils import SuggestionGenerator

    suggestion_generator = SuggestionGenerator("google/gemma-3-1b-it")
    use_llm_interface = False

# Test the model to make sure it's working
logger.info("Testing model connection...")
test_result = suggestion_generator.test_model()
logger.info(f"Model test result: {test_result}")

# If the model didn't load, try Ollama as fallback
if not suggestion_generator.model_loaded:
    logger.info("Online model not available, trying Ollama model...")
    suggestion_generator = LLMInterface("ollama/gemma:7b")
    test_result = suggestion_generator.test_model()
    logger.info(f"Ollama model test result: {test_result}")

    # If Ollama also fails, try OpenAI as fallback
    if not suggestion_generator.model_loaded:
        logger.info("Ollama not available, trying OpenAI model...")
        suggestion_generator = LLMInterface("gpt-3.5-turbo")
        test_result = suggestion_generator.test_model()
        logger.info(f"OpenAI model test result: {test_result}")

# Test the model to make sure it's working
test_result = suggestion_generator.test_model()
logger.info(f"Model test result: {test_result}")

# If the model didn't load, use the fallback responses
if not suggestion_generator.model_loaded:
    logger.warning("Model failed to load, using fallback responses...")
    # The SuggestionGenerator class has built-in fallback responses

# Initialize Whisper model (using the smallest model for speed)
try:
    whisper_model = whisper.load_model("tiny")
    whisper_loaded = True
except Exception as e:
    logger.error(f"Error loading Whisper model: {e}")
    whisper_loaded = False

# Generate training text for PPM if it doesn't exist
training_text_path = "training_text.txt"
if not os.path.exists(training_text_path):
    logger.info("Generating training text from social graph")
    generate_training_text(social_graph.graph, training_text_path)

# Initialize the Enhanced PPM predictor
ppm_predictor = EnhancedPPMPredictor(max_order=5)

# Initialize the continuous learning manager
continuous_learning_manager = ContinuousLearningManager(
    ppm_predictor=ppm_predictor,
    social_graph=social_graph.graph,
    model_path="enhanced_ppm_model.pkl",
    context_weight=2.0,  # Weight for contextual data (higher = more influence)
    recency_weight=1.5,  # Weight for recent interactions (higher = more influence)
)

# Initialize the predictive text component with the continuous learning manager
ppm_predictive_text = LocalPPMPredictiveTextInput(
    ppm_predictor=ppm_predictor,
    training_text_path=training_text_path,
    model_path="enhanced_ppm_model.pkl",  # Use a different model path for the enhanced model
    continuous_learning_manager=continuous_learning_manager,  # Pass the continuous learning manager
)

# Import existing functions from app.py
try:
    from app import (
        get_people_choices,
        get_topics_for_person,
        get_suggestion_categories,
        on_person_change,
        change_model,
        generate_suggestions,
        transcribe_audio,
        save_conversation,
    )
except ImportError:
    # If that fails, try with the full path
    logger.info("Trying to import app.py with full path...")
    import importlib.util
    import os

    app_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
    )
    spec = importlib.util.spec_from_file_location("app", app_path)
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)

    # Get the functions from the imported module
    get_people_choices = app.get_people_choices
    get_topics_for_person = app.get_topics_for_person
    get_suggestion_categories = app.get_suggestion_categories
    on_person_change = app.on_person_change
    change_model = app.change_model
    generate_suggestions = app.generate_suggestions
    transcribe_audio = app.transcribe_audio
    save_conversation = app.save_conversation

# Create the Gradio interface
with gr.Blocks(title="Will's AAC Communication Aid", css="custom.css") as demo:
    gr.Markdown("# Will's AAC Communication Aid (with Enhanced PPM Prediction)")
    gr.Markdown(
        """
    This demo simulates an AAC system from Will's perspective (a 38-year-old with MND). Its based on a social graph of people in Will's life and their common phrases. The idea is that this graph is generated on device securely. You can see this [here](https://github.com/willwade/skg-llm-mvp/blob/main/social_graph.json)

    **How to use this demo:**
    1. Select who you (Will) are talking to from the dropdown
    2. Optionally select a conversation topic
    3. Enter or record what the other person said to you
    4. Get suggested responses based on your relationship with that person
    5. Use the predictive text input to compose your own response with word completion
    """
    )

    # Display information about Will
    with gr.Accordion("About Me (Will)", open=False):
        gr.Markdown(
            """
        I'm Will, a 38-year-old computer programmer from Manchester with MND (diagnosed 5 months ago).
        I live with my wife Emma and two children (Mabel, 4 and Billy, 7).
        Originally from South East London, I enjoy technology, Manchester United, and have fond memories of cycling and hiking.
        I'm increasingly using this AAC system as my speech becomes more difficult.
        """
        )

    with gr.Row():
        with gr.Column(scale=1):
            # Person selection
            person_dropdown = gr.Dropdown(
                choices=get_people_choices(),
                label="I'm talking to:",
                info="Select who you (Will) are talking to",
            )

            # Topic selection dropdown
            topic_dropdown = gr.Dropdown(
                choices=[],  # Will be populated when a person is selected
                label="Topic (optional):",
                info="Select a topic to discuss or respond about",
                allow_custom_value=True,
            )

            # Context display
            context_display = gr.Markdown(label="Relationship Context")

            # User input section
            with gr.Row():
                user_input = gr.Textbox(
                    label="What they said to me: (leave empty to start a conversation)",
                    placeholder='Examples:\n"How was your physio session today?"\n"The kids are asking if you want to watch a movie tonight"\n"I\'ve been looking at that new AAC software you mentioned"',
                    lines=3,
                )

            # Audio input with auto-transcription
            with gr.Column(elem_classes="audio-recorder-container"):
                gr.Markdown("### 🎤 Or record what they said")
                audio_input = gr.Audio(
                    label="",
                    type="filepath",
                    sources=["microphone"],
                    elem_classes="audio-recorder",
                )
                gr.Markdown(
                    "*Recording will auto-transcribe when stopped*",
                    elem_classes="auto-transcribe-hint",
                )

            # Suggestion type selection with emojis
            suggestion_type = gr.Radio(
                choices=[
                    "🤖 model",
                    "🔍 auto_detect",
                    "💬 common_phrases",
                ]
                + get_suggestion_categories(),
                value="🤖 model",  # Default to model for better results
                label="How should I respond?",
                info="Choose response type",
                elem_classes="emoji-response-options",
            )

            # Add a mood slider with emoji indicators at the ends
            with gr.Column(elem_classes="mood-slider-container"):
                mood_slider = gr.Slider(
                    minimum=1,
                    maximum=5,
                    value=3,
                    step=1,
                    label="How am I feeling today?",
                    info="This will influence the tone of your responses (😢 Sad → Happy 😄)",
                    elem_classes="mood-slider",
                )

            # Model selection
            with gr.Row():
                model_dropdown = gr.Dropdown(
                    choices=list(AVAILABLE_MODELS.keys()),
                    value="gemini-1.5-flash-8b-latest",
                    label="Language Model",
                    info="Select which AI model to use (all are online API models)",
                )

                temperature_slider = gr.Slider(
                    minimum=0.1,
                    maximum=1.5,
                    value=0.7,
                    step=0.1,
                    label="Temperature",
                    info="Controls randomness (higher = more creative, lower = more focused)",
                )

            # Generate button
            generate_btn = gr.Button(
                "Generate My Responses/Conversation Starters", variant="primary"
            )

            # Model status
            model_status = gr.Markdown(
                value=f"Current model: {suggestion_generator.model_name}",
                label="Model Status",
            )

        with gr.Column(scale=1):
            # Common phrases
            common_phrases = gr.Textbox(
                label="My Common Phrases",
                placeholder="Common phrases I often use with this person will appear here...",
                lines=5,
            )

            # Conversation history display
            conversation_history = gr.Markdown(
                label="Recent Conversations",
                value="Select a person to see recent conversations...",
                elem_id="conversation_history",
            )

            # Suggestions output
            suggestions_output = gr.Markdown(
                label="My Suggested Responses",
                value="Suggested responses will appear here...",
                elem_id="suggestions_output",  # Add an ID for easier debugging
            )

            # Add buttons to select and use a specific response
            with gr.Row():
                use_response_1 = gr.Button("Use Response 1", variant="secondary")
                use_response_2 = gr.Button("Use Response 2", variant="secondary")
                use_response_3 = gr.Button("Use Response 3", variant="secondary")

            # Add the local PPM predictive text input component
            gr.Markdown("### 📝 Or compose your own response with predictive text")
            predictive_components = ppm_predictive_text.create_interface()

            # Function to handle submitting a custom response
            def submit_custom_response(text, person_id, topic=None):
                if not text or not person_id:
                    return conversation_history.value

                # Get the people choices dictionary
                people_choices = get_people_choices()

                # Extract the actual ID if it's in the format "Name (role)"
                actual_person_id = person_id
                if person_id in people_choices:
                    actual_person_id = people_choices[person_id]

                # Update the continuous learning model with this response
                try:
                    logger.info(
                        f"Updating continuous learning model with custom response: {text[:50]}..."
                    )
                    continuous_learning_manager.record_interaction(
                        person_id=actual_person_id,
                        topic=topic,
                        user_input="",  # Empty since this is a custom response
                        response=text,
                    )
                    logger.info("Continuous learning model updated successfully")
                except Exception as e:
                    logger.error(f"Error updating continuous learning model: {e}")

                # Save the conversation (with empty user input since this is a custom response)
                success = save_conversation(actual_person_id, "", text)

                if success:
                    # Get updated conversation history
                    try:
                        _, _, _, updated_history = on_person_change(actual_person_id)
                        return updated_history
                    except Exception as e:
                        logger.error(
                            f"Error retrieving updated conversation history: {e}"
                        )
                        return "Conversation saved, but could not retrieve updated history."
                else:
                    return "Failed to save the conversation. Please try again."

            # Connect the predictive text submit button
            predictive_components["submit_btn"].click(
                submit_custom_response,
                inputs=[
                    predictive_components["text_input"],
                    person_dropdown,
                    topic_dropdown,
                ],
                outputs=[conversation_history],
            )

    # Set up event handlers
    def handle_person_change(person_id):
        """Handle person selection change and update UI elements."""
        context_info, phrases_text, _, history_text = on_person_change(person_id)

        # Get topics for this person
        topics = get_topics_for_person(person_id)

        # Get the people choices dictionary
        people_choices = get_people_choices()

        # Extract the actual ID if it's in the format "Name (role)"
        actual_person_id = person_id
        if person_id in people_choices:
            actual_person_id = people_choices[person_id]

        # Update the predictive text context for personalized predictions
        try:
            ppm_predictive_text.set_context(person_id=actual_person_id)
            logger.info(
                f"Updated predictive text context for person: {actual_person_id}"
            )
        except Exception as e:
            logger.error(f"Error updating predictive text context: {e}")

        # Update the context, phrases, conversation history, and topic dropdown
        return context_info, phrases_text, gr.update(choices=topics), history_text

    def handle_model_change(model_name):
        """Handle model selection change."""
        status = change_model(model_name)
        return status

    # Function to handle topic change
    def handle_topic_change(topic, person_id):
        """Handle topic selection change.

        Args:
            topic: Selected topic
            person_id: Current person ID

        Returns:
            None
        """
        if not person_id:
            return

        # Get the people choices dictionary
        people_choices = get_people_choices()

        # Extract the actual ID if it's in the format "Name (role)"
        actual_person_id = person_id
        if person_id in people_choices:
            actual_person_id = people_choices[person_id]

        # Update the predictive text context with the new topic
        try:
            ppm_predictive_text.set_context(person_id=actual_person_id, topic=topic)
            logger.info(
                f"Updated predictive text context: person={actual_person_id}, topic={topic}"
            )
        except Exception as e:
            logger.error(f"Error updating predictive text context: {e}")

    # Set up the person change event
    person_dropdown.change(
        handle_person_change,
        inputs=[person_dropdown],
        outputs=[context_display, common_phrases, topic_dropdown, conversation_history],
    )

    # Set up the topic change event
    topic_dropdown.change(
        handle_topic_change,
        inputs=[topic_dropdown, person_dropdown],
        outputs=[],
    )

    # Set up the model change event
    model_dropdown.change(
        handle_model_change,
        inputs=[model_dropdown],
        outputs=[model_status],
    )

    # Set up the generate button click event
    generate_btn.click(
        generate_suggestions,
        inputs=[
            person_dropdown,
            user_input,
            suggestion_type,
            topic_dropdown,
            model_dropdown,
            temperature_slider,
            mood_slider,
        ],
        outputs=[suggestions_output],
    )

    # Auto-transcribe audio to text when recording stops
    audio_input.stop_recording(
        transcribe_audio,
        inputs=[audio_input],
        outputs=[user_input],
    )

    # Function to extract a response from the suggestions output
    def extract_response(suggestions_text, response_number):
        """Extract a specific response from the suggestions output.

        Args:
            suggestions_text: The text containing all suggestions
            response_number: Which response to extract (1, 2, or 3)

        Returns:
            The extracted response or None if not found
        """
        logger.info(
            f"Extracting response {response_number} from suggestions text: {suggestions_text[:100]}..."
        )

        if not suggestions_text:
            logger.warning("Suggestions text is empty")
            return None

        if "AI-Generated Responses" not in suggestions_text:
            logger.warning("AI-Generated Responses not found in suggestions text")
            # Try to extract from any numbered list
            try:
                import re

                pattern = rf"{response_number}\.\s+(.*?)(?=\n\n\d+\.|\n\n$|$)"
                match = re.search(pattern, suggestions_text)
                if match:
                    extracted = match.group(1).strip()
                    logger.info(
                        f"Found response using generic pattern: {extracted[:50]}..."
                    )
                    return extracted
            except Exception as e:
                logger.error(f"Error extracting response with generic pattern: {e}")
            return None

        try:
            # Look for numbered responses like "1. Response text"
            import re

            pattern = rf"{response_number}\.\s+(.*?)(?=\n\n\d+\.|\n\n$|$)"
            match = re.search(pattern, suggestions_text)
            if match:
                extracted = match.group(1).strip()
                logger.info(f"Successfully extracted response: {extracted[:50]}...")
                return extracted
            else:
                logger.warning(f"No match found for response {response_number}")
                # Try a more lenient pattern
                pattern = rf"{response_number}\.\s+(.*)"
                match = re.search(pattern, suggestions_text)
                if match:
                    extracted = match.group(1).strip()
                    logger.info(
                        f"Found response using lenient pattern: {extracted[:50]}..."
                    )
                    return extracted
        except Exception as e:
            logger.error(f"Error extracting response: {e}")

        logger.warning(f"Failed to extract response {response_number}")
        return None

    # Function to handle using a response
    def use_response(
        suggestions_text, response_number, person_id, user_input_text, topic=None
    ):
        """Handle using a specific response.

        Args:
            suggestions_text: The text containing all suggestions
            response_number: Which response to use (1, 2, or 3)
            person_id: ID of the person in the conversation
            user_input_text: What the person said to Will
            topic: Selected topic (if any)

        Returns:
            Updated conversation history
        """
        logger.info(f"\n=== Using Response {response_number} ===")
        logger.info(f"Person ID: {person_id}")
        logger.info(f"User input: {user_input_text}")
        logger.info(f"Topic: {topic}")

        # Check if person_id is valid
        if not person_id:
            logger.warning("Error: No person_id provided")
            return "Please select a person first."

        # Get the people choices dictionary
        people_choices = get_people_choices()
        logger.info(f"People choices: {people_choices}")

        # Extract the actual ID if it's in the format "Name (role)"
        actual_person_id = person_id
        if person_id in people_choices:
            # If the person_id is a display name, get the actual ID
            actual_person_id = people_choices[person_id]
            logger.info(f"Extracted actual person ID: {actual_person_id}")

        logger.info(
            f"People in social graph: {list(social_graph.graph.get('people', {}).keys())}"
        )

        # Check if person exists in social graph
        if actual_person_id not in social_graph.graph.get("people", {}):
            logger.error(f"Error: Person {actual_person_id} not found in social graph")
            return f"Error: Person {actual_person_id} not found in social graph."

        # Extract the selected response
        selected_response = extract_response(suggestions_text, response_number)

        if not selected_response:
            logger.error("Error: Could not extract response")
            return "Could not find the selected response. Please try generating responses again."

        # Update the continuous learning model with this response
        try:
            logger.info(
                f"Updating continuous learning model with selected response: {selected_response[:50]}..."
            )
            continuous_learning_manager.record_interaction(
                person_id=actual_person_id,
                topic=topic,
                user_input=user_input_text,
                response=selected_response,
            )
            logger.info("Continuous learning model updated successfully")
        except Exception as e:
            logger.error(f"Error updating continuous learning model: {e}")

        # Save the conversation
        logger.info(f"Saving conversation with response: {selected_response[:50]}...")
        success = save_conversation(
            actual_person_id, user_input_text, selected_response
        )

        if success:
            logger.info("Successfully saved conversation")
            # Get updated conversation history
            try:
                _, _, _, updated_history = on_person_change(actual_person_id)
                logger.info("Successfully retrieved updated conversation history")
                return updated_history
            except Exception as e:
                logger.error(f"Error retrieving updated conversation history: {e}")
                return "Conversation saved, but could not retrieve updated history."
        else:
            logger.error("Failed to save conversation")
            return "Failed to save the conversation. Please try again."

    # Set up the response selection button events
    use_response_1.click(
        lambda text, person, input_text, topic: use_response(
            text, 1, person, input_text, topic
        ),
        inputs=[suggestions_output, person_dropdown, user_input, topic_dropdown],
        outputs=[conversation_history],
    )

    use_response_2.click(
        lambda text, person, input_text, topic: use_response(
            text, 2, person, input_text, topic
        ),
        inputs=[suggestions_output, person_dropdown, user_input, topic_dropdown],
        outputs=[conversation_history],
    )

    use_response_3.click(
        lambda text, person, input_text, topic: use_response(
            text, 3, person, input_text, topic
        ),
        inputs=[suggestions_output, person_dropdown, user_input, topic_dropdown],
        outputs=[conversation_history],
    )

# Launch the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Will's AAC Communication Aid")
    parser.add_argument(
        "--regenerate-training",
        action="store_true",
        help="Regenerate PPM training text",
    )
    parser.add_argument(
        "--max-order", type=int, default=5, help="Maximum context length for PPM"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug output")
    args = parser.parse_args()

    # Regenerate training text if requested
    if args.regenerate_training:
        logger.info("Regenerating PPM training text...")
        generate_training_text(social_graph.graph, training_text_path)

    logger.info("Starting application...")
    try:
        demo.launch()
    except Exception as e:
        logger.error(f"Error launching application: {e}")
