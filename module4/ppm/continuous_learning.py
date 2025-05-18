"""
Continuous learning module for the Enhanced PPM predictor.

This module provides functionality to continuously update the PPM model
based on user interactions, including:
1. Selected conversation partners
2. Selected topics
3. Previous conversations
4. Responses the user selects or creates

The goal is to make the PPM model more personalized and context-aware
by adapting to the user's communication patterns over time.
"""

import os
import pickle
import logging
import time
from typing import Dict, List, Any, Optional, Union
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ContinuousLearningManager:
    """Manager for continuous learning of the PPM model."""

    def __init__(
        self,
        ppm_predictor,
        social_graph: Dict[str, Any],
        model_path: str = "enhanced_ppm_model.pkl",
        context_weight: float = 2.0,
        recency_weight: float = 1.5,
        max_history_size: int = 100,
    ):
        """Initialize the continuous learning manager.

        Args:
            ppm_predictor: The PPM predictor to update
            social_graph: The social graph dictionary
            model_path: Path to save the updated model
            context_weight: Weight for contextual data (higher = more influence)
            recency_weight: Weight for recent interactions (higher = more influence)
            max_history_size: Maximum number of interactions to store
        """
        self.ppm_predictor = ppm_predictor
        self.social_graph = social_graph
        self.model_path = model_path
        self.context_weight = context_weight
        self.recency_weight = recency_weight
        self.max_history_size = max_history_size

        # Initialize interaction history
        self.interaction_history = []
        self.person_frequency = Counter()
        self.topic_frequency = Counter()
        self.response_history = []

        # Load existing history if available
        self.history_path = os.path.splitext(model_path)[0] + "_history.pkl"
        self.load_history()

    def load_history(self) -> bool:
        """Load interaction history from disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, "rb") as f:
                    history_data = pickle.load(f)
                    self.interaction_history = history_data.get("interactions", [])
                    self.person_frequency = history_data.get(
                        "person_frequency", Counter()
                    )
                    self.topic_frequency = history_data.get(
                        "topic_frequency", Counter()
                    )
                    self.response_history = history_data.get("responses", [])
                logger.info(
                    f"Loaded interaction history: {len(self.interaction_history)} entries"
                )
                return True
            return False
        except Exception as e:
            logger.error(f"Error loading interaction history: {e}")
            return False

    def save_history(self) -> bool:
        """Save interaction history to disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            history_data = {
                "interactions": self.interaction_history,
                "person_frequency": self.person_frequency,
                "topic_frequency": self.topic_frequency,
                "responses": self.response_history,
            }
            with open(self.history_path, "wb") as f:
                pickle.dump(history_data, f)
            logger.info(
                f"Saved interaction history: {len(self.interaction_history)} entries"
            )
            return True
        except Exception as e:
            logger.error(f"Error saving interaction history: {e}")
            return False

    def record_interaction(
        self,
        person_id: str,
        topic: Optional[str] = None,
        user_input: Optional[str] = None,
        response: Optional[str] = None,
    ) -> None:
        """Record a user interaction for continuous learning.

        Args:
            person_id: ID of the person in the conversation
            topic: Selected topic (if any)
            user_input: What the person said to the user
            response: The user's response
        """
        # Create interaction record
        interaction = {
            "timestamp": time.time(),
            "person_id": person_id,
            "topic": topic,
            "user_input": user_input,
            "response": response,
        }

        # Add to history
        self.interaction_history.append(interaction)

        # Limit history size
        if len(self.interaction_history) > self.max_history_size:
            self.interaction_history = self.interaction_history[
                -self.max_history_size :
            ]

        # Update frequency counters
        self.person_frequency[person_id] += 1
        if topic:
            self.topic_frequency[topic] += 1

        # Add response to history if available
        if response:
            self.response_history.append(response)
            # Limit response history size
            if len(self.response_history) > self.max_history_size:
                self.response_history = self.response_history[-self.max_history_size :]

        # Save history
        self.save_history()

        # Update the model
        self.update_model(person_id, topic, response)

    def update_model(
        self,
        person_id: str,
        topic: Optional[str] = None,
        response: Optional[str] = None,
    ) -> None:
        """Update the PPM model based on the current interaction.

        Args:
            person_id: ID of the person in the conversation
            topic: Selected topic (if any)
            response: The user's response
        """
        # Skip if no response to learn from
        if not response:
            return

        try:
            # 1. Update word frequencies with the response
            words = self._tokenize(response)
            for word in words:
                self.ppm_predictor.word_frequencies[word] += self.context_weight

            # 2. Update word recency
            current_time = time.time()
            for word in words:
                self.ppm_predictor.word_recency[word] = current_time

            # 3. Update word contexts
            for i in range(1, len(words)):
                prev_word = words[i - 1]
                curr_word = words[i]
                self.ppm_predictor.word_contexts[prev_word][
                    curr_word
                ] += self.context_weight

            # 4. Update bigrams
            for i in range(1, len(words)):
                prev_word = words[i - 1]
                curr_word = words[i]
                self.ppm_predictor.bigrams[prev_word][curr_word] += self.context_weight

            # 5. Update trigrams
            for i in range(2, len(words)):
                first_word = words[i - 2]
                second_word = words[i - 1]
                third_word = words[i]

                if first_word not in self.ppm_predictor.trigrams:
                    self.ppm_predictor.trigrams[first_word] = {}

                if second_word not in self.ppm_predictor.trigrams[first_word]:
                    self.ppm_predictor.trigrams[first_word][second_word] = Counter()

                self.ppm_predictor.trigrams[first_word][second_word][
                    third_word
                ] += self.context_weight

            # 6. Add contextual information from social graph
            self._add_contextual_information(person_id, topic)

            # 7. Save the updated model
            self.save_model()

            logger.info(f"Updated PPM model with response: {response[:50]}...")
        except Exception as e:
            logger.error(f"Error updating PPM model: {e}")

    def _add_contextual_information(
        self, person_id: str, topic: Optional[str] = None
    ) -> None:
        """Add contextual information from the social graph.

        Args:
            person_id: ID of the person in the conversation
            topic: Selected topic (if any)
        """
        try:
            # Get person data from social graph
            people = self.social_graph.get("people", {})
            if person_id not in people:
                return

            person_data = people[person_id]

            # Add common phrases for this person with higher weight
            if "common_phrases" in person_data:
                for phrase in person_data["common_phrases"]:
                    words = self._tokenize(phrase)
                    for word in words:
                        self.ppm_predictor.word_frequencies[word] += (
                            self.context_weight / 2
                        )

                    # Update bigrams and trigrams
                    for i in range(1, len(words)):
                        prev_word = words[i - 1]
                        curr_word = words[i]
                        self.ppm_predictor.bigrams[prev_word][curr_word] += (
                            self.context_weight / 2
                        )

                    for i in range(2, len(words)):
                        first_word = words[i - 2]
                        second_word = words[i - 1]
                        third_word = words[i]

                        if first_word not in self.ppm_predictor.trigrams:
                            self.ppm_predictor.trigrams[first_word] = {}

                        if second_word not in self.ppm_predictor.trigrams[first_word]:
                            self.ppm_predictor.trigrams[first_word][
                                second_word
                            ] = Counter()

                        self.ppm_predictor.trigrams[first_word][second_word][
                            third_word
                        ] += (self.context_weight / 2)

            # Add topic-related words if a topic is selected
            if topic and "topics" in person_data and topic in person_data["topics"]:
                # Add the topic words with higher weight
                topic_words = self._tokenize(topic)
                for word in topic_words:
                    self.ppm_predictor.word_frequencies[word] += self.context_weight
                    self.ppm_predictor.word_recency[word] = time.time()

            # Add recent conversation history
            if "conversation_history" in person_data:
                for conversation in person_data["conversation_history"][
                    -3:
                ]:  # Use last 3 conversations
                    if "messages" in conversation:
                        for message in conversation["messages"]:
                            if (
                                "speaker" in message
                                and message["speaker"] == "Will"
                                and "text" in message
                            ):
                                # Add user's previous responses to this person
                                words = self._tokenize(message["text"])
                                for word in words:
                                    self.ppm_predictor.word_frequencies[word] += (
                                        self.context_weight / 3
                                    )

        except Exception as e:
            logger.error(f"Error adding contextual information: {e}")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of words
        """
        if not text:
            return []

        # Use the predictor's tokenize method if available
        if hasattr(self.ppm_predictor, "_tokenize"):
            return self.ppm_predictor._tokenize(text)

        # Simple fallback tokenization
        import re

        return re.findall(r"\b\w+\b", text.lower())

    def save_model(self) -> bool:
        """Save the updated PPM model to disk.

        Returns:
            True if successful, False otherwise
        """
        try:
            # Instead of saving the entire predictor, save only the necessary components
            model_state = {
                "word_frequencies": self.ppm_predictor.word_frequencies,
                "word_recency": self.ppm_predictor.word_recency,
                "word_contexts": self.ppm_predictor.word_contexts,
                "bigrams": self.ppm_predictor.bigrams,
                "trigrams": self.ppm_predictor.trigrams,
            }

            # Save the model state
            with open(self.model_path, "wb") as f:
                pickle.dump(
                    model_state, f, protocol=4
                )  # Use protocol 4 for better compatibility
            logger.info(f"Saved updated PPM model to {self.model_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving PPM model: {e}")
            return False

    def get_personalized_predictions(
        self,
        text: str,
        person_id: Optional[str] = None,
        topic: Optional[str] = None,
        num_predictions: int = 5,
    ) -> List[str]:
        """Get personalized word predictions based on current context.

        Args:
            text: Current text input
            person_id: ID of the person in the conversation
            topic: Selected topic (if any)
            num_predictions: Number of predictions to return

        Returns:
            List of predicted words
        """
        # Get base predictions from the PPM predictor
        base_predictions = self.ppm_predictor.predict_next_words(
            text, num_predictions * 2
        )

        # If no context, return base predictions
        if not person_id:
            return base_predictions[:num_predictions]

        # Create a dictionary to store personalized scores
        personalized_scores = {}

        # Score each prediction
        for word in base_predictions:
            # Start with the base score (position in the list)
            score = len(base_predictions) - base_predictions.index(word)

            # Boost based on person frequency
            if self.person_frequency[person_id] > 0:
                # Check if this word appears in conversations with this person
                person_data = self.social_graph.get("people", {}).get(person_id, {})
                person_text = ""

                # Add common phrases
                if "common_phrases" in person_data:
                    person_text += " ".join(person_data["common_phrases"]) + " "

                # Add conversation history
                if "conversation_history" in person_data:
                    for conversation in person_data["conversation_history"]:
                        if "messages" in conversation:
                            for message in conversation["messages"]:
                                if (
                                    "speaker" in message
                                    and message["speaker"] == "Will"
                                    and "text" in message
                                ):
                                    person_text += message["text"] + " "

                # Count occurrences of the word in person-specific text
                if word.lower() in person_text.lower():
                    word_count = person_text.lower().count(word.lower())
                    score += word_count * self.context_weight

            # Boost based on topic
            if topic and self.topic_frequency[topic] > 0:
                # Simple check if word is in the topic
                if word.lower() in topic.lower():
                    score += self.context_weight * 2

            # Store the personalized score
            personalized_scores[word] = score

        # Sort predictions by personalized score
        sorted_predictions = sorted(
            personalized_scores.items(), key=lambda x: x[1], reverse=True
        )

        # Return top predictions
        return [word for word, _ in sorted_predictions[:num_predictions]]
