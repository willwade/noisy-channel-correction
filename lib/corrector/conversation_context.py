#!/usr/bin/env python3
"""
Conversation Context Manager for AAC input correction.

This module provides functionality for managing conversation-level context
to improve the accuracy of the noisy channel corrector. It tracks conversation
history, speaker turns, and topics to provide richer context for correction.
"""

import logging
import re
import time
from typing import List, Optional
from collections import Counter, defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class ConversationContext:
    """
    Class for managing conversation-level context for AAC input correction.

    This class tracks conversation history, speaker turns, and topics to provide
    richer context for correction. It implements methods for extracting relevant
    context for a given correction task and for updating the context with new
    utterances.
    """

    def __init__(
        self,
        max_history: int = 10,
        recency_weight: float = 0.8,
        speaker_specific: bool = True,
        topic_aware: bool = True,
    ):
        """
        Initialize a conversation context manager.

        Args:
            max_history: Maximum number of utterances to keep in history
            recency_weight: Weight for recency in context relevance (0-1)
            speaker_specific: Whether to track speaker-specific patterns
            topic_aware: Whether to use topic modeling for context
        """
        self.max_history = max_history
        self.recency_weight = recency_weight
        self.speaker_specific = speaker_specific
        self.topic_aware = topic_aware

        # Initialize conversation history
        self.history = []  # List of (speaker, utterance, timestamp) tuples
        self.speakers = set()  # Set of speakers in the conversation
        self.speaker_vocabularies = defaultdict(Counter)  # Speaker-specific word frequencies
        self.global_vocabulary = Counter()  # Global word frequencies
        self.topic_keywords = Counter()  # Current topic keywords
        self.current_topic = None  # Current topic label (if available)

        # Initialize time tracking
        self.start_time = time.time()
        self.last_update_time = self.start_time

    def add_utterance(
        self, utterance: str, speaker: str = "user", timestamp: Optional[float] = None
    ) -> None:
        """
        Add an utterance to the conversation history.

        Args:
            utterance: The utterance text
            speaker: The speaker identifier
            timestamp: Optional timestamp (defaults to current time)
        """
        # Use current time if no timestamp provided
        if timestamp is None:
            timestamp = time.time()

        # Add to history
        self.history.append((speaker, utterance, timestamp))

        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # Update speaker set
        self.speakers.add(speaker)

        # Update vocabularies
        words = self._tokenize(utterance)
        self.global_vocabulary.update(words)
        if self.speaker_specific:
            self.speaker_vocabularies[speaker].update(words)

        # Update topic keywords
        if self.topic_aware:
            self._update_topic_keywords(words)

        # Update last update time
        self.last_update_time = timestamp

    def get_context_for_correction(
        self, current_speaker: str = "user", max_words: int = 20
    ) -> List[str]:
        """
        Get relevant context words for correction.

        Args:
            current_speaker: The current speaker
            max_words: Maximum number of context words to return

        Returns:
            List of context words, most relevant first
        """
        # Extract all words from history with their relevance scores
        word_scores = {}

        # Process history in reverse order (most recent first)
        for i, (speaker, utterance, timestamp) in enumerate(reversed(self.history)):
            # Calculate recency score (higher for more recent utterances)
            recency_score = self.recency_weight ** i

            # Calculate speaker relevance (higher for same speaker)
            speaker_score = 1.0 if speaker == current_speaker else 0.5

            # Get words from utterance
            words = self._tokenize(utterance)

            # Score each word
            for word in words:
                # Base score is recency * speaker relevance
                base_score = recency_score * speaker_score

                # Add topic relevance if topic-aware
                if self.topic_aware and word in self.topic_keywords:
                    topic_score = 0.5 * (self.topic_keywords[word] / max(self.topic_keywords.values()))
                    base_score += topic_score

                # Add frequency relevance
                freq_score = 0.3 * (self.global_vocabulary[word] / max(self.global_vocabulary.values()))
                base_score += freq_score

                # Update word score (take max if word appears multiple times)
                word_scores[word] = max(word_scores.get(word, 0), base_score)

        # Sort words by score and return top N
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_words]]

    def get_speaker_specific_context(self, speaker: str, max_words: int = 10) -> List[str]:
        """
        Get speaker-specific context words.

        Args:
            speaker: The speaker to get context for
            max_words: Maximum number of words to return

        Returns:
            List of speaker-specific context words
        """
        if not self.speaker_specific or speaker not in self.speaker_vocabularies:
            return []

        # Get the most common words for this speaker
        common_words = self.speaker_vocabularies[speaker].most_common(max_words)
        return [word for word, _ in common_words]

    def get_topic_keywords(self, max_keywords: int = 10) -> List[str]:
        """
        Get current topic keywords.

        Args:
            max_keywords: Maximum number of keywords to return

        Returns:
            List of topic keywords
        """
        if not self.topic_aware:
            return []

        # Get the most common topic keywords
        common_keywords = self.topic_keywords.most_common(max_keywords)
        return [word for word, _ in common_keywords]

    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text into words.

        Args:
            text: Text to tokenize

        Returns:
            List of words
        """
        # Replace punctuation with spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Condense all whitespace to a single space
        text = re.sub(r"\s+", " ", text)

        # Trim leading and trailing whitespaces
        text = text.strip()

        # Split the text into tokens and convert to lowercase
        tokens = [word.lower() for word in text.split()]

        return tokens

    def _update_topic_keywords(self, words: List[str]) -> None:
        """
        Update topic keywords based on new words.

        Args:
            words: List of words from the latest utterance
        """
        # Simple approach: just count word frequencies across all utterances
        # A more sophisticated approach would use actual topic modeling
        self.topic_keywords.update(words)

        # Remove common stopwords from topic keywords
        stopwords = {"the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
                    "be", "been", "being", "have", "has", "had", "do", "does", "did",
                    "to", "from", "in", "out", "on", "off", "over", "under", "again",
                    "further", "then", "once", "here", "there", "when", "where", "why",
                    "how", "all", "any", "both", "each", "few", "more", "most", "other",
                    "some", "such", "no", "nor", "not", "only", "own", "same", "so",
                    "than", "too", "very", "s", "t", "can", "will", "just", "don",
                    "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
                    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma",
                    "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
                    "won", "wouldn"}

        for word in stopwords:
            if word in self.topic_keywords:
                del self.topic_keywords[word]
