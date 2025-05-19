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
import math
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
        self.speaker_vocabularies = defaultdict(
            Counter
        )  # Speaker-specific word frequencies
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
            self.history = self.history[-self.max_history :]

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

        # Use math for logarithmic calculations

        # Process history in reverse order (most recent first)
        for i, (speaker, utterance, _) in enumerate(reversed(self.history)):
            # Calculate recency score (higher for more recent utterances)
            # Use a logarithmic decay instead of exponential for more gradual falloff
            recency_score = 1.0 / (1.0 + 0.5 * i)  # Starts at 1.0, decays more slowly

            # Calculate speaker relevance (higher for same speaker)
            # Enhanced: give higher weight to the current speaker's recent utterances
            if speaker == current_speaker:
                speaker_score = 1.0 + (
                    0.5 if i < 2 else 0
                )  # Boost for very recent utterances
            else:
                # Different weighting for different speakers
                # System/assistant utterances are more relevant than other users
                speaker_score = (
                    0.7 if speaker.lower() in ["system", "assistant"] else 0.5
                )

            # Get words from utterance
            words = self._tokenize(utterance)

            # Score each word
            for word in words:
                # Skip very short words and common stopwords
                if len(word) <= 1 or word in [
                    "the",
                    "a",
                    "an",
                    "and",
                    "or",
                    "but",
                    "is",
                    "are",
                ]:
                    continue

                # Base score is recency * speaker relevance
                base_score = recency_score * speaker_score

                # Add topic relevance if topic-aware with enhanced weighting
                if self.topic_aware and word in self.topic_keywords:
                    # Normalize by the maximum topic keyword count
                    max_count = max(self.topic_keywords.values())
                    # Higher weight for topic keywords (0.7 instead of 0.5)
                    topic_score = 0.7 * (self.topic_keywords[word] / max(1, max_count))
                    base_score += topic_score

                # Add frequency relevance with diminishing returns for very common words
                if word in self.global_vocabulary:
                    word_freq = self.global_vocabulary[word]
                    max_freq = max(self.global_vocabulary.values())
                    # Apply logarithmic scaling to avoid over-weighting common words
                    freq_factor = math.log(1 + word_freq) / math.log(1 + max_freq)
                    freq_score = 0.3 * freq_factor
                    base_score += freq_score

                # Add speaker-specific relevance if available
                if (
                    self.speaker_specific
                    and speaker in self.speaker_vocabularies
                    and word in self.speaker_vocabularies[speaker]
                ):
                    speaker_word_freq = self.speaker_vocabularies[speaker][word]
                    max_speaker_freq = max(self.speaker_vocabularies[speaker].values())
                    speaker_freq_factor = speaker_word_freq / max(1, max_speaker_freq)
                    speaker_score = 0.4 * speaker_freq_factor
                    base_score += speaker_score

                # Update word score (take max if word appears multiple times)
                word_scores[word] = max(word_scores.get(word, 0), base_score)

        # Enhance context with bigram information
        # This helps preserve phrases that are important for context
        if len(self.history) >= 2:
            # Extract bigrams from recent utterances
            bigrams = []
            for _, utterance, _ in self.history[-3:]:  # Last 3 utterances
                words = self._tokenize(utterance)
                for i in range(len(words) - 1):
                    bigrams.append((words[i], words[i + 1]))

            # Count bigram frequencies
            bigram_counts = Counter(bigrams)

            # Boost words that appear in frequent bigrams
            for (word1, word2), count in bigram_counts.items():
                if count > 1 and word1 in word_scores and word2 in word_scores:
                    # Boost both words in the bigram
                    word_scores[word1] *= 1.2
                    word_scores[word2] *= 1.2

        # Sort words by score and return top N
        sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:max_words]]

    def get_speaker_specific_context(
        self, speaker: str, max_words: int = 10
    ) -> List[str]:
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
        # Enhanced approach: use TF-IDF-like weighting to identify important words
        # 1. Update raw word counts
        self.topic_keywords.update(words)

        # 2. Remove common stopwords from topic keywords
        stopwords = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "to",
            "from",
            "in",
            "out",
            "on",
            "off",
            "over",
            "under",
            "again",
            "further",
            "then",
            "once",
            "here",
            "there",
            "when",
            "where",
            "why",
            "how",
            "all",
            "any",
            "both",
            "each",
            "few",
            "more",
            "most",
            "other",
            "some",
            "such",
            "no",
            "nor",
            "not",
            "only",
            "own",
            "same",
            "so",
            "than",
            "too",
            "very",
            "s",
            "t",
            "can",
            "will",
            "just",
            "don",
            "should",
            "now",
            "d",
            "ll",
            "m",
            "o",
            "re",
            "ve",
            "y",
            "ain",
            "aren",
            "couldn",
            "didn",
            "doesn",
            "hadn",
            "hasn",
            "haven",
            "isn",
            "ma",
            "mightn",
            "mustn",
            "needn",
            "shan",
            "shouldn",
            "wasn",
            "weren",
            "won",
            "wouldn",
            "i",
            "me",
            "my",
            "myself",
            "we",
            "our",
            "ours",
            "ourselves",
            "you",
            "your",
            "yours",
            "yourself",
            "yourselves",
            "he",
            "him",
            "his",
            "himself",
            "she",
            "her",
            "hers",
            "herself",
            "it",
            "its",
            "itself",
            "they",
            "them",
            "their",
            "theirs",
            "themselves",
            "what",
            "which",
            "who",
            "whom",
            "this",
            "that",
            "these",
            "those",
            "am",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "having",
            "do",
            "does",
            "did",
            "doing",
            "would",
            "should",
            "could",
            "ought",
            "i'm",
            "you're",
            "he's",
            "she's",
            "it's",
            "we're",
            "they're",
            "i've",
            "you've",
            "we've",
            "they've",
            "i'd",
            "you'd",
            "he'd",
            "she'd",
            "we'd",
            "they'd",
            "i'll",
            "you'll",
            "he'll",
            "she'll",
            "we'll",
            "they'll",
            "isn't",
            "aren't",
            "wasn't",
            "weren't",
            "hasn't",
            "haven't",
            "hadn't",
            "doesn't",
            "don't",
            "didn't",
            "won't",
            "wouldn't",
            "shan't",
            "shouldn't",
            "can't",
            "cannot",
            "couldn't",
            "mustn't",
            "let's",
            "that's",
            "who's",
            "what's",
            "here's",
            "there's",
            "when's",
            "where's",
            "why's",
            "how's",
        }

        for word in stopwords:
            if word in self.topic_keywords:
                del self.topic_keywords[word]

        # 3. Detect topic shifts by comparing current keywords with previous ones
        # If we have at least 3 utterances, check for topic shifts
        if len(self.history) >= 3:
            # Get words from the last 3 utterances
            recent_words = []
            for i in range(min(3, len(self.history))):
                _, utterance, _ = self.history[-(i + 1)]
                recent_words.extend(self._tokenize(utterance))

            # Count recent words
            recent_counts = Counter(recent_words)

            # Remove stopwords
            for word in stopwords:
                if word in recent_counts:
                    del recent_counts[word]

            # Calculate overlap between recent words and overall topic keywords
            overlap = sum(1 for word in recent_counts if word in self.topic_keywords)
            overlap_ratio = overlap / max(1, len(recent_counts))

            # If overlap is low, we might have a topic shift
            if overlap_ratio < 0.3 and len(recent_counts) >= 5:
                # Reset topic keywords to focus on the new topic
                self.topic_keywords = recent_counts
                logger.info("Topic shift detected. Resetting topic keywords.")

        # 4. Boost keywords that appear in multiple utterances
        # This helps identify consistent themes in the conversation
        if len(self.history) >= 2:
            # Count how many utterances each word appears in
            utterance_counts = Counter()
            for _, utterance, _ in self.history:
                # Get unique words in this utterance
                unique_words = set(self._tokenize(utterance))
                utterance_counts.update(unique_words)

            # Boost words that appear in multiple utterances
            for word, count in utterance_counts.items():
                if word in self.topic_keywords and count > 1:
                    # Boost by the number of utterances it appears in
                    self.topic_keywords[word] *= 1 + 0.5 * (count - 1)
