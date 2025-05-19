"""
Generate training text for PPM-API from the social graph.

This script extracts text from the social graph to create a training corpus
for the PPM model, which will be used for word prediction.
"""

import json
import argparse
import random
from typing import List, Dict, Any


def load_social_graph(file_path: str) -> Dict[str, Any]:
    """Load the social graph from a JSON file.

    Args:
        file_path: Path to the social graph JSON file

    Returns:
        The social graph as a dictionary
    """
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading social graph: {e}")
        return {}


def extract_common_utterances(social_graph: Dict[str, Any]) -> List[str]:
    """Extract common utterances from the social graph.

    Args:
        social_graph: The social graph dictionary

    Returns:
        List of common utterances
    """
    utterances = []

    # Extract utterances from common_utterances section
    if "common_utterances" in social_graph:
        for category, phrases in social_graph["common_utterances"].items():
            utterances.extend(phrases)

    return utterances


def extract_common_phrases(social_graph: Dict[str, Any]) -> List[str]:
    """Extract common phrases from each person in the social graph.

    Args:
        social_graph: The social graph dictionary

    Returns:
        List of common phrases
    """
    phrases = []

    # Extract phrases from each person
    if "people" in social_graph:
        for person_id, person_data in social_graph["people"].items():
            if "common_phrases" in person_data:
                phrases.extend(person_data["common_phrases"])

    return phrases


def extract_conversation_history(social_graph: Dict[str, Any]) -> List[str]:
    """Extract conversation history from each person in the social graph.

    Args:
        social_graph: The social graph dictionary

    Returns:
        List of conversation messages
    """
    messages = []

    # Extract conversation history from each person
    if "people" in social_graph:
        for person_id, person_data in social_graph["people"].items():
            if "conversation_history" in person_data:
                for conversation in person_data["conversation_history"]:
                    if "messages" in conversation:
                        for message in conversation["messages"]:
                            if "text" in message:
                                # Clean up the text (remove quotes, etc.)
                                text = message["text"].strip("\"'")
                                messages.append(text)

    return messages


def extract_topics(social_graph: Dict[str, Any]) -> List[str]:
    """Extract topics from each person in the social graph.

    Args:
        social_graph: The social graph dictionary

    Returns:
        List of topics
    """
    topics = []

    # Extract topics from each person
    if "people" in social_graph:
        for person_id, person_data in social_graph["people"].items():
            if "topics" in person_data:
                topics.extend(person_data["topics"])

    return topics


def generate_training_text(
    social_graph: Dict[str, Any],
    output_file: str,
    repeat_factor: int = 3,
    additional_text_files: List[str] = None,
    include_common_phrases: bool = True,
) -> None:
    """Generate training text from the social graph.

    Args:
        social_graph: The social graph dictionary
        output_file: Path to the output file
        repeat_factor: How many times to repeat important phrases
        additional_text_files: List of additional text files to include
        include_common_phrases: Whether to include common English phrases

    Returns:
        None
    """
    # Extract text from different parts of the social graph
    utterances = extract_common_utterances(social_graph)
    phrases = extract_common_phrases(social_graph)
    messages = extract_conversation_history(social_graph)
    topics = extract_topics(social_graph)

    # Combine all text
    all_text = []

    # Add utterances and phrases multiple times (they're more important)
    for _ in range(repeat_factor):
        all_text.extend(utterances)
        all_text.extend(phrases)

    # Add messages and topics
    all_text.extend(messages)
    all_text.extend(topics)

    # Add common English phrases if requested
    if include_common_phrases:
        common_phrases = [
            "hello world",
            "good morning",
            "good afternoon",
            "good evening",
            "how are you",
            "I'm fine thank you",
            "nice to meet you",
            "what time is it",
            "what day is it today",
            "what's the weather like",
            "I need help",
            "can you help me",
            "thank you very much",
            "you're welcome",
            "excuse me",
            "I'm sorry",
            "no problem",
            "that sounds good",
            "I agree",
            "I disagree",
            "maybe later",
            "see you later",
            "see you tomorrow",
            "have a nice day",
            "I'm hungry",
            "I'm thirsty",
            "I'm tired",
            "I need to rest",
            "I need to use the bathroom",
            "I'm feeling better",
            "I'm not feeling well",
            "I'd like to go outside",
            "I'd like to watch TV",
            "I'd like to listen to music",
            "can you turn the lights on",
            "can you turn the lights off",
            "can you open the window",
            "can you close the window",
            "it's too hot",
            "it's too cold",
            "it's comfortable",
            "I love you",
            "I miss you",
            "I'm thinking of you",
            "that's interesting",
            "tell me more",
            "I didn't understand",
            "can you repeat that",
            "speak more slowly please",
            "yes please",
            "no thank you",
            "maybe",
            "definitely",
            "I don't know",
            "I'll think about it",
            "let me consider that",
            "that's a good idea",
            "I have a question",
            "I have a suggestion",
            "what do you think",
            "what's your opinion",
            "I think that",
            "I believe",
            "in my opinion",
            "from my perspective",
            "I need more time",
            "take your time",
            "there's no rush",
            "I'm looking forward to it",
            "I'm excited",
            "I'm nervous",
            "I'm worried",
            "I'm happy",
            "I'm sad",
            "I'm angry",
            "I'm surprised",
            "I'm confused",
            "I'm disappointed",
            "that's great news",
            "that's unfortunate",
            "congratulations",
            "I'm proud of you",
            "well done",
            "good job",
            "let's talk about something else",
            "changing the subject",
            "by the way",
            "speaking of which",
            "that reminds me",
            "to get back to what we were discussing",
            "as I was saying",
            "in conclusion",
            "to sum up",
            "finally",
            "lastly",
            "first of all",
            "secondly",
            "furthermore",
            "moreover",
            "however",
            "nevertheless",
            "on the other hand",
            "in contrast",
            "similarly",
            "likewise",
            "for example",
            "such as",
            "in particular",
            "specifically",
            "in general",
            "usually",
            "typically",
            "occasionally",
            "rarely",
            "never",
            "always",
            "sometimes",
            "often",
            "seldom",
        ]
        all_text.extend(common_phrases)
        print(f"Added {len(common_phrases)} common phrases to training text")

    # Add text from additional files if provided
    if additional_text_files:
        for file_path in additional_text_files:
            try:
                with open(file_path, "r") as f:
                    additional_text = f.read().split("\n")
                    # Filter out empty lines
                    additional_text = [line for line in additional_text if line.strip()]
                all_text.extend(additional_text)
                print(
                    f"Added {len(additional_text)} lines from {file_path} to training text"
                )
            except Exception as e:
                print(f"Could not read additional text file {file_path}: {e}")

    # Shuffle the text to avoid biasing the model
    random.shuffle(all_text)

    # Write to output file
    try:
        with open(output_file, "w") as f:
            f.write("\n".join(all_text))
        print(f"Training text generated successfully: {output_file}")
        print(f"Total sentences: {len(all_text)}")
    except Exception as e:
        print(f"Error writing training text: {e}")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate training text from social graph"
    )
    parser.add_argument(
        "--social-graph",
        default="social_graph.json",
        help="Path to social graph JSON file",
    )
    parser.add_argument(
        "--output", default="training_text.txt", help="Path to output file"
    )
    parser.add_argument(
        "--repeat", type=int, default=3, help="Repeat factor for important phrases"
    )
    parser.add_argument(
        "--additional-text", nargs="+", help="Additional text files to include"
    )
    parser.add_argument(
        "--no-common-phrases",
        action="store_true",
        help="Don't include common English phrases",
    )
    args = parser.parse_args()

    # Load social graph
    social_graph = load_social_graph(args.social_graph)

    if not social_graph:
        print("Failed to load social graph. Exiting.")
        return

    # Generate training text
    generate_training_text(
        social_graph=social_graph,
        output_file=args.output,
        repeat_factor=args.repeat,
        additional_text_files=args.additional_text,
        include_common_phrases=not args.no_common_phrases,
    )


if __name__ == "__main__":
    main()
