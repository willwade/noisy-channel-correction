#!/usr/bin/env python3
"""
Tests for the ConversationLevelEvaluator class, focusing on context handling.
"""

import pytest
from unittest.mock import MagicMock

from scripts.evaluation.conversation_level_evaluator import (
    ConversationLevelEvaluator
)
from lib.corrector.corrector import NoisyChannelCorrector




@pytest.fixture
def mock_corrector() -> MagicMock:
    """Fixture to create a mock NoisyChannelCorrector."""
    mock = MagicMock(spec=NoisyChannelCorrector)
    # Simulate corrector.correct returning the noisy input as the top correction
    mock.correct.side_effect = lambda noisy_input, context=None, max_candidates=5, max_edit_distance=2, keyboard_layout='qwerty', noise_level='minimal', target_utterance=None: [(noisy_input, 0.9)]
    mock.lexicon = {"hello", "world", "test", "context", "gold", "user", "aac"} 
    return mock


@pytest.fixture
def evaluator_with_mock_corrector(mock_corrector: MagicMock) -> ConversationLevelEvaluator:
    """Fixture for ConversationLevelEvaluator using the mock_corrector."""
    return ConversationLevelEvaluator(
        corrector=mock_corrector,
        context_window_size=2,  # Default window size for these tests
        use_gold_context=False,  # Default, can be overridden in specific tests
        target_field="utterance_intended"  # Important for metric calculation, but not primary for context logic test
    )


# Test for gold context usage
def test_evaluate_conversation_gold_context(mock_corrector: MagicMock):
    """Test that gold context (intended utterances of same speaker) is correctly passed."""
    evaluator = ConversationLevelEvaluator(
        corrector=mock_corrector,
        context_window_size=2,
        use_gold_context=True,  # Enable gold context
        target_field="utterance_intended"
    )

    # Sample conversation: AAC User, Partner, AAC User, AAC User
    conversation = [
        {"speaker": "AAC User", "utterance_intended": "Gold 1 AAC", "noisy_qwerty_minimal": "noisy 1", "noisy_field_key": "noisy_qwerty_minimal"},
        {"speaker": "Partner", "utterance_intended": "Gold Partner", "noisy_qwerty_minimal": "noisy P", "noisy_field_key": "noisy_qwerty_minimal"},
        {"speaker": "AAC User", "utterance_intended": "Gold 2 AAC", "noisy_qwerty_minimal": "noisy 2", "noisy_field_key": "noisy_qwerty_minimal"},
        {"speaker": "AAC User", "utterance_intended": "Gold 3 AAC", "noisy_qwerty_minimal": "noisy 3", "noisy_field_key": "noisy_qwerty_minimal"},
    ]

    evaluator.evaluate_conversation(conversation)

    assert mock_corrector.correct.call_count == 4
    calls = mock_corrector.correct.call_args_list

    # Call 1 (AAC User): Noisy "noisy 1", Context: [] (no prior AAC utterances)
    # Passed args are (noisy_input, **kwargs), context is in kwargs
    assert calls[0][0][0] == "noisy 1"
    assert calls[0][1]['context'] == []
    assert calls[0][1]['target_utterance'] == "Gold 1 AAC"

    # Call 2 (Partner): Noisy "noisy P", Context: [] (no prior Partner utterances, AAC context ignored for partner)
    assert calls[1][0][0] == "noisy P"
    assert calls[1][1]['context'] == [] 
    assert calls[1][1]['target_utterance'] == "Gold Partner"

    # Call 3 (AAC User): Noisy "noisy 2", Context: ["Gold 1 AAC"]
    assert calls[2][0][0] == "noisy 2"
    assert calls[2][1]['context'] == ["Gold 1 AAC"]
    assert calls[2][1]['target_utterance'] == "Gold 2 AAC"

    # Call 4 (AAC User): Noisy "noisy 3", Context: ["Gold 1 AAC", "Gold 2 AAC"] (window size 2)
    assert calls[3][0][0] == "noisy 3"
    assert calls[3][1]['context'] == ["Gold 1 AAC", "Gold 2 AAC"]
    assert calls[3][1]['target_utterance'] == "Gold 3 AAC"


# Test for generated (non-gold) context usage
def test_evaluate_conversation_generated_context(mock_corrector: MagicMock):
    """Test that generated context (corrector's output) is correctly passed."""
    evaluator = ConversationLevelEvaluator(
        corrector=mock_corrector,
        context_window_size=2,
        use_gold_context=False,  # Use generated context
        target_field="utterance_intended"
    )

    # Define side effects for corrector.correct to simulate different outputs
    # Corrected output will be used for next turn's context if speaker is AAC User
    mock_corrector.correct.side_effect = [
        [("Corrected 1 AAC", 0.9)],  # Output for "noisy 1"
        [("Corrected P", 0.9)],      # Output for "noisy P"
        [("Corrected 2 AAC", 0.9)],  # Output for "noisy 2"
        [("Corrected 3 AAC", 0.9)],  # Output for "noisy 3"
    ]

    conversation = [
        {"speaker": "AAC User", "utterance_intended": "Gold 1 AAC", "noisy_qwerty_minimal": "noisy 1", "noisy_field_key": "noisy_qwerty_minimal"},
        {"speaker": "Partner", "utterance_intended": "Gold Partner", "noisy_qwerty_minimal": "noisy P", "noisy_field_key": "noisy_qwerty_minimal"},
        {"speaker": "AAC User", "utterance_intended": "Gold 2 AAC", "noisy_qwerty_minimal": "noisy 2", "noisy_field_key": "noisy_qwerty_minimal"},
        {"speaker": "AAC User", "utterance_intended": "Gold 3 AAC", "noisy_qwerty_minimal": "noisy 3", "noisy_field_key": "noisy_qwerty_minimal"},
    ]

    evaluator.evaluate_conversation(conversation)

    assert mock_corrector.correct.call_count == 4
    calls = mock_corrector.correct.call_args_list

    # Call 1 (AAC User): Noisy "noisy 1", Context: []
    assert calls[0][0][0] == "noisy 1"
    assert calls[0][1]['context'] == []

    # Call 2 (Partner): Noisy "noisy P", Context: [] (generated context only tracks AAC User by default in script)
    # The ConversationLevelEvaluator's current_context is only appended to if speaker is 'Patient (AAC)' or 'AAC User'.
    assert calls[1][0][0] == "noisy P"
    assert calls[1][1]['context'] == [] 

    # Call 3 (AAC User): Noisy "noisy 2", Context: ["Corrected 1 AAC"]
    assert calls[2][0][0] == "noisy 2"
    assert calls[2][1]['context'] == ["Corrected 1 AAC"]

    # Call 4 (AAC User): Noisy "noisy 3", Context: ["Corrected 1 AAC", "Corrected 2 AAC"]
    assert calls[3][0][0] == "noisy 3"
    assert calls[3][1]['context'] == ["Corrected 1 AAC", "Corrected 2 AAC"]


# Test for context window size limit
def test_evaluate_conversation_context_window_size(mock_corrector: MagicMock):
    """Test that context window size is respected for both gold and generated context."""
    
    # Test with Gold Context
    evaluator_gold = ConversationLevelEvaluator(
        corrector=mock_corrector, context_window_size=1, use_gold_context=True, target_field="utterance_intended"
    )
    conversation = [
        {"speaker": "AAC User", "utterance_intended": "Gold 1", "noisy_qwerty_minimal": "n1", "noisy_field_key": "noisy_qwerty_minimal"},
        {"speaker": "AAC User", "utterance_intended": "Gold 2", "noisy_qwerty_minimal": "n2", "noisy_field_key": "noisy_qwerty_minimal"},
        {"speaker": "AAC User", "utterance_intended": "Gold 3", "noisy_qwerty_minimal": "n3", "noisy_field_key": "noisy_qwerty_minimal"},
    ]
    mock_corrector.correct.reset_mock() # Reset call count and history
    evaluator_gold.evaluate_conversation(conversation)
    calls_gold = mock_corrector.correct.call_args_list
    assert calls_gold[2][1]['context'] == ["Gold 2"] # Only last one due to window size 1

    # Test with Generated Context
    evaluator_gen = ConversationLevelEvaluator(
        corrector=mock_corrector, context_window_size=1, use_gold_context=False, target_field="utterance_intended"
    )
    mock_corrector.correct.reset_mock()
    mock_corrector.correct.side_effect = [
        [("Corr 1", 0.9)], [("Corr 2", 0.9)], [("Corr 3", 0.9)]
    ]
    evaluator_gen.evaluate_conversation(conversation)
    calls_gen = mock_corrector.correct.call_args_list
    assert calls_gen[2][1]['context'] == ["Corr 2"] # Only last corrected one
