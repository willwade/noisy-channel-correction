"""
Metrics for AAC scanning and prediction evaluation.
"""

# Global counters for prediction accuracy
total_predictions = 0
correct_predictions = 0

def reset_metrics():
    """Reset all metrics counters."""
    global total_predictions, correct_predictions
    total_predictions = 0
    correct_predictions = 0

def get_prediction_accuracy():
    """Calculate the prediction accuracy."""
    if total_predictions == 0:
        return 0.0
    return correct_predictions / total_predictions

def print_metrics():
    """Print the current metrics."""
    accuracy = get_prediction_accuracy()
    print(f"Prediction Accuracy: {accuracy:.2%} ({correct_predictions}/{total_predictions})")
