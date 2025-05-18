"""
Correction Engine (Noisy Channel Model) for AAC input correction.

This module implements a noisy channel model for correcting noisy AAC input.
It combines a PPM language model for P(intended) and a confusion matrix for
P(noisy | intended) to rank candidate corrections.
"""

from .corrector import NoisyChannelCorrector, correct

__version__ = "0.1.0"
__all__ = ["NoisyChannelCorrector", "correct"]
