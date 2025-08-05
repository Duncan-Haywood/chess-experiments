"""Training utilities for chess ML experiments."""

from .trainer import Trainer, TrainingConfig
from .train_cnn import train_cnn_evaluator
from .train_simple import train_simple_evaluator

__all__ = [
    "Trainer",
    "TrainingConfig",
    "train_cnn_evaluator",
    "train_simple_evaluator",
]