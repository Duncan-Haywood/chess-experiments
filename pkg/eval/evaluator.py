"""
Evaluator interface for chess position evaluation.

This module defines the base interface for position evaluators.
"""

from abc import ABC, abstractmethod
import chess


class Evaluator(ABC):
    """Abstract base class for position evaluators."""
    
    @abstractmethod
    def eval(self, position: chess.Board) -> int:
        """
        Evaluate a chess position.
        
        Args:
            position: chess.Board object representing the position
            
        Returns:
            Integer evaluation score
        """
        pass