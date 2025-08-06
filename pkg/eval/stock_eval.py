"""
Stock evaluator implementation.

This module provides a basic stock evaluator that returns a constant value.
"""

import chess
from .evaluator import Evaluator


class StockEvaluator(Evaluator):
    """Basic stock evaluator implementation."""
    
    def eval(self, position: chess.Board) -> int:
        """
        Evaluate a chess position.
        
        Args:
            position: chess.Board object representing the position
            
        Returns:
            Integer evaluation score (always 0 for stock evaluator)
        """
        return 0