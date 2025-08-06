"""
Chess position evaluators.

This module provides different evaluation functions for chess positions,
including material and positional evaluators.
"""

import chess
from abc import ABC, abstractmethod
from typing import Dict
import sys
sys.path.append('/workspace')
from pkg.chess.game import Game
from .piece_square_tables import PieceSquareTables


class Evaluator(ABC):
    """Abstract base class for position evaluation."""
    
    @abstractmethod
    def evaluate(self, game: Game) -> float:
        """
        Evaluate the position.
        
        Args:
            game: Game object
            
        Returns:
            Evaluation score from the perspective of the side to move
        """
        pass


class MaterialEvaluator(Evaluator):
    """Evaluates positions based on material balance."""
    
    def __init__(self):
        """Initialize with standard piece values."""
        self.piece_values: Dict[chess.PieceType, float] = {
            chess.PAWN: 1.0,
            chess.KNIGHT: 3.0,
            chess.BISHOP: 3.0,
            chess.ROOK: 5.0,
            chess.QUEEN: 9.0,
            chess.KING: 0.0,  # King has no material value
        }
    
    def evaluate(self, game: Game) -> float:
        """
        Return the material evaluation of the position.
        
        Args:
            game: Game object
            
        Returns:
            Material evaluation from perspective of side to move
        """
        # Check for game over states
        outcome = game.outcome()
        if outcome == 'white_won':
            return 1000.0
        elif outcome == 'black_won':
            return -1000.0
        elif outcome == 'draw':
            return 0.0
        
        board = game.position
        score = 0.0
        
        # Calculate material balance
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            value = self.piece_values[piece.piece_type]
            if piece.color == chess.WHITE:
                score += value
            else:
                score -= value
        
        # Return score from the perspective of the side to move
        if board.turn == chess.BLACK:
            score = -score
        
        return score


class PositionalEvaluator(Evaluator):
    """Adds positional factors to material evaluation."""
    
    def __init__(self):
        """Initialize with material evaluator and piece square tables."""
        self.material = MaterialEvaluator()
        self.pst = PieceSquareTables()
    
    def evaluate(self, game: Game) -> float:
        """
        Return the positional evaluation.
        
        Args:
            game: Game object
            
        Returns:
            Positional evaluation from perspective of side to move
        """
        # Start with material evaluation
        score = self.material.evaluate(game)
        
        # Don't add positional factors for game over positions
        if score >= 900 or score <= -900:
            return score
        
        board = game.position
        positional_score = 0.0
        
        # Add piece-square table values
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is None:
                continue
            
            ps_value = self.pst.get_value(piece, square)
            if piece.color == chess.WHITE:
                positional_score += ps_value
            else:
                positional_score -= ps_value
        
        # Add mobility bonus (number of legal moves)
        mobility_bonus = len(list(board.legal_moves)) * 0.1
        
        # Combine scores
        total_score = score + positional_score / 100.0 + mobility_bonus
        
        # Return from perspective of side to move
        if board.turn == chess.BLACK:
            total_score = -total_score
        
        return total_score