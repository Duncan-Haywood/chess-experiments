"""
Minimax chess engine with alpha-beta pruning.

This module implements the minimax algorithm for chess move selection
with alpha-beta pruning for improved performance.
"""

import math
import chess
from typing import Optional, Tuple
import sys
sys.path.append('/workspace')
from pkg.chess.game import Game
from .evaluator import Evaluator


class MinimaxEngine:
    """Implements the minimax algorithm with alpha-beta pruning."""
    
    def __init__(self, depth: int, evaluator: Evaluator):
        """
        Initialize the minimax engine.
        
        Args:
            depth: Search depth
            evaluator: Position evaluator
        """
        self.depth = depth
        self.evaluator = evaluator
    
    def find_best_move(self, game: Game) -> Tuple[Optional[chess.Move], float]:
        """
        Find the best move for the current position.
        
        Args:
            game: Current game state
            
        Returns:
            Tuple of (best_move, evaluation_score)
        """
        moves = game.valid_moves()
        if not moves:
            return None, 0.0
        
        best_move = moves[0]
        best_score = -math.inf
        alpha = -math.inf
        beta = math.inf
        
        for move in moves:
            # Make the move
            new_game = game.clone()
            new_game.move(move)
            
            # Evaluate the position
            score = -self._minimax(new_game, self.depth - 1, -beta, -alpha)
            
            # Update best move
            if score > best_score:
                best_score = score
                best_move = move
            
            # Update alpha
            if score > alpha:
                alpha = score
        
        return best_move, best_score
    
    def _minimax(self, game: Game, depth: int, alpha: float, beta: float) -> float:
        """
        Minimax algorithm with alpha-beta pruning.
        
        Args:
            game: Current game state
            depth: Remaining search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            
        Returns:
            Position evaluation
        """
        # Terminal node - return evaluation
        if depth == 0 or game.outcome() != 'no_outcome':
            return self.evaluator.evaluate(game)
        
        moves = game.valid_moves()
        if not moves:
            return self.evaluator.evaluate(game)
        
        # Maximize
        best_score = -math.inf
        for move in moves:
            new_game = game.clone()
            new_game.move(move)
            
            score = -self._minimax(new_game, depth - 1, -beta, -alpha)
            
            if score > best_score:
                best_score = score
            
            if score > alpha:
                alpha = score
            
            # Alpha-beta cutoff
            if alpha >= beta:
                break
        
        return best_score
    
    def set_depth(self, depth: int) -> None:
        """
        Update the search depth.
        
        Args:
            depth: New search depth
        """
        self.depth = depth
    
    def get_depth(self) -> int:
        """
        Get the current search depth.
        
        Returns:
            Current search depth
        """
        return self.depth