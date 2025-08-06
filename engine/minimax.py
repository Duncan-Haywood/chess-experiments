"""
Simple minimax engine without alpha-beta pruning.

This module provides a basic minimax search implementation.
"""

import sys
sys.path.append('/workspace')
import math
import chess
from typing import Optional, Tuple
from pkg.chess.game import Game


def evaluate_board(game: Game) -> float:
    """
    Basic board evaluation function.
    
    Args:
        game: Game object
        
    Returns:
        Evaluation score
    """
    # This is a placeholder - should use proper evaluator
    board = game.position
    score = 0.0
    
    piece_values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        value = piece_values.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value
    
    return score


def minimax(game: Game, depth: int) -> Tuple[float, Optional[chess.Move]]:
    """
    Perform a basic minimax search to given depth.
    
    Args:
        game: Current game state
        depth: Search depth
        
    Returns:
        Tuple of (best_score, best_move)
    """
    if depth == 0:
        return evaluate_board(game), None
    
    best_score = -math.inf
    best_move = None
    if game.position.turn == chess.BLACK:
        best_score = math.inf
    
    for move in game.valid_moves():
        clone = game.clone()
        clone.move(move)
        score, _ = minimax(clone, depth - 1)
        
        if game.position.turn == chess.WHITE:
            if score > best_score:
                best_score = score
                best_move = move
        else:
            if score < best_score:
                best_score = score
                best_move = move
    
    return best_score, best_move


# Note: EvaluateBoard is assumed to be in another package or import accordingly.