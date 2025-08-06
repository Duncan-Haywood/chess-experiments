#!/usr/bin/env python3
"""
Stock evaluation command line tool.

This tool demonstrates basic chess position evaluation using material balance.
"""

import sys
sys.path.append('/workspace')
import chess
from pkg.eval import StockEvaluator
from pkg.chess.game import Game


def evaluate_board(game: Game) -> float:
    """
    Return a simple score based on material balance.
    
    Positive score favors white, negative favors black.
    This can be replaced or augmented with ML models later.
    
    Args:
        game: Game object
        
    Returns:
        Material evaluation score in centipawns
    """
    board = game.position
    score = 0.0
    
    # Piece values (basic centipawn equivalents)
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        # King not valued for material
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


def main():
    """Main function for stock evaluation demo."""
    game = Game.new_game()
    
    # Generate all valid moves
    valid_moves = game.valid_moves()
    
    # Make a move
    if valid_moves:
        game.move(valid_moves[0])
    
    # Get material difference using stock evaluator
    evaluator = StockEvaluator()
    print(evaluator.eval(game.position))
    
    # Also show our custom evaluation
    print(f"Material balance: {evaluate_board(game)}")


if __name__ == "__main__":
    main()