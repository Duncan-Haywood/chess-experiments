#!/usr/bin/env python3
"""
Test script to verify the Python conversion of Go code.

This script tests the basic functionality of the converted modules.
"""

import sys
sys.path.append('/workspace')

from pkg.chess.game import Game
from pkg.engine import MaterialEvaluator, PositionalEvaluator, MinimaxEngine
from pkg.eval import StockEvaluator
import chess


def test_game():
    """Test the Game class."""
    print("Testing Game class...")
    game = Game.new_game()
    print(f"Initial position: {game.position.fen()}")
    
    moves = game.valid_moves()
    print(f"Number of legal moves: {len(moves)}")
    
    if moves:
        game.move(moves[0])
        print(f"After first move: {game.position.fen()}")
    
    print("✓ Game class works\n")


def test_evaluators():
    """Test the evaluator classes."""
    print("Testing Evaluators...")
    game = Game.new_game()
    
    # Test Material Evaluator
    material_eval = MaterialEvaluator()
    score = material_eval.evaluate(game)
    print(f"Material evaluation of starting position: {score}")
    
    # Test Positional Evaluator
    pos_eval = PositionalEvaluator()
    score = pos_eval.evaluate(game)
    print(f"Positional evaluation of starting position: {score}")
    
    # Test Stock Evaluator
    stock_eval = StockEvaluator()
    score = stock_eval.eval(game.position)
    print(f"Stock evaluation of starting position: {score}")
    
    print("✓ Evaluators work\n")


def test_minimax():
    """Test the minimax engine."""
    print("Testing Minimax Engine...")
    game = Game.new_game()
    
    # Test with material evaluator
    evaluator = MaterialEvaluator()
    engine = MinimaxEngine(depth=3, evaluator=evaluator)
    
    best_move, score = engine.find_best_move(game)
    print(f"Best move at depth 3: {best_move}")
    print(f"Evaluation: {score}")
    
    print("✓ Minimax engine works\n")


def test_complex_position():
    """Test with a more complex position."""
    print("Testing complex position...")
    
    # Italian Game position
    fen = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
    board = chess.Board(fen)
    game = Game(board)
    
    evaluator = PositionalEvaluator()
    engine = MinimaxEngine(depth=3, evaluator=evaluator)
    
    best_move, score = engine.find_best_move(game)
    print(f"Best move in Italian Game: {best_move}")
    print(f"Evaluation: {score}")
    
    print("✓ Complex position analysis works\n")


def main():
    """Run all tests."""
    print("=" * 50)
    print("Python Chess Conversion Test Suite")
    print("=" * 50)
    print()
    
    try:
        test_game()
        test_evaluators()
        test_minimax()
        test_complex_position()
        
        print("=" * 50)
        print("All tests passed! ✓")
        print("=" * 50)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()