"""
Chess game wrapper for Python.

This module provides a simple wrapper around the python-chess library
to match the Go implementation structure.
"""

import chess


class Game:
    """Wrapper class for chess.Board to match Go implementation."""
    
    def __init__(self, board=None):
        """
        Initialize a new game.
        
        Args:
            board: Optional chess.Board object. If None, creates a new game.
        """
        self.board = board if board is not None else chess.Board()
    
    @classmethod
    def new_game(cls):
        """Create a new game with starting position."""
        return cls()
    
    def clone(self):
        """Create a deep copy of the current game."""
        return Game(self.board.copy())
    
    def move(self, move):
        """
        Make a move on the board.
        
        Args:
            move: chess.Move object
        """
        self.board.push(move)
    
    def valid_moves(self):
        """Get all legal moves in the current position."""
        return list(self.board.legal_moves)
    
    def outcome(self):
        """
        Get the game outcome.
        
        Returns:
            'white_won', 'black_won', 'draw', or 'no_outcome'
        """
        outcome = self.board.outcome()
        if outcome is None:
            return 'no_outcome'
        
        if outcome.winner == chess.WHITE:
            return 'white_won'
        elif outcome.winner == chess.BLACK:
            return 'black_won'
        else:
            return 'draw'
    
    @property
    def position(self):
        """Get the current position (board)."""
        return self.board
    
    def __str__(self):
        """String representation of the board."""
        return str(self.board)