"""
Base classes for chess ML models.

This module defines abstract base classes that all chess ML models should inherit from.
"""

from abc import ABC, abstractmethod
import chess
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import torch
import torch.nn as nn


class ChessModel(ABC):
    """Abstract base class for all chess ML models."""
    
    @abstractmethod
    def predict(self, board: chess.Board) -> Any:
        """Make a prediction for the given board position."""
        pass
    
    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """Perform one training step and return metrics."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        pass


class ChessEvaluator(ChessModel):
    """Base class for position evaluation models."""
    
    @abstractmethod
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a chess position.
        
        Args:
            board: Chess position to evaluate
            
        Returns:
            Evaluation score (positive favors white, negative favors black)
        """
        pass
    
    def predict(self, board: chess.Board) -> float:
        """Alias for evaluate to satisfy ChessModel interface."""
        return self.evaluate(board)
    
    def evaluate_batch(self, boards: List[chess.Board]) -> np.ndarray:
        """
        Evaluate multiple positions at once.
        
        Default implementation calls evaluate() for each board.
        Subclasses should override for better performance.
        
        Args:
            boards: List of chess positions
            
        Returns:
            Array of evaluation scores
        """
        return np.array([self.evaluate(board) for board in boards])


class ChessPolicyNetwork(ChessModel):
    """Base class for move prediction models."""
    
    @abstractmethod
    def predict_moves(self, board: chess.Board) -> Dict[str, float]:
        """
        Predict move probabilities for a position.
        
        Args:
            board: Chess position
            
        Returns:
            Dictionary mapping move (in UCI notation) to probability
        """
        pass
    
    def predict(self, board: chess.Board) -> Dict[str, float]:
        """Alias for predict_moves to satisfy ChessModel interface."""
        return self.predict_moves(board)
    
    def get_best_move(self, board: chess.Board) -> str:
        """
        Get the best move according to the policy network.
        
        Args:
            board: Chess position
            
        Returns:
            Best move in UCI notation
        """
        move_probs = self.predict_moves(board)
        return max(move_probs.items(), key=lambda x: x[1])[0]


class TorchChessModel(ChessModel, nn.Module):
    """Base class for PyTorch-based chess models."""
    
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def save(self, path: str) -> None:
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_config': self.get_config()
        }, path)
    
    def load(self, path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        if 'model_config' in checkpoint:
            self.load_config(checkpoint['model_config'])
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration. Override in subclasses."""
        return {}
    
    def load_config(self, config: Dict[str, Any]) -> None:
        """Load model configuration. Override in subclasses."""
        pass
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        pass