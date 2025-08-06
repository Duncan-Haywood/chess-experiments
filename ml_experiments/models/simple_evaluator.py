"""
Simple ML-based chess evaluator using traditional features.

This module implements evaluators using classical machine learning algorithms
(Random Forest, XGBoost, etc.) with hand-crafted chess features.
"""

import chess
import numpy as np
import pickle
from typing import Dict, List, Any, Optional
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import joblib

from .base import ChessEvaluator
from ..utils.board_representation import encode_position


class SimpleMLEvaluator(ChessEvaluator):
    """
    Simple ML evaluator using traditional features and scikit-learn models.
    
    Supports various ML algorithms:
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    - Neural Network (MLP)
    """
    
    def __init__(self, model_type: str = "random_forest", **model_kwargs):
        """
        Initialize the evaluator.
        
        Args:
            model_type: Type of model to use ("linear", "random_forest", "gradient_boosting", "mlp")
            **model_kwargs: Additional arguments passed to the model constructor
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, **model_kwargs)
        self.is_trained = False
    
    def _create_model(self, model_type: str, **kwargs):
        """Create the underlying ML model."""
        if model_type == "linear":
            return LinearRegression(**kwargs)
        elif model_type == "random_forest":
            default_kwargs = {
                "n_estimators": 100,
                "max_depth": 20,
                "random_state": 42,
                "n_jobs": -1
            }
            default_kwargs.update(kwargs)
            return RandomForestRegressor(**default_kwargs)
        elif model_type == "gradient_boosting":
            default_kwargs = {
                "n_estimators": 100,
                "max_depth": 5,
                "learning_rate": 0.1,
                "random_state": 42
            }
            default_kwargs.update(kwargs)
            return GradientBoostingRegressor(**default_kwargs)
        elif model_type == "mlp":
            default_kwargs = {
                "hidden_layer_sizes": (100, 50),
                "activation": "relu",
                "solver": "adam",
                "random_state": 42,
                "max_iter": 1000
            }
            default_kwargs.update(kwargs)
            return MLPRegressor(**default_kwargs)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def extract_features(self, board: chess.Board) -> np.ndarray:
        """
        Extract features from a chess position.
        
        Uses encode_position from utils, but can be extended with more features.
        
        Args:
            board: Chess position
            
        Returns:
            Feature vector
        """
        basic_features = encode_position(board)
        
        # Add more sophisticated features
        additional_features = []
        
        # Center control
        center_squares = [chess.E4, chess.D4, chess.E5, chess.D5]
        center_control = 0
        for square in center_squares:
            piece = board.piece_at(square)
            if piece:
                center_control += 1 if piece.color == chess.WHITE else -1
        additional_features.append(center_control)
        
        # King safety (simplified - distance from edge)
        white_king = board.king(chess.WHITE)
        black_king = board.king(chess.BLACK)
        if white_king:
            white_king_file = chess.square_file(white_king)
            white_king_rank = chess.square_rank(white_king)
            white_king_safety = min(white_king_file, 7 - white_king_file, 
                                   white_king_rank, 7 - white_king_rank)
        else:
            white_king_safety = 0
            
        if black_king:
            black_king_file = chess.square_file(black_king)
            black_king_rank = chess.square_rank(black_king)
            black_king_safety = min(black_king_file, 7 - black_king_file,
                                   black_king_rank, 7 - black_king_rank)
        else:
            black_king_safety = 0
            
        additional_features.extend([white_king_safety, black_king_safety])
        
        # Mobility (number of legal moves)
        mobility = len(list(board.legal_moves))
        additional_features.append(mobility)
        
        # Pawn structure features
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)
        
        # Doubled pawns
        white_doubled = 0
        black_doubled = 0
        for file_idx in range(8):
            white_on_file = sum(1 for sq in white_pawns if chess.square_file(sq) == file_idx)
            black_on_file = sum(1 for sq in black_pawns if chess.square_file(sq) == file_idx)
            if white_on_file > 1:
                white_doubled += white_on_file - 1
            if black_on_file > 1:
                black_doubled += black_on_file - 1
        
        additional_features.extend([white_doubled, black_doubled])
        
        # Combine all features
        return np.concatenate([basic_features, additional_features])
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a chess position.
        
        Args:
            board: Chess position to evaluate
            
        Returns:
            Evaluation score
        """
        if not self.is_trained:
            # Return material balance as fallback
            return self._material_balance(board)
        
        features = self.extract_features(board).reshape(1, -1)
        return self.model.predict(features)[0]
    
    def evaluate_batch(self, boards: List[chess.Board]) -> np.ndarray:
        """Evaluate multiple positions at once."""
        if not self.is_trained:
            return np.array([self._material_balance(board) for board in boards])
        
        features = np.array([self.extract_features(board) for board in boards])
        return self.model.predict(features)
    
    def _material_balance(self, board: chess.Board) -> float:
        """Calculate simple material balance."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0
        }
        
        balance = 0
        for piece_type, value in piece_values.items():
            balance += value * (
                len(board.pieces(piece_type, chess.WHITE)) -
                len(board.pieces(piece_type, chess.BLACK))
            )
        
        return balance / 10.0  # Normalize
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Train the model on a batch of positions.
        
        Args:
            batch: Dictionary containing:
                - 'positions': List of chess.Board objects
                - 'values': Target evaluation scores
                
        Returns:
            Dictionary of metrics
        """
        positions = batch['positions']
        values = batch['values']
        
        # Extract features
        X = np.array([self.extract_features(board) for board in positions])
        y = np.array(values)
        
        # Fit the model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate metrics
        predictions = self.model.predict(X)
        mse = np.mean((predictions - y) ** 2)
        mae = np.mean(np.abs(predictions - y))
        
        return {
            'mse': mse,
            'mae': mae,
            'r2_score': self.model.score(X, y) if hasattr(self.model, 'score') else 0.0
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, path)
    
    def load(self, path: str) -> None:
        """Load model from disk."""
        model_data = joblib.load(path)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']