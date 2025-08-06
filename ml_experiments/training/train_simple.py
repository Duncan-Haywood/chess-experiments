"""
Training script for simple ML-based chess evaluator.

This script trains traditional ML models (Random Forest, XGBoost, etc.) on chess positions.
"""

import click
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from pathlib import Path

from ..models.simple_evaluator import SimpleMLEvaluator
from ..data.data_loader import (
    load_pgn_games,
    create_training_data,
    split_dataset,
    create_balanced_dataset
)


@click.command()
@click.option('--data-path', type=click.Path(exists=True), required=True,
              help='Path to PGN file with training games')
@click.option('--model-type', type=click.Choice(['linear', 'random_forest', 'gradient_boosting', 'mlp']),
              default='random_forest', help='Type of ML model to train')
@click.option('--max-games', type=int, default=5000,
              help='Maximum number of games to load')
@click.option('--output-path', type=click.Path(), default='./models/simple_evaluator.pkl',
              help='Path to save trained model')
@click.option('--balance-dataset/--no-balance-dataset', default=True,
              help='Whether to balance the dataset')
def train_simple_evaluator(
    data_path: str,
    model_type: str,
    max_games: int,
    output_path: str,
    balance_dataset: bool
):
    """Train a simple ML-based chess position evaluator."""
    
    print(f"Training {model_type} evaluator...")
    
    # Load games
    print("Loading chess games...")
    games = load_pgn_games(data_path, max_games=max_games)
    print(f"Loaded {len(games)} games")
    
    # Create training data
    print("Creating training data...")
    positions, evaluations = create_training_data(
        games,
        use_game_result=True,
        position_limit=30  # Limit positions per game for simple models
    )
    print(f"Created {len(positions)} positions")
    
    # Balance dataset if requested
    if balance_dataset:
        print("Balancing dataset...")
        positions, evaluations = create_balanced_dataset(positions, evaluations, bins=5)
        print(f"Balanced dataset size: {len(positions)}")
    
    # Split dataset
    print("Splitting dataset...")
    splits = split_dataset(positions, evaluations)
    
    train_positions, train_evaluations = splits['train']
    val_positions, val_evaluations = splits['val']
    test_positions, test_evaluations = splits['test']
    
    print(f"Train size: {len(train_positions)}")
    print(f"Val size: {len(val_positions)}")
    print(f"Test size: {len(test_positions)}")
    
    # Create model
    print(f"\nCreating {model_type} model...")
    model = SimpleMLEvaluator(model_type=model_type)
    
    # Train model
    print("Training model...")
    train_metrics = model.train_step({
        'positions': train_positions,
        'values': train_evaluations
    })
    
    print("Training metrics:")
    for key, value in train_metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_predictions = model.evaluate_batch(val_positions)
    val_mse = mean_squared_error(val_evaluations, val_predictions)
    val_mae = mean_absolute_error(val_evaluations, val_predictions)
    val_r2 = r2_score(val_evaluations, val_predictions)
    
    print(f"Validation MSE: {val_mse:.4f}")
    print(f"Validation MAE: {val_mae:.4f}")
    print(f"Validation R²: {val_r2:.4f}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_predictions = model.evaluate_batch(test_positions)
    test_mse = mean_squared_error(test_evaluations, test_predictions)
    test_mae = mean_absolute_error(test_evaluations, test_predictions)
    test_r2 = r2_score(test_evaluations, test_predictions)
    
    print(f"Test MSE: {test_mse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    
    # Feature importance (for tree-based models)
    if hasattr(model.model, 'feature_importances_'):
        print("\nTop 10 most important features:")
        feature_names = [
            'white_pawns', 'white_knights', 'white_bishops', 'white_rooks', 'white_queens', 'white_kings',
            'black_pawns', 'black_knights', 'black_bishops', 'black_rooks', 'black_queens', 'black_kings',
            'material_balance', 'white_kingside_castle', 'white_queenside_castle',
            'black_kingside_castle', 'black_queenside_castle', 'white_to_move', 'move_number',
            'center_control', 'white_king_safety', 'black_king_safety', 'mobility',
            'white_doubled_pawns', 'black_doubled_pawns'
        ]
        
        importances = model.model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        for i, idx in enumerate(indices):
            if idx < len(feature_names):
                print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save model
    print(f"\nSaving model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    model.save(output_path)
    
    print("Training completed!")
    
    # Print some example evaluations
    print("\nExample evaluations:")
    for i in range(min(5, len(test_positions))):
        position = test_positions[i]
        true_eval = test_evaluations[i]
        pred_eval = model.evaluate(position)
        print(f"  Position {i+1}: True={true_eval:.3f}, Predicted={pred_eval:.3f}")


if __name__ == "__main__":
    train_simple_evaluator()