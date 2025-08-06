"""
Training script for CNN-based chess evaluator.

This script provides a complete training pipeline for the CNN evaluator model.
"""

import click
import torch
from torch.utils.data import DataLoader
import os
from pathlib import Path

from ..models.cnn_evaluator import CNNEvaluator
from ..data.dataset import ChessPositionDataset
from ..data.data_loader import (
    load_pgn_games,
    create_training_data,
    split_dataset,
    create_balanced_dataset
)
from .trainer import Trainer, TrainingConfig


@click.command()
@click.option('--data-path', type=click.Path(exists=True), required=True,
              help='Path to PGN file with training games')
@click.option('--max-games', type=int, default=10000,
              help='Maximum number of games to load')
@click.option('--batch-size', type=int, default=32,
              help='Batch size for training')
@click.option('--epochs', type=int, default=50,
              help='Number of epochs to train')
@click.option('--learning-rate', type=float, default=1e-3,
              help='Learning rate')
@click.option('--num-blocks', type=int, default=10,
              help='Number of residual blocks in CNN')
@click.option('--channels', type=int, default=256,
              help='Number of channels in CNN')
@click.option('--checkpoint-dir', type=click.Path(), default='./checkpoints/cnn',
              help='Directory to save checkpoints')
@click.option('--balance-dataset/--no-balance-dataset', default=True,
              help='Whether to balance the dataset')
def train_cnn_evaluator(
    data_path: str,
    max_games: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    num_blocks: int,
    channels: int,
    checkpoint_dir: str,
    balance_dataset: bool
):
    """Train a CNN-based chess position evaluator."""
    
    print("Loading chess games...")
    games = load_pgn_games(data_path, max_games=max_games)
    print(f"Loaded {len(games)} games")
    
    print("Creating training data...")
    positions, evaluations = create_training_data(
        games,
        use_game_result=True,
        position_limit=50  # Limit positions per game
    )
    print(f"Created {len(positions)} positions")
    
    # Balance dataset if requested
    if balance_dataset:
        print("Balancing dataset...")
        positions, evaluations = create_balanced_dataset(positions, evaluations)
        print(f"Balanced dataset size: {len(positions)}")
    
    # Split dataset
    print("Splitting dataset...")
    splits = split_dataset(positions, evaluations)
    
    # Create datasets
    train_dataset = ChessPositionDataset(
        positions=splits['train'][0],
        evaluations=splits['train'][1],
        augment=True
    )
    
    val_dataset = ChessPositionDataset(
        positions=splits['val'][0],
        evaluations=splits['val'][1],
        augment=False
    )
    
    test_dataset = ChessPositionDataset(
        positions=splits['test'][0],
        evaluations=splits['test'][1],
        augment=False
    )
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print("Creating CNN model...")
    model = CNNEvaluator(
        num_residual_blocks=num_blocks,
        channels=channels
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create training config
    config = TrainingConfig(
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=epochs,
        checkpoint_dir=checkpoint_dir,
        tensorboard_dir=os.path.join(checkpoint_dir, 'runs'),
        early_stopping_metric='value_loss',
        early_stopping_patience=10,
        checkpoint_interval=5
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader
    )
    
    # Train model
    print("Starting training...")
    trainer.train()
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    model.eval()
    test_loss = 0
    test_mae = 0
    
    with torch.no_grad():
        for positions, values in test_loader:
            positions = positions.to(model.device)
            values = values.to(model.device)
            
            predictions = model(positions).squeeze()
            
            test_loss += torch.nn.functional.mse_loss(predictions, values).item()
            test_mae += torch.nn.functional.l1_loss(predictions, values).item()
    
    test_loss /= len(test_loader)
    test_mae /= len(test_loader)
    
    print(f"Test MSE Loss: {test_loss:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    
    print("\nTraining completed!")


if __name__ == "__main__":
    train_cnn_evaluator()