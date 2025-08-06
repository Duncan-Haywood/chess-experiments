"""
Example usage of chess ML experiments.

This script demonstrates how to use the various ML models and utilities
in the chess experiments package.
"""

import chess
import numpy as np
from pathlib import Path

# Import ML experiment modules
from ml_experiments.models import CNNEvaluator, SimpleMLEvaluator
from ml_experiments.data import load_pgn_games, create_training_data, split_dataset
from ml_experiments.utils import board_to_tensor, fen_to_board


def demo_simple_evaluator():
    """Demonstrate simple ML evaluator."""
    print("=== Simple ML Evaluator Demo ===\n")
    
    # Create evaluator
    evaluator = SimpleMLEvaluator(model_type="random_forest")
    
    # Test positions
    positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"),  # Open position
        chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 0 1"),  # K vs K endgame
        chess.Board("8/8/8/4k3/8/8/4K3/4R3 w - - 0 1"),  # K+R vs K endgame
    ]
    
    print("Evaluations (untrained model - using material balance):")
    for i, board in enumerate(positions):
        eval_score = evaluator.evaluate(board)
        print(f"Position {i+1}: {eval_score:.3f}")
    
    # Create synthetic training data
    print("\nGenerating synthetic training data...")
    training_positions = []
    training_evaluations = []
    
    for _ in range(100):
        board = chess.Board()
        # Make random moves
        for _ in range(np.random.randint(0, 20)):
            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break
            move = np.random.choice(legal_moves)
            board.push(move)
        
        training_positions.append(board.copy())
        # Simple evaluation based on material
        material = 0
        piece_values = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, 
                       chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
        for piece_type, value in piece_values.items():
            material += value * (len(board.pieces(piece_type, chess.WHITE)) - 
                               len(board.pieces(piece_type, chess.BLACK)))
        training_evaluations.append(material / 10.0)
    
    # Train model
    print(f"Training on {len(training_positions)} positions...")
    metrics = evaluator.train_step({
        'positions': training_positions,
        'values': training_evaluations
    })
    
    print("Training metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    
    # Re-evaluate positions
    print("\nEvaluations (trained model):")
    for i, board in enumerate(positions):
        eval_score = evaluator.evaluate(board)
        print(f"Position {i+1}: {eval_score:.3f}")


def demo_cnn_evaluator():
    """Demonstrate CNN evaluator."""
    print("\n\n=== CNN Evaluator Demo ===\n")
    
    # Create small CNN for demo
    evaluator = CNNEvaluator(
        num_residual_blocks=2,
        channels=32
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in evaluator.parameters())
    print(f"CNN model has {total_params:,} parameters")
    
    # Test position
    board = chess.Board()
    
    # Show tensor representation
    tensor = board_to_tensor(board)
    print(f"\nBoard tensor shape: {tensor.shape}")
    print(f"Number of input planes: {tensor.shape[2]}")
    
    # Evaluate position
    eval_score = evaluator.evaluate(board)
    print(f"\nStarting position evaluation: {eval_score:.3f}")
    
    # Batch evaluation example
    boards = [chess.Board() for _ in range(10)]
    evaluations = evaluator.evaluate_batch(boards)
    print(f"\nBatch evaluation shape: {evaluations.shape}")
    print(f"Mean evaluation: {evaluations.mean():.3f}")


def demo_data_loading():
    """Demonstrate data loading utilities."""
    print("\n\n=== Data Loading Demo ===\n")
    
    # Create sample PGN
    sample_pgn = """[Event "Example Game"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 d6 1-0
"""
    
    # Save to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as f:
        f.write(sample_pgn)
        pgn_path = f.name
    
    try:
        # Load games
        games = load_pgn_games(pgn_path)
        print(f"Loaded {len(games)} game(s)")
        
        # Create training data
        positions, evaluations = create_training_data(games)
        print(f"Created {len(positions)} training positions")
        
        # Split dataset
        splits = split_dataset(positions, evaluations)
        print(f"Train size: {len(splits['train'][0])}")
        print(f"Val size: {len(splits['val'][0])}")
        print(f"Test size: {len(splits['test'][0])}")
        
    finally:
        # Clean up
        import os
        os.unlink(pgn_path)


def main():
    """Run all demos."""
    print("Chess ML Experiments - Example Usage\n")
    print("This demonstrates the various ML models and utilities available.\n")
    
    demo_simple_evaluator()
    demo_cnn_evaluator()
    demo_data_loading()
    
    print("\n\nTo train models on real data:")
    print("1. Download PGN files from lichess.org or other sources")
    print("2. Use the training scripts:")
    print("   python -m ml_experiments.training.train_cnn --data-path games.pgn")
    print("   python -m ml_experiments.training.train_simple --data-path games.pgn")


if __name__ == "__main__":
    main()