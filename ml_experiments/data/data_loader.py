"""
Data loading utilities for chess ML experiments.

This module provides functions to load chess games from various sources
and convert them into training data.
"""

import chess
import chess.pgn
import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Iterator
import os
import glob
from pathlib import Path
import requests
import gzip
import io
from tqdm import tqdm

from ..utils.chess_utils import pgn_to_positions, get_game_result


def load_pgn_games(
    pgn_path: str,
    max_games: Optional[int] = None,
    min_elo: Optional[int] = None
) -> List[chess.pgn.Game]:
    """
    Load games from a PGN file.
    
    Args:
        pgn_path: Path to PGN file
        max_games: Maximum number of games to load
        min_elo: Minimum ELO rating for both players
        
    Returns:
        List of chess.pgn.Game objects
    """
    games = []
    
    with open(pgn_path, 'r') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            
            # Filter by ELO if specified
            if min_elo is not None:
                white_elo = game.headers.get("WhiteElo", "0")
                black_elo = game.headers.get("BlackElo", "0")
                
                try:
                    if int(white_elo) < min_elo or int(black_elo) < min_elo:
                        continue
                except ValueError:
                    continue
            
            games.append(game)
            
            if max_games is not None and len(games) >= max_games:
                break
    
    return games


def load_lichess_dataset(
    dataset_path: str,
    max_games: Optional[int] = None,
    rating_range: Optional[Tuple[int, int]] = None
) -> Iterator[chess.pgn.Game]:
    """
    Load games from Lichess dataset (compressed PGN).
    
    Args:
        dataset_path: Path to compressed PGN file (.pgn.gz)
        max_games: Maximum number of games to yield
        rating_range: Tuple of (min_rating, max_rating)
        
    Yields:
        chess.pgn.Game objects
    """
    games_yielded = 0
    
    # Handle both .gz and regular files
    if dataset_path.endswith('.gz'):
        open_func = gzip.open
        mode = 'rt'
    else:
        open_func = open
        mode = 'r'
    
    with open_func(dataset_path, mode) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            
            # Filter by rating if specified
            if rating_range is not None:
                white_elo = game.headers.get("WhiteElo", "0")
                black_elo = game.headers.get("BlackElo", "0")
                
                try:
                    white_elo = int(white_elo)
                    black_elo = int(black_elo)
                    avg_elo = (white_elo + black_elo) // 2
                    
                    if avg_elo < rating_range[0] or avg_elo > rating_range[1]:
                        continue
                except ValueError:
                    continue
            
            yield game
            games_yielded += 1
            
            if max_games is not None and games_yielded >= max_games:
                break


def create_training_data(
    games: List[chess.pgn.Game],
    use_game_result: bool = True,
    use_engine_eval: bool = False,
    position_limit: Optional[int] = None
) -> Tuple[List[chess.Board], List[float]]:
    """
    Create training data from games.
    
    Args:
        games: List of chess games
        use_game_result: Use game result as evaluation
        use_engine_eval: Use engine evaluation if available in comments
        position_limit: Maximum positions per game to use
        
    Returns:
        Tuple of (positions, evaluations)
    """
    all_positions = []
    all_evaluations = []
    
    for game in tqdm(games, desc="Processing games"):
        # Get game result
        result = game.headers.get("Result", "*")
        if result == "1-0":
            game_result = 1.0
        elif result == "0-1":
            game_result = -1.0
        elif result == "1/2-1/2":
            game_result = 0.0
        else:
            continue  # Skip unfinished games
        
        # Extract positions
        positions_with_eval = pgn_to_positions(
            game.accept(chess.pgn.StringExporter()),
            include_evaluations=use_engine_eval
        )
        
        # Limit positions if specified
        if position_limit is not None:
            positions_with_eval = positions_with_eval[:position_limit]
        
        for board, engine_eval in positions_with_eval:
            all_positions.append(board)
            
            if use_engine_eval and engine_eval is not None:
                # Use engine evaluation
                all_evaluations.append(engine_eval)
            elif use_game_result:
                # Use game result with decay based on position
                # Positions later in the game are more indicative of result
                move_number = board.fullmove_number
                weight = min(move_number / 40.0, 1.0)  # Full weight after move 40
                
                # From current player's perspective
                if board.turn == chess.WHITE:
                    evaluation = game_result * weight
                else:
                    evaluation = -game_result * weight
                
                all_evaluations.append(evaluation)
            else:
                # No evaluation available
                all_evaluations.append(0.0)
    
    return all_positions, all_evaluations


def split_dataset(
    positions: List[chess.Board],
    evaluations: List[float],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_seed: int = 42
) -> Dict[str, Tuple[List[chess.Board], List[float]]]:
    """
    Split dataset into train/validation/test sets.
    
    Args:
        positions: List of chess positions
        evaluations: List of evaluations
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with 'train', 'val', 'test' keys
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n_samples = len(positions)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    
    # Calculate split points
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # Split indices
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    # Create splits
    splits = {
        'train': (
            [positions[i] for i in train_indices],
            [evaluations[i] for i in train_indices]
        ),
        'val': (
            [positions[i] for i in val_indices],
            [evaluations[i] for i in val_indices]
        ),
        'test': (
            [positions[i] for i in test_indices],
            [evaluations[i] for i in test_indices]
        )
    }
    
    return splits


def download_lichess_database(
    year: int,
    month: int,
    output_dir: str = "./data",
    variant: str = "standard"
) -> str:
    """
    Download Lichess database for a specific month.
    
    Args:
        year: Year (e.g., 2024)
        month: Month (1-12)
        output_dir: Directory to save the file
        variant: Chess variant (standard, chess960, etc.)
        
    Returns:
        Path to downloaded file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct URL
    month_str = f"{month:02d}"
    filename = f"lichess_db_{variant}_rated_{year}-{month_str}.pgn.gz"
    url = f"https://database.lichess.org/{variant}/lichess_db_{variant}_rated_{year}-{month_str}.pgn.gz"
    
    output_path = os.path.join(output_dir, filename)
    
    # Download if not exists
    if not os.path.exists(output_path):
        print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    return output_path


def create_balanced_dataset(
    positions: List[chess.Board],
    evaluations: List[float],
    bins: int = 10
) -> Tuple[List[chess.Board], List[float]]:
    """
    Create a balanced dataset by sampling equally from evaluation bins.
    
    Args:
        positions: List of chess positions
        evaluations: List of evaluations
        bins: Number of bins to divide evaluations into
        
    Returns:
        Balanced dataset
    """
    # Create bins
    eval_array = np.array(evaluations)
    bin_edges = np.percentile(eval_array, np.linspace(0, 100, bins + 1))
    
    # Assign positions to bins
    binned_positions = [[] for _ in range(bins)]
    binned_evaluations = [[] for _ in range(bins)]
    
    for pos, eval_score in zip(positions, evaluations):
        bin_idx = np.digitize(eval_score, bin_edges) - 1
        bin_idx = max(0, min(bin_idx, bins - 1))
        binned_positions[bin_idx].append(pos)
        binned_evaluations[bin_idx].append(eval_score)
    
    # Sample equally from each bin
    min_bin_size = min(len(bin_pos) for bin_pos in binned_positions if len(bin_pos) > 0)
    
    balanced_positions = []
    balanced_evaluations = []
    
    for bin_pos, bin_eval in zip(binned_positions, binned_evaluations):
        if len(bin_pos) > 0:
            indices = np.random.choice(len(bin_pos), min_bin_size, replace=False)
            balanced_positions.extend([bin_pos[i] for i in indices])
            balanced_evaluations.extend([bin_eval[i] for i in indices])
    
    return balanced_positions, balanced_evaluations