"""
PyTorch dataset classes for chess ML experiments.

This module provides dataset classes for loading and preprocessing chess positions
for training neural networks.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import chess
from typing import List, Tuple, Optional, Dict, Any
import h5py
import os

from ..utils.board_representation import board_to_tensor
from ..utils.chess_utils import get_game_result


class ChessDataset(Dataset):
    """
    Base PyTorch dataset for chess positions.
    
    This is an abstract base class that handles common functionality.
    """
    
    def __init__(self, transform=None):
        """
        Initialize dataset.
        
        Args:
            transform: Optional transform to apply to positions
        """
        self.transform = transform
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self, idx):
        raise NotImplementedError


class ChessPositionDataset(ChessDataset):
    """
    Dataset for chess positions with evaluations.
    
    Stores positions and their corresponding evaluations for training
    position evaluation models.
    """
    
    def __init__(
        self,
        positions: List[chess.Board],
        evaluations: List[float],
        transform=None,
        augment: bool = True
    ):
        """
        Initialize dataset with positions and evaluations.
        
        Args:
            positions: List of chess.Board objects
            evaluations: List of evaluation scores
            transform: Optional transform to apply
            augment: Whether to apply data augmentation
        """
        super().__init__(transform)
        self.positions = positions
        self.evaluations = evaluations
        self.augment = augment
        
        assert len(positions) == len(evaluations), \
            "Number of positions must match number of evaluations"
    
    def __len__(self):
        return len(self.positions)
    
    def __getitem__(self, idx):
        """
        Get a position and its evaluation.
        
        Returns:
            Tuple of (position_tensor, evaluation)
        """
        board = self.positions[idx]
        evaluation = self.evaluations[idx]
        
        # Apply augmentation if enabled
        if self.augment and np.random.random() < 0.5:
            # Flip the board horizontally (mirror)
            board = self._flip_board(board)
            # Evaluation stays the same from white's perspective
        
        # Convert to tensor
        position_tensor = board_to_tensor(board)
        
        # Apply additional transforms if any
        if self.transform:
            position_tensor = self.transform(position_tensor)
        
        return torch.from_numpy(position_tensor).float(), torch.tensor(evaluation).float()
    
    def _flip_board(self, board: chess.Board) -> chess.Board:
        """Flip board horizontally for data augmentation."""
        flipped = chess.Board()
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                # Flip file (a-h becomes h-a)
                file = 7 - chess.square_file(square)
                rank = chess.square_rank(square)
                new_square = chess.square(file, rank)
                flipped.set_piece_at(new_square, piece)
        
        # Copy other board state
        flipped.turn = board.turn
        flipped.castling_rights = board.castling_rights
        flipped.ep_square = board.ep_square
        flipped.halfmove_clock = board.halfmove_clock
        flipped.fullmove_number = board.fullmove_number
        
        return flipped


class ChessHDF5Dataset(ChessDataset):
    """
    Dataset that loads chess positions from HDF5 files.
    
    HDF5 format allows efficient storage and loading of large datasets.
    """
    
    def __init__(
        self,
        hdf5_path: str,
        transform=None,
        load_to_memory: bool = False
    ):
        """
        Initialize HDF5 dataset.
        
        Args:
            hdf5_path: Path to HDF5 file
            transform: Optional transform to apply
            load_to_memory: Whether to load entire dataset to memory
        """
        super().__init__(transform)
        self.hdf5_path = hdf5_path
        self.load_to_memory = load_to_memory
        
        # Open file to get dataset info
        with h5py.File(hdf5_path, 'r') as f:
            self.length = len(f['positions'])
            
            # Load to memory if requested
            if load_to_memory:
                self.positions = f['positions'][:]
                self.evaluations = f['evaluations'][:]
                self.file = None
            else:
                self.positions = None
                self.evaluations = None
                self.file = None
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        """Get item from HDF5 file."""
        if self.load_to_memory:
            position = self.positions[idx]
            evaluation = self.evaluations[idx]
        else:
            # Open file if not already open
            if self.file is None:
                self.file = h5py.File(self.hdf5_path, 'r')
            
            position = self.file['positions'][idx]
            evaluation = self.file['evaluations'][idx]
        
        # Apply transform if any
        if self.transform:
            position = self.transform(position)
        
        return torch.from_numpy(position).float(), torch.tensor(evaluation).float()
    
    def __del__(self):
        """Close HDF5 file when dataset is deleted."""
        if self.file is not None:
            self.file.close()
    
    @staticmethod
    def create_from_positions(
        positions: List[chess.Board],
        evaluations: List[float],
        output_path: str,
        chunk_size: int = 1000
    ):
        """
        Create HDF5 dataset from positions and evaluations.
        
        Args:
            positions: List of chess positions
            evaluations: List of evaluation scores
            output_path: Path to save HDF5 file
            chunk_size: Chunk size for HDF5 storage
        """
        # Convert positions to tensors
        position_tensors = [board_to_tensor(board) for board in positions]
        
        # Get shape info
        n_positions = len(positions)
        tensor_shape = position_tensors[0].shape
        
        # Create HDF5 file
        with h5py.File(output_path, 'w') as f:
            # Create datasets
            pos_dataset = f.create_dataset(
                'positions',
                shape=(n_positions, *tensor_shape),
                dtype='float32',
                chunks=(min(chunk_size, n_positions), *tensor_shape),
                compression='gzip'
            )
            
            eval_dataset = f.create_dataset(
                'evaluations',
                shape=(n_positions,),
                dtype='float32',
                chunks=(min(chunk_size * 10, n_positions),),
                compression='gzip'
            )
            
            # Write data in chunks
            for i in range(0, n_positions, chunk_size):
                end_idx = min(i + chunk_size, n_positions)
                pos_dataset[i:end_idx] = position_tensors[i:end_idx]
                eval_dataset[i:end_idx] = evaluations[i:end_idx]
            
            # Add metadata
            f.attrs['total_positions'] = n_positions
            f.attrs['tensor_shape'] = tensor_shape