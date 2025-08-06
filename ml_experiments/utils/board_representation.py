"""
Board representation utilities for neural networks.

This module provides functions to convert chess positions into tensor representations
suitable for neural network input, following approaches similar to AlphaZero.
"""

import numpy as np
import chess
from typing import Optional, Tuple, List


def board_to_tensor(board: chess.Board, history_length: int = 8) -> np.ndarray:
    """
    Convert a chess board to a tensor representation.
    
    The representation includes:
    - Current position (12 planes: 6 piece types × 2 colors)
    - History of positions (for detecting repetitions)
    - Metadata (castling rights, en passant, turn, move count)
    
    Args:
        board: python-chess Board object
        history_length: Number of previous positions to include
        
    Returns:
        Tensor of shape (8, 8, num_planes)
    """
    # Initialize tensor with appropriate number of planes
    # 12 planes per position (6 piece types × 2 colors)
    # Plus metadata planes
    num_position_planes = 12
    num_metadata_planes = 7  # castling (4), en passant (1), turn (1), move count (1)
    total_planes = num_position_planes * (history_length + 1) + num_metadata_planes
    
    tensor = np.zeros((8, 8, total_planes), dtype=np.float32)
    
    # Current position planes
    plane_idx = 0
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in range(1, 7):  # PAWN to KING
            piece_mask = board.pieces_mask(piece_type, color)
            for square in chess.scan_forward(piece_mask):
                row, col = divmod(square, 8)
                tensor[row, col, plane_idx] = 1.0
            plane_idx += 1
    
    # TODO: Add history planes (requires maintaining game history)
    # For now, skip to metadata planes
    plane_idx = num_position_planes * (history_length + 1)
    
    # Castling rights
    tensor[:, :, plane_idx] = float(board.has_kingside_castling_rights(chess.WHITE))
    tensor[:, :, plane_idx + 1] = float(board.has_queenside_castling_rights(chess.WHITE))
    tensor[:, :, plane_idx + 2] = float(board.has_kingside_castling_rights(chess.BLACK))
    tensor[:, :, plane_idx + 3] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # En passant
    if board.ep_square is not None:
        row, col = divmod(board.ep_square, 8)
        tensor[row, col, plane_idx + 4] = 1.0
    
    # Turn
    tensor[:, :, plane_idx + 5] = float(board.turn)
    
    # Move count (normalized)
    tensor[:, :, plane_idx + 6] = min(board.fullmove_number / 100.0, 1.0)
    
    return tensor


def tensor_to_board(tensor: np.ndarray) -> chess.Board:
    """
    Convert a tensor representation back to a chess board.
    
    Note: This only recovers the current position, not the full game history.
    
    Args:
        tensor: Tensor representation of shape (8, 8, num_planes)
        
    Returns:
        python-chess Board object
    """
    board = chess.Board.empty()
    
    # Reconstruct pieces from first 12 planes
    plane_idx = 0
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in range(1, 7):  # PAWN to KING
            for row in range(8):
                for col in range(8):
                    if tensor[row, col, plane_idx] > 0.5:
                        square = row * 8 + col
                        board.set_piece_at(square, chess.Piece(piece_type, color))
            plane_idx += 1
    
    # TODO: Reconstruct castling rights and other metadata
    
    return board


def encode_position(board: chess.Board) -> np.ndarray:
    """
    Encode a chess position as a flat feature vector.
    
    This is a simpler encoding suitable for traditional ML methods.
    
    Args:
        board: python-chess Board object
        
    Returns:
        1D numpy array of features
    """
    features = []
    
    # Piece counts
    for color in [chess.WHITE, chess.BLACK]:
        for piece_type in range(1, 7):
            count = len(board.pieces(piece_type, color))
            features.append(count)
    
    # Material balance
    material_balance = 0
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }
    
    for piece_type, value in piece_values.items():
        white_count = len(board.pieces(piece_type, chess.WHITE))
        black_count = len(board.pieces(piece_type, chess.BLACK))
        material_balance += value * (white_count - black_count)
    
    features.append(material_balance)
    
    # Castling rights
    features.extend([
        float(board.has_kingside_castling_rights(chess.WHITE)),
        float(board.has_queenside_castling_rights(chess.WHITE)),
        float(board.has_kingside_castling_rights(chess.BLACK)),
        float(board.has_queenside_castling_rights(chess.BLACK))
    ])
    
    # Turn
    features.append(float(board.turn))
    
    # Move number
    features.append(board.fullmove_number)
    
    return np.array(features, dtype=np.float32)


def decode_position(features: np.ndarray) -> dict:
    """
    Decode a feature vector back to position information.
    
    Note: This returns a dictionary of position features, not a full Board object.
    
    Args:
        features: 1D feature vector
        
    Returns:
        Dictionary containing position information
    """
    idx = 0
    position_info = {}
    
    # Piece counts
    for color in ['white', 'black']:
        position_info[f'{color}_pieces'] = {}
        for piece_type in ['pawn', 'knight', 'bishop', 'rook', 'queen', 'king']:
            position_info[f'{color}_pieces'][piece_type] = int(features[idx])
            idx += 1
    
    # Material balance
    position_info['material_balance'] = features[idx]
    idx += 1
    
    # Castling rights
    position_info['castling'] = {
        'white_kingside': bool(features[idx]),
        'white_queenside': bool(features[idx + 1]),
        'black_kingside': bool(features[idx + 2]),
        'black_queenside': bool(features[idx + 3])
    }
    idx += 4
    
    # Turn
    position_info['white_to_move'] = bool(features[idx])
    idx += 1
    
    # Move number
    position_info['move_number'] = int(features[idx])
    
    return position_info