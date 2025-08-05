"""
Chess utility functions for ML experiments.

This module provides helper functions for working with chess positions,
games, and moves using the python-chess library.
"""

import chess
import chess.pgn
from typing import List, Optional, Iterator, Tuple
import io


def fen_to_board(fen: str) -> chess.Board:
    """
    Convert a FEN string to a chess board.
    
    Args:
        fen: FEN notation string
        
    Returns:
        python-chess Board object
    """
    return chess.Board(fen)


def board_to_fen(board: chess.Board) -> str:
    """
    Convert a chess board to FEN notation.
    
    Args:
        board: python-chess Board object
        
    Returns:
        FEN notation string
    """
    return board.fen()


def pgn_to_positions(pgn_text: str, include_evaluations: bool = False) -> List[Tuple[chess.Board, Optional[float]]]:
    """
    Extract all positions from a PGN game.
    
    Args:
        pgn_text: PGN format game text
        include_evaluations: Whether to extract evaluation scores if available
        
    Returns:
        List of (board, evaluation) tuples
    """
    positions = []
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    
    if game is None:
        return positions
    
    board = game.board()
    positions.append((board.copy(), None))
    
    for node in game.mainline():
        board.push(node.move)
        
        # Extract evaluation if requested and available
        evaluation = None
        if include_evaluations and node.comment:
            # Simple extraction - assumes eval in format [%eval X.XX]
            if "[%eval" in node.comment:
                try:
                    eval_str = node.comment.split("[%eval")[1].split("]")[0].strip()
                    evaluation = float(eval_str)
                except (IndexError, ValueError):
                    pass
        
        positions.append((board.copy(), evaluation))
    
    return positions


def is_valid_move(board: chess.Board, move_uci: str) -> bool:
    """
    Check if a move in UCI notation is valid for the current position.
    
    Args:
        board: Current board position
        move_uci: Move in UCI notation (e.g., "e2e4")
        
    Returns:
        True if move is valid, False otherwise
    """
    try:
        move = chess.Move.from_uci(move_uci)
        return move in board.legal_moves
    except ValueError:
        return False


def get_legal_moves(board: chess.Board) -> List[str]:
    """
    Get all legal moves in the current position.
    
    Args:
        board: Current board position
        
    Returns:
        List of moves in UCI notation
    """
    return [move.uci() for move in board.legal_moves]


def move_to_index(move: chess.Move) -> int:
    """
    Convert a move to a unique index (for neural network output).
    
    Uses a simple encoding: from_square * 64 + to_square + promotion_offset
    
    Args:
        move: chess.Move object
        
    Returns:
        Integer index representing the move
    """
    index = move.from_square * 64 + move.to_square
    
    # Add offset for promotions
    if move.promotion:
        # Queen=1, Rook=2, Bishop=3, Knight=4
        promotion_offset = 64 * 64 * (move.promotion - 1)
        index += promotion_offset
    
    return index


def index_to_move(index: int, board: chess.Board) -> Optional[chess.Move]:
    """
    Convert an index back to a move.
    
    Args:
        index: Move index
        board: Current board position (needed to validate move)
        
    Returns:
        chess.Move object or None if invalid
    """
    # Handle promotions
    promotion = None
    if index >= 64 * 64:
        promotion_idx = index // (64 * 64)
        promotion = promotion_idx + 1  # 1=Queen, 2=Rook, etc.
        index = index % (64 * 64)
    
    from_square = index // 64
    to_square = index % 64
    
    move = chess.Move(from_square, to_square, promotion=promotion)
    
    # Validate move
    if move in board.legal_moves:
        return move
    return None


def get_game_result(board: chess.Board) -> Optional[float]:
    """
    Get the result of the game if it's finished.
    
    Args:
        board: Current board position
        
    Returns:
        1.0 for white win, 0.0 for black win, 0.5 for draw, None if game not finished
    """
    if not board.is_game_over():
        return None
    
    result = board.result()
    if result == "1-0":
        return 1.0
    elif result == "0-1":
        return 0.0
    else:  # Draw
        return 0.5


def mirror_board(board: chess.Board) -> chess.Board:
    """
    Mirror the board (swap colors).
    
    Useful for data augmentation in training.
    
    Args:
        board: Original board
        
    Returns:
        Mirrored board with colors swapped
    """
    mirrored = chess.Board.empty()
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Mirror the square vertically and swap color
            mirrored_square = chess.square_mirror(square)
            mirrored_piece = chess.Piece(piece.piece_type, not piece.color)
            mirrored.set_piece_at(mirrored_square, mirrored_piece)
    
    # Swap turn
    mirrored.turn = not board.turn
    
    # Mirror castling rights
    mirrored.castling_rights = 0
    if board.has_kingside_castling_rights(chess.WHITE):
        mirrored.castling_rights |= chess.BB_H1
    if board.has_queenside_castling_rights(chess.WHITE):
        mirrored.castling_rights |= chess.BB_A1
    if board.has_kingside_castling_rights(chess.BLACK):
        mirrored.castling_rights |= chess.BB_H8
    if board.has_queenside_castling_rights(chess.BLACK):
        mirrored.castling_rights |= chess.BB_A8
    
    # Mirror en passant
    if board.ep_square:
        mirrored.ep_square = chess.square_mirror(board.ep_square)
    
    return mirrored