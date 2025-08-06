#!/usr/bin/env python3
"""
Chess web UI server.

This module provides a Flask-based web server for the chess analysis UI.
"""

import sys
sys.path.append('/workspace')
import os
import json
from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import chess
import chess.pgn
import io
from pkg.chess.game import Game
from pkg.engine import MaterialEvaluator, PositionalEvaluator, MinimaxEngine


app = Flask(__name__, static_folder='static')
CORS(app)


@app.route('/')
def serve_index():
    """Serve the main index page."""
    static_path = os.path.join(app.static_folder, 'index.html')
    if os.path.exists(static_path):
        return send_file(static_path)
    return "Chess UI - index.html not found", 404


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files."""
    return send_from_directory('static', path)


@app.route('/api/analyze', methods=['POST'])
def handle_analyze():
    """Analyze a chess position."""
    data = request.get_json()
    
    fen = data.get('fen', '')
    engine_type = data.get('engine', 'minimax')
    depth = data.get('depth', 4)
    
    # Parse FEN and create game
    try:
        board = chess.Board(fen)
        game = Game(board)
    except ValueError:
        return jsonify({'error': 'Invalid FEN'}), 400
    
    # Create evaluator
    evaluator_name = 'material'
    if engine_type == 'positional':
        evaluator = PositionalEvaluator()
        evaluator_name = 'positional'
    else:
        evaluator = MaterialEvaluator()
    
    # Create engine and find best move
    engine = MinimaxEngine(depth, evaluator)
    best_move, score = engine.find_best_move(game)
    
    result = {
        'score': score,
        'bestMove': str(best_move) if best_move else '',
        'depth': depth,
        'engineName': f"{engine_type}_{evaluator_name}",
        'evaluation': f"{score:.2f}"
    }
    
    return jsonify(result)


@app.route('/api/game/load', methods=['POST'])
def handle_load_game():
    """Load a PGN game."""
    data = request.get_json()
    pgn_text = data.get('pgn', '')
    
    # Parse PGN
    try:
        pgn_io = io.StringIO(pgn_text)
        pgn_game = chess.pgn.read_game(pgn_io)
        if pgn_game is None:
            return jsonify({'error': 'Invalid PGN'}), 400
        
        # Extract moves
        board = pgn_game.board()
        moves = []
        for move in pgn_game.mainline_moves():
            moves.append(move.uci())
            board.push(move)
        
        # Get final position
        fen = board.fen()
        
        state = {
            'fen': fen,
            'pgn': pgn_text,
            'moves': moves,
            'position': len(moves)
        }
        
        return jsonify(state)
    
    except Exception as e:
        return jsonify({'error': f'Invalid PGN: {str(e)}'}), 400


def evaluate_position(game: Game) -> float:
    """
    Basic position evaluation (material only).
    
    Args:
        game: Game object
        
    Returns:
        Material evaluation score
    """
    board = game.position
    score = 0.0
    
    piece_values = {
        chess.PAWN: 1.0,
        chess.KNIGHT: 3.0,
        chess.BISHOP: 3.0,
        chess.ROOK: 5.0,
        chess.QUEEN: 9.0,
    }
    
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        
        value = piece_values.get(piece.piece_type, 0)
        if piece.color == chess.WHITE:
            score += value
        else:
            score -= value
    
    return score


def main():
    """Start the Flask web server."""
    print("Chess UI server starting on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080, debug=True)


if __name__ == '__main__':
    main()