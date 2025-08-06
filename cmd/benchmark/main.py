#!/usr/bin/env python3
"""
Chess engine benchmark tool.

This tool benchmarks different chess engines on various test positions
to measure performance and move quality.
"""

import sys
sys.path.append('/workspace')
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
import chess
from pkg.chess.game import Game
from pkg.engine import MaterialEvaluator, PositionalEvaluator, MinimaxEngine


@dataclass
class TestPosition:
    """Test position for benchmarking."""
    name: str
    fen: str
    best_move: Optional[str] = None  # Optional: known best move for validation


@dataclass
class BenchmarkResult:
    """Result from a benchmark run."""
    engine_name: str
    position: str
    depth: int
    time_seconds: float
    nodes_evaluated: int
    best_move: str
    score: float


def main():
    """Main benchmark function."""
    # Test positions - mix of tactical and positional
    positions = [
        TestPosition(
            name="Starting Position",
            fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        ),
        TestPosition(
            name="Italian Opening",
            fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"
        ),
        TestPosition(
            name="Middlegame Position",
            fen="r1bq1rk1/pp2nppp/2n1p3/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 8"
        ),
        TestPosition(
            name="Endgame - Rook vs Pawns",
            fen="8/pp3p1p/2p2kp1/8/8/5P2/PPP3PP/4R1K1 w - - 0 1"
        ),
        TestPosition(
            name="Tactical Position",
            fen="r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
        ),
    ]
    
    # Engines to test
    engines = [
        ("Material", MaterialEvaluator()),
        ("Positional", PositionalEvaluator()),
    ]
    
    # Test depths
    depths = [2, 3, 4]
    
    print("Chess Engine Benchmark")
    print("======================")
    print()
    
    results: List[BenchmarkResult] = []
    
    for pos in positions:
        print(f"Testing position: {pos.name}")
        print(f"FEN: {pos.fen}")
        print()
        
        try:
            board = chess.Board(pos.fen)
            game = Game(board)
        except ValueError as e:
            print(f"Invalid FEN for {pos.name}: {e}")
            continue
        
        for eng_name, evaluator in engines:
            for depth in depths:
                print(f"  {eng_name} (depth {depth}): ", end="", flush=True)
                
                minimax_engine = MinimaxEngine(depth, evaluator)
                
                start = time.time()
                best_move, score = minimax_engine.find_best_move(game)
                elapsed = time.time() - start
                
                move_str = "none"
                if best_move is not None:
                    move_str = str(best_move)
                
                result = BenchmarkResult(
                    engine_name=eng_name,
                    position=pos.name,
                    depth=depth,
                    time_seconds=elapsed,
                    nodes_evaluated=0,  # Not tracked in current implementation
                    best_move=move_str,
                    score=score
                )
                results.append(result)
                
                print(f"Move: {move_str}, Score: {score:.2f}, Time: {elapsed:.3f}s")
        
        print()
    
    # Summary
    print("\nSummary")
    print("=======")
    print()
    
    # Average time by engine and depth
    avg_times: Dict[str, Dict[int, float]] = {}
    counts: Dict[str, Dict[int, int]] = {}
    
    for result in results:
        if result.engine_name not in avg_times:
            avg_times[result.engine_name] = {}
            counts[result.engine_name] = {}
        
        if result.depth not in avg_times[result.engine_name]:
            avg_times[result.engine_name][result.depth] = 0.0
            counts[result.engine_name][result.depth] = 0
        
        avg_times[result.engine_name][result.depth] += result.time_seconds
        counts[result.engine_name][result.depth] += 1
    
    print("Average Time per Move:")
    for eng_name, _ in engines:
        print(f"\n{eng_name} Engine:")
        for depth in depths:
            if eng_name in counts and depth in counts[eng_name]:
                count = counts[eng_name][depth]
                if count > 0:
                    avg = avg_times[eng_name][depth] / count
                    print(f"  Depth {depth}: {avg:.3f}s")
    
    # Move agreement analysis
    print("\nMove Agreement Analysis:")
    for pos in positions:
        print(f"\n{pos.name}:")
        moves: Dict[str, List[str]] = {}
        
        for result in results:
            if result.position == pos.name:
                key = f"{result.engine_name}-d{result.depth}"
                if result.best_move not in moves:
                    moves[result.best_move] = []
                moves[result.best_move].append(key)
        
        for move, engine_list in moves.items():
            print(f"  {move}: {engine_list}")


if __name__ == "__main__":
    main()