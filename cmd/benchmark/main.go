package main

import (
	"fmt"
	"log"
	"time"

	"github.com/notnil/chess"
	"github.com/Duncan-Haywood/chess-experiments/pkg/engine"
)

type TestPosition struct {
	Name     string
	FEN      string
	BestMove string // Optional: known best move for validation
}

type BenchmarkResult struct {
	EngineName string
	Position   string
	Depth      int
	Time       time.Duration
	NodesEvaluated int
	BestMove   string
	Score      float64
}

func main() {
	// Test positions - mix of tactical and positional
	positions := []TestPosition{
		{
			Name: "Starting Position",
			FEN:  "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
		},
		{
			Name: "Italian Opening",
			FEN:  "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
		},
		{
			Name: "Middlegame Position",
			FEN:  "r1bq1rk1/pp2nppp/2n1p3/3p4/2PP4/2N1PN2/PP3PPP/R1BQKB1R w KQ - 0 8",
		},
		{
			Name: "Endgame - Rook vs Pawns",
			FEN:  "8/pp3p1p/2p2kp1/8/8/5P2/PPP3PP/4R1K1 w - - 0 1",
		},
		{
			Name: "Tactical Position",
			FEN:  "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
		},
	}

	// Engines to test
	engines := []struct {
		Name      string
		Evaluator engine.Evaluator
	}{
		{"Material", engine.NewMaterialEvaluator()},
		{"Positional", engine.NewPositionalEvaluator()},
	}

	// Test depths
	depths := []int{2, 3, 4}

	fmt.Println("Chess Engine Benchmark")
	fmt.Println("======================")
	fmt.Println()

	results := []BenchmarkResult{}

	for _, pos := range positions {
		fmt.Printf("Testing position: %s\n", pos.Name)
		fmt.Printf("FEN: %s\n", pos.FEN)
		fmt.Println()

		fen, err := chess.FEN(pos.FEN)
		if err != nil {
			log.Printf("Invalid FEN for %s: %v", pos.Name, err)
			continue
		}

		game := chess.NewGame(fen)

		for _, eng := range engines {
			for _, depth := range depths {
				fmt.Printf("  %s (depth %d): ", eng.Name, depth)
				
				minimaxEngine := engine.NewMinimaxEngine(depth, eng.Evaluator)
				
				start := time.Now()
				bestMove, score := minimaxEngine.FindBestMove(game)
				elapsed := time.Since(start)

				moveStr := "none"
				if bestMove != nil {
					moveStr = bestMove.String()
				}

				result := BenchmarkResult{
					EngineName: eng.Name,
					Position:   pos.Name,
					Depth:      depth,
					Time:       elapsed,
					BestMove:   moveStr,
					Score:      score,
				}
				results = append(results, result)

				fmt.Printf("Move: %s, Score: %.2f, Time: %v\n", moveStr, score, elapsed)
			}
		}
		fmt.Println()
	}

	// Summary
	fmt.Println("\nSummary")
	fmt.Println("=======")
	fmt.Println()

	// Average time by engine and depth
	avgTimes := make(map[string]map[int]time.Duration)
	counts := make(map[string]map[int]int)

	for _, result := range results {
		if avgTimes[result.EngineName] == nil {
			avgTimes[result.EngineName] = make(map[int]time.Duration)
			counts[result.EngineName] = make(map[int]int)
		}
		avgTimes[result.EngineName][result.Depth] += result.Time
		counts[result.EngineName][result.Depth]++
	}

	fmt.Println("Average Time per Move:")
	for _, eng := range engines {
		fmt.Printf("\n%s Engine:\n", eng.Name)
		for _, depth := range depths {
			if count := counts[eng.Name][depth]; count > 0 {
				avg := avgTimes[eng.Name][depth] / time.Duration(count)
				fmt.Printf("  Depth %d: %v\n", depth, avg)
			}
		}
	}

	// Move agreement analysis
	fmt.Println("\nMove Agreement Analysis:")
	for _, pos := range positions {
		fmt.Printf("\n%s:\n", pos.Name)
		moves := make(map[string][]string)
		
		for _, result := range results {
			if result.Position == pos.Name {
				key := fmt.Sprintf("%s-d%d", result.EngineName, result.Depth)
				moves[result.BestMove] = append(moves[result.BestMove], key)
			}
		}

		for move, engines := range moves {
			fmt.Printf("  %s: %v\n", move, engines)
		}
	}
}