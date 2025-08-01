package engine

import (
	"math"

	"github.com/notnil/chess"
)

// MinimaxEngine implements the minimax algorithm with alpha-beta pruning
type MinimaxEngine struct {
	depth     int
	evaluator Evaluator
}

// NewMinimaxEngine creates a new minimax engine
func NewMinimaxEngine(depth int, evaluator Evaluator) *MinimaxEngine {
	return &MinimaxEngine{
		depth:     depth,
		evaluator: evaluator,
	}
}

// FindBestMove returns the best move for the current position
func (e *MinimaxEngine) FindBestMove(game *chess.Game) (*chess.Move, float64) {
	moves := game.ValidMoves()
	if len(moves) == 0 {
		return nil, 0
	}

	bestMove := moves[0]
	bestScore := math.Inf(-1)
	alpha := math.Inf(-1)
	beta := math.Inf(1)

	for _, move := range moves {
		// Make the move
		newGame := game.Clone()
		newGame.Move(move)

		// Evaluate the position
		score := -e.minimax(newGame, e.depth-1, -beta, -alpha)

		// Update best move
		if score > bestScore {
			bestScore = score
			bestMove = move
		}

		// Update alpha
		if score > alpha {
			alpha = score
		}
	}

	return bestMove, bestScore
}

// minimax implements the minimax algorithm with alpha-beta pruning
func (e *MinimaxEngine) minimax(game *chess.Game, depth int, alpha, beta float64) float64 {
	// Terminal node - return evaluation
	if depth == 0 || game.Outcome() != chess.NoOutcome {
		return e.evaluator.Evaluate(game)
	}

	moves := game.ValidMoves()
	if len(moves) == 0 {
		return e.evaluator.Evaluate(game)
	}

	// Maximize
	bestScore := math.Inf(-1)
	for _, move := range moves {
		newGame := game.Clone()
		newGame.Move(move)

		score := -e.minimax(newGame, depth-1, -beta, -alpha)
		
		if score > bestScore {
			bestScore = score
		}

		if score > alpha {
			alpha = score
		}

		// Alpha-beta cutoff
		if alpha >= beta {
			break
		}
	}

	return bestScore
}

// SetDepth updates the search depth
func (e *MinimaxEngine) SetDepth(depth int) {
	e.depth = depth
}

// GetDepth returns the current search depth
func (e *MinimaxEngine) GetDepth() int {
	return e.depth
}