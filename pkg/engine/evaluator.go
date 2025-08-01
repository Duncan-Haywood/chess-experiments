package engine

import (
	"github.com/notnil/chess"
)

// Evaluator interface for position evaluation
type Evaluator interface {
	Evaluate(game *chess.Game) float64
}

// MaterialEvaluator evaluates positions based on material balance
type MaterialEvaluator struct {
	pieceValues map[chess.PieceType]float64
}

// NewMaterialEvaluator creates a new material evaluator
func NewMaterialEvaluator() *MaterialEvaluator {
	return &MaterialEvaluator{
		pieceValues: map[chess.PieceType]float64{
			chess.Pawn:   1.0,
			chess.Knight: 3.0,
			chess.Bishop: 3.0,
			chess.Rook:   5.0,
			chess.Queen:  9.0,
			chess.King:   0.0, // King has no material value
		},
	}
}

// Evaluate returns the material evaluation of the position
func (e *MaterialEvaluator) Evaluate(game *chess.Game) float64 {
	// Check for game over states
	outcome := game.Outcome()
	if outcome == chess.WhiteWon {
		return 1000.0
	} else if outcome == chess.BlackWon {
		return -1000.0
	} else if outcome == chess.Draw {
		return 0.0
	}

	board := game.Position().Board()
	score := 0.0

	for sq := chess.A1; sq <= chess.H8; sq++ {
		piece := board.Piece(sq)
		if piece == chess.NoPiece {
			continue
		}

		value := e.pieceValues[piece.Type()]
		if piece.Color() == chess.White {
			score += value
		} else {
			score -= value
		}
	}

	// Return score from the perspective of the side to move
	if game.Position().Turn() == chess.Black {
		score = -score
	}

	return score
}

// PositionalEvaluator adds positional factors to material evaluation
type PositionalEvaluator struct {
	material *MaterialEvaluator
	pst      PieceSquareTables
}

// NewPositionalEvaluator creates a new positional evaluator
func NewPositionalEvaluator() *PositionalEvaluator {
	return &PositionalEvaluator{
		material: NewMaterialEvaluator(),
		pst:      NewPieceSquareTables(),
	}
}

// Evaluate returns the positional evaluation
func (e *PositionalEvaluator) Evaluate(game *chess.Game) float64 {
	// Start with material evaluation
	score := e.material.Evaluate(game)
	
	// Don't add positional factors for game over positions
	if score >= 900 || score <= -900 {
		return score
	}

	board := game.Position().Board()
	positionalScore := 0.0

	// Add piece-square table values
	for sq := chess.A1; sq <= chess.H8; sq++ {
		piece := board.Piece(sq)
		if piece == chess.NoPiece {
			continue
		}

		psValue := e.pst.GetValue(piece, sq)
		if piece.Color() == chess.White {
			positionalScore += psValue
		} else {
			positionalScore -= psValue
		}
	}

	// Add mobility bonus (number of legal moves)
	mobilityBonus := float64(len(game.ValidMoves())) * 0.1

	// Combine scores
	totalScore := score + positionalScore/100.0 + mobilityBonus

	// Return from perspective of side to move
	if game.Position().Turn() == chess.Black {
		totalScore = -totalScore
	}

	return totalScore
}