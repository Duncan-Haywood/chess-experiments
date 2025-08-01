package main

import (
	"fmt"

	"github.com/georgesp/chess-experiments/pkg/eval"
	"github.com/notnil/chess"
)

// EvaluateBoard returns a simple score based on material balance.
// Positive score favors white, negative favors black.
// This can be replaced or augmented with ML models later.
func EvaluateBoard(game *chess.Game) float64 {
	board := game.Position().Board()
	score := 0.0

	// Piece values (basic centipawn equivalents)
	pieceValues := map[chess.PieceType]float64{
		chess.Pawn:   100,
		chess.Knight: 320,
		chess.Bishop: 330,
		chess.Rook:   500,
		chess.Queen:  900,
		// King not valued for material
	}

	for sq := chess.A1; sq <= chess.H8; sq++ {
		piece := board.Piece(sq)
		if piece == chess.NoPiece {
			continue
		}
		value := pieceValues[piece.Type()]
		if piece.Color() == chess.White {
			score += value
		} else {
			score -= value
		}
	}

	return score
}

func main() {
	game := chess.NewGame()
	// generate all valid moves
	validMoves := game.ValidMoves()
	// make a move
	game.Move(validMoves[0])
	// get material difference
	evaluator := eval.StockEvaluator{}
	fmt.Println(evaluator.Eval(game.Position()))
}
