package engine

import (
	"github.com/notnil/chess"
)

// PieceSquareTables provides positional values for pieces on different squares
type PieceSquareTables struct {
	tables map[chess.PieceType][64]float64
}

// NewPieceSquareTables creates piece-square tables with default values
func NewPieceSquareTables() PieceSquareTables {
	pst := PieceSquareTables{
		tables: make(map[chess.PieceType][64]float64),
	}

	// Pawn table (from white's perspective)
	pst.tables[chess.Pawn] = [64]float64{
		0, 0, 0, 0, 0, 0, 0, 0,
		50, 50, 50, 50, 50, 50, 50, 50,
		10, 10, 20, 30, 30, 20, 10, 10,
		5, 5, 10, 25, 25, 10, 5, 5,
		0, 0, 0, 20, 20, 0, 0, 0,
		5, -5, -10, 0, 0, -10, -5, 5,
		5, 10, 10, -20, -20, 10, 10, 5,
		0, 0, 0, 0, 0, 0, 0, 0,
	}

	// Knight table
	pst.tables[chess.Knight] = [64]float64{
		-50, -40, -30, -30, -30, -30, -40, -50,
		-40, -20, 0, 0, 0, 0, -20, -40,
		-30, 0, 10, 15, 15, 10, 0, -30,
		-30, 5, 15, 20, 20, 15, 5, -30,
		-30, 0, 15, 20, 20, 15, 0, -30,
		-30, 5, 10, 15, 15, 10, 5, -30,
		-40, -20, 0, 5, 5, 0, -20, -40,
		-50, -40, -30, -30, -30, -30, -40, -50,
	}

	// Bishop table
	pst.tables[chess.Bishop] = [64]float64{
		-20, -10, -10, -10, -10, -10, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 0, 5, 10, 10, 5, 0, -10,
		-10, 5, 5, 10, 10, 5, 5, -10,
		-10, 0, 10, 10, 10, 10, 0, -10,
		-10, 10, 10, 10, 10, 10, 10, -10,
		-10, 5, 0, 0, 0, 0, 5, -10,
		-20, -10, -10, -10, -10, -10, -10, -20,
	}

	// Rook table
	pst.tables[chess.Rook] = [64]float64{
		0, 0, 0, 0, 0, 0, 0, 0,
		5, 10, 10, 10, 10, 10, 10, 5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		-5, 0, 0, 0, 0, 0, 0, -5,
		0, 0, 0, 5, 5, 0, 0, 0,
	}

	// Queen table
	pst.tables[chess.Queen] = [64]float64{
		-20, -10, -10, -5, -5, -10, -10, -20,
		-10, 0, 0, 0, 0, 0, 0, -10,
		-10, 0, 5, 5, 5, 5, 0, -10,
		-5, 0, 5, 5, 5, 5, 0, -5,
		0, 0, 5, 5, 5, 5, 0, -5,
		-10, 5, 5, 5, 5, 5, 0, -10,
		-10, 0, 5, 0, 0, 0, 0, -10,
		-20, -10, -10, -5, -5, -10, -10, -20,
	}

	// King middle game table
	pst.tables[chess.King] = [64]float64{
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-30, -40, -40, -50, -50, -40, -40, -30,
		-20, -30, -30, -40, -40, -30, -30, -20,
		-10, -20, -20, -20, -20, -20, -20, -10,
		20, 20, 0, 0, 0, 0, 20, 20,
		20, 30, 10, 0, 0, 10, 30, 20,
	}

	return pst
}

// GetValue returns the piece-square value for a piece on a square
func (pst *PieceSquareTables) GetValue(piece chess.Piece, square chess.Square) float64 {
	table, exists := pst.tables[piece.Type()]
	if !exists {
		return 0
	}

	// Convert square to array index
	rank := int(square / 8)
	file := int(square % 8)
	
	// For black pieces, flip the board vertically
	if piece.Color() == chess.Black {
		rank = 7 - rank
	}
	
	index := rank*8 + file
	return table[index]
}

// squareToIndex converts a chess.Square to array index
func squareToIndex(sq chess.Square) int {
	return int(sq)
}