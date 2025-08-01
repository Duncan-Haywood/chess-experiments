package eval

import "github.com/notnil/chess"

type StockEvaluator struct{}

func (s StockEvaluator) Eval(p *chess.Position) int {
	return 0
}