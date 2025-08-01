package eval

type Evaluator interface {
	Eval(p *chess.Position) int
}