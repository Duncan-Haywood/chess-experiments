package chess

import "github.com/notnil/chess"

type Game struct {
	*chess.Game
}

func NewGame() *Game {
	return &Game{
		Game: chess.NewGame(),
	}
}