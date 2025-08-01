package main

import (
	"embed"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"

	"github.com/notnil/chess"
	"github.com/Duncan-Haywood/chess-experiments/pkg/engine"
)

//go:embed static/*
var staticFiles embed.FS

type GameState struct {
	FEN      string   `json:"fen"`
	PGN      string   `json:"pgn"`
	Moves    []string `json:"moves"`
	Position int      `json:"position"`
}

type EvaluationResult struct {
	Score       float64 `json:"score"`
	BestMove    string  `json:"bestMove"`
	Depth       int     `json:"depth"`
	EngineName  string  `json:"engineName"`
	Evaluation  string  `json:"evaluation"`
}

func main() {
	// Serve static files
	http.Handle("/static/", http.FileServer(http.FS(staticFiles)))
	
	// API endpoints
	http.HandleFunc("/", serveIndex)
	http.HandleFunc("/api/analyze", handleAnalyze)
	http.HandleFunc("/api/game/load", handleLoadGame)
	
	fmt.Println("Chess UI server starting on http://localhost:8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}

func serveIndex(w http.ResponseWriter, r *http.Request) {
	http.ServeFile(w, r, "static/index.html")
}

func handleAnalyze(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var req struct {
		FEN        string `json:"fen"`
		Engine     string `json:"engine"`
		Depth      int    `json:"depth"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Default values
	if req.Engine == "" {
		req.Engine = "minimax"
	}
	if req.Depth == 0 {
		req.Depth = 4
	}
	
	// Parse FEN and create game
	fen, err := chess.FEN(req.FEN)
	if err != nil {
		http.Error(w, "Invalid FEN", http.StatusBadRequest)
		return
	}
	
	game := chess.NewGame(fen)
	
	// Create evaluator
	var evaluator engine.Evaluator
	evaluatorName := "material"
	if req.Engine == "positional" {
		evaluator = engine.NewPositionalEvaluator()
		evaluatorName = "positional"
	} else {
		evaluator = engine.NewMaterialEvaluator()
	}
	
	// Create engine and find best move
	eng := engine.NewMinimaxEngine(req.Depth, evaluator)
	bestMove, score := eng.FindBestMove(game)
	
	result := EvaluationResult{
		Score:      score,
		BestMove:   "",
		Depth:      req.Depth,
		EngineName: req.Engine + "_" + evaluatorName,
		Evaluation: fmt.Sprintf("%.2f", score),
	}
	
	if bestMove != nil {
		result.BestMove = bestMove.String()
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(result)
}

func handleLoadGame(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
		return
	}
	
	var req struct {
		PGN string `json:"pgn"`
	}
	
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	
	// Parse PGN
	pgn, err := chess.PGN(strings.NewReader(req.PGN))
	if err != nil {
		http.Error(w, "Invalid PGN", http.StatusBadRequest)
		return
	}
	
	game := chess.NewGame(pgn)
	moves := game.Moves()
	moveStrings := make([]string, len(moves))
	for i, move := range moves {
		moveStrings[i] = move.String()
	}
	
	state := GameState{
		FEN:      game.FEN(),
		PGN:      req.PGN,
		Moves:    moveStrings,
		Position: len(moves),
	}
	
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(state)
}

// Basic position evaluation (material only for now)
func evaluatePosition(game *chess.Game) float64 {
	board := game.Position().Board()
	score := 0.0
	
	pieceValues := map[chess.PieceType]float64{
		chess.Pawn:   1.0,
		chess.Knight: 3.0,
		chess.Bishop: 3.0,
		chess.Rook:   5.0,
		chess.Queen:  9.0,
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