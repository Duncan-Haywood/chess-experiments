package main

import (
	"net/http"
)

func main() {
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		w.Write([]byte("Chess Game Reviewer UI - TODO"))
	})
	http.ListenAndServe(":8080", nil)
}

// TODO: Add endpoints for loading PGN, displaying board, etc.
// Integrate with frontend JS for chessboard.