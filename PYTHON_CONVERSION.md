# Python Conversion of Chess Engine

This document describes the Python conversion of the Go chess engine code.

## Converted Files

### Core Chess Package (`pkg/chess/`)
- `game.py` - Chess game wrapper around python-chess library
- `__init__.py` - Package initialization

### Engine Package (`pkg/engine/`)
- `evaluator.py` - Material and positional evaluators
- `minimax.py` - Minimax engine with alpha-beta pruning
- `piece_square_tables.py` - Positional evaluation tables
- `__init__.py` - Package initialization

### Evaluation Package (`pkg/eval/`)
- `evaluator.py` - Base evaluator interface
- `stock_eval.py` - Stock evaluator implementation
- `__init__.py` - Package initialization

### Command Line Tools (`cmd/`)
- `stock-eval/main.py` - Stock evaluation demo
- `benchmark/main.py` - Engine benchmark tool
- `web/main.py` - Flask-based web UI server

### Web Server (`web/`)
- `server.py` - Simple HTTP server implementation

### Additional Files
- `engine/minimax.py` - Simple minimax without alpha-beta pruning
- `requirements.txt` - Python dependencies
- `test_conversion.py` - Test script to verify conversion

## Key Changes from Go to Python

1. **Chess Library**: Using `python-chess` instead of `github.com/notnil/chess`
2. **Web Framework**: Using Flask instead of Go's built-in HTTP server
3. **Type Hints**: Added Python type hints for better code clarity
4. **Class Structure**: Converted Go structs to Python classes
5. **Error Handling**: Python exceptions instead of Go error returns

## Installation

```bash
pip install -r requirements.txt
```

## Running the Applications

### Stock Evaluation Demo
```bash
python cmd/stock-eval/main.py
```

### Benchmark Tool
```bash
python cmd/benchmark/main.py
```

### Web UI Server
```bash
python cmd/web/main.py
# Then open http://localhost:8080
```

### Test Script
```bash
python test_conversion.py
```

## Architecture

The Python conversion maintains the same architecture as the Go version:

- **Game Package**: Provides game state management
- **Engine Package**: Contains the chess engine logic (minimax, evaluators)
- **Eval Package**: Simple evaluation interfaces
- **CMD Package**: Command-line applications
- **Web Package**: Web server implementations

## Performance Notes

The Python version may be slower than the Go version due to:
- Python's interpreted nature vs Go's compiled code
- GIL (Global Interpreter Lock) limiting true parallelism
- Less efficient memory management

For production use, consider:
- Using PyPy for better performance
- Implementing critical paths in Cython
- Adding multiprocessing for parallel position analysis