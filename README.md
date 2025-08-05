# Chess Experiments

Just for fun experiments in ML chess engines. Maybe an LLM chess arena.

## Overview

This project provides a platform for experimenting with different chess engine algorithms, from traditional minimax with alpha-beta pruning to modern machine learning approaches. It includes:

- **Web UI**: Interactive chess board for game review and analysis (Go)
- **Traditional Algorithms**: Minimax with alpha-beta pruning (Go)
- **Evaluation Functions**: Material and positional evaluation (Go)
- **Benchmarking Tools**: Compare different engines and settings (Go)
- **ML Experiments**: Python-based machine learning models and training pipelines
  - CNN-based position evaluator (AlphaZero-inspired)
  - Traditional ML models (Random Forest, XGBoost, etc.)
  - Data loading and preprocessing utilities
  - Training scripts with TensorBoard integration
- **ML Report**: Comprehensive guide on neural network approaches

## Quick Start

### Prerequisites

- Go 1.23 or higher (for traditional engines and web UI)
- Python 3.8 or higher (for ML experiments)
- Make (optional, but recommended)
- Web browser for the UI
- CUDA-capable GPU (optional, for faster ML training)

### Installation

```bash
# Clone the repository
git clone https://github.com/Duncan-Haywood/chess-experiments.git
cd chess-experiments

# Install Go dependencies
go mod download

# Build Go components
make build

# Set up Python environment (recommended: use virtual environment)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python ML dependencies
pip install -r ml_experiments/requirements.txt
```

### Running the Web UI

```bash
# Start the web server
make run-web

# Or directly:
go run cmd/web/main.go
```

Open http://localhost:8080 in your browser. Features include:
- Interactive chess board
- PGN game loading
- Position analysis with different engines
- Move-by-move game review
- Real-time evaluation

### Running Benchmarks

```bash
# Run engine comparisons
make run-benchmark

# Or directly:
go run cmd/benchmark/main.go
```

This will test different engines across various positions and depths.

### Running ML Experiments

```bash
# Train a CNN evaluator
python -m ml_experiments.training.train_cnn \
    --data-path path/to/games.pgn \
    --epochs 50 \
    --batch-size 32

# Train a simple ML evaluator (Random Forest)
python -m ml_experiments.training.train_simple \
    --data-path path/to/games.pgn \
    --model-type random_forest

# Run example demonstrations
python ml_experiments/example_usage.py
```

## Project Structure

```
chess-experiments/
├── cmd/                    # Go application entry points
│   ├── web/               # Web UI server
│   ├── evaluator/         # Simple evaluation demo
│   └── benchmark/         # Engine comparison tool
├── pkg/                   # Go core library code
│   └── engine/           # Chess engine implementations
├── ml_experiments/        # Python ML experiments
│   ├── models/           # ML model implementations
│   │   ├── cnn_evaluator.py    # CNN-based evaluator
│   │   └── simple_evaluator.py # Traditional ML evaluators
│   ├── data/             # Data loading and preprocessing
│   ├── training/         # Training scripts and utilities
│   ├── utils/            # Helper functions
│   └── requirements.txt  # Python dependencies
├── docs/                  # Documentation
│   └── ml_approaches_report.md  # ML/NN approaches guide
└── static/               # Web UI assets
```

## Available Engines

### Go-based Engines

#### 1. Material Evaluator
- Simple piece counting
- Fast evaluation
- Good baseline for comparison

#### 2. Positional Evaluator
- Piece-square tables
- Mobility bonus
- Better positional understanding

### Python ML-based Engines

#### 1. CNN Evaluator
- Convolutional Neural Network with residual blocks
- Inspired by AlphaZero architecture
- Configurable depth and width
- Supports batch evaluation for efficiency

#### 2. Simple ML Evaluators
- Random Forest
- Gradient Boosting
- Neural Network (MLP)
- Linear Regression
- Hand-crafted features including material, position, and pawn structure

## Development

### Adding a New Go Engine

1. Implement the `Evaluator` interface in `pkg/engine/`
2. Add to benchmark suite in `cmd/benchmark/main.go`
3. Update web UI options in `cmd/web/main.go`

### Adding a New ML Model

1. Create a new model class inheriting from `ChessEvaluator` or `ChessPolicyNetwork` in `ml_experiments/models/`
2. Implement required methods: `evaluate()`, `train_step()`, `save()`, `load()`
3. Add to `ml_experiments/models/__init__.py`
4. Create a training script in `ml_experiments/training/`

### Running Tests

```bash
# Go tests
make test

# Python tests
pytest ml_experiments/
```

### Code Formatting

```bash
# Go formatting
make fmt

# Python formatting
black ml_experiments/
flake8 ml_experiments/
```

### Getting Chess Data

1. **Lichess Database**: Download monthly PGN archives from https://database.lichess.org/
2. **Use the download utility**:
   ```python
   from ml_experiments.data import download_lichess_database
   path = download_lichess_database(year=2024, month=1)
   ```

## Future Work

See [ML Approaches Report](docs/ml_approaches_report.md) for detailed plans on implementing:
- Convolutional Neural Networks
- Reinforcement Learning (AlphaZero-style)
- Transformer-based approaches
- Hybrid traditional/ML engines

## Contributing

Feel free to experiment and add new algorithms! The project is designed to make it easy to compare different approaches.

## License

MIT
