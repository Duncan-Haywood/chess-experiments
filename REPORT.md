# Report on Neural Networks and ML Algorithms for Chess

## Traditional vs ML Approaches
Traditional chess engines use handcrafted evaluation functions and tree search. ML can enhance evaluation and move selection.

## Recommended NN/ML Algorithms
1. **Neural Network Evaluation (NNUE)**: Efficiently updatable neural networks for board evaluation, as in Stockfish. Simple feedforward nets trained on positions.

2. **Deep Reinforcement Learning (AlphaZero-style)**: Use MCTS with policy and value networks. Train via self-play. Requires significant compute.

3. **LLM-based (e.g., GPT models)**: Fine-tune LLMs on chess games for move prediction or analysis. Useful for 'LLM chess arena' idea.

4. **Other**: Convolutional NNs for board representation, or hybrid models combining traditional and ML.

## Sensible Choices for This Project
- Start with NNUE for evaluator replacement.
- Experiment with small AlphaZero-like setup using Go ML libraries like Gorgonia or Gonum.
- For LLM, integrate with APIs like OpenAI for quick prototypes.

## Next Steps
Implement a basic NN evaluator trainer using existing datasets.