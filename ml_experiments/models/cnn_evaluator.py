"""
CNN-based chess position evaluator.

This module implements a convolutional neural network for chess position evaluation,
inspired by the AlphaZero architecture but simplified for easier training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import chess
from typing import Dict, List, Optional, Any

from .base import ChessEvaluator, TorchChessModel
from ..utils.board_representation import board_to_tensor


class ResidualBlock(nn.Module):
    """Residual block with convolutional layers."""
    
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class CNNEvaluator(ChessEvaluator, TorchChessModel):
    """
    CNN-based chess position evaluator.
    
    Architecture:
    - Input: 8x8xN tensor (N channels for piece positions and metadata)
    - Convolutional layers with residual connections
    - Value head outputting position evaluation
    """
    
    def __init__(
        self,
        input_channels: int = 115,  # Default for 8 history positions + metadata
        num_residual_blocks: int = 10,
        channels: int = 256,
        value_head_hidden_size: int = 256
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_residual_blocks = num_residual_blocks
        self.channels = channels
        self.value_head_hidden_size = value_head_hidden_size
        
        # Initial convolution
        self.conv_initial = nn.Conv2d(input_channels, channels, kernel_size=3, padding=1)
        self.bn_initial = nn.BatchNorm2d(channels)
        
        # Residual blocks
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(channels) for _ in range(num_residual_blocks)
        ])
        
        # Value head
        self.value_conv = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(64, value_head_hidden_size)
        self.value_fc2 = nn.Linear(value_head_hidden_size, 1)
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 8, 8, channels)
            
        Returns:
            Value output of shape (batch_size, 1)
        """
        # Reshape from (B, 8, 8, C) to (B, C, 8, 8)
        x = x.permute(0, 3, 1, 2)
        
        # Initial convolution
        x = F.relu(self.bn_initial(self.conv_initial(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return value
    
    def evaluate(self, board: chess.Board) -> float:
        """
        Evaluate a single chess position.
        
        Args:
            board: Chess position to evaluate
            
        Returns:
            Evaluation score between -1 and 1
        """
        # Convert board to tensor
        board_tensor = board_to_tensor(board)
        
        # Add batch dimension and convert to torch tensor
        x = torch.from_numpy(board_tensor).unsqueeze(0).float().to(self.device)
        
        # Get evaluation
        with torch.no_grad():
            value = self.forward(x)
        
        return value.item()
    
    def evaluate_batch(self, boards: List[chess.Board]) -> np.ndarray:
        """
        Evaluate multiple positions at once for efficiency.
        
        Args:
            boards: List of chess positions
            
        Returns:
            Array of evaluation scores
        """
        # Convert all boards to tensors
        board_tensors = [board_to_tensor(board) for board in boards]
        
        # Stack into batch
        x = torch.from_numpy(np.stack(board_tensors)).float().to(self.device)
        
        # Get evaluations
        with torch.no_grad():
            values = self.forward(x)
        
        return values.cpu().numpy().squeeze()
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Perform one training step.
        
        Args:
            batch: Dictionary containing:
                - 'positions': Tensor of board positions
                - 'values': Target values
                - 'optimizer': Torch optimizer (optional)
                
        Returns:
            Dictionary of metrics
        """
        positions = batch['positions'].to(self.device)
        target_values = batch['values'].to(self.device)
        
        # Forward pass
        predicted_values = self.forward(positions)
        
        # Calculate loss
        value_loss = F.mse_loss(predicted_values.squeeze(), target_values)
        
        # Backward pass if optimizer provided
        if 'optimizer' in batch:
            optimizer = batch['optimizer']
            optimizer.zero_grad()
            value_loss.backward()
            optimizer.step()
        
        return {
            'value_loss': value_loss.item(),
            'mean_absolute_error': F.l1_loss(predicted_values.squeeze(), target_values).item()
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            'input_channels': self.input_channels,
            'num_residual_blocks': self.num_residual_blocks,
            'channels': self.channels,
            'value_head_hidden_size': self.value_head_hidden_size
        }
    
    def load_config(self, config: Dict[str, Any]) -> None:
        """Load model configuration."""
        # Configuration is set during initialization
        pass