"""
Generic trainer class for chess ML models.

This module provides a flexible trainer that can work with different model types
and training configurations.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from typing import Dict, Optional, Any, Callable, List
from dataclasses import dataclass
import os
from tqdm import tqdm
import time
import json

from ..models.base import ChessModel, TorchChessModel


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 100
    warmup_epochs: int = 5
    
    # Optimizer settings
    optimizer_type: str = "adam"  # adam, sgd, adamw
    weight_decay: float = 1e-4
    momentum: float = 0.9  # For SGD
    
    # Learning rate schedule
    lr_scheduler_type: str = "cosine"  # cosine, step, exponential, none
    lr_decay_factor: float = 0.1
    lr_decay_epochs: List[int] = None
    
    # Logging and checkpointing
    log_interval: int = 100
    eval_interval: int = 1000
    checkpoint_interval: int = 5000
    checkpoint_dir: str = "./checkpoints"
    tensorboard_dir: str = "./runs"
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_metric: str = "val_loss"
    early_stopping_mode: str = "min"  # min or max
    
    # Device settings
    device: str = "auto"  # auto, cuda, cpu
    num_workers: int = 4
    
    def __post_init__(self):
        if self.lr_decay_epochs is None:
            self.lr_decay_epochs = [30, 60, 90]
        
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


class Trainer:
    """Generic trainer for chess ML models."""
    
    def __init__(
        self,
        model: ChessModel,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        custom_loss_fn: Optional[Callable] = None
    ):
        """
        Initialize trainer.
        
        Args:
            model: Chess ML model to train
            config: Training configuration
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            custom_loss_fn: Optional custom loss function
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.custom_loss_fn = custom_loss_fn
        
        # Setup device
        self.device = torch.device(config.device)
        if isinstance(model, TorchChessModel):
            model.to(self.device)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup learning rate scheduler
        self.lr_scheduler = self._create_lr_scheduler()
        
        # Setup logging
        self.writer = SummaryWriter(config.tensorboard_dir)
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_metric = float('inf') if config.early_stopping_mode == 'min' else float('-inf')
        self.patience_counter = 0
    
    def _create_optimizer(self):
        """Create optimizer based on config."""
        if not isinstance(self.model, nn.Module):
            return None
        
        params = self.model.parameters()
        
        if self.config.optimizer_type == "adam":
            return torch.optim.Adam(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "adamw":
            return torch.optim.AdamW(
                params,
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type == "sgd":
            return torch.optim.SGD(
                params,
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {self.config.optimizer_type}")
    
    def _create_lr_scheduler(self):
        """Create learning rate scheduler based on config."""
        if self.optimizer is None or self.config.lr_scheduler_type == "none":
            return None
        
        if self.config.lr_scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.lr_scheduler_type == "step":
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.lr_decay_epochs,
                gamma=self.config.lr_decay_factor
            )
        elif self.config.lr_scheduler_type == "exponential":
            return torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=0.95
            )
        else:
            raise ValueError(f"Unknown scheduler type: {self.config.lr_scheduler_type}")
    
    def train(self):
        """Run the full training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Batch size: {self.config.batch_size}")
        print(f"Learning rate: {self.config.learning_rate}")
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            
            # Training phase
            train_metrics = self._train_epoch()
            
            # Validation phase
            if self.val_loader is not None:
                val_metrics = self._validate()
                self._check_early_stopping(val_metrics)
            else:
                val_metrics = {}
            
            # Update learning rate
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
            
            # Log epoch metrics
            self._log_epoch_metrics(train_metrics, val_metrics)
            
            # Save checkpoint
            if (epoch + 1) % self.config.checkpoint_interval == 0:
                self._save_checkpoint()
            
            # Early stopping check
            if self.patience_counter >= self.config.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Save final model
        self._save_checkpoint(is_final=True)
        self.writer.close()
        print("Training completed!")
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        if isinstance(self.model, nn.Module):
            self.model.train()
        
        epoch_metrics = {}
        batch_metrics = []
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Prepare batch
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                # Standard (input, target) format
                inputs, targets = batch
                batch_dict = {
                    'positions': inputs,
                    'values': targets,
                    'optimizer': self.optimizer
                }
            else:
                # Assume batch is already a dictionary
                batch_dict = batch
                if self.optimizer is not None:
                    batch_dict['optimizer'] = self.optimizer
            
            # Forward pass and optimization
            metrics = self.model.train_step(batch_dict)
            batch_metrics.append(metrics)
            
            # Update progress bar
            pbar.set_postfix(metrics)
            
            # Log to tensorboard
            if self.global_step % self.config.log_interval == 0:
                for key, value in metrics.items():
                    self.writer.add_scalar(f"train/{key}", value, self.global_step)
            
            self.global_step += 1
        
        # Aggregate epoch metrics
        for key in batch_metrics[0].keys():
            epoch_metrics[key] = np.mean([m[key] for m in batch_metrics])
        
        return epoch_metrics
    
    def _validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        if isinstance(self.model, nn.Module):
            self.model.eval()
        
        val_metrics = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Prepare batch
                if isinstance(batch, (list, tuple)) and len(batch) == 2:
                    inputs, targets = batch
                    batch_dict = {
                        'positions': inputs,
                        'values': targets
                    }
                else:
                    batch_dict = batch
                
                # Forward pass only
                metrics = self.model.train_step(batch_dict)
                val_metrics.append(metrics)
        
        # Aggregate validation metrics
        aggregated = {}
        for key in val_metrics[0].keys():
            aggregated[f"val_{key}"] = np.mean([m[key] for m in val_metrics])
        
        return aggregated
    
    def _check_early_stopping(self, val_metrics: Dict[str, float]):
        """Check early stopping criteria."""
        metric_key = f"val_{self.config.early_stopping_metric}"
        if metric_key not in val_metrics:
            return
        
        current_metric = val_metrics[metric_key]
        
        if self.config.early_stopping_mode == "min":
            is_better = current_metric < self.best_metric
        else:
            is_better = current_metric > self.best_metric
        
        if is_better:
            self.best_metric = current_metric
            self.patience_counter = 0
            self._save_checkpoint(is_best=True)
        else:
            self.patience_counter += 1
    
    def _log_epoch_metrics(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log metrics at end of epoch."""
        # Console logging
        log_str = f"Epoch {self.epoch}: "
        log_str += ", ".join([f"{k}={v:.4f}" for k, v in train_metrics.items()])
        if val_metrics:
            log_str += " | " + ", ".join([f"{k}={v:.4f}" for k, v in val_metrics.items()])
        print(log_str)
        
        # Tensorboard logging
        for key, value in train_metrics.items():
            self.writer.add_scalar(f"epoch/train_{key}", value, self.epoch)
        for key, value in val_metrics.items():
            self.writer.add_scalar(f"epoch/{key}", value, self.epoch)
        
        # Log learning rate
        if self.optimizer is not None:
            lr = self.optimizer.param_groups[0]['lr']
            self.writer.add_scalar("epoch/learning_rate", lr, self.epoch)
    
    def _save_checkpoint(self, is_best: bool = False, is_final: bool = False):
        """Save model checkpoint."""
        if is_best:
            path = os.path.join(self.config.checkpoint_dir, "best_model.pt")
        elif is_final:
            path = os.path.join(self.config.checkpoint_dir, "final_model.pt")
        else:
            path = os.path.join(self.config.checkpoint_dir, f"checkpoint_epoch_{self.epoch}.pt")
        
        # Save model
        self.model.save(path)
        
        # Save training state
        state_path = path.replace(".pt", "_state.json")
        state = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "best_metric": self.best_metric,
            "config": self.config.__dict__
        }
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        print(f"Checkpoint saved to {path}")