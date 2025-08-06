"""Evaluation package for chess positions."""

from .evaluator import Evaluator
from .stock_eval import StockEvaluator

__all__ = ['Evaluator', 'StockEvaluator']