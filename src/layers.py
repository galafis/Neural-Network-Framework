"""
Neural Network Layers Module
Dense layer implementation with forward and backward pass.
"""

import numpy as np
from typing import Optional, Tuple


class Dense:
    """Fully connected (dense) layer."""

    def __init__(self, input_size: int, output_size: int, seed: Optional[int] = None):
        rng = np.random.RandomState(seed)
        # He initialization
        self.weights = rng.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.biases = np.zeros((1, output_size))
        self.input = None
        # Gradient storage
        self.dweights = None
        self.dbiases = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass: output = x @ W + b."""
        self.input = x
        return x @ self.weights + self.biases

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        """Backward pass: compute gradients and return d_input."""
        self.dweights = self.input.T @ d_output
        self.dbiases = np.sum(d_output, axis=0, keepdims=True)
        return d_output @ self.weights.T

    @property
    def parameters(self) -> list:
        return [self.weights, self.biases]

    @property
    def gradients(self) -> list:
        return [self.dweights, self.dbiases]
