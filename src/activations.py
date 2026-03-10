"""
Activation Functions Module
ReLU, Sigmoid, Softmax, and Tanh implementations.
"""

import numpy as np


class ReLU:
    """Rectified Linear Unit activation."""

    def __init__(self):
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        return d_output * (self.input > 0).astype(float)


class Sigmoid:
    """Sigmoid activation."""

    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
        return self.output

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        return d_output * self.output * (1.0 - self.output)


class Tanh:
    """Hyperbolic tangent activation."""

    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.output = np.tanh(x)
        return self.output

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        return d_output * (1.0 - self.output ** 2)


class Softmax:
    """Softmax activation (typically used with cross-entropy loss)."""

    def __init__(self):
        self.output = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return self.output

    def backward(self, d_output: np.ndarray) -> np.ndarray:
        # When paired with cross-entropy, gradient is (y_pred - y_true)
        return d_output
