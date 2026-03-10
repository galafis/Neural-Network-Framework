"""
Loss Functions Module
MSE and Cross-Entropy loss implementations.
"""

import numpy as np


class MSELoss:
    """Mean Squared Error loss."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = y_pred
        self.y_true = y_true
        return np.mean((y_pred - y_true) ** 2)

    def backward(self) -> np.ndarray:
        n = self.y_pred.shape[0]
        return 2.0 * (self.y_pred - self.y_true) / n


class CrossEntropyLoss:
    """Categorical cross-entropy loss (expects softmax output)."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Args:
            y_pred: Softmax probabilities (batch_size, num_classes).
            y_true: One-hot encoded targets (batch_size, num_classes).
        """
        self.y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        self.y_true = y_true
        n = y_pred.shape[0]
        return -np.sum(y_true * np.log(self.y_pred)) / n

    def backward(self) -> np.ndarray:
        """Gradient for softmax + cross-entropy combined."""
        n = self.y_pred.shape[0]
        return (self.y_pred - self.y_true) / n


class BinaryCrossEntropyLoss:
    """Binary cross-entropy loss (expects sigmoid output)."""

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
        self.y_pred = np.clip(y_pred, 1e-12, 1.0 - 1e-12)
        self.y_true = y_true
        n = y_pred.shape[0]
        return -np.sum(
            y_true * np.log(self.y_pred) + (1 - y_true) * np.log(1 - self.y_pred)
        ) / n

    def backward(self) -> np.ndarray:
        n = self.y_pred.shape[0]
        return (self.y_pred - self.y_true) / (
            self.y_pred * (1.0 - self.y_pred) * n + 1e-12
        )
