"""
Neural Network Module
Sequential network with training loop.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple

from .layers import Dense
from .activations import ReLU, Sigmoid, Softmax, Tanh
from .losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from .optimizers import SGD, Adam


class Sequential:
    """Sequential neural network model."""

    def __init__(self):
        self.layers = []
        self.loss_fn = None
        self.optimizer = None
        self.history = {"loss": [], "val_loss": []}

    def add(self, layer):
        """Add a layer to the network."""
        self.layers.append(layer)
        return self

    def compile(self, optimizer="sgd", loss="mse", learning_rate=0.01, **kwargs):
        """Configure optimizer and loss."""
        if isinstance(optimizer, str):
            if optimizer.lower() == "sgd":
                self.optimizer = SGD(learning_rate=learning_rate, **kwargs)
            elif optimizer.lower() == "adam":
                self.optimizer = Adam(learning_rate=learning_rate, **kwargs)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer}")
        else:
            self.optimizer = optimizer

        if isinstance(loss, str):
            if loss.lower() == "mse":
                self.loss_fn = MSELoss()
            elif loss.lower() in ("crossentropy", "cross_entropy"):
                self.loss_fn = CrossEntropyLoss()
            elif loss.lower() in ("bce", "binary_crossentropy"):
                self.loss_fn = BinaryCrossEntropyLoss()
            else:
                raise ValueError(f"Unknown loss: {loss}")
        else:
            self.loss_fn = loss

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, d_output: np.ndarray):
        """Backward pass through all layers in reverse."""
        for layer in reversed(self.layers):
            d_output = layer.backward(d_output)

    def fit(self, x: np.ndarray, y: np.ndarray, epochs: int = 100,
            batch_size: Optional[int] = None, verbose: bool = True,
            validation_data: Optional[Tuple] = None) -> Dict:
        """
        Train the network.

        Args:
            x: Training input data.
            y: Training target data.
            epochs: Number of training epochs.
            batch_size: Mini-batch size (None for full batch).
            verbose: Print training progress.
            validation_data: Optional (x_val, y_val) tuple.

        Returns:
            Training history dictionary.
        """
        n_samples = x.shape[0]
        if batch_size is None:
            batch_size = n_samples

        self.history = {"loss": [], "val_loss": []}

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                output = self.forward(x_batch)

                # Compute loss
                loss = self.loss_fn.forward(output, y_batch)
                epoch_loss += loss
                n_batches += 1

                # Backward pass
                d_loss = self.loss_fn.backward()
                self.backward(d_loss)

                # Update parameters
                self.optimizer.update(self.layers)

            avg_loss = epoch_loss / n_batches
            self.history["loss"].append(avg_loss)

            # Validation
            if validation_data is not None:
                val_output = self.forward(validation_data[0])
                val_loss = self.loss_fn.forward(val_output, validation_data[1])
                self.history["val_loss"].append(val_loss)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - loss: {avg_loss:.6f}"
                if validation_data is not None:
                    msg += f" - val_loss: {val_loss:.6f}"
                print(msg)

        return self.history

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions."""
        return self.forward(x)

    def summary(self) -> str:
        """Print network architecture summary."""
        lines = ["=" * 50, "Network Summary", "=" * 50]
        total_params = 0
        for i, layer in enumerate(self.layers):
            name = layer.__class__.__name__
            if hasattr(layer, 'weights'):
                shape = layer.weights.shape
                params = layer.weights.size + layer.biases.size
                total_params += params
                lines.append(f"Layer {i}: {name} - shape {shape} - params: {params}")
            else:
                lines.append(f"Layer {i}: {name}")
        lines.append("=" * 50)
        lines.append(f"Total parameters: {total_params}")
        result = "\n".join(lines)
        return result
