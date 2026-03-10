"""
Optimizers Module
SGD and Adam optimizer implementations.
"""

import numpy as np
from typing import List, Tuple


class SGD:
    """Stochastic Gradient Descent optimizer."""

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        self.lr = learning_rate
        self.momentum = momentum
        self.velocities = []

    def update(self, layers: list):
        """Update layer parameters using SGD."""
        if not self.velocities:
            for layer in layers:
                if hasattr(layer, 'parameters'):
                    layer_vels = [np.zeros_like(p) for p in layer.parameters]
                    self.velocities.append(layer_vels)
                else:
                    self.velocities.append(None)

        vel_idx = 0
        for layer in layers:
            if not hasattr(layer, 'parameters') or layer.gradients[0] is None:
                vel_idx += 1
                continue
            for i, (param, grad) in enumerate(zip(layer.parameters, layer.gradients)):
                if self.momentum > 0:
                    self.velocities[vel_idx][i] = (
                        self.momentum * self.velocities[vel_idx][i] - self.lr * grad
                    )
                    param += self.velocities[vel_idx][i]
                else:
                    param -= self.lr * grad
            vel_idx += 1


class Adam:
    """Adam optimizer."""

    def __init__(self, learning_rate: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, epsilon: float = 1e-8):
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = []
        self.v = []

    def update(self, layers: list):
        """Update parameters using Adam."""
        self.t += 1

        if not self.m:
            for layer in layers:
                if hasattr(layer, 'parameters'):
                    self.m.append([np.zeros_like(p) for p in layer.parameters])
                    self.v.append([np.zeros_like(p) for p in layer.parameters])
                else:
                    self.m.append(None)
                    self.v.append(None)

        idx = 0
        for layer in layers:
            if not hasattr(layer, 'parameters') or layer.gradients[0] is None:
                idx += 1
                continue
            for i, (param, grad) in enumerate(zip(layer.parameters, layer.gradients)):
                self.m[idx][i] = self.beta1 * self.m[idx][i] + (1 - self.beta1) * grad
                self.v[idx][i] = self.beta2 * self.v[idx][i] + (1 - self.beta2) * grad ** 2

                m_hat = self.m[idx][i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[idx][i] / (1 - self.beta2 ** self.t)

                param -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
            idx += 1
