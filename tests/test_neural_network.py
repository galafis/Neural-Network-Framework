"""
Tests for the Neural Network Framework.
"""

import sys
import numpy as np
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from layers import Dense
from activations import ReLU, Sigmoid, Softmax, Tanh
from losses import MSELoss, CrossEntropyLoss, BinaryCrossEntropyLoss
from optimizers import SGD, Adam
from network import Sequential


class TestDenseLayer:
    def test_forward_shape(self):
        layer = Dense(4, 3, seed=42)
        x = np.random.randn(5, 4)
        output = layer.forward(x)
        assert output.shape == (5, 3)

    def test_backward_shape(self):
        layer = Dense(4, 3, seed=42)
        x = np.random.randn(5, 4)
        layer.forward(x)
        d_output = np.random.randn(5, 3)
        d_input = layer.backward(d_output)
        assert d_input.shape == (5, 4)
        assert layer.dweights.shape == (4, 3)
        assert layer.dbiases.shape == (1, 3)


class TestActivations:
    def test_relu_forward(self):
        relu = ReLU()
        x = np.array([[-1, 0, 1, 2]])
        out = relu.forward(x)
        np.testing.assert_array_equal(out, [[0, 0, 1, 2]])

    def test_relu_backward(self):
        relu = ReLU()
        x = np.array([[-1.0, 0.0, 1.0, 2.0]])
        relu.forward(x)
        d_out = np.ones_like(x)
        grad = relu.backward(d_out)
        np.testing.assert_array_equal(grad, [[0, 0, 1, 1]])

    def test_sigmoid_range(self):
        sig = Sigmoid()
        x = np.array([[-10, 0, 10]])
        out = sig.forward(x)
        assert np.all(out >= 0) and np.all(out <= 1)

    def test_sigmoid_backward(self):
        sig = Sigmoid()
        x = np.array([[0.0]])
        out = sig.forward(x)
        grad = sig.backward(np.ones_like(out))
        assert grad.shape == x.shape

    def test_tanh_range(self):
        tanh = Tanh()
        x = np.array([[-10, 0, 10]])
        out = tanh.forward(x)
        assert np.all(out >= -1) and np.all(out <= 1)

    def test_softmax_sums_to_one(self):
        sm = Softmax()
        x = np.array([[1, 2, 3], [4, 5, 6]])
        out = sm.forward(x)
        np.testing.assert_allclose(out.sum(axis=1), [1.0, 1.0], atol=1e-6)

    def test_softmax_positive(self):
        sm = Softmax()
        x = np.array([[-100, 0, 100]])
        out = sm.forward(x)
        assert np.all(out >= 0)


class TestLossFunctions:
    def test_mse_zero_loss(self):
        loss = MSELoss()
        y = np.array([[1.0, 2.0]])
        l = loss.forward(y, y)
        assert abs(l) < 1e-10

    def test_mse_backward_shape(self):
        loss = MSELoss()
        y_pred = np.array([[1.0, 2.0]])
        y_true = np.array([[0.0, 3.0]])
        loss.forward(y_pred, y_true)
        grad = loss.backward()
        assert grad.shape == y_pred.shape

    def test_cross_entropy_perfect(self):
        loss = CrossEntropyLoss()
        y_pred = np.array([[0.99, 0.01]])
        y_true = np.array([[1.0, 0.0]])
        l = loss.forward(y_pred, y_true)
        assert l < 0.1

    def test_cross_entropy_backward(self):
        loss = CrossEntropyLoss()
        y_pred = np.array([[0.7, 0.3]])
        y_true = np.array([[1.0, 0.0]])
        loss.forward(y_pred, y_true)
        grad = loss.backward()
        assert grad.shape == y_pred.shape


class TestOptimizers:
    def test_sgd_update(self):
        layer = Dense(2, 1, seed=42)
        optimizer = SGD(learning_rate=0.1)
        x = np.array([[1.0, 2.0]])
        layer.forward(x)
        layer.dweights = np.ones_like(layer.weights)
        layer.dbiases = np.ones_like(layer.biases)
        w_before = layer.weights.copy()
        optimizer.update([layer])
        assert not np.array_equal(layer.weights, w_before)

    def test_adam_update(self):
        layer = Dense(2, 1, seed=42)
        optimizer = Adam(learning_rate=0.01)
        x = np.array([[1.0, 2.0]])
        layer.forward(x)
        layer.dweights = np.ones_like(layer.weights)
        layer.dbiases = np.ones_like(layer.biases)
        w_before = layer.weights.copy()
        optimizer.update([layer])
        assert not np.array_equal(layer.weights, w_before)


class TestXORProblem:
    """Test that the network can learn XOR."""

    def test_xor_convergence(self):
        np.random.seed(42)
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
        y = np.array([[0], [1], [1], [0]], dtype=np.float64)

        model = Sequential()
        model.add(Dense(2, 8, seed=42))
        model.add(ReLU())
        model.add(Dense(8, 1, seed=43))
        model.add(Sigmoid())
        model.compile(optimizer="adam", loss="mse", learning_rate=0.05)

        history = model.fit(X, y, epochs=500, verbose=False)

        predictions = model.predict(X)
        rounded = (predictions > 0.5).astype(int)

        # Should correctly learn XOR
        assert history["loss"][-1] < history["loss"][0]
        np.testing.assert_array_equal(rounded.flatten(), y.flatten())


class TestNetworkSummary:
    def test_summary(self):
        model = Sequential()
        model.add(Dense(4, 8, seed=42))
        model.add(ReLU())
        model.add(Dense(8, 2, seed=43))
        model.add(Softmax())
        summary = model.summary()
        assert "Dense" in summary
        assert "Total parameters" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
