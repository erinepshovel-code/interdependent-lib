"""Tests for interdependent_lib.pcna."""

import math
import pytest

from interdependent_lib.pcna.activations import (
    relu, relu_grad, sigmoid, sigmoid_grad, tanh, tanh_grad,
    linear, linear_grad, apply, apply_grad,
)
from interdependent_lib.pcna.layer import PCNALayer
from interdependent_lib.pcna.network import PCNANetwork


# ---------------------------------------------------------------------------
# Activations
# ---------------------------------------------------------------------------

class TestActivations:
    def test_relu_positive(self):
        assert relu(3.0) == 3.0

    def test_relu_negative(self):
        assert relu(-1.0) == 0.0

    def test_relu_grad(self):
        assert relu_grad(1.0) == 1.0
        assert relu_grad(-1.0) == 0.0

    def test_sigmoid_range(self):
        assert 0.0 < sigmoid(0.0) < 1.0
        assert sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)
        assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)

    def test_sigmoid_at_zero(self):
        assert sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_grad_at_zero(self):
        assert sigmoid_grad(0.0) == pytest.approx(0.25)

    def test_tanh_range(self):
        assert tanh(0.0) == pytest.approx(0.0)
        assert tanh(100.0) == pytest.approx(1.0, abs=1e-6)

    def test_tanh_grad_at_zero(self):
        assert tanh_grad(0.0) == pytest.approx(1.0)

    def test_linear(self):
        assert linear(5.5) == 5.5
        assert linear_grad(99.0) == 1.0

    def test_apply_dispatch(self):
        assert apply("relu", -1.0) == 0.0
        assert apply("linear", 3.0) == 3.0

    def test_apply_grad_dispatch(self):
        assert apply_grad("relu", 1.0) == 1.0
        assert apply_grad("linear", 99.0) == 1.0

    def test_apply_invalid(self):
        with pytest.raises(ValueError):
            apply("unknown", 1.0)

    def test_apply_grad_invalid(self):
        with pytest.raises(ValueError):
            apply_grad("unknown", 1.0)


# ---------------------------------------------------------------------------
# PCNALayer
# ---------------------------------------------------------------------------

class TestPCNALayer:
    def test_forward_output_size(self):
        layer = PCNALayer(4, 3, "relu")
        out = layer.forward([1.0, 2.0, 3.0, 4.0])
        assert len(out) == 3

    def test_forward_wrong_input_size(self):
        layer = PCNALayer(4, 3)
        with pytest.raises(ValueError):
            layer.forward([1.0, 2.0])

    def test_backward_requires_forward_first(self):
        layer = PCNALayer(2, 2)
        with pytest.raises(RuntimeError):
            layer.backward([1.0, 1.0])

    def test_backward_shapes(self):
        layer = PCNALayer(3, 2, "linear")
        layer.forward([1.0, 2.0, 3.0])
        grad_in, grad_w, grad_b = layer.backward([1.0, 1.0])
        assert len(grad_in) == 3
        assert len(grad_w) == 3 * 2
        assert len(grad_b) == 2

    def test_gradient_descent_reduces_loss(self):
        # Simple 1-in, 1-out linear layer: should converge to target weight
        layer = PCNALayer(1, 1, "linear")
        layer.weights = [0.0]
        layer.biases = [0.0]

        target = 2.0
        lr = 0.1
        for _ in range(50):
            out = layer.forward([1.0])
            grad_out = [2.0 * (out[0] - target)]  # MSE grad
            _, gw, gb = layer.backward(grad_out)
            layer.update(gw, gb, lr)

        out = layer.forward([1.0])
        assert out[0] == pytest.approx(target, abs=0.01)

    def test_as_tensor_length(self):
        layer = PCNALayer(3, 4)
        t = layer.as_tensor()
        assert len(t) == 3 * 4 + 4  # weights + biases

    def test_repr(self):
        layer = PCNALayer(2, 3, "sigmoid")
        assert "PCNALayer" in repr(layer)
        assert "sigmoid" in repr(layer)


# ---------------------------------------------------------------------------
# PCNANetwork
# ---------------------------------------------------------------------------

class TestPCNANetwork:
    def test_construction(self):
        net = PCNANetwork([4, 8, 2])
        assert len(net.layers) == 2

    def test_construction_needs_two_sizes(self):
        with pytest.raises(ValueError):
            PCNANetwork([4])

    def test_activation_count_mismatch(self):
        with pytest.raises(ValueError):
            PCNANetwork([4, 8, 2], activations=["relu"])

    def test_forward_output_size(self):
        net = PCNANetwork([3, 5, 2])
        out = net.forward([0.1, 0.2, 0.3])
        assert len(out) == 2

    def test_parameter_count(self):
        net = PCNANetwork([2, 3, 1])
        # layer0: 2*3 + 3 = 9; layer1: 3*1 + 1 = 4
        assert net.parameter_count == 13

    def test_mse_loss_perfect(self):
        loss, grad = PCNANetwork.mse_loss([1.0, 0.0], [1.0, 0.0])
        assert loss == pytest.approx(0.0)
        assert all(g == pytest.approx(0.0) for g in grad)

    def test_mse_loss_grad_direction(self):
        # output > target → gradient positive
        _, grad = PCNANetwork.mse_loss([2.0], [1.0])
        assert grad[0] > 0.0

    def test_backward_returns_input_grad(self):
        net = PCNANetwork([3, 4, 2])
        out = net.forward([0.5, 0.5, 0.5])
        _, loss_grad = PCNANetwork.mse_loss(out, [1.0, 0.0])
        input_grad = net.backward(loss_grad)
        assert len(input_grad) == 3

    def test_update_requires_backward(self):
        net = PCNANetwork([2, 2])
        with pytest.raises(RuntimeError):
            net.update()

    def test_training_converges(self):
        # Learn identity: output ≈ input for a single sample
        net = PCNANetwork([1, 4, 1], activations=["relu", "linear"])
        target = [3.0]
        inputs = [1.0]
        for _ in range(300):
            out = net.forward(inputs)
            loss, grad = PCNANetwork.mse_loss(out, target)
            net.backward(grad)
            net.update(learning_rate=0.05)
        out = net.forward(inputs)
        assert out[0] == pytest.approx(target[0], abs=0.5)

    def test_as_tensor_length(self):
        net = PCNANetwork([2, 3, 1])
        t = net.as_tensor()
        assert len(t) == net.parameter_count

    def test_repr(self):
        net = PCNANetwork([2, 4, 1])
        assert "PCNANetwork" in repr(net)
