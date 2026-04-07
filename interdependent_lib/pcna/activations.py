# GPT/Claude generated; context, prompt Erin Spencer
"""
Activation functions for PCNA layers.

All functions operate on plain Python floats or lists of floats.
No external dependencies.
"""

from __future__ import annotations

import math


def relu(x: float) -> float:
    return max(0.0, x)


def relu_grad(x: float) -> float:
    """Derivative of ReLU at pre-activation value x."""
    return 1.0 if x > 0.0 else 0.0


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def sigmoid_grad(x: float) -> float:
    """Derivative of sigmoid at pre-activation value x."""
    s = sigmoid(x)
    return s * (1.0 - s)


def tanh(x: float) -> float:
    return math.tanh(x)


def tanh_grad(x: float) -> float:
    """Derivative of tanh at pre-activation value x."""
    t = math.tanh(x)
    return 1.0 - t * t


def linear(x: float) -> float:
    return x


def linear_grad(x: float) -> float:
    return 1.0


_ACTIVATIONS = {
    "relu":    (relu,    relu_grad),
    "sigmoid": (sigmoid, sigmoid_grad),
    "tanh":    (tanh,    tanh_grad),
    "linear":  (linear,  linear_grad),
}


def apply(name: str, x: float) -> float:
    """Apply named activation function."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}. Valid: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name][0](x)


def apply_grad(name: str, x: float) -> float:
    """Apply derivative of named activation at pre-activation value x."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}. Valid: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name][1](x)
