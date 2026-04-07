# GPT/Claude generated; context, prompt Erin Spencer
"""
PCNALayer — a single fully-connected layer with backpropagation.

Weights are stored as a flat list in row-major order:
    weights[i * in_size + j]  →  weight from input j to neuron i

Backpropagation computes:
    grad_inputs  : gradient w.r.t. layer inputs  (len = in_size)
    grad_weights : gradient w.r.t. weights        (len = out_size * in_size)
    grad_biases  : gradient w.r.t. biases         (len = out_size)
"""

from __future__ import annotations

import os

from interdependent_lib.pcna.activations import apply, apply_grad


class PCNALayer:
    """
    Fully-connected layer with a configurable activation.

    Parameters
    ----------
    in_size:
        Number of input features.
    out_size:
        Number of neurons (output features).
    activation:
        One of ``'relu'``, ``'sigmoid'``, ``'tanh'``, ``'linear'``.
    """

    def __init__(
        self,
        in_size: int,
        out_size: int,
        activation: str = "relu",
    ) -> None:
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation

        # Xavier initialisation scaled to [−limit, +limit]
        limit = (6.0 / (in_size + out_size)) ** 0.5
        raw = os.urandom(in_size * out_size * 4)
        self.weights: list[float] = [
            (int.from_bytes(raw[i * 4:(i + 1) * 4], "big") / 0xFFFFFFFF * 2 - 1) * limit
            for i in range(in_size * out_size)
        ]
        self.biases: list[float] = [0.0] * out_size

        # Cache for backprop
        self._last_inputs: list[float] = []
        self._last_pre_act: list[float] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: list[float]) -> list[float]:
        """
        Compute activations for this layer.

        Caches inputs and pre-activation values for :meth:`backward`.
        """
        if len(inputs) != self.in_size:
            raise ValueError(
                f"Expected {self.in_size} inputs, got {len(inputs)}"
            )
        self._last_inputs = inputs
        pre_act: list[float] = []
        outputs: list[float] = []
        for i in range(self.out_size):
            z = self.biases[i]
            for j in range(self.in_size):
                z += self.weights[i * self.in_size + j] * inputs[j]
            pre_act.append(z)
            outputs.append(apply(self.activation, z))
        self._last_pre_act = pre_act
        return outputs

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(
        self,
        grad_output: list[float],
    ) -> tuple[list[float], list[float], list[float]]:
        """
        Backpropagate gradients through this layer.

        Parameters
        ----------
        grad_output:
            Gradient of the loss w.r.t. this layer's output (len = out_size).

        Returns
        -------
        (grad_inputs, grad_weights, grad_biases)
        """
        if not self._last_inputs:
            raise RuntimeError("forward() must be called before backward()")

        # δ_i = grad_output_i * activation'(pre_act_i)
        delta = [
            grad_output[i] * apply_grad(self.activation, self._last_pre_act[i])
            for i in range(self.out_size)
        ]

        # grad_weights[i * in_size + j] = δ_i * input_j
        grad_weights = [
            delta[i] * self._last_inputs[j]
            for i in range(self.out_size)
            for j in range(self.in_size)
        ]

        # grad_biases[i] = δ_i
        grad_biases = list(delta)

        # grad_inputs[j] = Σ_i δ_i * weights[i, j]
        grad_inputs = [0.0] * self.in_size
        for i in range(self.out_size):
            for j in range(self.in_size):
                grad_inputs[j] += delta[i] * self.weights[i * self.in_size + j]

        return grad_inputs, grad_weights, grad_biases

    # ------------------------------------------------------------------
    # Parameter update
    # ------------------------------------------------------------------

    def update(
        self,
        grad_weights: list[float],
        grad_biases: list[float],
        learning_rate: float,
    ) -> None:
        """Apply a gradient-descent step."""
        for k in range(len(self.weights)):
            self.weights[k] -= learning_rate * grad_weights[k]
        for k in range(len(self.biases)):
            self.biases[k] -= learning_rate * grad_biases[k]

    # ------------------------------------------------------------------
    # Tensor view
    # ------------------------------------------------------------------

    def as_tensor(self) -> list[float]:
        """Flat representation: weights followed by biases."""
        return list(self.weights) + list(self.biases)

    def __repr__(self) -> str:
        return (
            f"PCNALayer(in={self.in_size}, out={self.out_size}, "
            f"activation={self.activation!r})"
        )
