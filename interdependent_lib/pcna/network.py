# GPT/Claude generated; context, prompt Erin Spencer
"""
PCNANetwork — a back-propagating neural network composed of PCNALayers.

Typical usage
-------------
::

    from interdependent_lib.pcna.network import PCNANetwork

    net = PCNANetwork(layer_sizes=[4, 8, 4, 2], activations=["relu", "relu", "linear"])
    outputs = net.forward([0.5, 0.1, 0.9, 0.3])

    loss, loss_grad = net.mse_loss(outputs, targets=[1.0, 0.0])
    net.backward(loss_grad)
    net.update(learning_rate=0.01)
"""

from __future__ import annotations

from typing import Any

from interdependent_lib.pcna.layer import PCNALayer


class PCNANetwork:
    """
    Multi-layer perceptron with back-propagation.

    Parameters
    ----------
    layer_sizes:
        List of layer widths including input.  E.g. ``[4, 8, 2]`` creates
        one hidden layer of width 8 and an output layer of width 2.
    activations:
        Activation for each layer transition (len = len(layer_sizes) - 1).
        Defaults to ``'relu'`` for hidden layers and ``'linear'`` for output.
    """

    def __init__(
        self,
        layer_sizes: list[int],
        activations: list[str] | None = None,
    ) -> None:
        if len(layer_sizes) < 2:
            raise ValueError("layer_sizes must have at least 2 entries")

        n_layers = len(layer_sizes) - 1
        if activations is None:
            activations = ["relu"] * (n_layers - 1) + ["linear"]
        if len(activations) != n_layers:
            raise ValueError(
                f"Expected {n_layers} activations, got {len(activations)}"
            )

        self.layers: list[PCNALayer] = [
            PCNALayer(layer_sizes[i], layer_sizes[i + 1], activations[i])
            for i in range(n_layers)
        ]
        self.layer_sizes = layer_sizes

        # Caches set during forward/backward
        self._last_grads_w: list[list[float]] = []
        self._last_grads_b: list[list[float]] = []

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, inputs: list[float]) -> list[float]:
        """Run a forward pass; returns the network output."""
        x = inputs
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # ------------------------------------------------------------------
    # Loss helpers
    # ------------------------------------------------------------------

    @staticmethod
    def mse_loss(
        outputs: list[float],
        targets: list[float],
    ) -> tuple[float, list[float]]:
        """
        Mean-squared error loss.

        Returns
        -------
        (loss, grad_output)
            ``loss`` is the scalar MSE.
            ``grad_output`` is dL/d(output) for backpropagation.
        """
        if len(outputs) != len(targets):
            raise ValueError("outputs and targets must have the same length")
        n = len(outputs)
        loss = sum((o - t) ** 2 for o, t in zip(outputs, targets)) / n
        grad = [(2.0 / n) * (o - t) for o, t in zip(outputs, targets)]
        return loss, grad

    @staticmethod
    def binary_cross_entropy(
        outputs: list[float],
        targets: list[float],
        eps: float = 1e-12,
    ) -> tuple[float, list[float]]:
        """
        Binary cross-entropy loss.  Outputs should be sigmoid-activated.

        Returns
        -------
        (loss, grad_output)
        """
        if len(outputs) != len(targets):
            raise ValueError("outputs and targets must have the same length")
        n = len(outputs)
        loss = -sum(
            t * (o + eps) ** 0 if o <= 0 else  # clip guard
            t * __import__("math").log(o + eps) + (1 - t) * __import__("math").log(1 - o + eps)
            for o, t in zip(outputs, targets)
        ) / n
        import math
        loss = -sum(
            t * math.log(max(o, eps)) + (1 - t) * math.log(max(1 - o, eps))
            for o, t in zip(outputs, targets)
        ) / n
        grad = [
            (-t / max(o, eps) + (1 - t) / max(1 - o, eps)) / n
            for o, t in zip(outputs, targets)
        ]
        return loss, grad

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, grad_output: list[float]) -> list[float]:
        """
        Backpropagate ``grad_output`` through all layers.

        Stores per-layer gradients internally for :meth:`update`.
        Returns the gradient w.r.t. the network inputs.
        """
        grads_w: list[list[float]] = []
        grads_b: list[list[float]] = []

        grad = grad_output
        for layer in reversed(self.layers):
            grad, gw, gb = layer.backward(grad)
            grads_w.insert(0, gw)
            grads_b.insert(0, gb)

        self._last_grads_w = grads_w
        self._last_grads_b = grads_b
        return grad

    # ------------------------------------------------------------------
    # Parameter update
    # ------------------------------------------------------------------

    def update(self, learning_rate: float = 0.01) -> None:
        """Apply gradient-descent step using gradients from last :meth:`backward`."""
        if not self._last_grads_w:
            raise RuntimeError("backward() must be called before update()")
        for layer, gw, gb in zip(self.layers, self._last_grads_w, self._last_grads_b):
            layer.update(gw, gb, learning_rate)

    # ------------------------------------------------------------------
    # Tensor view
    # ------------------------------------------------------------------

    def as_tensor(self) -> list[float]:
        """Flat concatenation of all layer tensors (weights + biases)."""
        result: list[float] = []
        for layer in self.layers:
            result.extend(layer.as_tensor())
        return result

    @property
    def parameter_count(self) -> int:
        """Total number of trainable parameters."""
        return sum(
            layer.in_size * layer.out_size + layer.out_size
            for layer in self.layers
        )

    def __repr__(self) -> str:
        return (
            f"PCNANetwork(sizes={self.layer_sizes}, "
            f"params={self.parameter_count})"
        )
