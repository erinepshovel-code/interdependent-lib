# GPT/Claude generated; context, prompt Erin Spencer
"""
PCTACircle — a circle of exactly 7 PCNANetworks, audited as a single tensor.

Each PCTA circle:
- Contains exactly 7 PCNA tensors (networks).
- Is itself a tensor: its flat representation is the concatenation of all
  7 member network tensors, used by the PTCA seed layer above it.
- Audits its members: computes summary statistics across the 7 networks
  and records discrepancies (gradient alignment, weight divergence, loss spread).

In the PTCA hierarchy:
    PCNA network  →  base tensor
    PCTA circle   →  7 PCNA tensors, audited as one tensor
    PTCA seed     →  7 PCTA circles
    PTCA core     →  53 seeds (4 sentinel seeds → 9 S-channels)
"""

from __future__ import annotations

from typing import Any

from interdependent_lib.pcna.network import PCNANetwork

CIRCLE_SIZE = 7  # number of PCNA members per circle


class PCTACircle:
    """
    A PCTA circle: 7 PCNA networks audited as a single tensor.

    Parameters
    ----------
    members:
        Exactly 7 :class:`~interdependent_lib.pcna.network.PCNANetwork` instances.
    circle_id:
        Optional identifier for this circle within its parent seed.
    """

    def __init__(
        self,
        members: list[PCNANetwork],
        circle_id: int = 0,
    ) -> None:
        if len(members) != CIRCLE_SIZE:
            raise ValueError(
                f"A PCTACircle requires exactly {CIRCLE_SIZE} PCNA members, "
                f"got {len(members)}"
            )
        self.members: list[PCNANetwork] = members
        self.circle_id = circle_id
        self._last_audit: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Tensor view
    # ------------------------------------------------------------------

    def as_tensor(self) -> list[float]:
        """
        Flat concatenation of all 7 member network tensors.

        This is the tensor representation used by the PTCA seed layer.
        """
        result: list[float] = []
        for net in self.members:
            result.extend(net.as_tensor())
        return result

    @property
    def tensor_size(self) -> int:
        """Total number of floats in this circle's tensor representation."""
        return sum(net.parameter_count for net in self.members)

    # ------------------------------------------------------------------
    # Forward pass across all members
    # ------------------------------------------------------------------

    def forward(self, inputs: list[list[float]]) -> list[list[float]]:
        """
        Run a forward pass through all 7 members.

        Parameters
        ----------
        inputs:
            List of 7 input vectors, one per member network.

        Returns
        -------
        List of 7 output vectors.
        """
        if len(inputs) != CIRCLE_SIZE:
            raise ValueError(
                f"Expected {CIRCLE_SIZE} input vectors, got {len(inputs)}"
            )
        return [net.forward(inp) for net, inp in zip(self.members, inputs)]

    # ------------------------------------------------------------------
    # Audit
    # ------------------------------------------------------------------

    def audit(self) -> dict[str, Any]:
        """
        Audit the 7 member networks and return a summary.

        Computes:
        - ``weight_norms``   : L2 norm of each member's weight vector
        - ``weight_mean``    : mean norm across members
        - ``weight_spread``  : max norm − min norm (divergence signal)
        - ``param_counts``   : parameter count per member
        - ``circle_id``      : this circle's identifier

        The audit result is cached in ``_last_audit`` for inspection.
        """
        norms: list[float] = []
        for net in self.members:
            t = net.as_tensor()
            norm = sum(x * x for x in t) ** 0.5
            norms.append(norm)

        mean_norm = sum(norms) / len(norms)
        spread = max(norms) - min(norms)
        param_counts = [net.parameter_count for net in self.members]

        audit: dict[str, Any] = {
            "circle_id": self.circle_id,
            "weight_norms": norms,
            "weight_mean": mean_norm,
            "weight_spread": spread,
            "param_counts": param_counts,
            "total_params": sum(param_counts),
        }
        self._last_audit = audit
        return audit

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return CIRCLE_SIZE

    def __repr__(self) -> str:
        return (
            f"PCTACircle(id={self.circle_id}, "
            f"members={CIRCLE_SIZE}, "
            f"tensor_size={self.tensor_size})"
        )
