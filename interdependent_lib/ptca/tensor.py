"""
PTCA Tensor — 53 × 9 × 8 × 7 routing structure.

Dimensions
----------
axis 0 – node    : 53 prime-indexed routing nodes
axis 1 – sentinel: 9 sentinel channels (S1–S9)
axis 2 – phase   : 8 processing phases
axis 3 – slot    : 7 heptagram slots

The tensor is backed by a flat list of floats for zero-dependency
efficiency. All indexing is done via the helper ``_idx``.

Each cell stores a float score initialised to 0.0.  Callers write
exchange results (weighted sums from exchange.py) into the tensor and
can later aggregate across any axis.
"""

from __future__ import annotations

from typing import Sequence

from interdependent_lib.ptca.constants import NODES, SENTINELS, PHASES, SLOTS


class PTCATensor:
    """
    Zero-dependency 4-D tensor backed by a flat Python list.

    Shape: (NODES, SENTINELS, PHASES, SLOTS) = (53, 9, 8, 7)
    Total cells: 26 796
    """

    SHAPE: tuple[int, int, int, int] = (NODES, SENTINELS, PHASES, SLOTS)
    SIZE: int = NODES * SENTINELS * PHASES * SLOTS

    def __init__(self) -> None:
        self._data: list[float] = [0.0] * self.SIZE

    # ------------------------------------------------------------------
    # Internal indexing
    # ------------------------------------------------------------------

    @staticmethod
    def _idx(node: int, sentinel: int, phase: int, slot: int) -> int:
        if not (0 <= node < NODES):
            raise IndexError(f"node {node} out of range [0, {NODES})")
        if not (0 <= sentinel < SENTINELS):
            raise IndexError(f"sentinel {sentinel} out of range [0, {SENTINELS})")
        if not (0 <= phase < PHASES):
            raise IndexError(f"phase {phase} out of range [0, {PHASES})")
        if not (0 <= slot < SLOTS):
            raise IndexError(f"slot {slot} out of range [0, {SLOTS})")
        return (
            node * (SENTINELS * PHASES * SLOTS)
            + sentinel * (PHASES * SLOTS)
            + phase * SLOTS
            + slot
        )

    # ------------------------------------------------------------------
    # Cell access
    # ------------------------------------------------------------------

    def get(self, node: int, sentinel: int, phase: int, slot: int) -> float:
        return self._data[self._idx(node, sentinel, phase, slot)]

    def set(self, node: int, sentinel: int, phase: int, slot: int, value: float) -> None:
        self._data[self._idx(node, sentinel, phase, slot)] = float(value)

    def add(self, node: int, sentinel: int, phase: int, slot: int, delta: float) -> None:
        idx = self._idx(node, sentinel, phase, slot)
        self._data[idx] += float(delta)

    # ------------------------------------------------------------------
    # Slice helpers (return plain lists — no numpy dependency)
    # ------------------------------------------------------------------

    def node_slice(self, node: int) -> list[float]:
        """All values for a given node (SENTINELS × PHASES × SLOTS cells)."""
        start = node * (SENTINELS * PHASES * SLOTS)
        return self._data[start: start + SENTINELS * PHASES * SLOTS]

    def sentinel_slice(self, sentinel: int) -> list[float]:
        """All values for a given sentinel channel across all nodes/phases/slots."""
        result: list[float] = []
        for n in range(NODES):
            for ph in range(PHASES):
                for sl in range(SLOTS):
                    result.append(self._data[self._idx(n, sentinel, ph, sl)])
        return result

    def phase_slice(self, phase: int) -> list[float]:
        """All values for a given phase across all nodes/sentinels/slots."""
        result: list[float] = []
        for n in range(NODES):
            for s in range(SENTINELS):
                for sl in range(SLOTS):
                    result.append(self._data[self._idx(n, s, phase, sl)])
        return result

    def slot_slice(self, slot: int) -> list[float]:
        """All values for a given heptagram slot across all nodes/sentinels/phases."""
        result: list[float] = []
        for n in range(NODES):
            for s in range(SENTINELS):
                for ph in range(PHASES):
                    result.append(self._data[self._idx(n, s, ph, slot)])
        return result

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    @staticmethod
    def _mean(values: Sequence[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    @staticmethod
    def _sum(values: Sequence[float]) -> float:
        return sum(values)

    def aggregate(
        self,
        method: str = "mean",
        *,
        node: int | None = None,
        sentinel: int | None = None,
        phase: int | None = None,
        slot: int | None = None,
    ) -> float:
        """
        Aggregate all cells matching the supplied fixed axes.

        Any axis left as ``None`` is summed/averaged over.
        ``method`` is ``'mean'`` or ``'sum'``.
        """
        nodes = [node] if node is not None else list(range(NODES))
        sentinels = [sentinel] if sentinel is not None else list(range(SENTINELS))
        phases = [phase] if phase is not None else list(range(PHASES))
        slots = [slot] if slot is not None else list(range(SLOTS))

        values = [
            self._data[self._idx(n, s, ph, sl)]
            for n in nodes
            for s in sentinels
            for ph in phases
            for sl in slots
        ]

        if method == "mean":
            return self._mean(values)
        if method == "sum":
            return self._sum(values)
        raise ValueError(f"Unknown aggregation method: {method!r}")

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Zero every cell."""
        self._data = [0.0] * self.SIZE

    def reset_node(self, node: int) -> None:
        """Zero all cells for a given node."""
        start = node * (SENTINELS * PHASES * SLOTS)
        for i in range(SENTINELS * PHASES * SLOTS):
            self._data[start + i] = 0.0

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"PTCATensor(shape={self.SHAPE}, "
            f"nonzero={sum(1 for v in self._data if v != 0.0)})"
        )

    def __len__(self) -> int:
        return self.SIZE
