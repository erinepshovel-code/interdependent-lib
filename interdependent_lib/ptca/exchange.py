"""
Exchange mechanics — weighted scoring across sentinel channels.

An *exchange* is a single routing event that writes a weighted score
into one or more tensor cells.  The score is computed from:

  score = DELTA * (
      ALPHA * s1_weight
    + BETA  * s2_weight
    + GAMMA * s5_weight
    + ALPHA * s8_weight
    + bonus
  )

where the sentinel weights are normalised values supplied by the
caller (typically 0.0–1.0) representing signal strength on each
channel.

The module also provides seed aggregation (AGG_SEEDS) and S6 identity
aggregation (AGG6), both currently "mean".

Typical usage
-------------
::

    from interdependent_lib.ptca.exchange import Exchange
    from interdependent_lib.ptca.tensor import PTCATensor
    from interdependent_lib.ptca.sentinels import SentinelState

    tensor = PTCATensor()
    state  = SentinelState()

    exc = Exchange(tensor, state)
    result = exc.route(
        node=0, phase=0, slot=0,
        s1=1.0, s2=0.5, s5=0.8, s8=0.1,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from interdependent_lib.ptca.constants import (
    ALPHA, BETA, GAMMA, DELTA,
    AGG6, AGG_SEEDS,
    SENTINEL_WEIGHTS,
)
from interdependent_lib.ptca.tensor import PTCATensor
from interdependent_lib.ptca.sentinels import SentinelState


# ---------------------------------------------------------------------------
# Exchange result
# ---------------------------------------------------------------------------

@dataclass
class ExchangeResult:
    """Outcome of a single routing exchange."""
    node: int
    sentinel_idx: int
    phase: int
    slot: int
    score: float
    components: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core exchange helpers
# ---------------------------------------------------------------------------

def compute_score(
    *,
    s1: float = 0.0,
    s2: float = 0.0,
    s3: float = 0.0,
    s5: float = 0.0,
    s8: float = 0.0,
    bonus: float = 0.0,
) -> tuple[float, dict[str, float]]:
    """
    Compute the scalar exchange score and its component breakdown.

    Parameters
    ----------
    s1:
        S1_PROVENANCE signal strength [0.0, 1.0].
    s2:
        S2_POLICY signal strength [0.0, 1.0].
    s3:
        S3_BOUNDS signal strength [0.0, 1.0].
    s5:
        S5_CONTEXT signal strength [0.0, 1.0].
    s8:
        S8_RISK signal strength [0.0, 1.0].
    bonus:
        Caller-supplied additive bonus.

    Returns
    -------
    (score, components)
        ``score`` is the final scalar.
        ``components`` is a dict of individual weighted contributions.
    """
    c1 = ALPHA * s1
    c2 = BETA  * s2
    c3 = GAMMA * s3
    c5 = GAMMA * s5
    c8 = ALPHA * s8

    components = {
        "s1": c1,
        "s2": c2,
        "s3": c3,
        "s5": c5,
        "s8": c8,
        "bonus": bonus,
    }
    score = DELTA * (c1 + c2 + c3 + c5 + c8 + bonus)
    return score, components


def aggregate_seeds(values: list[float], method: str = AGG_SEEDS) -> float:
    """
    Aggregate a list of seed values using *method* (``'mean'`` or ``'sum'``).

    Defaults to ``AGG_SEEDS`` (``'mean'``).
    """
    if not values:
        return 0.0
    if method == "mean":
        return sum(values) / len(values)
    if method == "sum":
        return sum(values)
    raise ValueError(f"Unknown aggregation method: {method!r}")


def aggregate_identity(values: list[float], method: str = AGG6) -> float:
    """
    Aggregate S6 identity scores using *method*.

    Defaults to ``AGG6`` (``'mean'``).
    """
    return aggregate_seeds(values, method)


# ---------------------------------------------------------------------------
# Stateful exchange router
# ---------------------------------------------------------------------------

class Exchange:
    """
    Routes a single tensor exchange, updating the tensor and sentinel
    state in one call.

    Parameters
    ----------
    tensor:
        The live ``PTCATensor`` to write into.
    sentinel_state:
        The live ``SentinelState`` whose channels inform and record
        each exchange.
    """

    def __init__(self, tensor: PTCATensor, sentinel_state: SentinelState) -> None:
        self.tensor = tensor
        self.state = sentinel_state

    def route(
        self,
        *,
        node: int,
        phase: int,
        slot: int,
        s1: float = 0.0,
        s2: float = 0.0,
        s3: float = 0.0,
        s5: float = 0.0,
        s8: float = 0.0,
        bonus: float = 0.0,
        sentinel_idx: int = 0,
        audit_event: str = "exchange",
        **audit_details: Any,
    ) -> ExchangeResult:
        """
        Compute a score, write it to the tensor, and record in S9 audit.

        Parameters
        ----------
        node:
            Prime-node index (0-based, 0–52).
        phase:
            Phase index (0-based, 0–7).
        slot:
            Heptagram slot index (0-based, 0–6).
        s1 … s8:
            Per-sentinel signal strengths used in scoring.
        bonus:
            Additive bonus to the raw score.
        sentinel_idx:
            Which sentinel axis to write the score into (default 0 = S1).
        audit_event:
            Label for the S9 audit entry.
        **audit_details:
            Extra fields appended to the S9 audit entry.

        Returns
        -------
        ExchangeResult
        """
        score, components = compute_score(
            s1=s1, s2=s2, s3=s3, s5=s5, s8=s8, bonus=bonus,
        )

        self.tensor.add(node, sentinel_idx, phase, slot, score)

        self.state.s9.record(
            audit_event,
            node=node,
            sentinel_idx=sentinel_idx,
            phase=phase,
            slot=slot,
            score=score,
            **audit_details,
        )

        return ExchangeResult(
            node=node,
            sentinel_idx=sentinel_idx,
            phase=phase,
            slot=slot,
            score=score,
            components=components,
        )

    def batch_route(
        self,
        exchanges: list[dict[str, Any]],
    ) -> list[ExchangeResult]:
        """
        Execute a list of exchange dicts in order.

        Each dict is passed as keyword arguments to :meth:`route`.
        """
        return [self.route(**exc) for exc in exchanges]
