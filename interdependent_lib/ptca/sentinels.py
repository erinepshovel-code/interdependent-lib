"""
Sentinel channel definitions and live state.

Nine sentinel channels gate and annotate every tensor exchange:

  S1_PROVENANCE  – origin hash + chain of custody
  S2_POLICY      – applicable policy rules
  S3_BOUNDS      – numeric bounds / constraint envelope
  S4_APPROVAL    – boolean gate (must be True to commit)
  S5_CONTEXT     – live context window tokens/summary
  S6_IDENTITY    – model / caller identity record
  S7_MEMORY      – persistent memory log
  S8_RISK        – running risk score [0.0, 1.0]
  S9_AUDIT       – append-only audit trail

SentinelState is a plain dataclass so it is trivially serialisable to
a dict (via ``dataclasses.asdict``) and reconstructable from one.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from interdependent_lib.ptca.constants import SENTINEL_NAMES


# ---------------------------------------------------------------------------
# Individual channel state
# ---------------------------------------------------------------------------

@dataclass
class S1ProvenanceState:
    """Origin hash and chain-of-custody list."""
    origin_hash: str = ""
    chain: list[str] = field(default_factory=list)

    def append(self, hash_value: str) -> None:
        self.chain.append(hash_value)


@dataclass
class S2PolicyState:
    """Active policy rule identifiers."""
    rules: list[str] = field(default_factory=list)

    def set_rules(self, rules: list[str]) -> None:
        self.rules = list(rules)


@dataclass
class S3BoundsState:
    """Numeric constraint envelope."""
    lower: float = float("-inf")
    upper: float = float("inf")
    constraints: dict[str, Any] = field(default_factory=dict)

    def within(self, value: float) -> bool:
        return self.lower <= value <= self.upper


@dataclass
class S4ApprovalState:
    """Boolean approval gate."""
    approved: bool = False
    reason: str = ""

    def approve(self, reason: str = "") -> None:
        self.approved = True
        self.reason = reason

    def revoke(self, reason: str = "") -> None:
        self.approved = False
        self.reason = reason


@dataclass
class S5ContextState:
    """Live context window: ordered list of context entries."""
    entries: list[dict[str, Any]] = field(default_factory=list)
    max_entries: int = 256

    def push(self, entry: dict[str, Any]) -> None:
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries:]

    def clear(self) -> None:
        self.entries = []

    @property
    def token_count(self) -> int:
        return sum(e.get("tokens", 0) for e in self.entries)


@dataclass
class S6IdentityState:
    """Caller / model identity record."""
    model_id: str = ""
    caller_id: str = ""
    session_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def set_identity(
        self,
        model_id: str = "",
        caller_id: str = "",
        session_id: str = "",
        **metadata: Any,
    ) -> None:
        self.model_id = model_id
        self.caller_id = caller_id
        self.session_id = session_id
        self.metadata.update(metadata)


@dataclass
class S7MemoryState:
    """Persistent memory log: key-value store + ordered recall list."""
    store: dict[str, Any] = field(default_factory=dict)
    recall: list[dict[str, Any]] = field(default_factory=list)

    def remember(self, key: str, value: Any) -> None:
        self.store[key] = value

    def recall_entry(self, entry: dict[str, Any]) -> None:
        self.recall.append(entry)

    def retrieve(self, key: str, default: Any = None) -> Any:
        return self.store.get(key, default)


@dataclass
class S8RiskState:
    """Running risk score in [0.0, 1.0] and contributing factors."""
    score: float = 0.0
    factors: list[dict[str, Any]] = field(default_factory=list)

    def update(self, delta: float, factor: str = "", **details: Any) -> None:
        self.score = max(0.0, min(1.0, self.score + delta))
        self.factors.append({"delta": delta, "factor": factor, **details})

    def reset(self) -> None:
        self.score = 0.0
        self.factors = []


@dataclass
class S9AuditState:
    """Append-only audit trail."""
    log: list[dict[str, Any]] = field(default_factory=list)

    def record(self, event: str, **details: Any) -> None:
        self.log.append({
            "ts": time.time(),
            "event": event,
            **details,
        })

    def tail(self, n: int = 10) -> list[dict[str, Any]]:
        return self.log[-n:]


# ---------------------------------------------------------------------------
# Composite sentinel state (all nine channels together)
# ---------------------------------------------------------------------------

@dataclass
class SentinelState:
    """
    All nine sentinel channels as a single coherent unit.

    Attributes mirror SENTINEL_NAMES in order:
      s1 … s9
    """
    s1: S1ProvenanceState = field(default_factory=S1ProvenanceState)
    s2: S2PolicyState = field(default_factory=S2PolicyState)
    s3: S3BoundsState = field(default_factory=S3BoundsState)
    s4: S4ApprovalState = field(default_factory=S4ApprovalState)
    s5: S5ContextState = field(default_factory=S5ContextState)
    s6: S6IdentityState = field(default_factory=S6IdentityState)
    s7: S7MemoryState = field(default_factory=S7MemoryState)
    s8: S8RiskState = field(default_factory=S8RiskState)
    s9: S9AuditState = field(default_factory=S9AuditState)

    def channel(self, name: str) -> Any:
        """Return a channel by its sentinel name (e.g. ``'S5_CONTEXT'``)."""
        mapping = {
            "S1_PROVENANCE": self.s1,
            "S2_POLICY": self.s2,
            "S3_BOUNDS": self.s3,
            "S4_APPROVAL": self.s4,
            "S5_CONTEXT": self.s5,
            "S6_IDENTITY": self.s6,
            "S7_MEMORY": self.s7,
            "S8_RISK": self.s8,
            "S9_AUDIT": self.s9,
        }
        if name not in mapping:
            raise KeyError(f"Unknown sentinel: {name!r}. Valid names: {SENTINEL_NAMES}")
        return mapping[name]

    def to_dict(self) -> dict[str, Any]:
        """Shallow serialisation suitable for provenance blocks."""
        import dataclasses
        return {
            "S1_PROVENANCE": dataclasses.asdict(self.s1),
            "S2_POLICY": dataclasses.asdict(self.s2),
            "S3_BOUNDS": dataclasses.asdict(self.s3),
            "S4_APPROVAL": dataclasses.asdict(self.s4),
            "S5_CONTEXT": dataclasses.asdict(self.s5),
            "S6_IDENTITY": dataclasses.asdict(self.s6),
            "S7_MEMORY": dataclasses.asdict(self.s7),
            "S8_RISK": dataclasses.asdict(self.s8),
            "S9_AUDIT": dataclasses.asdict(self.s9),
        }
