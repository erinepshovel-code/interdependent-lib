"""
PTCAInstance — a PTCA-aware stateful model session.

A ``PTCAInstance`` wraps a single model session and carries live
sentinel state — S5 context window, S6 identity, S7 memory, S8 risk
score, S9 audit log — as first-class fields.  It also owns the
``PTCATensor`` and ``Exchange`` router so that every exchange is
automatically reflected in the tensor and audit trail.

``PTCAInstance`` is designed to be used standalone **or** composed
with ``aimmh_lib.ModelInstance`` — pass the instance's
``sentinel_state`` and ``tensor`` to any consumer that knows about
the PTCA schema.

Typical standalone usage
------------------------
::

    from interdependent_lib.ptca.instance import PTCAInstance

    inst = PTCAInstance(
        model_id="claude-sonnet-4-6",
        caller_id="user:alice",
        session_id="sess_xyz",
    )

    # Push a context turn
    inst.push_context({"role": "user", "content": "Hello", "tokens": 5})

    # Record a provenance block
    inst.record_provenance(payload={"prompt_tokens": 5})

    # Route an exchange into the tensor
    result = inst.route(node=0, phase=0, slot=0, s1=1.0, s5=0.9)

    # Inspect live state
    print(inst.risk_score)
    print(inst.audit_tail())
"""

from __future__ import annotations

import uuid
from typing import Any

from interdependent_lib.ptca.constants import NODES, PHASES, SLOTS
from interdependent_lib.ptca.exchange import Exchange, ExchangeResult
from interdependent_lib.ptca.provenance import build_block, extend_chain, hash_block
from interdependent_lib.ptca.sentinels import SentinelState
from interdependent_lib.ptca.tensor import PTCATensor


class PTCAInstance:
    """
    A PTCA-aware stateful session for a single model / conversation.

    Parameters
    ----------
    model_id:
        Identifier of the model backing this instance.
    caller_id:
        Identifier of the caller / user.
    session_id:
        Unique session identifier; auto-generated as a UUID4 hex if
        not supplied.
    policy_rules:
        Initial S2 policy rule identifiers.
    bounds:
        Mapping of ``lower``/``upper`` and/or named constraints for S3.
    approved:
        Whether S4 starts in an approved state.
    max_context_entries:
        Maximum number of S5 context entries to retain.
    """

    def __init__(
        self,
        *,
        model_id: str = "",
        caller_id: str = "",
        session_id: str = "",
        policy_rules: list[str] | None = None,
        bounds: dict[str, Any] | None = None,
        approved: bool = False,
        max_context_entries: int = 256,
    ) -> None:
        self.session_id = session_id or uuid.uuid4().hex

        # Core PTCA objects
        self.tensor = PTCATensor()
        self.sentinel_state = SentinelState()
        self._exchange = Exchange(self.tensor, self.sentinel_state)

        # Provenance chain (S1)
        self._provenance_chain: list[dict[str, Any]] = []

        # Initialise sentinel channels
        self.sentinel_state.s6.set_identity(
            model_id=model_id,
            caller_id=caller_id,
            session_id=self.session_id,
        )
        if policy_rules:
            self.sentinel_state.s2.set_rules(policy_rules)
        if bounds:
            lower = bounds.get("lower", float("-inf"))
            upper = bounds.get("upper", float("inf"))
            constraints = {k: v for k, v in bounds.items() if k not in ("lower", "upper")}
            self.sentinel_state.s3.lower = lower
            self.sentinel_state.s3.upper = upper
            self.sentinel_state.s3.constraints = constraints
        if approved:
            self.sentinel_state.s4.approve(reason="initialised approved")
        self.sentinel_state.s5.max_entries = max_context_entries

        # Genesis provenance block
        self._genesis_block = build_block(
            model_id=model_id,
            caller_id=caller_id,
            session_id=self.session_id,
            payload={"event": "genesis"},
        )
        self._provenance_chain.append(self._genesis_block)
        self.sentinel_state.s1.origin_hash = hash_block(self._genesis_block)
        self.sentinel_state.s9.record(
            "genesis",
            model_id=model_id,
            caller_id=caller_id,
            session_id=self.session_id,
        )

    # ------------------------------------------------------------------
    # Convenience properties (S5-S9 quick access)
    # ------------------------------------------------------------------

    @property
    def model_id(self) -> str:
        return self.sentinel_state.s6.model_id

    @property
    def caller_id(self) -> str:
        return self.sentinel_state.s6.caller_id

    @property
    def risk_score(self) -> float:
        return self.sentinel_state.s8.score

    @property
    def approved(self) -> bool:
        return self.sentinel_state.s4.approved

    @property
    def context_entries(self) -> list[dict[str, Any]]:
        return self.sentinel_state.s5.entries

    @property
    def memory_store(self) -> dict[str, Any]:
        return self.sentinel_state.s7.store

    # ------------------------------------------------------------------
    # S1 — Provenance
    # ------------------------------------------------------------------

    def record_provenance(
        self,
        *,
        payload: dict[str, Any] | None = None,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        """
        Extend the provenance chain with a new block and update S1.

        Returns the new block.
        """
        block = extend_chain(
            self._provenance_chain,
            model_id=self.model_id,
            caller_id=self.caller_id,
            session_id=self.session_id,
            payload=payload,
            timestamp=timestamp,
        )
        h = hash_block(block)
        self.sentinel_state.s1.append(h)
        return block

    @property
    def provenance_chain(self) -> list[dict[str, Any]]:
        return self._provenance_chain

    # ------------------------------------------------------------------
    # S2 — Policy
    # ------------------------------------------------------------------

    def set_policy(self, rules: list[str]) -> None:
        self.sentinel_state.s2.set_rules(rules)

    # ------------------------------------------------------------------
    # S3 — Bounds
    # ------------------------------------------------------------------

    def set_bounds(self, lower: float = float("-inf"), upper: float = float("inf")) -> None:
        self.sentinel_state.s3.lower = lower
        self.sentinel_state.s3.upper = upper

    def within_bounds(self, value: float) -> bool:
        return self.sentinel_state.s3.within(value)

    # ------------------------------------------------------------------
    # S4 — Approval
    # ------------------------------------------------------------------

    def approve(self, reason: str = "") -> None:
        self.sentinel_state.s4.approve(reason)
        self.sentinel_state.s9.record("approval_granted", reason=reason)

    def revoke(self, reason: str = "") -> None:
        self.sentinel_state.s4.revoke(reason)
        self.sentinel_state.s9.record("approval_revoked", reason=reason)

    # ------------------------------------------------------------------
    # S5 — Context
    # ------------------------------------------------------------------

    def push_context(self, entry: dict[str, Any]) -> None:
        """Push a context entry (e.g. a conversation turn) onto S5."""
        self.sentinel_state.s5.push(entry)

    def clear_context(self) -> None:
        self.sentinel_state.s5.clear()

    # ------------------------------------------------------------------
    # S7 — Memory
    # ------------------------------------------------------------------

    def remember(self, key: str, value: Any) -> None:
        self.sentinel_state.s7.remember(key, value)

    def recall(self, key: str, default: Any = None) -> Any:
        return self.sentinel_state.s7.retrieve(key, default)

    # ------------------------------------------------------------------
    # S8 — Risk
    # ------------------------------------------------------------------

    def update_risk(self, delta: float, factor: str = "", **details: Any) -> None:
        self.sentinel_state.s8.update(delta, factor=factor, **details)
        self.sentinel_state.s9.record(
            "risk_update",
            delta=delta,
            factor=factor,
            new_score=self.risk_score,
            **details,
        )

    def reset_risk(self) -> None:
        self.sentinel_state.s8.reset()
        self.sentinel_state.s9.record("risk_reset")

    # ------------------------------------------------------------------
    # S9 — Audit
    # ------------------------------------------------------------------

    def audit_tail(self, n: int = 10) -> list[dict[str, Any]]:
        return self.sentinel_state.s9.tail(n)

    # ------------------------------------------------------------------
    # Exchange routing
    # ------------------------------------------------------------------

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
        **audit_details: Any,
    ) -> ExchangeResult:
        """
        Route a tensor exchange through this instance.

        All parameters are forwarded to :meth:`Exchange.route`.
        """
        return self._exchange.route(
            node=node,
            phase=phase,
            slot=slot,
            s1=s1,
            s2=s2,
            s3=s3,
            s5=s5,
            s8=s8,
            bonus=bonus,
            sentinel_idx=sentinel_idx,
            **audit_details,
        )

    def batch_route(self, exchanges: list[dict[str, Any]]) -> list[ExchangeResult]:
        """Route a list of exchange dicts; see :meth:`Exchange.batch_route`."""
        return self._exchange.batch_route(exchanges)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """
        Return a JSON-serialisable snapshot of live sentinel state.

        This is suitable for persisting instance state between
        processes or attaching to an API response as a
        ``sentinel_context`` block (compatible with the aimmh backend
        format for S5–S9).
        """
        s = self.sentinel_state
        return {
            "session_id": self.session_id,
            "S5_CONTEXT": {
                "entries": s.s5.entries,
                "token_count": s.s5.token_count,
            },
            "S6_IDENTITY": {
                "model_id": s.s6.model_id,
                "caller_id": s.s6.caller_id,
                "session_id": s.s6.session_id,
                "metadata": s.s6.metadata,
            },
            "S7_MEMORY": {
                "store": s.s7.store,
            },
            "S8_RISK": {
                "score": s.s8.score,
                "factors": s.s8.factors,
            },
            "S9_AUDIT": {
                "log": s.s9.log,
            },
        }

    def __repr__(self) -> str:
        return (
            f"PTCAInstance(model_id={self.model_id!r}, "
            f"session_id={self.session_id!r}, "
            f"risk={self.risk_score:.3f}, "
            f"approved={self.approved})"
        )
