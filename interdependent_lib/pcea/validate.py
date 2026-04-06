# GPT/Claude generated; context, prompt Erin Spencer
"""
Invariant checks for Guardian state objects.

validate_invariant() raises InvariantViolation on any structural violation.
All checks are pure comparisons — no crypto, no I/O.
"""

from __future__ import annotations

from .types import MetaShares, SealedState, WrappedLiveKey


class InvariantViolation(Exception):
    """Raised when a Guardian state invariant is violated."""


def validate_invariant(
    wrapped: WrappedLiveKey,
    sealed: SealedState,
    meta_shares: MetaShares,
) -> None:
    """
    Check that the three core persistent objects form a consistent set.

    Checks:
    1. All three objects share the same epoch.
    2. wrapped.key_id == sealed.key_id.
    3. meta_shares.threshold >= 2.
    4. meta_shares.threshold <= meta_shares.total_shares.
    5. len(meta_shares.shares) == meta_shares.total_shares.
    6. wrapped.wrapped_live_key is non-empty (i.e., is ciphertext, not absent).
    7. sealed.ciphertext is non-empty.
    8. sealed.nonce is exactly 12 bytes.
    9. All share index values are unique and in [1, total_shares].

    Raises:
        InvariantViolation: with a descriptive message on the first failure found.
    """
    if wrapped.epoch != sealed.epoch:
        raise InvariantViolation(
            f"epoch mismatch: wrapped.epoch={wrapped.epoch}, sealed.epoch={sealed.epoch}"
        )
    if wrapped.epoch != meta_shares.epoch:
        raise InvariantViolation(
            f"epoch mismatch: wrapped.epoch={wrapped.epoch}, meta_shares.epoch={meta_shares.epoch}"
        )
    if wrapped.key_id != sealed.key_id:
        raise InvariantViolation(
            f"key_id mismatch: wrapped.key_id={wrapped.key_id!r}, sealed.key_id={sealed.key_id!r}"
        )
    if meta_shares.threshold < 2:
        raise InvariantViolation(
            f"threshold must be >= 2, got {meta_shares.threshold}"
        )
    if meta_shares.threshold > meta_shares.total_shares:
        raise InvariantViolation(
            f"threshold ({meta_shares.threshold}) exceeds total_shares ({meta_shares.total_shares})"
        )
    if len(meta_shares.shares) != meta_shares.total_shares:
        raise InvariantViolation(
            f"shares list length ({len(meta_shares.shares)}) != total_shares ({meta_shares.total_shares})"
        )
    if not wrapped.wrapped_live_key:
        raise InvariantViolation("wrapped_live_key must not be empty")
    if not sealed.ciphertext:
        raise InvariantViolation("sealed.ciphertext must not be empty")
    if len(sealed.nonce) != 12:
        raise InvariantViolation(
            f"sealed.nonce must be 12 bytes, got {len(sealed.nonce)}"
        )

    # Share index uniqueness and range
    indices = [s["index"] for s in meta_shares.shares]
    if len(set(indices)) != len(indices):
        raise InvariantViolation("share indices are not unique")
    for idx in indices:
        if not (1 <= idx <= meta_shares.total_shares):
            raise InvariantViolation(
                f"share index {idx} out of range [1, {meta_shares.total_shares}]"
            )
