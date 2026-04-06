# GPT/Claude generated; context, prompt Erin Spencer
"""
Commitment generation over Shamir share bytes.

make_commitment() produces a deterministic hex digest over the ordered,
length-prefixed share bytes. This lets anyone holding the shares verify
that the set matches what was committed to at ceremony time, without
revealing anything about the shares themselves.
"""

from __future__ import annotations

import hashlib
import struct


def make_commitment(shares: list) -> str:
    """
    SHA-256 commitment over ordered share bytes.

    shares: list of dicts with keys {sentinel_id, share: bytes, index: int}
    The list is sorted by index before hashing to ensure determinism
    regardless of the order supplied by the caller.

    Returns:
        hex-encoded SHA-256 digest
    """
    sorted_shares = sorted(shares, key=lambda s: s["index"])
    h = hashlib.sha256()
    for s in sorted_shares:
        share_bytes = s["share"]
        # Length-prefix each share to prevent concatenation collisions
        h.update(struct.pack(">I", s["index"]))
        h.update(struct.pack(">I", len(share_bytes)))
        h.update(share_bytes)
    return h.hexdigest()


def verify_commitment(shares: list, commitment: str) -> bool:
    """
    Verify that shares match a previously computed commitment.

    Returns True if the commitment matches, False otherwise.
    Does not raise on mismatch; callers that treat mismatch as fatal
    should check the return value and raise their own exception.
    """
    return make_commitment(shares) == commitment
