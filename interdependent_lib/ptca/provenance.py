"""
Provenance hashing — S1_PROVENANCE support.

A provenance block captures the origin and chain-of-custody of a
tensor exchange.  It is hashed with SHA-256 (stdlib ``hashlib``) so
every block is both content-addressed and tamper-evident.

Typical usage
-------------
::

    from interdependent_lib.ptca.provenance import build_block, hash_block, chain_hashes

    block = build_block(
        model_id="claude-sonnet-4-6",
        caller_id="user:alice",
        session_id="sess_abc123",
        payload={"prompt_tokens": 42},
    )
    h = hash_block(block)
    # h is a 64-char hex string

    # extend a chain: include parent hash in new block
    block2 = build_block(
        model_id="claude-sonnet-4-6",
        parent_hash=h,
        payload={"completion_tokens": 17},
    )
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any


# ---------------------------------------------------------------------------
# Block construction
# ---------------------------------------------------------------------------

def build_block(
    *,
    model_id: str = "",
    caller_id: str = "",
    session_id: str = "",
    parent_hash: str = "",
    payload: dict[str, Any] | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """
    Construct a provenance block (plain dict, JSON-serialisable).

    Parameters
    ----------
    model_id:
        Identifier of the model that produced this exchange.
    caller_id:
        Identifier of the caller / user.
    session_id:
        Session or conversation identifier.
    parent_hash:
        SHA-256 hex digest of the immediately preceding block in the
        chain (empty string for genesis blocks).
    payload:
        Arbitrary JSON-serialisable metadata to attach.
    timestamp:
        Unix timestamp; defaults to ``time.time()``.

    Returns
    -------
    dict
        The provenance block.  Pass to :func:`hash_block` to obtain
        its content-addressed digest.
    """
    return {
        "model_id": model_id,
        "caller_id": caller_id,
        "session_id": session_id,
        "parent_hash": parent_hash,
        "payload": payload or {},
        "ts": timestamp if timestamp is not None else time.time(),
    }


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------

def _canonical(block: dict[str, Any]) -> bytes:
    """Deterministic JSON bytes for a provenance block."""
    return json.dumps(block, sort_keys=True, separators=(",", ":")).encode("utf-8")


def hash_block(block: dict[str, Any]) -> str:
    """
    Return the SHA-256 hex digest of *block*.

    The block is serialised to canonical JSON (sorted keys, no
    whitespace) before hashing, so the digest is deterministic.
    """
    return hashlib.sha256(_canonical(block)).hexdigest()


# ---------------------------------------------------------------------------
# Chain helpers
# ---------------------------------------------------------------------------

def chain_hashes(blocks: list[dict[str, Any]]) -> list[str]:
    """
    Return the list of SHA-256 digests for an ordered list of blocks.

    The list is computed independently of any ``parent_hash`` fields
    already stored in the blocks — it reflects the *content* of each
    block as supplied.
    """
    return [hash_block(b) for b in blocks]


def verify_chain(blocks: list[dict[str, Any]]) -> bool:
    """
    Verify that each block's ``parent_hash`` matches the hash of the
    preceding block.

    Returns ``True`` if the chain is intact, ``False`` otherwise.
    The genesis block (index 0) is valid when its ``parent_hash`` is
    an empty string.
    """
    if not blocks:
        return True
    if blocks[0].get("parent_hash", "") != "":
        return False
    for i in range(1, len(blocks)):
        expected = hash_block(blocks[i - 1])
        if blocks[i].get("parent_hash", "") != expected:
            return False
    return True


def extend_chain(
    blocks: list[dict[str, Any]],
    *,
    model_id: str = "",
    caller_id: str = "",
    session_id: str = "",
    payload: dict[str, Any] | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    """
    Build a new block whose ``parent_hash`` is the hash of the last
    block in *blocks*, append it to *blocks* in-place, and return it.
    """
    parent_hash = hash_block(blocks[-1]) if blocks else ""
    block = build_block(
        model_id=model_id,
        caller_id=caller_id,
        session_id=session_id,
        parent_hash=parent_hash,
        payload=payload,
        timestamp=timestamp,
    )
    blocks.append(block)
    return block
