# GPT/Claude generated; context, prompt Erin Spencer
"""
HKDF-SHA256 key derivation and per-seal nonce derivation.

derive_keys produces two independent 32-byte keys (live key, meta key)
from input key material plus binding context.

derive_nonce produces a 12-byte nonce that is unique per (epoch, key_id,
seal_counter, guardian_node_id) tuple — per the frozen spec requirement.
"""

from __future__ import annotations

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from .codec import encode_key_info, encode_nonce_input

_HASH = hashes.SHA256()
_KEY_LEN = 32
_NONCE_LEN = 12


def _hkdf_expand(ikm: bytes, info: bytes, length: int) -> bytes:
    return HKDF(
        algorithm=hashes.SHA256(),
        length=length,
        salt=None,
        info=info,
    ).derive(ikm)


def derive_keys(
    ikm: bytes,
    epoch: int,
    key_id: str,
    guardian_node_id: str,
) -> tuple[bytes, bytes]:
    """
    Derive (live_key, meta_key) from input key material.

    Both keys are 32 bytes (AES-256 / HKDF-SHA256).
    Different info strings guarantee independence.

    Returns:
        (live_key, meta_key) — both bytes of length 32
    """
    live_key = _hkdf_expand(
        ikm,
        encode_key_info("guardian:live", epoch, key_id),
        _KEY_LEN,
    )
    meta_key = _hkdf_expand(
        ikm,
        encode_key_info("guardian:meta", epoch, key_id),
        _KEY_LEN,
    )
    return live_key, meta_key


def derive_nonce(
    epoch: int,
    key_id: str,
    seal_counter: int,
    guardian_node_id: str,
    ikm: bytes,
) -> bytes:
    """
    Derive a 12-byte nonce that is unique per seal operation.

    Nonce = HKDF(ikm, info=encode_nonce_input(...))[0:12]

    Using ikm (the live key) as the HKDF input means the nonce space is
    bound to the key, making collisions structurally impossible across
    different epochs or key IDs even if the seal_counter resets.
    """
    return _hkdf_expand(
        ikm,
        encode_nonce_input(epoch, key_id, seal_counter, guardian_node_id),
        _NONCE_LEN,
    )
