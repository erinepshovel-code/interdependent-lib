# GPT/Claude generated; context, prompt Erin Spencer
"""
Guardian state operations: seal, unseal.

Orchestrates kdf + aead to produce SealedState from LiveState and vice versa.
LiveState is encoded to/from JSON bytes for serialization inside the AEAD envelope.

seal_live_state()   → SealedState
unseal_live_state() → LiveState
"""

from __future__ import annotations

import json
import time

from .aead import seal, unseal
from .codec import encode_aad
from .kdf import derive_nonce
from .types import LiveState, SealedState


def _encode_live_state(state: LiveState) -> bytes:
    """
    Serialize LiveState to bytes for AEAD encryption.

    Uses JSON with explicit handling of complex types via a custom default.
    The 'cores' and 'density_matrix' and 'transport' fields may hold
    non-JSON-native objects; callers that use custom tensor types must
    ensure those objects are JSON-serializable (e.g., convert numpy arrays
    to nested lists before passing to seal_live_state).
    """
    d = {
        "epoch": state.epoch,
        "spiral": state.spiral,
        "cores": state.cores,
        "density_matrix": state.density_matrix,
        "coherence": state.coherence,
        "transport": state.transport,
        "last_renorm": state.last_renorm,
    }
    return json.dumps(d, separators=(",", ":")).encode("utf-8")


def _decode_live_state(raw: bytes) -> LiveState:
    """Deserialize LiveState from JSON bytes."""
    d = json.loads(raw.decode("utf-8"))
    return LiveState(
        epoch=d["epoch"],
        spiral=d["spiral"],
        cores=d["cores"],
        density_matrix=d["density_matrix"],
        coherence=d["coherence"],
        transport=d["transport"],
        last_renorm=d["last_renorm"],
    )


def seal_live_state(
    state: LiveState,
    live_key: bytes,
    epoch: int,
    key_id: str,
    seal_counter: int,
    guardian_node_id: str,
    sealed_by: str,
) -> SealedState:
    """
    Encrypt LiveState under live_key.

    The caller is responsible for maintaining a monotonic seal_counter per
    (key_id, guardian_node_id) pair to guarantee nonce uniqueness.

    Returns:
        SealedState with ciphertext, nonce, and binding AAD.
    """
    nonce = derive_nonce(epoch, key_id, seal_counter, guardian_node_id, live_key)
    aad = encode_aad(epoch, key_id, sealed_by)
    plaintext = _encode_live_state(state)
    ciphertext = seal(live_key, nonce, plaintext, aad)
    return SealedState(
        epoch=epoch,
        key_id=key_id,
        ciphertext=ciphertext,
        nonce=nonce,
        aad=aad,
        sealed_by=sealed_by,
        sealed_at=time.time(),
    )


def unseal_live_state(sealed: SealedState, live_key: bytes) -> LiveState:
    """
    Decrypt SealedState and return LiveState.

    Raises:
        AuthenticationError: if live_key is wrong or ciphertext is tampered.
    """
    plaintext = unseal(live_key, sealed.nonce, sealed.ciphertext, sealed.aad)
    return _decode_live_state(plaintext)
