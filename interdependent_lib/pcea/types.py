# GPT/Claude generated; context, prompt Erin Spencer
"""
Frozen canonical types for Guardian state.

Invariants enforced at the type level:
- WrappedLiveKey holds only ciphertext, never plaintext liveKey
- UnsealGrant is an authorization token, never a key container
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class LiveState:
    """
    In-memory only. Never serialized directly. Guardian-internal.

    Fields map to the density-matrix / propagator notation used in the spec.
    'cores', 'density_matrix', and 'transport' are intentionally typed as Any
    here; concrete tensor types are provided by the runtime that owns LiveState.
    """
    epoch: int
    spiral: dict                     # {phase: float, magnitude: float, base: float}
    cores: dict                      # {Phi: Any, Psi: Any, Omega: Any}
    density_matrix: Any              # SigmaMatrix
    coherence: float
    transport: Any                   # TransportOperator
    last_renorm: float


@dataclass
class SealedState:
    """
    Durable carrier of the private present. The only form that may be persisted.
    ciphertext = AEAD(K_live, nonce, plaintext_of_LiveState, aad)
    """
    epoch: int
    key_id: str
    ciphertext: bytes
    nonce: bytes
    aad: bytes
    sealed_by: str
    sealed_at: float = field(default_factory=time.time)


@dataclass
class WrappedLiveKey:
    """
    K_live encrypted under K_meta. MUST NOT contain a plaintext live_key field.
    wrapped_live_key = AEAD(K_meta, nonce, K_live, aad)
    """
    key_id: str
    epoch: int
    wrapped_live_key: bytes          # ciphertext — never plaintext
    wrap_key_hash: str               # hex digest of K_meta used for wrapping
    wrapped_at: float = field(default_factory=time.time)


@dataclass
class MetaShares:
    """
    Governance object. Assembled only during ceremony; not a storage object.
    shares holds cleartext share bytes in assembled form.
    At-rest encryption of individual shares is a service-layer concern.
    """
    epoch: int
    total_shares: int
    threshold: int
    shares: list                     # list of {sentinel_id: str, share: bytes, index: int}
    commitment: str                  # hex SHA-256 over ordered share bytes


@dataclass
class UnsealGrant:
    """
    Authorization artifact only. MUST NOT contain a reconstructed_meta_key field.
    Authorizes use of the reconstruction process; does not cache the result.
    """
    epoch: int
    key_id: str
    grant_signature: str
    granted_to: str
    validity_window: float           # seconds
    purpose: Literal["unseal", "rekey", "migrate"]


@dataclass
class RekeyEpoch:
    """
    Record of a completed rekey operation.
    spectral_snapshot carries diagnostic data; no contract is enforced on its values.
    """
    from_epoch: int
    to_epoch: int
    new_session_secret_commitment: str
    meta_shares_updated: bool
    renorm_confirmed: bool
    spectral_snapshot: list          # list of float; diagnostic only
