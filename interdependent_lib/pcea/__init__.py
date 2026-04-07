# GPT/Claude generated; context, prompt Erin Spencer
"""PCEA — Guardian state encryption, key management, and secret sharing."""

from .aead import AuthenticationError, seal, unseal
from .codec import encode_aad, encode_key_info, encode_nonce_input, encode_wrap_aad
from .commitment import make_commitment, verify_commitment
from .guardian import seal_live_state, unseal_live_state
from .kdf import derive_keys, derive_nonce
from .rekey import reconstruct_meta_key, rekey_epoch, split_meta_key
from .threshold import reconstruct_secret, split_secret
from .types import (
    LiveState,
    MetaShares,
    RekeyEpoch,
    SealedState,
    UnsealGrant,
    WrappedLiveKey,
)
from .validate import InvariantViolation, validate_invariant
from .wipe import wipe, wipe_bytearray, wipe_bytes
from .wrap import unwrap_live_key, wrap_live_key

__all__ = [
    "AuthenticationError",
    "InvariantViolation",
    "LiveState",
    "MetaShares",
    "RekeyEpoch",
    "SealedState",
    "UnsealGrant",
    "WrappedLiveKey",
    "derive_keys",
    "derive_nonce",
    "encode_aad",
    "encode_key_info",
    "encode_nonce_input",
    "encode_wrap_aad",
    "make_commitment",
    "reconstruct_meta_key",
    "reconstruct_secret",
    "rekey_epoch",
    "seal",
    "seal_live_state",
    "split_meta_key",
    "split_secret",
    "unseal",
    "unseal_live_state",
    "unwrap_live_key",
    "validate_invariant",
    "verify_commitment",
    "wipe",
    "wipe_bytearray",
    "wipe_bytes",
    "wrap_live_key",
]
