# GPT/Claude generated; context, prompt Erin Spencer
"""
Key wrapping and unwrapping using AES-256-GCM.

wrap_key()   encrypts K_live under K_meta, binding epoch + key_id in AAD.
unwrap_key() decrypts and returns K_live ephemerally — caller must zeroize.

The wrapped output never contains the plaintext key; the WrappedLiveKey
type enforces this at the field level.
"""

from __future__ import annotations

import hashlib
import os
import time

from .aead import AuthenticationError, seal, unseal
from .codec import encode_wrap_aad
from .types import WrappedLiveKey

_WRAP_NONCE_LEN = 12


def wrap_live_key(
    live_key: bytes,
    meta_key: bytes,
    epoch: int,
    key_id: str,
) -> WrappedLiveKey:
    """
    Encrypt K_live under K_meta.

    A fresh random nonce is generated for each wrap operation.
    The nonce is prepended to wrapped_live_key so it travels with the ciphertext.

    Returns:
        WrappedLiveKey with wrapped_live_key = nonce || ciphertext
    """
    nonce = os.urandom(_WRAP_NONCE_LEN)
    aad = encode_wrap_aad(epoch, key_id)
    ciphertext = seal(meta_key, nonce, live_key, aad)
    wrap_key_hash = hashlib.sha256(meta_key).hexdigest()
    return WrappedLiveKey(
        key_id=key_id,
        epoch=epoch,
        wrapped_live_key=nonce + ciphertext,
        wrap_key_hash=wrap_key_hash,
        wrapped_at=time.time(),
    )


def unwrap_live_key(wrapped: WrappedLiveKey, meta_key: bytes) -> bytes:
    """
    Decrypt and return K_live.

    The caller is responsible for zeroizing the returned bytes after use.

    Raises:
        AuthenticationError: if the meta_key is wrong or data is tampered
        ValueError: if wrapped_live_key is too short to contain nonce + tag
    """
    blob = wrapped.wrapped_live_key
    if len(blob) < _WRAP_NONCE_LEN + 16:
        raise ValueError("wrapped_live_key too short")
    nonce = blob[:_WRAP_NONCE_LEN]
    ciphertext = blob[_WRAP_NONCE_LEN:]
    aad = encode_wrap_aad(wrapped.epoch, wrapped.key_id)
    return unseal(meta_key, nonce, ciphertext, aad)
