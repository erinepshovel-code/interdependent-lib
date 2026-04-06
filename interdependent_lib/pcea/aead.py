# GPT/Claude generated; context, prompt Erin Spencer
"""
AES-256-GCM authenticated encryption / decryption.

seal()   → ciphertext + 16-byte GCM tag appended
unseal() → plaintext, raises InvalidTag on any authentication failure

No plaintext is returned on auth failure; the exception is raised before
any output is produced.
"""

from __future__ import annotations

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class AuthenticationError(Exception):
    """Raised when GCM authentication fails during unseal."""


def seal(key: bytes, nonce: bytes, plaintext: bytes, aad: bytes) -> bytes:
    """
    AES-256-GCM encrypt.

    Args:
        key:       32-byte AES key
        nonce:     12-byte nonce (must be unique per (key, message))
        plaintext: message to encrypt
        aad:       additional authenticated data (not encrypted)

    Returns:
        ciphertext with 16-byte GCM tag appended
    """
    if len(key) != 32:
        raise ValueError(f"key must be 32 bytes, got {len(key)}")
    if len(nonce) != 12:
        raise ValueError(f"nonce must be 12 bytes, got {len(nonce)}")
    return AESGCM(key).encrypt(nonce, plaintext, aad)


def unseal(key: bytes, nonce: bytes, ciphertext: bytes, aad: bytes) -> bytes:
    """
    AES-256-GCM decrypt and verify.

    Args:
        key:        32-byte AES key
        nonce:      12-byte nonce used during sealing
        ciphertext: encrypted bytes with 16-byte GCM tag appended
        aad:        additional authenticated data used during sealing

    Returns:
        plaintext

    Raises:
        AuthenticationError: if tag verification fails
    """
    if len(key) != 32:
        raise ValueError(f"key must be 32 bytes, got {len(key)}")
    if len(nonce) != 12:
        raise ValueError(f"nonce must be 12 bytes, got {len(nonce)}")
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, aad)
    except InvalidTag as exc:
        raise AuthenticationError("AEAD authentication failed") from exc
