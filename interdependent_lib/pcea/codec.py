# GPT/Claude generated; context, prompt Erin Spencer
"""
Deterministic encoding for AAD construction and commitment inputs.

Uses big-endian struct packing for fixed-size fields and length-prefixed
UTF-8 for variable-length strings. No external serialization dependency.
"""

from __future__ import annotations

import struct


def encode_aad(epoch: int, key_id: str, sealed_by: str) -> bytes:
    """
    Produce deterministic AAD bytes for AEAD sealing.

    Format:
      epoch        : uint64 big-endian
      key_id_len   : uint16 big-endian
      key_id       : UTF-8 bytes
      sealed_by_len: uint16 big-endian
      sealed_by    : UTF-8 bytes
    """
    kid = key_id.encode("utf-8")
    sb = sealed_by.encode("utf-8")
    return (
        struct.pack(">Q", epoch)
        + struct.pack(">H", len(kid)) + kid
        + struct.pack(">H", len(sb)) + sb
    )


def encode_wrap_aad(epoch: int, key_id: str) -> bytes:
    """
    Produce deterministic AAD bytes for key wrapping.

    Format:
      epoch      : uint64 big-endian
      key_id_len : uint16 big-endian
      key_id     : UTF-8 bytes
    """
    kid = key_id.encode("utf-8")
    return struct.pack(">Q", epoch) + struct.pack(">H", len(kid)) + kid


def encode_nonce_input(
    epoch: int,
    key_id: str,
    seal_counter: int,
    guardian_node_id: str,
) -> bytes:
    """
    Produce the deterministic info string fed into HKDF for nonce derivation.

    Format:
      b"nonce:"         : literal prefix (6 bytes)
      epoch             : uint64 big-endian
      seal_counter      : uint64 big-endian
      key_id_len        : uint16 big-endian
      key_id            : UTF-8 bytes
      node_id_len       : uint16 big-endian
      guardian_node_id  : UTF-8 bytes
    """
    kid = key_id.encode("utf-8")
    nid = guardian_node_id.encode("utf-8")
    return (
        b"nonce:"
        + struct.pack(">Q", epoch)
        + struct.pack(">Q", seal_counter)
        + struct.pack(">H", len(kid)) + kid
        + struct.pack(">H", len(nid)) + nid
    )


def encode_key_info(label: str, epoch: int, key_id: str) -> bytes:
    """
    Produce the info string for HKDF key derivation.

    Format:
      label_len : uint16 big-endian
      label     : UTF-8 bytes
      epoch     : uint64 big-endian
      key_id_len: uint16 big-endian
      key_id    : UTF-8 bytes
    """
    lbl = label.encode("utf-8")
    kid = key_id.encode("utf-8")
    return (
        struct.pack(">H", len(lbl)) + lbl
        + struct.pack(">Q", epoch)
        + struct.pack(">H", len(kid)) + kid
    )
