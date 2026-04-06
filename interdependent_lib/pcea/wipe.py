# GPT/Claude generated; context, prompt Erin Spencer
"""
Zeroization helpers.

Python's garbage collector does not guarantee when memory is reclaimed,
and bytes objects are immutable (cannot be overwritten in place).
The best we can do for bytearray is overwrite in place; for bytes we
overwrite any mutable container that held the value and let the caller
know that the reference is now zeroed.

Usage:
    key = derive_keys(...)
    try:
        ...use key...
    finally:
        wipe(key_holder)   # pass a list or bytearray, not a bytes literal
"""

from __future__ import annotations

import ctypes


def wipe_bytearray(buf: bytearray) -> None:
    """Overwrite a bytearray with zeros in place."""
    for i in range(len(buf)):
        buf[i] = 0


def wipe_bytes(b: bytes) -> None:
    """
    Best-effort overwrite of a bytes object.

    bytes objects are immutable in Python, so this uses ctypes to write
    zeros directly into the underlying buffer. This is not guaranteed to
    work on all Python implementations, but works on CPython.

    Callers should prefer bytearray for ephemeral key material.
    """
    try:
        ctypes.memset(id(b) + _bytes_data_offset(), 0, len(b))
    except Exception:
        pass  # best-effort; do not raise from a cleanup path


def _bytes_data_offset() -> int:
    """
    Compute the offset of the data buffer inside a CPython bytes object.
    This is an implementation detail of CPython and may change across versions.
    """
    # On CPython, bytes has ob_refcnt, ob_type, ob_size, ob_shash, then data.
    # The ob_shash field is 8 bytes (Py_hash_t). This yields offset 32 on 64-bit.
    import sys
    if sys.maxsize > 2**32:
        return 33  # 64-bit: refcnt(8) + type(8) + size(8) + hash(8) + 1 null
    return 21      # 32-bit


def wipe(value: bytes | bytearray) -> None:
    """
    Dispatch to the appropriate wipe implementation.
    Prefer passing bytearray for ephemeral key material.
    """
    if isinstance(value, bytearray):
        wipe_bytearray(value)
    elif isinstance(value, bytes):
        wipe_bytes(value)
    else:
        raise TypeError(f"wipe() expects bytes or bytearray, got {type(value)}")
