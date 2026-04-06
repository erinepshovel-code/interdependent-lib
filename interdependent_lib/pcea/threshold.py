# GPT/Claude generated; context, prompt Erin Spencer
"""
Shamir's Secret Sharing over GF(256).

Implemented inline over the field GF(2^8) with irreducible polynomial
x^8 + x^4 + x^3 + x^2 + 1 (0x11d), where 2 is a primitive root of order 255.
This avoids any external Shamir dependency and its associated supply-chain risk.

Note: the AES polynomial (0x11b) has 3 as its primitive root, not 2.
We use 0x11d so that generator g=2 works correctly.

split_secret()        → list of (index, share_bytes) pairs
reconstruct_secret()  → original secret bytes

Secrets of arbitrary byte length are supported: each byte position is
treated as an independent GF(256) polynomial evaluation.
"""

from __future__ import annotations

import os

# GF(256) using irreducible polynomial x^8+x^4+x^3+x^2+1 = 0x11d
# Under this polynomial, 2 (= x) is a primitive root of order 255.
_GF_EXP = [0] * 512
_GF_LOG = [0] * 256

# Build log and exp tables with generator g=2
_x = 1
for _i in range(255):
    _GF_EXP[_i] = _x
    _GF_LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11d
    _x &= 0xFF
for _i in range(255, 512):
    _GF_EXP[_i] = _GF_EXP[_i - 255]


def _gf_mul(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] + _GF_LOG[b]) % 255]


def _gf_div(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("division by zero in GF(256)")
    if a == 0:
        return 0
    return _GF_EXP[(_GF_LOG[a] - _GF_LOG[b]) % 255]


def _gf_pow(x: int, power: int) -> int:
    return _GF_EXP[(_GF_LOG[x] * power) % 255] if x != 0 else 0


def _eval_poly(coeffs: list[int], x: int) -> int:
    """Evaluate polynomial with GF(256) arithmetic at point x."""
    result = 0
    for c in reversed(coeffs):
        result = _gf_mul(result, x) ^ c
    return result


def _lagrange_interpolate(x: int, x_points: list[int], y_points: list[int]) -> int:
    """Lagrange interpolation at x=0 (constant term recovery) in GF(256)."""
    result = 0
    for i, xi in enumerate(x_points):
        num = 1
        den = 1
        for j, xj in enumerate(x_points):
            if i == j:
                continue
            # num *= (x - xj), but in GF(256) subtraction is XOR
            num = _gf_mul(num, x ^ xj)
            den = _gf_mul(den, xi ^ xj)
        result ^= _gf_mul(_gf_div(num, den), y_points[i])
    return result


def split_secret(secret: bytes, threshold: int, n: int) -> list[tuple[int, bytes]]:
    """
    Split secret into n shares requiring threshold to reconstruct.

    Returns:
        list of (index, share_bytes) where index is in [1..n]
        The index is 1-based so that x=0 corresponds to the secret.

    Raises:
        ValueError: if threshold > n or threshold < 2 or secret is empty
    """
    if not secret:
        raise ValueError("secret must not be empty")
    if threshold < 2:
        raise ValueError("threshold must be at least 2")
    if threshold > n:
        raise ValueError("threshold cannot exceed n")
    if n > 254:
        raise ValueError("n cannot exceed 254 (GF(256) constraint)")

    shares = [(i, bytearray()) for i in range(1, n + 1)]

    for byte in secret:
        # Random polynomial of degree (threshold - 1) with constant term = byte
        coeffs = [byte] + [int(b) for b in os.urandom(threshold - 1)]
        for idx, share_bytes in shares:
            share_bytes.append(_eval_poly(coeffs, idx))

    return [(idx, bytes(sb)) for idx, sb in shares]


def reconstruct_secret(shares: list[tuple[int, bytes]]) -> bytes:
    """
    Reconstruct the secret from a list of (index, share_bytes) pairs.

    Requires exactly threshold or more shares (any subset of the correct
    threshold size works). Providing wrong shares produces garbage, not
    an exception — the caller should verify the result against a commitment.

    Returns:
        reconstructed secret bytes
    """
    if not shares:
        raise ValueError("shares must not be empty")
    length = len(shares[0][1])
    if any(len(sb) != length for _, sb in shares):
        raise ValueError("all shares must have the same byte length")

    x_points = [idx for idx, _ in shares]
    result = bytearray()
    for pos in range(length):
        y_points = [sb[pos] for _, sb in shares]
        result.append(_lagrange_interpolate(0, x_points, y_points))
    return bytes(result)
