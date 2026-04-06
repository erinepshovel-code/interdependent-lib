"""
Prime-node axis for the PTCA tensor.

The 53 routing nodes are indexed by the first 53 prime numbers.
Each prime p_i is the canonical address of node i in the tensor's
first dimension.
"""

from __future__ import annotations

from interdependent_lib.ptca.constants import NODES

# First 53 primes (the 53rd prime is 241)
PRIME_NODES: tuple[int, ...] = (
    2,   3,   5,   7,   11,  13,  17,  19,  23,  29,
    31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
    73,  79,  83,  89,  97,  101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173,
    179, 181, 191, 193, 197, 199, 211, 223, 227, 229,
    233, 239, 241,
)

assert len(PRIME_NODES) == NODES, (
    f"Expected {NODES} primes, got {len(PRIME_NODES)}"
)

# Reverse lookup: prime value → node index
PRIME_TO_NODE: dict[int, int] = {p: i for i, p in enumerate(PRIME_NODES)}


def node_for_prime(p: int) -> int:
    """Return the node index (0-based) for a given prime, or raise KeyError."""
    return PRIME_TO_NODE[p]


def prime_for_node(idx: int) -> int:
    """Return the prime for a node index (0-based)."""
    if not (0 <= idx < NODES):
        raise IndexError(f"Node index {idx} out of range [0, {NODES})")
    return PRIME_NODES[idx]


def is_prime_node(p: int) -> bool:
    """Return True if *p* is one of the 53 routing primes."""
    return p in PRIME_TO_NODE
