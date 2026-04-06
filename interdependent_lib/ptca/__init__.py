# GPT/Claude generated; context, prompt Erin Spencer
"""PTCA — Prime-Tensor-Channel Architecture: sentinel state, tensor routing, exchange."""

from .constants import (
    AGG6,
    AGG_SEEDS,
    ALPHA,
    BETA,
    DELTA,
    GAMMA,
    NODES,
    PHASES,
    SENTINEL_INDEX,
    SENTINEL_NAMES,
    SENTINEL_WEIGHTS,
    SENTINELS,
    SLOTS,
)
from .exchange import Exchange, ExchangeResult, aggregate_identity, aggregate_seeds, compute_score
from .instance import PTCAInstance
from .primes import PRIME_NODES, PRIME_TO_NODE, is_prime_node, node_for_prime, prime_for_node
from .provenance import (
    build_block,
    chain_hashes,
    extend_chain,
    hash_block,
    verify_chain,
)
from .sentinels import (
    S1ProvenanceState,
    S2PolicyState,
    S3BoundsState,
    S4ApprovalState,
    S5ContextState,
    S6IdentityState,
    S7MemoryState,
    S8RiskState,
    S9AuditState,
    SentinelState,
)
from .tensor import PTCATensor

__all__ = [
    "AGG6",
    "AGG_SEEDS",
    "ALPHA",
    "BETA",
    "DELTA",
    "GAMMA",
    "NODES",
    "PHASES",
    "PRIME_NODES",
    "PRIME_TO_NODE",
    "PTCAInstance",
    "PTCATensor",
    "S1ProvenanceState",
    "S2PolicyState",
    "S3BoundsState",
    "S4ApprovalState",
    "S5ContextState",
    "S6IdentityState",
    "S7MemoryState",
    "S8RiskState",
    "S9AuditState",
    "SENTINEL_INDEX",
    "SENTINEL_NAMES",
    "SENTINEL_WEIGHTS",
    "SENTINELS",
    "SLOTS",
    "SentinelState",
    "Exchange",
    "ExchangeResult",
    "aggregate_identity",
    "aggregate_seeds",
    "build_block",
    "chain_hashes",
    "compute_score",
    "extend_chain",
    "hash_block",
    "is_prime_node",
    "node_for_prime",
    "prime_for_node",
    "verify_chain",
]
