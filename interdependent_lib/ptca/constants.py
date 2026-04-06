"""
PTCA schema constants.

Tensor dimensions
-----------------
NODES      : 53  (prime-indexed routing nodes)
SENTINELS  : 9   (S1–S9 sentinel channels)
PHASES     : 8   (processing phases)
SLOTS      : 7   (heptagram slots)

Exchange constants
------------------
DELTA      : base exchange unit
ALPHA      : provenance / S1 weight
BETA       : policy / S2 weight
GAMMA      : bounds+context / S3+S5 weight
AGG6       : aggregation method for S6 (identity)
AGG_SEEDS  : aggregation method for seed values
"""

# --- Tensor dimensions ---
NODES: int = 53
SENTINELS: int = 9
PHASES: int = 8
SLOTS: int = 7

# --- Exchange constants ---
DELTA: int = 1
ALPHA: float = 0.10
BETA: float = 0.20
GAMMA: float = 0.10
AGG6: str = "mean"
AGG_SEEDS: str = "mean"

# --- Sentinel channel names (index 0 = S1) ---
SENTINEL_NAMES: tuple[str, ...] = (
    "S1_PROVENANCE",
    "S2_POLICY",
    "S3_BOUNDS",
    "S4_APPROVAL",
    "S5_CONTEXT",
    "S6_IDENTITY",
    "S7_MEMORY",
    "S8_RISK",
    "S9_AUDIT",
)

# Convenience mapping: name → 0-based index
SENTINEL_INDEX: dict[str, int] = {name: i for i, name in enumerate(SENTINEL_NAMES)}

# Sentinel weights used in exchange scoring (parallel to SENTINEL_NAMES)
SENTINEL_WEIGHTS: tuple[float, ...] = (
    ALPHA,   # S1_PROVENANCE
    BETA,    # S2_POLICY
    GAMMA,   # S3_BOUNDS
    0.0,     # S4_APPROVAL  (boolean gate, not a weighted channel)
    GAMMA,   # S5_CONTEXT
    0.0,     # S6_IDENTITY  (aggregated separately via AGG6)
    0.0,     # S7_MEMORY    (carries forward, not scored per-exchange)
    ALPHA,   # S8_RISK
    0.0,     # S9_AUDIT     (append-only log, not scored)
)
