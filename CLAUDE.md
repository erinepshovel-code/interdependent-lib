# CLAUDE.md

## Repository overview

`interdependent-lib` is a pure-Python library that implements five interdependent
subsystems arranged in a strict hierarchy:

```
PCNA  ←  back-propagating neural network (base tensor)
PCTA  ←  circle: 7 PCNA tensors, audited as one tensor
PTCA  ←  seed: 7 PCTA circles → core: 53 seeds (4 sentinel seeds → 9 S-channels)
PCEA  ←  encryption layer over PTCA guardian state
EDCM  ←  transcript analysis using bone/marker vocabulary
```

- **PCNA** (`interdependent_lib/pcna/`) — back-propagating neural network: the
  base tensor layer. Pure-Python MLP with forward pass, MSE/BCE loss, and
  gradient-descent backpropagation.
- **PCTA** (`interdependent_lib/pcta/`) — Prime Circle Tensor Architecture: a
  circle of exactly 7 PCNA tensors, audited and exposed as a single tensor.
- **PTCA** (`interdependent_lib/ptca/`) — Prime Tensor Circular Architecture: a
  seed holds 7 PCTA circles; a full core is 53 seeds. 4 of those seeds are
  sentinels, upgraded to 9 sentinel channels (S1–S9). Tensor shape: 53×9×8×7.
- **PCEA** (`interdependent_lib/pcea/`) — Prime Circle Encryption Algorithm:
  AES-256-GCM sealing/unsealing, HKDF-SHA256 key derivation, Shamir secret
  sharing over GF(256), key wrapping, rekey ceremonies, and zeroization helpers.
- **EDCM** (`interdependent_lib/edcm/`) — Energy Dissonance Circuit Model: bone
  inventory and marker tables for transcript analysis. 253 bone words mapped to
  PKQTS families, 35 multiword joins, morphological affixes, punctuation, and 9
  marker families (C/R/D/N/L/O/F/E/I). Data sourced from `edcmbone_canon_data_v1`.

## Development branch

All work goes on `claude/fix-remaining-issues-bYnZm`. Never push directly to
`main`.

## Install

```bash
pip install -e .
```

Requires Python ≥ 3.9 and `cryptography>=41`.

## Package structure

```
interdependent_lib/
├── __init__.py
├── pcna/
│   ├── __init__.py       # re-exports full public API
│   ├── activations.py    # relu, sigmoid, tanh, linear + derivatives
│   ├── layer.py          # PCNALayer (weights, forward, backward, update)
│   └── network.py        # PCNANetwork (MLP, mse_loss, binary_cross_entropy, as_tensor)
├── pcta/
│   ├── __init__.py       # re-exports full public API
│   └── circle.py         # PCTACircle (7 PCNANetworks, audit, as_tensor)
├── ptca/
│   ├── __init__.py       # re-exports full public API
│   ├── constants.py      # NODES, SENTINELS, PHASES, SLOTS, weights
│   ├── primes.py         # 53-prime node index
│   ├── tensor.py         # PTCATensor (flat-list 4-D tensor)
│   ├── sentinels.py      # SentinelState + S1–S9 channel dataclasses
│   ├── provenance.py     # SHA-256 block chain helpers
│   ├── exchange.py       # compute_score, Exchange router
│   └── instance.py       # PTCAInstance (high-level session object)
├── pcea/
│   ├── __init__.py       # re-exports full public API
│   ├── types.py          # LiveState, SealedState, WrappedLiveKey, …
│   ├── codec.py          # deterministic AAD / info encoding (stdlib only)
│   ├── aead.py           # AES-256-GCM seal / unseal
│   ├── kdf.py            # HKDF derive_keys / derive_nonce
│   ├── guardian.py       # seal_live_state / unseal_live_state
│   ├── wrap.py           # wrap_live_key / unwrap_live_key
│   ├── threshold.py      # GF(256) Shamir split_secret / reconstruct_secret
│   ├── commitment.py     # SHA-256 share commitment
│   ├── validate.py       # validate_invariant (pure structural checks)
│   ├── rekey.py          # rekey_epoch, split_meta_key, reconstruct_meta_key
│   └── wipe.py           # wipe / wipe_bytearray / wipe_bytes
└── edcm/
    ├── __init__.py       # re-exports full public API
    ├── bones.py          # bones(), words_by_family(), bone_set(), affixes(), punctuation()
    ├── markers.py        # markers(), family(), marker_set()
    ├── canon/
    │   └── __init__.py   # CanonLoader: unified lookup for words/affixes/punct/markers
    ├── data/
    │   ├── bones_words_v1.json    # 253 bones + 35 multiword joins
    │   ├── bones_affixes_v1.json  # inflectional + derivational affixes
    │   ├── bones_punct_v1.json    # punctuation entries
    │   └── markers_v1.json        # 9 behavioral metrics (C–I) with marker phrase lists
    └── parser/
        └── turns_rounds.py        # parse_transcript() → ParsedTranscript
                                   # BoneToken, FleshToken, Turn, Round, ParsedTranscript
                                   # _BoneClassifier: multiword→word→prefix→suffix→punct→flesh
```

## Key design notes

- **No numpy**: `PTCATensor` is backed by a flat Python `list[float]`.
- **No external Shamir library**: GF(256) is implemented inline in
  `threshold.py` using irreducible polynomial `0x11d` with generator `g=2`.
- **Zeroization**: `wipe.py` uses `ctypes.memset` into the CPython bytes
  object at the correct data offset (32 bytes on 64-bit, 16 bytes on 32-bit).
  This is best-effort and CPython-specific.
- **`pcea` depends on `cryptography>=41`**; `ptca` and `edcm` are stdlib-only.
- **EDCM data** is loaded lazily via `importlib.resources` and cached with
  `lru_cache`; JSON files ship as package data inside `edcm/data/`.
- **`CanonLoader`** (`edcm/canon/__init__.py`) is the single data-access layer
  used by the parser. It indexes words, multiword joins, affixes, punctuation,
  and behavioral markers for O(1) lookup. Mirrors the upstream
  `The-Interdependency/edcmbone` `Backend/src/edcmbone/canon/loader.py`.
- **`parse_transcript`** returns a `ParsedTranscript` object (not a dict).
  Access `.turns`, `.rounds`, `.speakers`, `.family_counts()`, `.bone_count()`.
  Each `Turn` has `.bone_tokens` (list of `BoneToken`), `.flesh_tokens`, and
  `.family_counts` (Counter). Accepts `str` or `list[dict]` input.
- **9 behavioral metrics** in `markers_v1.json` (C/R/D/N/L/O/F/E/I): each metric
  has `formula`, `computable_from_markers`, `requires_embeddings`, and a
  `markers` dict of category → phrase list. Access via `CanonLoader.metric_info()`
  or `CanonLoader.all_marker_phrases()`.

## Common tasks

Build a source + wheel distribution:
```bash
python -m build
```

Check what packages setuptools discovers:
```bash
python -c "from setuptools import find_packages; print(find_packages(include=['interdependent_lib*']))"
```
