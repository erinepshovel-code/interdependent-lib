# CLAUDE.md

## Repository overview

`interdependent-lib` is a pure-Python library that implements three subsystems:

- **PTCA** (`interdependent_lib/ptca/`) — Prime Tensor Circular Architecture: a
  53×9×8×7 routing tensor, nine sentinel channels (S1–S9), exchange scoring,
  provenance hashing, and a high-level `PTCAInstance` session object.
- **PCEA** (`interdependent_lib/pcea/`) — Guardian state encryption: AES-256-GCM
  sealing/unsealing, HKDF-SHA256 key derivation, Shamir secret sharing over
  GF(256), key wrapping, rekey ceremonies, and best-effort zeroization helpers.
- **EDCM** (`interdependent_lib/edcm/`) — Bone inventory and marker tables for
  EDCM-PCNA-PCTA transcript analysis: 253 bone words mapped to PKQTS families,
  35 multiword joins, morphological affixes, punctuation, and 9 marker families
  (C/R/D/N/L/O/F/E/I). Data sourced from `edcmbone_canon_data_v1`.

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
    ├── data/
    │   ├── bones_words_v1.json    # 253 bones + 35 multiword joins
    │   ├── bones_affixes_v1.json  # inflectional + derivational affixes
    │   ├── bones_punct_v1.json    # punctuation entries
    │   └── markers_v1.json        # 9 marker families × 6 entries
    └── parser/
        └── turns_rounds.py        # parse_transcript(): join → tokenize → tag → PKQTS counts
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

## Common tasks

Build a source + wheel distribution:
```bash
python -m build
```

Check what packages setuptools discovers:
```bash
python -c "from setuptools import find_packages; print(find_packages(include=['interdependent_lib*']))"
```
