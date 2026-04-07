# interdependent-lib

Pure-Python implementation of three interdependent subsystems used in
EDCM-PCNA-PCTA transcript analysis and guardian-state encryption.

## Subsystems

### PTCA — Prime Tensor Circular Architecture

A 53 × 9 × 8 × 7 routing tensor indexed by prime nodes, nine sentinel channels
(S1–S9), eight processing phases, and seven heptagram slots.  Every exchange is
scored, written to the tensor, and recorded in the S9 audit trail.

```python
from interdependent_lib.ptca import PTCAInstance

inst = PTCAInstance(
    model_id="claude-sonnet-4-6",
    caller_id="user:alice",
    approved=True,
)

inst.push_context({"role": "user", "content": "Hello", "tokens": 5})
result = inst.route(node=0, phase=0, slot=0, s1=1.0, s5=0.9)
print(result.score)          # weighted exchange score
print(inst.audit_tail(3))    # last 3 S9 audit entries
```

**No external dependencies.** Tensor backed by a flat `list[float]`.

---

### PCEA — Guardian State Encryption

AES-256-GCM sealing/unsealing of `LiveState`, HKDF-SHA256 key derivation,
Shamir secret sharing over GF(256), key wrapping, and rekey ceremonies.

```python
from interdependent_lib.pcea import (
    derive_keys, seal_live_state, unseal_live_state,
    split_meta_key, reconstruct_meta_key, wipe,
)

live_key, meta_key = derive_keys(ikm, epoch=1, key_id="k1", guardian_node_id="g0")
sealed = seal_live_state(state, live_key, epoch=1, key_id="k1",
                         seal_counter=0, guardian_node_id="g0", sealed_by="g0")
recovered = unseal_live_state(sealed, live_key)
wipe(live_key)
```

**Requires** `cryptography >= 41`.  GF(256) Shamir is implemented inline
(irreducible polynomial `0x11d`, generator `g=2`) — no external Shamir library.

---

### EDCM — Bone Inventory & Marker Tables

253 English *bone* words (operator/structural words that create, redirect, or
resolve constraint relationships) mapped to PKQTS families, plus 35 multiword
joins, morphological affixes, punctuation, and 9 discourse marker families.

```python
from interdependent_lib.edcm import bones, words_by_family, bone_set, markers, family

print(len(bones()))              # 253
print(len(bone_set()))           # 288 (bones + multiword joins)
print(words_by_family("T")[:3]) # temporal/aspectual bones

from interdependent_lib.edcm.parser.turns_rounds import parse_transcript

result = parse_transcript([
    {"speaker": "A", "text": "I will not do that again."},
    {"speaker": "B", "text": "But of course you should."},
])
for turn in result["turns"]:
    print(turn["speaker"], turn["bone_counts"])
```

**No external dependencies.**  Data loaded lazily via `importlib.resources`.

PKQTS families:

| Family | Meaning |
|--------|---------|
| P | Polarity — negation and affirmation |
| K | Conditionality / contingency |
| Q | Quantity / scope |
| T | Temporal / aspectual |
| S | Structural / relational |

---

## Install

```bash
pip install interdependent-lib
```

Requires Python ≥ 3.9.  `pcea` requires `cryptography >= 41`; `ptca` and `edcm`
are stdlib-only.

## Development

```bash
git clone https://github.com/wayseer00/interdependent-lib
cd interdependent-lib
pip install -e .
```

All development goes on `claude/fix-remaining-issues-bYnZm`; never push
directly to `main`.

## License

MIT — see `LICENSE`.
