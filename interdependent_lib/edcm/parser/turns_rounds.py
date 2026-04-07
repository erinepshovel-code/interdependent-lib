# GPT/Claude generated; context, prompt Erin Spencer
"""
Transcript parser for EDCM-PCNA-PCTA analysis.

Parses a transcript into turns, applies multiword joins, tokenizes, and tags
each token against the bone inventory.  Returns a structured result suitable
for downstream PKQTS family counting.

Transcript formats accepted
---------------------------
1. List of dicts:  [{"speaker": "A", "text": "..."}, ...]
2. Plain string:   "A: hello\\nB: world\\n..."  (Speaker: text per line)
3. Plain string without speaker labels — treated as a single anonymous turn.

Output structure
----------------
::

    {
        "turns": [
            {
                "speaker": "A",
                "raw_text": "of course I will not do that again.",
                "normalized_text": "ofcourse I will not do that again.",
                "tokens": ["ofcourse", "I", "will", "not", "do", "that", "again", "."],
                "tagged": [
                    {"token": "ofcourse", "bone": True, "primary": "P", "families": ["P"],
                     "join": True, "original": "of course"},
                    {"token": "I",       "bone": False},
                    {"token": "will",    "bone": True, "primary": "T", "families": ["T"]},
                    {"token": "not",     "bone": True, "primary": "P", "families": ["P"]},
                    ...
                ],
                "bone_counts": {"P": 2, "K": 0, "Q": 0, "T": 1, "S": 0},
                "bone_tokens": ["ofcourse", "will", "not"],
            },
            ...
        ],
        "totals": {"P": 2, "K": 0, "Q": 0, "T": 1, "S": 0},
        "join_log": [
            {"turn": 0, "original": "of course", "joined": "ofcourse", "start": 0}
        ],
    }
"""

from __future__ import annotations

import re
import string
from typing import Any

from interdependent_lib.edcm.bones import bones as _bones_list
from interdependent_lib.edcm.bones import multiword_joins as _mw_joins


# ---------------------------------------------------------------------------
# Build lookup tables (module-level, built once)
# ---------------------------------------------------------------------------

def _build_bone_index() -> dict[str, dict[str, Any]]:
    """word → bone entry (lowercase keys)."""
    return {entry["word"].lower(): entry for entry in _bones_list()}


def _build_join_index() -> list[tuple[str, dict[str, Any]]]:
    """
    Sorted list of (original_lower, entry) pairs, longest-match-first.
    """
    joins = [(j["original"].lower(), j) for j in _mw_joins()]
    joins.sort(key=lambda x: -len(x[0]))  # longest first
    return joins


_BONE_INDEX: dict[str, dict[str, Any]] = {}
_JOIN_INDEX: list[tuple[str, dict[str, Any]]] = []


def _ensure_indexes() -> None:
    global _BONE_INDEX, _JOIN_INDEX
    if not _BONE_INDEX:
        _BONE_INDEX = _build_bone_index()
        _JOIN_INDEX = _build_join_index()


# ---------------------------------------------------------------------------
# Normalisation: apply multiword joins (longest-match-first)
# ---------------------------------------------------------------------------

def _apply_joins(
    text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Replace multiword sequences with their joined forms.

    Returns (normalized_text, join_events).
    join_events records each replacement for losslessness / audit.
    """
    _ensure_indexes()
    events: list[dict[str, Any]] = []
    text_lower = text.lower()

    # We'll build a replacement map: (start, end) → joined form
    replacements: list[tuple[int, int, str, str]] = []  # (start, end, original, joined)
    covered: set[int] = set()

    for original, entry in _JOIN_INDEX:
        pos = 0
        while True:
            idx = text_lower.find(original, pos)
            if idx == -1:
                break
            end = idx + len(original)
            # Check it isn't already covered by a longer match
            span = set(range(idx, end))
            if span & covered:
                pos = idx + 1
                continue
            # Boundary check: must be at word boundary
            before_ok = idx == 0 or not text_lower[idx - 1].isalpha()
            after_ok = end == len(text_lower) or not text_lower[end].isalpha()
            if before_ok and after_ok:
                replacements.append((idx, end, text[idx:end], entry["joined"]))
                covered |= span
                events.append({
                    "original": text[idx:end],
                    "joined": entry["joined"],
                    "start": idx,
                })
            pos = idx + 1

    if not replacements:
        return text, events

    # Apply replacements in reverse order so indices stay valid
    replacements.sort(key=lambda x: -x[0])
    result = list(text)
    for start, end, _, joined in replacements:
        result[start:end] = list(joined)
    return "".join(result), events


# ---------------------------------------------------------------------------
# Tokenisation
# ---------------------------------------------------------------------------

_PUNCT = set(string.punctuation)


def _tokenize(text: str) -> list[str]:
    """
    Split on whitespace, then split off leading/trailing punctuation as
    separate tokens.  Preserves internal punctuation (contractions, hyphens).
    """
    tokens: list[str] = []
    for raw in text.split():
        # Strip leading punctuation
        leading = []
        while raw and raw[0] in _PUNCT and raw[0] not in ("'", "-"):
            leading.append(raw[0])
            raw = raw[1:]
        # Strip trailing punctuation
        trailing = []
        while raw and raw[-1] in _PUNCT and raw[-1] not in ("'", "-"):
            trailing.append(raw[-1])
            raw = raw[:-1]
        tokens.extend(leading)
        if raw:
            tokens.append(raw)
        tokens.extend(reversed(trailing))
    return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# Bone tagging
# ---------------------------------------------------------------------------

def _tag_token(token: str) -> dict[str, Any]:
    """Return a tagged dict for a single token."""
    _ensure_indexes()
    key = token.lower()
    if key in _BONE_INDEX:
        entry = _BONE_INDEX[key]
        return {
            "token": token,
            "bone": True,
            "primary": entry["primary"],
            "families": entry["families"],
            "notes": entry.get("notes", ""),
        }
    return {"token": token, "bone": False}


def _tag_tokens(
    tokens: list[str],
    join_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tag all tokens; annotate joined tokens with their original form."""
    joined_forms = {e["joined"].lower(): e["original"] for e in join_events}
    tagged: list[dict[str, Any]] = []
    for token in tokens:
        t = _tag_token(token)
        if t["bone"] and token.lower() in joined_forms:
            t["join"] = True
            t["original"] = joined_forms[token.lower()]
        tagged.append(t)
    return tagged


# ---------------------------------------------------------------------------
# Turn parsing
# ---------------------------------------------------------------------------

_SPEAKER_RE = re.compile(r"^([A-Za-z0-9_\-]+)\s*:\s*(.*)")


def _parse_raw(transcript: Any) -> list[dict[str, str]]:
    """
    Normalise input to a list of {"speaker": ..., "text": ...} dicts.
    """
    if isinstance(transcript, list):
        turns = []
        for item in transcript:
            if isinstance(item, dict):
                turns.append({
                    "speaker": str(item.get("speaker", "")),
                    "text": str(item.get("text", "")),
                })
            else:
                turns.append({"speaker": "", "text": str(item)})
        return turns

    if isinstance(transcript, str):
        lines = [l.rstrip() for l in transcript.splitlines() if l.strip()]
        turns = []
        for line in lines:
            m = _SPEAKER_RE.match(line)
            if m:
                turns.append({"speaker": m.group(1), "text": m.group(2)})
            else:
                turns.append({"speaker": "", "text": line})
        return turns

    raise TypeError(f"transcript must be str or list, got {type(transcript)}")


def _count_families(tagged: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {"P": 0, "K": 0, "Q": 0, "T": 0, "S": 0}
    for t in tagged:
        if t["bone"]:
            counts[t["primary"]] = counts.get(t["primary"], 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_transcript(transcript: Any) -> dict[str, Any]:
    """
    Parse a transcript into tagged turns with PKQTS bone counts.

    Parameters
    ----------
    transcript:
        A list of ``{"speaker": str, "text": str}`` dicts **or** a plain
        string with one turn per line, optionally prefixed ``Speaker: text``.

    Returns
    -------
    dict with keys:
        turns     : list of per-turn analysis dicts
        totals    : aggregate PKQTS counts across all turns
        join_log  : all multiword-join events (for audit / losslessness)
    """
    raw_turns = _parse_raw(transcript)

    out_turns: list[dict[str, Any]] = []
    totals: dict[str, int] = {"P": 0, "K": 0, "Q": 0, "T": 0, "S": 0}
    join_log: list[dict[str, Any]] = []

    for i, turn in enumerate(raw_turns):
        raw_text = turn["text"]
        normalized, join_events = _apply_joins(raw_text)

        for ev in join_events:
            join_log.append({"turn": i, **ev})

        tokens = _tokenize(normalized)
        tagged = _tag_tokens(tokens, join_events)
        counts = _count_families(tagged)

        for family, n in counts.items():
            totals[family] = totals.get(family, 0) + n

        out_turns.append({
            "speaker": turn["speaker"],
            "raw_text": raw_text,
            "normalized_text": normalized,
            "tokens": tokens,
            "tagged": tagged,
            "bone_counts": counts,
            "bone_tokens": [t["token"] for t in tagged if t["bone"]],
        })

    return {"turns": out_turns, "totals": totals, "join_log": join_log}
