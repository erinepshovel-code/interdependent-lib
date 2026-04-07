# GPT/Claude generated; context, prompt Erin Spencer
"""
EDCM marker loader.

Markers annotate discourse structure.  The nine marker families are:

  C — Causal
  R — Resultative
  D — Discourse / meta
  N — Narrative / sequential
  L — Logical / inferential
  O — Oppositional / concessive
  F — Factive / epistemic
  E — Evaluative
  I — Interrogative / clarificatory

Each family contains 6 entries.

Usage
-----
::

    from interdependent_lib.edcm.markers import markers, family

    all_markers = markers()
    causal      = family("C")
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


@lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    return json.loads(
        files("interdependent_lib.edcm.data")
        .joinpath("markers_v1.json")
        .read_text(encoding="utf-8")
    )


def meta() -> dict[str, Any]:
    """Return the _meta block from markers_v1.json."""
    return _data()["_meta"]


def markers() -> dict[str, list[Any]]:
    """
    Return all marker families as a dict keyed by family letter.

    Keys: C, R, D, N, L, O, F, E, I
    """
    return {k: v for k, v in _data().items() if not k.startswith("_")}


def family(letter: str) -> list[Any]:
    """
    Return the entries for a single marker family.

    Parameters
    ----------
    letter:
        One of ``'C'``, ``'R'``, ``'D'``, ``'N'``, ``'L'``, ``'O'``,
        ``'F'``, ``'E'``, ``'I'``.

    Raises
    ------
    KeyError
        If *letter* is not a valid marker family.
    """
    d = _data()
    if letter not in d:
        valid = [k for k in d if not k.startswith("_")]
        raise KeyError(f"Unknown marker family {letter!r}. Valid: {valid}")
    return d[letter]


def marker_set() -> frozenset[str]:
    """Return a frozenset of all marker word/phrase strings (for fast membership tests)."""
    result: set[str] = set()
    for entries in markers().values():
        for entry in entries:
            if isinstance(entry, dict):
                w = entry.get("word") or entry.get("marker") or entry.get("phrase")
                if w:
                    result.add(w)
            elif isinstance(entry, str):
                result.add(entry)
    return frozenset(result)
