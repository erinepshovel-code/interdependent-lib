# GPT/Claude generated; context, prompt Erin Spencer
"""
EDCM bone inventory loader.

Bones are the structural/operator words in EDCM — words that create, redirect,
or resolve constraint relationships (PKQTS families).  Flesh words (degree
adverbs, pure content words) are excluded.

Families
--------
P  — Polarity (negation, affirmation)
K  — Conditionality / contingency
Q  — Quantity / scope
T  — Temporal / aspectual
S  — Structural / relational

Usage
-----
::

    from interdependent_lib.edcm.bones import bones, multiword_joins, words_by_family

    all_bones = bones()           # list of bone dicts
    joins     = multiword_joins() # list of multiword-join dicts
    t_bones   = words_by_family("T")
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


def _load() -> dict[str, Any]:
    return json.loads(
        files("interdependent_lib.edcm.data")
        .joinpath("bones_words_v1.json")
        .read_text(encoding="utf-8")
    )


@lru_cache(maxsize=1)
def _data() -> dict[str, Any]:
    return _load()


def meta() -> dict[str, Any]:
    """Return the _meta block from bones_words_v1.json."""
    return _data()["_meta"]


def bones() -> list[dict[str, Any]]:
    """
    Return the full bone word list.

    Each entry is a dict with keys:
      word     : str
      primary  : str   — primary PKQTS family
      families : list[str]
      notes    : str
    """
    return _data()["words"]


def multiword_joins() -> list[dict[str, Any]]:
    """
    Return the multiword-join table.

    Each entry has:
      joined   : str   — smashed form (e.g. 'ofcourse')
      original : str   — original form (e.g. 'of course')
      primary  : str
      families : list[str]
      notes    : str
    """
    return _data()["multiword_joins"]


def words_by_family(family: str) -> list[dict[str, Any]]:
    """
    Return all bone entries whose *primary* family matches *family*.

    Parameters
    ----------
    family:
        One of ``'P'``, ``'K'``, ``'Q'``, ``'T'``, ``'S'``.
    """
    return [w for w in bones() if w["primary"] == family]


def bone_set() -> frozenset[str]:
    """Return a frozenset of all bone word strings (for fast membership tests)."""
    words = {w["word"] for w in bones()}
    joins = {j["joined"] for j in multiword_joins()}
    return frozenset(words | joins)


def affixes() -> dict[str, Any]:
    """Return the full bones_affixes_v1 data."""
    return json.loads(
        files("interdependent_lib.edcm.data")
        .joinpath("bones_affixes_v1.json")
        .read_text(encoding="utf-8")
    )


def punctuation() -> dict[str, Any]:
    """Return the full bones_punct_v1 data."""
    return json.loads(
        files("interdependent_lib.edcm.data")
        .joinpath("bones_punct_v1.json")
        .read_text(encoding="utf-8")
    )
