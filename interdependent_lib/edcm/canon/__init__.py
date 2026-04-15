# GPT/Claude generated; context, prompt Erin Spencer
"""
EDCM CanonLoader — loads and exposes the v1 canon data files.

Provides lookup access to:
- bones_words_v1.json   free word bones (PKQTS families)
- bones_affixes_v1.json bound bone affixes
- bones_punct_v1.json   punctuation bones
- markers_v1.json       behavioral-layer markers (9-metric vector)

Usage::

    from interdependent_lib.edcm.canon import CanonLoader

    canon = CanonLoader()
    canon.lookup_word("not")        # -> {"word": "not", "primary": "P", ...}
    canon.lookup_affix("un-")       # -> {"affix": "un-", "primary": "P", ...}
    canon.lookup_punct("?")         # -> {"mark": "?", "primary": "Q", ...}
    canon.metric_names()            # -> ["C", "R", "D", "N", "L", "O", "F", "E", "I"]
    canon.metric_info("R")          # -> {metric, formula, computable_from_markers, ...}
    canon.marker_phrases("R", "refusal")  # -> ["I can't", "I won't", ...]
"""

from __future__ import annotations

import json
from importlib.resources import files
from typing import Any


def _load(filename: str) -> dict[str, Any]:
    return json.loads(
        files("interdependent_lib.edcm.data")
        .joinpath(filename)
        .read_text(encoding="utf-8")
    )


class CanonLoader:
    """Provides lookup access to all v1 canon data."""

    def __init__(self) -> None:
        self._words_data = _load("bones_words_v1.json")
        self._affixes_data = _load("bones_affixes_v1.json")
        self._punct_data = _load("bones_punct_v1.json")
        self._markers_data = _load("markers_v1.json")

        # Build lookup indexes
        self._word_index: dict[str, Any] = {
            entry["word"].lower(): entry
            for entry in self._words_data["words"]
        }
        self._multiword_index: dict[str, Any] = {
            entry["joined"].lower(): entry
            for entry in self._words_data.get("multiword_joins", [])
        }
        self._affix_index: dict[str, Any] = {}
        for section in ("inflectional", "derivational_prefixes", "derivational_suffixes"):
            for entry in self._affixes_data[section]["affixes"]:
                self._affix_index[entry["affix"].lower()] = entry
        self._punct_index: dict[str, Any] = {
            entry["mark"]: entry
            for entry in self._punct_data["punctuation"]
        }

    # ------------------------------------------------------------------
    # Bone lookups
    # ------------------------------------------------------------------

    def lookup_word(self, word: str) -> dict[str, Any] | None:
        """Return the bone entry for a free word, or None.

        Checks multiword joins first (using the joined/normalised form),
        then single-word entries.
        """
        key = word.lower().replace(" ", "")
        result = self._multiword_index.get(key)
        if result is None:
            result = self._word_index.get(word.lower())
        return result

    def lookup_affix(self, affix: str) -> dict[str, Any] | None:
        """Return the bone entry for an affix (e.g. ``'un-'``, ``'-ness'``), or None."""
        return self._affix_index.get(affix.lower())

    def lookup_punct(self, mark: str) -> dict[str, Any] | None:
        """Return the bone entry for a punctuation mark, or None."""
        return self._punct_index.get(mark)

    def all_words(self) -> list[dict[str, Any]]:
        """Return all free-word bone entries."""
        return list(self._words_data["words"])

    def all_multiword_joins(self) -> list[dict[str, Any]]:
        """Return all multiword-join entries."""
        return list(self._words_data.get("multiword_joins", []))

    def all_affixes(self) -> list[dict[str, Any]]:
        """Return all affix bone entries across all sections."""
        return list(self._affix_index.values())

    def all_punct(self) -> list[dict[str, Any]]:
        """Return all punctuation bone entries."""
        return list(self._punct_data["punctuation"])

    # ------------------------------------------------------------------
    # Marker lookups
    # ------------------------------------------------------------------

    def metric_names(self) -> list[str]:
        """Return the ordered list of behavioral metric keys (C–I)."""
        return [k for k in self._markers_data if k != "_meta"]

    def metric_info(self, metric: str) -> dict[str, Any]:
        """Return the full metric dict for a given key (e.g. ``'C'``, ``'R'``).

        Keys include: metric, formula, computable_from_markers,
        requires_embeddings, explanation, markers.
        """
        if metric not in self._markers_data or metric == "_meta":
            raise KeyError(
                f"Unknown metric {metric!r}. Available: {self.metric_names()}"
            )
        return self._markers_data[metric]

    def marker_phrases(self, metric: str, category: str) -> list[str]:
        """Return the phrase list for a marker category within a metric.

        Parameters
        ----------
        metric:
            e.g. ``"R"``
        category:
            a key inside ``metric["markers"]``, e.g. ``"refusal"``
        """
        info = self.metric_info(metric)
        mcat = info.get("markers", {})
        if category not in mcat:
            raise KeyError(
                f"Unknown category {category!r} for metric {metric!r}. "
                f"Available: {list(mcat.keys())}"
            )
        return list(mcat[category])

    def all_marker_phrases(self, metric: str) -> list[str]:
        """Return a flat list of all marker phrases across all categories for a metric."""
        info = self.metric_info(metric)
        phrases: list[str] = []
        for phrases_list in info.get("markers", {}).values():
            if isinstance(phrases_list, list):
                phrases.extend(p for p in phrases_list if isinstance(p, str))
        return phrases

    # ------------------------------------------------------------------
    # Meta
    # ------------------------------------------------------------------

    def meta(self, dataset: str) -> dict[str, Any]:
        """Return the ``_meta`` block for a dataset.

        Parameters
        ----------
        dataset:
            One of ``"words"``, ``"affixes"``, ``"punct"``, ``"markers"``.
        """
        mapping = {
            "words": self._words_data,
            "affixes": self._affixes_data,
            "punct": self._punct_data,
            "markers": self._markers_data,
        }
        if dataset not in mapping:
            raise KeyError(
                f"Unknown dataset {dataset!r}. Choose from: {list(mapping.keys())}"
            )
        return mapping[dataset].get("_meta", {})
