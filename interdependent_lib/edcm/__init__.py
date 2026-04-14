# GPT/Claude generated; context, prompt Erin Spencer
"""EDCM — Energy Dissonance Circuit Model: bone inventory, marker tables, and transcript parser."""

from .bones import affixes, bone_set, bones, meta, multiword_joins, punctuation, words_by_family
from .canon import CanonLoader
from .markers import family, marker_set, markers
from .parser.turns_rounds import (
    BoneToken,
    FleshToken,
    ParsedTranscript,
    Round,
    Turn,
    parse_transcript,
)

__all__ = [
    # bones
    "affixes",
    "bone_set",
    "bones",
    "meta",
    "multiword_joins",
    "punctuation",
    "words_by_family",
    # canon
    "CanonLoader",
    # markers
    "family",
    "marker_set",
    "markers",
    # parser
    "BoneToken",
    "FleshToken",
    "ParsedTranscript",
    "Round",
    "Turn",
    "parse_transcript",
]
