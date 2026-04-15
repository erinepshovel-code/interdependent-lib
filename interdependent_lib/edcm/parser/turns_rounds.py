# GPT/Claude generated; context, prompt Erin Spencer
"""
EDCM transcript parser.

Parses a transcript into turns and rounds, classifying each token against the
bone canon (CanonLoader).  Classification pipeline:

  1. multiword join (2-gram smash)
  2. single-word lookup
  3. prefix strip → affix lookup
  4. suffix strip → affix lookup
  5. punctuation lookup (only entries that emit tokens)
  6. flesh (unclassified)

Public API
----------
parse_transcript(transcript, round_strategy="cycle", canon=None)
    -> ParsedTranscript

Data classes
------------
BoneToken, FleshToken, Turn, Round, ParsedTranscript

Transcript formats accepted
---------------------------
1. List of dicts:  [{"speaker": "A", "text": "..."}, ...]
2. Plain string:   "A: hello\\nB: world\\n..."  (Speaker: text per line)
3. Plain string without speaker labels — treated as one anonymous turn.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

from interdependent_lib.edcm.canon import CanonLoader

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class BoneToken:
    """A single token that matched the bone canon."""

    __slots__ = ("surface", "normalized", "bone_type", "primary", "families", "entry")

    def __init__(
        self,
        surface: str,
        normalized: str,
        bone_type: str,
        primary: str,
        families: list[str],
        entry: dict[str, Any],
    ) -> None:
        self.surface = surface      # original text as it appeared
        self.normalized = normalized  # key used for lookup
        self.bone_type = bone_type  # "word" | "multiword" | "affix" | "punct"
        self.primary = primary      # dominant PKQTS letter
        self.families = families    # full family list, e.g. ["P", "K"]
        self.entry = entry          # raw canon dict

    def __repr__(self) -> str:
        return f"BoneToken({self.surface!r}, {self.primary})"


class FleshToken:
    """A token that did not match any bone canon entry."""

    __slots__ = ("surface",)

    def __init__(self, surface: str) -> None:
        self.surface = surface

    def __repr__(self) -> str:
        return f"FleshToken({self.surface!r})"


class Turn:
    """One speaker utterance."""

    def __init__(
        self,
        speaker: str,
        text: str,
        tokens: list[BoneToken | FleshToken],
    ) -> None:
        self.speaker = speaker
        self.text = text
        self.tokens = tokens
        self.bone_tokens: list[BoneToken] = [
            t for t in tokens if isinstance(t, BoneToken)
        ]
        self.flesh_tokens: list[FleshToken] = [
            t for t in tokens if isinstance(t, FleshToken)
        ]

        # Family counts for this turn (primary family only)
        self.family_counts: Counter[str] = Counter()
        for t in self.bone_tokens:
            self.family_counts[t.primary] += 1

        self.bone_count = len(self.bone_tokens)
        self.token_count = len(self.tokens)

    def __repr__(self) -> str:
        return f"Turn({self.speaker!r}, bones={self.bone_count})"


class Round:
    """A group of turns forming one exchange cycle."""

    def __init__(self, index: int, turns: list[Turn]) -> None:
        self.index = index
        self.turns = turns

        # Aggregate across turns
        self.all_tokens: list[BoneToken | FleshToken] = []
        self.bone_tokens: list[BoneToken] = []
        self.family_counts: Counter[str] = Counter()

        for t in turns:
            self.all_tokens.extend(t.tokens)
            self.bone_tokens.extend(t.bone_tokens)
            self.family_counts.update(t.family_counts)

        self.bone_count = len(self.bone_tokens)
        self.token_count = len(self.all_tokens)
        self.speakers = [t.speaker for t in turns]

    def __repr__(self) -> str:
        return (
            f"Round(index={self.index}, turns={len(self.turns)}, "
            f"bones={self.bone_count})"
        )


class ParsedTranscript:
    """Full parse result."""

    def __init__(self, rounds: list[Round], turns: list[Turn]) -> None:
        self.rounds = rounds
        self.turns = turns
        self.speakers = _ordered_unique(t.speaker for t in turns)

    def bone_count(self) -> int:
        """Total bone token count across all turns."""
        return sum(t.bone_count for t in self.turns)

    def family_counts(self) -> dict[str, int]:
        """Aggregate PKQTS family counts across all turns."""
        total: Counter[str] = Counter()
        for t in self.turns:
            total.update(t.family_counts)
        return dict(total)

    def __repr__(self) -> str:
        return (
            f"ParsedTranscript(rounds={len(self.rounds)}, "
            f"turns={len(self.turns)}, speakers={self.speakers})"
        )


# ---------------------------------------------------------------------------
# Turn splitter — detects speaker prefixes in common transcript formats
# ---------------------------------------------------------------------------

# Patterns tried in order; each must have named groups "speaker" and "text".
_TURN_PATTERNS = [
    # **Speaker**: text  (markdown bold)
    re.compile(r"^\*\*(?P<speaker>[^\*]+)\*\*\s*:\s*(?P<text>.+)$", re.MULTILINE),
    # [Speaker]: text
    re.compile(r"^\[(?P<speaker>[^\]]+)\]\s*:\s*(?P<text>.+)$", re.MULTILINE),
    # Speaker (role): text
    re.compile(
        r"^(?P<speaker>[A-Za-z][A-Za-z0-9 _\-]{0,30})\s*\([^)]*\)\s*:\s*(?P<text>.+)$",
        re.MULTILINE,
    ),
    # Speaker: text  (plain label — shortest reliable last)
    re.compile(
        r"^(?P<speaker>[A-Za-z][A-Za-z0-9 _\-]{0,30})\s*:\s*(?P<text>.+)$",
        re.MULTILINE,
    ),
]


def _split_turns_str(text: str) -> list[tuple[str, str]]:
    """Return list of (speaker, text) pairs from a raw transcript string."""
    for pattern in _TURN_PATTERNS:
        matches = list(pattern.finditer(text))
        if len(matches) >= 2:
            return [
                (m.group("speaker").strip(), m.group("text").strip())
                for m in matches
            ]
    # Fallback: treat the whole transcript as one anonymous turn
    stripped = text.strip()
    if stripped:
        return [("SPEAKER", stripped)]
    return []


def _split_turns(transcript: Any) -> list[tuple[str, str]]:
    """Normalise input to a list of (speaker, text) pairs."""
    if isinstance(transcript, list):
        result = []
        for item in transcript:
            if isinstance(item, dict):
                result.append((
                    str(item.get("speaker", "")),
                    str(item.get("text", "")),
                ))
            else:
                result.append(("", str(item)))
        return result
    if isinstance(transcript, str):
        return _split_turns_str(transcript)
    raise TypeError(f"transcript must be str or list, got {type(transcript)!r}")


# ---------------------------------------------------------------------------
# Round grouper
# ---------------------------------------------------------------------------


def _group_into_rounds(turns: list[Turn], strategy: str = "cycle") -> list[Round]:
    """Group Turn objects into Round objects.

    Parameters
    ----------
    strategy:
        ``"cycle"`` — a new round starts each time the first-seen speaker
        takes the floor again after at least one other speaker has spoken.
        Works for N speakers.

        ``"pairs"`` — every consecutive pair of turns is one round
        (ignores speaker identity).
    """
    if not turns:
        return []

    if strategy == "pairs":
        rounds: list[Round] = []
        for i in range(0, len(turns), 2):
            rounds.append(Round(len(rounds), turns[i : i + 2]))
        return rounds

    # strategy == "cycle"
    anchor = turns[0].speaker
    rounds = []
    current: list[Turn] = []
    seen_others = False

    for turn in turns:
        if turn.speaker == anchor and seen_others and current:
            rounds.append(Round(len(rounds), current))
            current = [turn]
            seen_others = False
        else:
            if turn.speaker != anchor:
                seen_others = True
            current.append(turn)

    if current:
        rounds.append(Round(len(rounds), current))

    return rounds


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

# Split into word-runs and individual punctuation characters
_WORD_RE = re.compile(r"[A-Za-z''\-]+|[^\w\s]|\d+")


def _raw_tokens(text: str) -> list[str]:
    """Return list of raw token strings from utterance text."""
    return _WORD_RE.findall(text)


# ---------------------------------------------------------------------------
# Bone classifier
# ---------------------------------------------------------------------------


class _BoneClassifier:
    """Classifies token sequences using CanonLoader as its model."""

    def __init__(self, canon: CanonLoader) -> None:
        self._canon = canon

        # Pre-sort prefixes/suffixes longest-first so greedy matching works
        self._prefixes: list[str] = sorted(
            [
                a["affix"].rstrip("-")
                for a in canon.all_affixes()
                if a.get("type") == "prefix"
            ],
            key=len,
            reverse=True,
        )
        self._suffixes: list[str] = sorted(
            [
                a["affix"].lstrip("-")
                for a in canon.all_affixes()
                if a.get("type") == "suffix"
            ],
            key=len,
            reverse=True,
        )
        self._affix_cache: dict[str, Any] = {}

    def _make_bone(
        self,
        surface: str,
        normalized: str,
        bone_type: str,
        entry: dict[str, Any],
    ) -> BoneToken:
        return BoneToken(
            surface=surface,
            normalized=normalized,
            bone_type=bone_type,
            primary=entry["primary"],
            families=entry.get("families", [entry["primary"]]),
            entry=entry,
        )

    def classify_sequence(
        self, raw_tokens: list[str]
    ) -> list[BoneToken | FleshToken]:
        """Classify a flat list of raw token strings into BoneToken/FleshToken.

        Tries (in order) for each position:

        1. Multiword join (2-gram smash)
        2. Single word lookup
        3. Prefix affix strip
        4. Suffix affix strip
        5. Punctuation lookup (only entries that emit tokens)
        6. Flesh
        """
        result: list[BoneToken | FleshToken] = []
        i = 0
        n = len(raw_tokens)

        while i < n:
            tok = raw_tokens[i]

            # 1. Try 2-gram multiword join
            if i + 1 < n:
                bigram = (tok + raw_tokens[i + 1]).lower().replace(" ", "")
                entry = self._canon.lookup_word(bigram)
                if entry and entry.get("joined"):
                    result.append(
                        self._make_bone(
                            tok + " " + raw_tokens[i + 1],
                            bigram,
                            "multiword",
                            entry,
                        )
                    )
                    i += 2
                    continue

            # 2. Single word lookup
            entry = self._canon.lookup_word(tok)
            if entry:
                result.append(self._make_bone(tok, tok.lower(), "word", entry))
                i += 1
                continue

            # 3. Prefix affix strip
            lower = tok.lower()
            matched_affix = False
            for pre in self._prefixes:
                if lower.startswith(pre) and len(lower) - len(pre) >= 2:
                    affix_key = pre + "-"
                    if affix_key not in self._affix_cache:
                        self._affix_cache[affix_key] = self._canon.lookup_affix(
                            affix_key
                        )
                    entry = self._affix_cache[affix_key]
                    if entry:
                        result.append(
                            self._make_bone(tok, affix_key, "affix", entry)
                        )
                        matched_affix = True
                        break

            if matched_affix:
                i += 1
                continue

            # 4. Suffix affix strip
            for suf in self._suffixes:
                if lower.endswith(suf) and len(lower) - len(suf) >= 2:
                    affix_key = "-" + suf
                    if affix_key not in self._affix_cache:
                        self._affix_cache[affix_key] = self._canon.lookup_affix(
                            affix_key
                        )
                    entry = self._affix_cache[affix_key]
                    if entry:
                        result.append(
                            self._make_bone(tok, affix_key, "affix", entry)
                        )
                        matched_affix = True
                        break

            if matched_affix:
                i += 1
                continue

            # 5. Punctuation — only count entries that actually emit tokens
            entry = self._canon.lookup_punct(tok)
            if entry and entry.get("tokens_emitted", 0) > 0:
                result.append(self._make_bone(tok, tok, "punct", entry))
                i += 1
                continue

            # 6. Flesh
            result.append(FleshToken(tok))
            i += 1

        return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_transcript(
    transcript: Any,
    round_strategy: str = "cycle",
    canon: CanonLoader | None = None,
) -> ParsedTranscript:
    """Parse a transcript into a :class:`ParsedTranscript`.

    Parameters
    ----------
    transcript:
        Either a **list of dicts** ``[{"speaker": str, "text": str}, ...]``
        or a **plain string** with speaker-prefixed lines
        (e.g. ``"A: hello\\nB: world"``).
        Multiple label formats are detected automatically:
        ``Speaker:``, ``[Speaker]:``, ``**Speaker**:``.

    round_strategy:
        ``"cycle"`` (default) — a round ends when the anchor speaker
        (first speaker seen) regains the floor after at least one other
        speaker has spoken.

        ``"pairs"`` — every consecutive pair of turns is one round.

    canon:
        Optional pre-loaded :class:`~interdependent_lib.edcm.canon.CanonLoader`.
        If *None*, a fresh instance is created.

    Returns
    -------
    ParsedTranscript
    """
    if canon is None:
        canon = CanonLoader()

    classifier = _BoneClassifier(canon)
    raw_turns = _split_turns(transcript)

    turns: list[Turn] = []
    for speaker, text in raw_turns:
        toks = _raw_tokens(text)
        classified = classifier.classify_sequence(toks)
        turns.append(Turn(speaker, text, classified))

    rounds = _group_into_rounds(turns, strategy=round_strategy)
    return ParsedTranscript(rounds=rounds, turns=turns)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ordered_unique(iterable: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in iterable:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result
