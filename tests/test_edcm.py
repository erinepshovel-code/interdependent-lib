"""Tests for interdependent_lib.edcm."""

import pytest

from interdependent_lib.edcm.bones import (
    bones, multiword_joins, bone_set, words_by_family,
    affixes, punctuation, meta,
)
from interdependent_lib.edcm.markers import markers, family, marker_set, meta as markers_meta
from interdependent_lib.edcm.canon import CanonLoader
from interdependent_lib.edcm.parser.turns_rounds import (
    parse_transcript,
    BoneToken, FleshToken, Turn, Round, ParsedTranscript,
)


# ---------------------------------------------------------------------------
# Bones
# ---------------------------------------------------------------------------

class TestBones:
    def test_count(self):
        assert len(bones()) == 253

    def test_multiword_joins_count(self):
        assert len(multiword_joins()) == 35

    def test_bone_set_size(self):
        bs = bone_set()
        assert len(bs) == 253 + 35

    def test_bone_set_is_frozenset(self):
        assert isinstance(bone_set(), frozenset)

    def test_bone_entry_fields(self):
        for bone in bones():
            assert "word" in bone
            assert "primary" in bone
            assert "families" in bone
            assert bone["primary"] in ("P", "K", "Q", "T", "S")

    def test_families_valid(self):
        valid = {"P", "K", "Q", "T", "S"}
        for bone in bones():
            for f in bone["families"]:
                assert f in valid

    def test_words_by_family_returns_subset(self):
        for fam in ("P", "K", "Q", "T", "S"):
            result = words_by_family(fam)
            assert all(b["primary"] == fam for b in result)

    def test_words_by_family_totals(self):
        total = sum(len(words_by_family(f)) for f in ("P", "K", "Q", "T", "S"))
        assert total == 253

    def test_known_words_present(self):
        bs = bone_set()
        assert "not" in bs
        assert "will" in bs
        assert "but" in bs
        assert "if" in bs

    def test_multiword_join_smash_convention(self):
        for join in multiword_joins():
            assert " " not in join["joined"]
            assert "_" not in join["joined"]

    def test_affixes_has_inflectional(self):
        a = affixes()
        assert "inflectional" in a
        assert "derivational_prefixes" in a
        assert "derivational_suffixes" in a

    def test_punctuation_has_entries(self):
        p = punctuation()
        assert "punctuation" in p

    def test_meta_has_version(self):
        m = meta()
        assert "version" in m

    def test_caching(self):
        # Calling twice returns the same list object (cached)
        assert bones() is bones()


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------

class TestMarkers:
    def test_family_keys(self):
        m = markers()
        assert set(m.keys()) == {"C", "R", "D", "N", "L", "O", "F", "E", "I"}

    def test_each_family_has_marker_dict(self):
        for letter, data in markers().items():
            assert isinstance(data, dict)
            assert "markers" in data
            assert len(data["markers"]) > 0, f"family {letter} has no marker categories"

    def test_family_accessor(self):
        for letter in ("C", "R", "D", "N", "L", "O", "F", "E", "I"):
            assert family(letter) == markers()[letter]

    def test_family_invalid(self):
        with pytest.raises(KeyError):
            family("Z")

    def test_marker_set_is_frozenset(self):
        assert isinstance(marker_set(), frozenset)

    def test_marker_set_nonempty(self):
        ms = marker_set()
        assert len(ms) > 0
        # Spot-check a known marker phrase
        assert "actually" in ms

    def test_meta_exists(self):
        assert markers_meta() is not None


# ---------------------------------------------------------------------------
# CanonLoader
# ---------------------------------------------------------------------------

class TestCanonLoader:
    def setup_method(self):
        self.canon = CanonLoader()

    def test_lookup_word_bone(self):
        entry = self.canon.lookup_word("not")
        assert entry is not None
        assert entry["primary"] == "P"

    def test_lookup_word_multiword(self):
        entry = self.canon.lookup_word("ofcourse")
        assert entry is not None
        assert entry.get("joined") == "ofcourse"

    def test_lookup_word_missing(self):
        assert self.canon.lookup_word("xyzzy") is None

    def test_lookup_affix(self):
        entry = self.canon.lookup_affix("un-")
        assert entry is not None
        assert entry["primary"] == "P"

    def test_lookup_punct_question(self):
        entry = self.canon.lookup_punct("?")
        assert entry is not None
        assert entry["primary"] == "Q"

    def test_metric_names(self):
        names = self.canon.metric_names()
        assert set(names) == {"C", "R", "D", "N", "L", "O", "F", "E", "I"}

    def test_metric_info(self):
        info = self.canon.metric_info("R")
        assert "metric" in info
        assert "markers" in info

    def test_metric_info_invalid(self):
        with pytest.raises(KeyError):
            self.canon.metric_info("Z")

    def test_marker_phrases(self):
        phrases = self.canon.marker_phrases("R", "refusal")
        assert isinstance(phrases, list)
        assert len(phrases) > 0

    def test_all_marker_phrases(self):
        phrases = self.canon.all_marker_phrases("R")
        assert len(phrases) > 0
        assert all(isinstance(p, str) for p in phrases)

    def test_meta_words(self):
        m = self.canon.meta("words")
        assert "version" in m

    def test_meta_invalid(self):
        with pytest.raises(KeyError):
            self.canon.meta("bogus")


# ---------------------------------------------------------------------------
# parse_transcript
# ---------------------------------------------------------------------------

class TestParseTranscript:
    def test_returns_parsed_transcript(self):
        result = parse_transcript([{"speaker": "A", "text": "not now"}])
        assert isinstance(result, ParsedTranscript)

    def test_list_format_basic(self):
        result = parse_transcript([
            {"speaker": "A", "text": "yes"},
            {"speaker": "B", "text": "no"},
        ])
        assert len(result.turns) == 2
        assert result.turns[0].speaker == "A"
        assert result.turns[1].speaker == "B"

    def test_string_format(self):
        result = parse_transcript("A: yes\nB: no")
        assert len(result.turns) == 2
        assert result.turns[0].speaker == "A"

    def test_string_no_speaker_fallback(self):
        result = parse_transcript("hello world")
        assert len(result.turns) == 1
        # upstream uses "SPEAKER" as anonymous fallback
        assert result.turns[0].speaker == "SPEAKER"

    def test_turn_has_tokens(self):
        result = parse_transcript([{"speaker": "A", "text": "not now"}])
        turn = result.turns[0]
        assert turn.token_count > 0
        assert all(isinstance(t, (BoneToken, FleshToken)) for t in turn.tokens)

    def test_bone_family_counts_type(self):
        result = parse_transcript([{"speaker": "A", "text": "x"}])
        from collections import Counter
        assert isinstance(result.turns[0].family_counts, Counter)

    def test_totals_matches_sum(self):
        result = parse_transcript([
            {"speaker": "A", "text": "I will not go."},
            {"speaker": "B", "text": "But if you must."},
        ])
        total = result.family_counts()
        for fam in ("P", "K", "Q", "T", "S"):
            expected = sum(t.family_counts.get(fam, 0) for t in result.turns)
            assert total.get(fam, 0) == expected

    def test_known_bone_tagged(self):
        result = parse_transcript([{"speaker": "A", "text": "not"}])
        assert any(
            t.surface.lower() == "not"
            for t in result.turns[0].bone_tokens
        )

    def test_not_is_P(self):
        result = parse_transcript([{"speaker": "A", "text": "not"}])
        not_tok = next(
            t for t in result.turns[0].bone_tokens
            if t.surface.lower() == "not"
        )
        assert not_tok.primary == "P"

    def test_multiword_join_applied(self):
        result = parse_transcript([{"speaker": "A", "text": "of course"}])
        mw_tokens = [
            t for t in result.turns[0].bone_tokens
            if t.bone_type == "multiword"
        ]
        assert len(mw_tokens) == 1
        assert mw_tokens[0].primary == "P"

    def test_raw_text_preserved(self):
        text = "of course I will not"
        result = parse_transcript([{"speaker": "A", "text": text}])
        assert result.turns[0].text == text

    def test_question_mark_is_Q(self):
        # "?" is in bones_words_v1 (bone_type="word") and bones_punct_v1 (bone_type="punct").
        # The word lookup fires first, so bone_type may be "word" or "punct" — either is fine.
        result = parse_transcript([{"speaker": "A", "text": "really?"}])
        q_bones = [t for t in result.turns[0].bone_tokens if t.primary == "Q"]
        assert len(q_bones) == 1

    def test_empty_text(self):
        result = parse_transcript([{"speaker": "A", "text": ""}])
        assert result.turns[0].tokens == []
        assert result.turns[0].bone_count == 0

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            parse_transcript(42)

    def test_rounds_cycle_strategy(self):
        result = parse_transcript("A: yes\nB: no\nA: ok")
        # A speaks, B speaks, A speaks again → new round at second A turn
        assert len(result.rounds) == 2

    def test_rounds_pairs_strategy(self):
        result = parse_transcript(
            [
                {"speaker": "A", "text": "yes"},
                {"speaker": "B", "text": "no"},
                {"speaker": "A", "text": "ok"},
                {"speaker": "B", "text": "sure"},
            ],
            round_strategy="pairs",
        )
        assert len(result.rounds) == 2

    def test_speakers_list(self):
        result = parse_transcript([
            {"speaker": "A", "text": "yes"},
            {"speaker": "B", "text": "no"},
        ])
        assert "A" in result.speakers
        assert "B" in result.speakers

    def test_bone_count_method(self):
        result = parse_transcript([{"speaker": "A", "text": "not now"}])
        # "not" → P, "now" → T
        assert result.bone_count() == 2

    def test_affix_prefix_classified(self):
        # "unhappy" — "un-" prefix should be classified as an affix bone (P)
        result = parse_transcript([{"speaker": "A", "text": "unhappy"}])
        affix_bones = [
            t for t in result.turns[0].bone_tokens
            if t.bone_type == "affix"
        ]
        assert len(affix_bones) >= 1
        assert any(t.primary == "P" for t in affix_bones)

    def test_flesh_tokens_unclassified(self):
        result = parse_transcript([{"speaker": "A", "text": "xyzzy"}])
        flesh = result.turns[0].flesh_tokens
        assert len(flesh) == 1
        assert flesh[0].surface == "xyzzy"
