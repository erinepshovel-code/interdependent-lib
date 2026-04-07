"""Tests for interdependent_lib.edcm."""

import pytest

from interdependent_lib.edcm.bones import (
    bones, multiword_joins, bone_set, words_by_family,
    affixes, punctuation, meta,
)
from interdependent_lib.edcm.markers import markers, family, marker_set, meta as markers_meta
from interdependent_lib.edcm.parser.turns_rounds import parse_transcript


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
        # Core bones that must be present
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

    def test_each_family_has_entries(self):
        for letter, entries in markers().items():
            assert len(entries) > 0, f"family {letter} is empty"

    def test_family_accessor(self):
        for letter in ("C", "R", "D", "N", "L", "O", "F", "E", "I"):
            assert family(letter) == markers()[letter]

    def test_family_invalid(self):
        with pytest.raises(KeyError):
            family("Z")

    def test_marker_set_is_frozenset(self):
        assert isinstance(marker_set(), frozenset)

    def test_marker_set_nonempty(self):
        assert len(marker_set()) > 0

    def test_meta_exists(self):
        assert markers_meta() is not None


# ---------------------------------------------------------------------------
# parse_transcript
# ---------------------------------------------------------------------------

class TestParseTranscript:
    def test_list_format_basic(self):
        result = parse_transcript([
            {"speaker": "A", "text": "yes"},
            {"speaker": "B", "text": "no"},
        ])
        assert len(result["turns"]) == 2
        assert result["turns"][0]["speaker"] == "A"
        assert result["turns"][1]["speaker"] == "B"

    def test_string_format(self):
        result = parse_transcript("A: yes\nB: no")
        assert len(result["turns"]) == 2
        assert result["turns"][0]["speaker"] == "A"

    def test_string_no_speaker(self):
        result = parse_transcript("hello world")
        assert len(result["turns"]) == 1
        assert result["turns"][0]["speaker"] == ""

    def test_output_keys(self):
        result = parse_transcript([{"speaker": "A", "text": "not now"}])
        turn = result["turns"][0]
        assert "raw_text" in turn
        assert "normalized_text" in turn
        assert "tokens" in turn
        assert "tagged" in turn
        assert "bone_counts" in turn
        assert "bone_tokens" in turn

    def test_bone_counts_families(self):
        result = parse_transcript([{"speaker": "A", "text": "x"}])
        counts = result["turns"][0]["bone_counts"]
        assert set(counts.keys()) == {"P", "K", "Q", "T", "S"}

    def test_totals_matches_sum(self):
        result = parse_transcript([
            {"speaker": "A", "text": "I will not go."},
            {"speaker": "B", "text": "But if you must."},
        ])
        for fam in ("P", "K", "Q", "T", "S"):
            expected = sum(t["bone_counts"][fam] for t in result["turns"])
            assert result["totals"][fam] == expected

    def test_known_bone_tagged(self):
        result = parse_transcript([{"speaker": "A", "text": "not"}])
        tagged = result["turns"][0]["tagged"]
        bone_tokens = [t for t in tagged if t["bone"]]
        assert any(t["token"].lower() == "not" for t in bone_tokens)

    def test_not_is_P(self):
        result = parse_transcript([{"speaker": "A", "text": "not"}])
        tagged = result["turns"][0]["tagged"]
        not_tag = next(t for t in tagged if t["token"].lower() == "not")
        assert not_tag["bone"] is True
        assert not_tag["primary"] == "P"

    def test_multiword_join_applied(self):
        result = parse_transcript([{"speaker": "A", "text": "of course"}])
        turn = result["turns"][0]
        assert "ofcourse" in turn["normalized_text"]
        assert len(result["join_log"]) == 1
        assert result["join_log"][0]["original"] == "of course"

    def test_join_log_has_turn_index(self):
        result = parse_transcript([
            {"speaker": "A", "text": "hello"},
            {"speaker": "B", "text": "of course"},
        ])
        assert result["join_log"][0]["turn"] == 1

    def test_raw_text_preserved(self):
        text = "of course I will not"
        result = parse_transcript([{"speaker": "A", "text": text}])
        assert result["turns"][0]["raw_text"] == text

    def test_punctuation_split(self):
        result = parse_transcript([{"speaker": "A", "text": "No."}])
        tokens = result["turns"][0]["tokens"]
        assert "No" in tokens or "no" in tokens
        assert "." in tokens

    def test_empty_text(self):
        result = parse_transcript([{"speaker": "A", "text": ""}])
        assert result["turns"][0]["tokens"] == []
        assert all(v == 0 for v in result["turns"][0]["bone_counts"].values())

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            parse_transcript(42)

    def test_join_log_empty_when_no_joins(self):
        result = parse_transcript([{"speaker": "A", "text": "hello world"}])
        assert result["join_log"] == []
