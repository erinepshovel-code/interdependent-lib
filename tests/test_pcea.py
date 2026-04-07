"""Tests for interdependent_lib.pcea (stdlib-only modules only).

Modules that depend on `cryptography` (aead, kdf, guardian, wrap, rekey)
are tested only if the package is importable; they are skipped otherwise.
"""

import hashlib
import struct

import pytest

# Import directly from submodules to avoid triggering pcea/__init__.py,
# which eagerly imports aead (cryptography-dependent).
import importlib
_codec     = importlib.import_module("interdependent_lib.pcea.codec")
_commit    = importlib.import_module("interdependent_lib.pcea.commitment")
_threshold = importlib.import_module("interdependent_lib.pcea.threshold")
_wipe_mod  = importlib.import_module("interdependent_lib.pcea.wipe")
_types     = importlib.import_module("interdependent_lib.pcea.types")
_validate  = importlib.import_module("interdependent_lib.pcea.validate")

encode_aad         = _codec.encode_aad
encode_wrap_aad    = _codec.encode_wrap_aad
encode_nonce_input = _codec.encode_nonce_input
encode_key_info    = _codec.encode_key_info

make_commitment    = _commit.make_commitment
verify_commitment  = _commit.verify_commitment

split_secret       = _threshold.split_secret
reconstruct_secret = _threshold.reconstruct_secret

wipe               = _wipe_mod.wipe
wipe_bytearray     = _wipe_mod.wipe_bytearray
wipe_bytes         = _wipe_mod.wipe_bytes
_bytes_data_offset = _wipe_mod._bytes_data_offset

LiveState          = _types.LiveState
SealedState        = _types.SealedState
WrappedLiveKey     = _types.WrappedLiveKey
MetaShares         = _types.MetaShares
UnsealGrant        = _types.UnsealGrant
RekeyEpoch         = _types.RekeyEpoch

validate_invariant = _validate.validate_invariant
InvariantViolation = _validate.InvariantViolation


# ---------------------------------------------------------------------------
# codec
# ---------------------------------------------------------------------------

class TestCodec:
    def test_encode_aad_deterministic(self):
        a = encode_aad(1, "key-id", "guardian-0")
        b = encode_aad(1, "key-id", "guardian-0")
        assert a == b

    def test_encode_aad_structure(self):
        epoch = 42
        key_id = "k1"
        sealed_by = "g0"
        out = encode_aad(epoch, key_id, sealed_by)
        # First 8 bytes: epoch as uint64 big-endian
        assert struct.unpack(">Q", out[:8])[0] == epoch

    def test_encode_aad_differs_on_epoch(self):
        assert encode_aad(1, "k", "g") != encode_aad(2, "k", "g")

    def test_encode_wrap_aad_deterministic(self):
        assert encode_wrap_aad(5, "key") == encode_wrap_aad(5, "key")

    def test_encode_wrap_aad_differs(self):
        assert encode_wrap_aad(1, "k1") != encode_wrap_aad(1, "k2")

    def test_encode_nonce_input_prefix(self):
        out = encode_nonce_input(1, "k", 0, "g")
        assert out[:6] == b"nonce:"

    def test_encode_nonce_input_deterministic(self):
        a = encode_nonce_input(1, "k", 0, "g")
        b = encode_nonce_input(1, "k", 0, "g")
        assert a == b

    def test_encode_nonce_input_unique_per_counter(self):
        a = encode_nonce_input(1, "k", 0, "g")
        b = encode_nonce_input(1, "k", 1, "g")
        assert a != b

    def test_encode_key_info_deterministic(self):
        a = encode_key_info("guardian:live", 1, "k1")
        b = encode_key_info("guardian:live", 1, "k1")
        assert a == b

    def test_encode_key_info_label_differs(self):
        live = encode_key_info("guardian:live", 1, "k1")
        meta = encode_key_info("guardian:meta", 1, "k1")
        assert live != meta


# ---------------------------------------------------------------------------
# commitment
# ---------------------------------------------------------------------------

def _make_shares(n=3):
    return [
        {"sentinel_id": f"s{i}", "share": bytes([i] * 8), "index": i + 1}
        for i in range(n)
    ]


class TestCommitment:
    def test_make_commitment_returns_hex64(self):
        c = make_commitment(_make_shares())
        assert len(c) == 64
        assert all(ch in "0123456789abcdef" for ch in c)

    def test_make_commitment_deterministic(self):
        shares = _make_shares()
        assert make_commitment(shares) == make_commitment(shares)

    def test_make_commitment_order_independent(self):
        shares = _make_shares()
        shuffled = list(reversed(shares))
        assert make_commitment(shares) == make_commitment(shuffled)

    def test_make_commitment_sensitive_to_content(self):
        shares = _make_shares()
        modified = [dict(s) for s in shares]
        modified[0] = dict(modified[0], share=bytes([99] * 8))
        assert make_commitment(shares) != make_commitment(modified)

    def test_verify_commitment_pass(self):
        shares = _make_shares()
        c = make_commitment(shares)
        assert verify_commitment(shares, c) is True

    def test_verify_commitment_fail(self):
        shares = _make_shares()
        c = make_commitment(shares)
        bad = [dict(s) for s in shares]
        bad[0] = dict(bad[0], share=b"\xff" * 8)
        assert verify_commitment(bad, c) is False


# ---------------------------------------------------------------------------
# threshold (GF-256 Shamir)
# ---------------------------------------------------------------------------

class TestShamir:
    def test_split_and_reconstruct_2of3(self):
        secret = b"hello-world-key!"
        shares = split_secret(secret, threshold=2, n=3)
        assert len(shares) == 3
        recovered = reconstruct_secret(shares[:2])
        assert recovered == secret

    def test_split_and_reconstruct_3of5(self):
        secret = bytes(range(32))
        shares = split_secret(secret, threshold=3, n=5)
        recovered = reconstruct_secret(shares[1:4])
        assert recovered == secret

    def test_any_threshold_subset_works(self):
        secret = b"test-secret"
        shares = split_secret(secret, threshold=3, n=5)
        for combo in [shares[0:3], shares[1:4], shares[2:5]]:
            assert reconstruct_secret(combo) == secret

    def test_share_indices_are_1_based(self):
        shares = split_secret(b"abc", threshold=2, n=3)
        indices = [idx for idx, _ in shares]
        assert indices == [1, 2, 3]

    def test_share_length_matches_secret(self):
        secret = b"sixteen-byte-key"
        shares = split_secret(secret, threshold=2, n=3)
        for _, share_bytes in shares:
            assert len(share_bytes) == len(secret)

    def test_invalid_threshold_lt_2(self):
        with pytest.raises(ValueError):
            split_secret(b"x", threshold=1, n=3)

    def test_invalid_threshold_gt_n(self):
        with pytest.raises(ValueError):
            split_secret(b"x", threshold=4, n=3)

    def test_empty_secret_raises(self):
        with pytest.raises(ValueError):
            split_secret(b"", threshold=2, n=3)

    def test_reconstruct_mismatched_lengths_raises(self):
        with pytest.raises(ValueError):
            reconstruct_secret([(1, b"ab"), (2, b"abc")])

    def test_reconstruct_empty_raises(self):
        with pytest.raises(ValueError):
            reconstruct_secret([])

    def test_shares_look_random(self):
        # Different secrets should produce different shares
        s1 = split_secret(b"secret-one", threshold=2, n=3)
        s2 = split_secret(b"secret-two", threshold=2, n=3)
        assert s1[0][1] != s2[0][1]


# ---------------------------------------------------------------------------
# wipe
# ---------------------------------------------------------------------------

class TestWipe:
    def test_bytes_data_offset_64bit(self):
        import sys
        offset = _bytes_data_offset()
        if sys.maxsize > 2**32:
            assert offset == 32
        else:
            assert offset == 16

    def test_wipe_bytearray(self):
        buf = bytearray(b"sensitive-data")
        wipe_bytearray(buf)
        assert all(b == 0 for b in buf)

    def test_wipe_dispatch_bytearray(self):
        buf = bytearray(b"secret")
        wipe(buf)
        assert all(b == 0 for b in buf)

    def test_wipe_bytes_best_effort(self):
        # Should not raise regardless of outcome
        b = b"ephemeral"
        wipe_bytes(b)  # best-effort; no assertion on content

    def test_wipe_wrong_type(self):
        with pytest.raises(TypeError):
            wipe("not bytes")  # type: ignore


# ---------------------------------------------------------------------------
# types (instantiation smoke tests)
# ---------------------------------------------------------------------------

class TestTypes:
    def test_live_state(self):
        ls = LiveState(
            epoch=1, spiral={}, cores={}, density_matrix=None,
            coherence=1.0, transport=None, last_renorm=0.0,
        )
        assert ls.epoch == 1

    def test_sealed_state(self):
        ss = SealedState(
            epoch=1, key_id="k", ciphertext=b"ct", nonce=b"n" * 12,
            aad=b"aad", sealed_by="g",
        )
        assert ss.epoch == 1

    def test_wrapped_live_key(self):
        w = WrappedLiveKey(
            key_id="k", epoch=1, wrapped_live_key=b"wlk",
            wrap_key_hash="abc",
        )
        assert w.key_id == "k"

    def test_meta_shares(self):
        ms = MetaShares(epoch=1, total_shares=3, threshold=2, shares=[], commitment="c")
        assert ms.threshold == 2

    def test_rekey_epoch(self):
        re = RekeyEpoch(
            from_epoch=1, to_epoch=2,
            new_session_secret_commitment="c",
            meta_shares_updated=True,
            renorm_confirmed=False,
            spectral_snapshot=[],
        )
        assert re.to_epoch == 2


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------

def _make_valid_objects():
    wrapped = WrappedLiveKey(
        key_id="k1", epoch=1,
        wrapped_live_key=b"x" * 32,
        wrap_key_hash="abc",
    )
    sealed = SealedState(
        epoch=1, key_id="k1", ciphertext=b"ct" * 8,
        nonce=b"n" * 12, aad=b"aad", sealed_by="g",
    )
    shares = [
        {"sentinel_id": f"s{i}", "share": b"x" * 8, "index": i + 1}
        for i in range(3)
    ]
    meta = MetaShares(
        epoch=1, total_shares=3, threshold=2,
        shares=shares, commitment="c",
    )
    return wrapped, sealed, meta


class TestValidate:
    def test_valid_passes(self):
        w, s, m = _make_valid_objects()
        validate_invariant(w, s, m)  # should not raise

    def test_epoch_mismatch_wrapped_sealed(self):
        w, s, m = _make_valid_objects()
        w2 = WrappedLiveKey(key_id=w.key_id, epoch=99,
                            wrapped_live_key=w.wrapped_live_key,
                            wrap_key_hash=w.wrap_key_hash)
        with pytest.raises(InvariantViolation, match="epoch"):
            validate_invariant(w2, s, m)

    def test_key_id_mismatch(self):
        w, s, m = _make_valid_objects()
        s2 = SealedState(epoch=1, key_id="different", ciphertext=s.ciphertext,
                         nonce=s.nonce, aad=s.aad, sealed_by=s.sealed_by)
        with pytest.raises(InvariantViolation, match="key_id"):
            validate_invariant(w, s2, m)

    def test_threshold_too_low(self):
        w, s, m = _make_valid_objects()
        m2 = MetaShares(epoch=1, total_shares=3, threshold=1,
                        shares=m.shares, commitment=m.commitment)
        with pytest.raises(InvariantViolation, match="threshold"):
            validate_invariant(w, s, m2)

    def test_threshold_exceeds_total(self):
        w, s, m = _make_valid_objects()
        m2 = MetaShares(epoch=1, total_shares=3, threshold=4,
                        shares=m.shares, commitment=m.commitment)
        with pytest.raises(InvariantViolation):
            validate_invariant(w, s, m2)

    def test_wrong_nonce_length(self):
        w, s, m = _make_valid_objects()
        s2 = SealedState(epoch=1, key_id="k1", ciphertext=s.ciphertext,
                         nonce=b"short", aad=s.aad, sealed_by=s.sealed_by)
        with pytest.raises(InvariantViolation, match="nonce"):
            validate_invariant(w, s2, m)

    def test_duplicate_share_indices(self):
        w, s, m = _make_valid_objects()
        bad_shares = [{"sentinel_id": "s0", "share": b"x" * 8, "index": 1},
                      {"sentinel_id": "s1", "share": b"y" * 8, "index": 1},
                      {"sentinel_id": "s2", "share": b"z" * 8, "index": 2}]
        m2 = MetaShares(epoch=1, total_shares=3, threshold=2,
                        shares=bad_shares, commitment="c")
        with pytest.raises(InvariantViolation, match="unique"):
            validate_invariant(w, s, m2)
