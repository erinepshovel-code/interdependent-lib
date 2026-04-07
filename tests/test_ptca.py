"""Tests for interdependent_lib.ptca."""

import pytest

from interdependent_lib.ptca.constants import (
    NODES, SENTINELS, PHASES, SLOTS,
    ALPHA, BETA, GAMMA, DELTA,
    SENTINEL_NAMES,
)
from interdependent_lib.ptca.primes import (
    PRIME_NODES, PRIME_TO_NODE,
    node_for_prime, prime_for_node, is_prime_node,
)
from interdependent_lib.ptca.tensor import PTCATensor
from interdependent_lib.ptca.sentinels import SentinelState
from interdependent_lib.ptca.provenance import (
    build_block, hash_block, verify_chain, extend_chain, chain_hashes,
)
from interdependent_lib.ptca.exchange import compute_score, Exchange, ExchangeResult
from interdependent_lib.ptca.instance import PTCAInstance


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_dimensions(self):
        assert NODES == 53
        assert SENTINELS == 9
        assert PHASES == 8
        assert SLOTS == 7

    def test_sentinel_names_count(self):
        assert len(SENTINEL_NAMES) == 9

    def test_sentinel_names_format(self):
        for name in SENTINEL_NAMES:
            assert name.startswith("S")


# ---------------------------------------------------------------------------
# Primes
# ---------------------------------------------------------------------------

class TestPrimes:
    def test_count(self):
        assert len(PRIME_NODES) == 53

    def test_first_and_last(self):
        assert PRIME_NODES[0] == 2
        assert PRIME_NODES[-1] == 241

    def test_reverse_lookup(self):
        for i, p in enumerate(PRIME_NODES):
            assert PRIME_TO_NODE[p] == i

    def test_node_for_prime(self):
        assert node_for_prime(2) == 0
        assert node_for_prime(241) == 52

    def test_node_for_prime_invalid(self):
        with pytest.raises(KeyError):
            node_for_prime(4)

    def test_prime_for_node(self):
        assert prime_for_node(0) == 2
        assert prime_for_node(52) == 241

    def test_prime_for_node_out_of_range(self):
        with pytest.raises(IndexError):
            prime_for_node(53)

    def test_is_prime_node(self):
        assert is_prime_node(2)
        assert is_prime_node(241)
        assert not is_prime_node(4)
        assert not is_prime_node(0)


# ---------------------------------------------------------------------------
# PTCATensor
# ---------------------------------------------------------------------------

class TestPTCATensor:
    def test_shape_and_size(self):
        t = PTCATensor()
        assert t.SHAPE == (53, 9, 8, 7)
        assert t.SIZE == 53 * 9 * 8 * 7
        assert len(t) == t.SIZE

    def test_initial_zeros(self):
        t = PTCATensor()
        assert t.get(0, 0, 0, 0) == 0.0
        assert t.get(52, 8, 7, 6) == 0.0

    def test_set_and_get(self):
        t = PTCATensor()
        t.set(1, 2, 3, 4, 9.5)
        assert t.get(1, 2, 3, 4) == 9.5

    def test_add(self):
        t = PTCATensor()
        t.add(0, 0, 0, 0, 1.0)
        t.add(0, 0, 0, 0, 2.5)
        assert t.get(0, 0, 0, 0) == pytest.approx(3.5)

    def test_out_of_range(self):
        t = PTCATensor()
        with pytest.raises(IndexError):
            t.get(53, 0, 0, 0)
        with pytest.raises(IndexError):
            t.get(0, 9, 0, 0)
        with pytest.raises(IndexError):
            t.get(0, 0, 8, 0)
        with pytest.raises(IndexError):
            t.get(0, 0, 0, 7)

    def test_reset(self):
        t = PTCATensor()
        t.set(5, 3, 2, 1, 42.0)
        t.reset()
        assert t.get(5, 3, 2, 1) == 0.0

    def test_reset_node(self):
        t = PTCATensor()
        t.set(2, 0, 0, 0, 1.0)
        t.set(3, 0, 0, 0, 2.0)
        t.reset_node(2)
        assert t.get(2, 0, 0, 0) == 0.0
        assert t.get(3, 0, 0, 0) == 2.0

    def test_node_slice_length(self):
        t = PTCATensor()
        assert len(t.node_slice(0)) == SENTINELS * PHASES * SLOTS

    def test_aggregate_sum(self):
        t = PTCATensor()
        t.set(0, 0, 0, 0, 1.0)
        t.set(0, 0, 0, 1, 2.0)
        total = t.aggregate("sum", node=0, sentinel=0, phase=0)
        assert total == pytest.approx(3.0)

    def test_aggregate_mean(self):
        t = PTCATensor()
        t.set(0, 0, 0, 0, 4.0)
        t.set(0, 0, 0, 1, 8.0)
        mean = t.aggregate("mean", node=0, sentinel=0, phase=0)
        # 7 slots, only 2 non-zero
        assert mean == pytest.approx(12.0 / 7)

    def test_aggregate_invalid_method(self):
        t = PTCATensor()
        with pytest.raises(ValueError):
            t.aggregate("median")

    def test_repr(self):
        t = PTCATensor()
        assert "PTCATensor" in repr(t)


# ---------------------------------------------------------------------------
# SentinelState
# ---------------------------------------------------------------------------

class TestSentinelState:
    def test_default_construction(self):
        s = SentinelState()
        assert s.s4.approved is False
        assert s.s8.score == 0.0
        assert s.s5.entries == []

    def test_channel_access(self):
        s = SentinelState()
        assert s.channel("S1_PROVENANCE") is s.s1
        assert s.channel("S9_AUDIT") is s.s9

    def test_channel_invalid(self):
        s = SentinelState()
        with pytest.raises(KeyError):
            s.channel("S99_FAKE")

    def test_s4_approve_revoke(self):
        s = SentinelState()
        s.s4.approve("test")
        assert s.s4.approved is True
        s.s4.revoke("nope")
        assert s.s4.approved is False

    def test_s8_risk_clamp(self):
        s = SentinelState()
        s.s8.update(2.0)
        assert s.s8.score == pytest.approx(1.0)
        s.s8.update(-5.0)
        assert s.s8.score == pytest.approx(0.0)

    def test_s5_max_entries(self):
        s = SentinelState()
        s.s5.max_entries = 3
        for i in range(5):
            s.s5.push({"i": i})
        assert len(s.s5.entries) == 3
        assert s.s5.entries[0]["i"] == 2

    def test_s9_audit_record_and_tail(self):
        s = SentinelState()
        s.s9.record("event_a")
        s.s9.record("event_b")
        tail = s.s9.tail(1)
        assert len(tail) == 1
        assert tail[0]["event"] == "event_b"

    def test_to_dict_keys(self):
        s = SentinelState()
        d = s.to_dict()
        assert set(d.keys()) == set(SENTINEL_NAMES)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_build_block_fields(self):
        b = build_block(model_id="m", caller_id="c", session_id="s")
        assert b["model_id"] == "m"
        assert b["parent_hash"] == ""
        assert isinstance(b["ts"], float)

    def test_hash_block_is_hex64(self):
        b = build_block()
        h = hash_block(b)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_block_deterministic(self):
        b = build_block(model_id="x", timestamp=1234.0, payload={"k": 1})
        assert hash_block(b) == hash_block(b)

    def test_verify_chain_empty(self):
        assert verify_chain([]) is True

    def test_verify_chain_valid(self):
        blocks = [build_block(timestamp=1.0)]
        extend_chain(blocks, timestamp=2.0)
        extend_chain(blocks, timestamp=3.0)
        assert verify_chain(blocks) is True

    def test_verify_chain_tampered(self):
        blocks = [build_block()]
        extend_chain(blocks)
        # Tamper with the first block; its hash will no longer match
        # blocks[1]["parent_hash"]
        blocks[0]["model_id"] = "tampered"
        assert verify_chain(blocks) is False

    def test_chain_hashes_length(self):
        blocks = [build_block()]
        extend_chain(blocks)
        assert len(chain_hashes(blocks)) == 2


# ---------------------------------------------------------------------------
# Exchange / compute_score
# ---------------------------------------------------------------------------

class TestComputeScore:
    def test_zero_inputs(self):
        score, components = compute_score()
        assert score == 0.0
        assert components["bonus"] == 0.0

    def test_s1_only(self):
        score, components = compute_score(s1=1.0)
        assert score == pytest.approx(DELTA * ALPHA)
        assert components["s1"] == pytest.approx(ALPHA)

    def test_all_channels(self):
        score, _ = compute_score(s1=1.0, s2=1.0, s3=1.0, s5=1.0, s8=1.0)
        expected = DELTA * (ALPHA + BETA + GAMMA + GAMMA + ALPHA)
        assert score == pytest.approx(expected)

    def test_bonus(self):
        score, _ = compute_score(bonus=5.0)
        assert score == pytest.approx(5.0)


class TestExchange:
    def test_route_returns_result(self):
        t = PTCATensor()
        s = SentinelState()
        exc = Exchange(t, s)
        result = exc.route(node=0, phase=0, slot=0, s1=1.0)
        assert isinstance(result, ExchangeResult)
        assert result.node == 0
        assert result.score == pytest.approx(DELTA * ALPHA)

    def test_route_writes_tensor(self):
        t = PTCATensor()
        s = SentinelState()
        exc = Exchange(t, s)
        exc.route(node=1, phase=2, slot=3, sentinel_idx=0, s1=1.0)
        assert t.get(1, 0, 2, 3) == pytest.approx(DELTA * ALPHA)

    def test_route_records_audit(self):
        t = PTCATensor()
        s = SentinelState()
        exc = Exchange(t, s)
        exc.route(node=0, phase=0, slot=0, audit_event="test_event")
        assert s.s9.log[-1]["event"] == "test_event"

    def test_batch_route(self):
        t = PTCATensor()
        s = SentinelState()
        exc = Exchange(t, s)
        results = exc.batch_route([
            {"node": 0, "phase": 0, "slot": 0},
            {"node": 1, "phase": 1, "slot": 1},
        ])
        assert len(results) == 2


# ---------------------------------------------------------------------------
# PTCAInstance
# ---------------------------------------------------------------------------

class TestPTCAInstance:
    def test_init_defaults(self):
        inst = PTCAInstance()
        assert inst.session_id != ""
        assert inst.risk_score == 0.0
        assert inst.approved is False

    def test_init_with_params(self):
        inst = PTCAInstance(model_id="m", caller_id="c", approved=True)
        assert inst.model_id == "m"
        assert inst.caller_id == "c"
        assert inst.approved is True

    def test_provenance_genesis(self):
        inst = PTCAInstance()
        assert len(inst.provenance_chain) == 1
        assert inst.sentinel_state.s1.origin_hash != ""

    def test_record_provenance(self):
        inst = PTCAInstance()
        block = inst.record_provenance(payload={"x": 1})
        assert block["payload"]["x"] == 1
        assert len(inst.provenance_chain) == 2

    def test_push_context(self):
        inst = PTCAInstance()
        inst.push_context({"role": "user", "tokens": 10})
        assert len(inst.context_entries) == 1

    def test_approve_revoke(self):
        inst = PTCAInstance()
        inst.approve("ok")
        assert inst.approved is True
        inst.revoke("no")
        assert inst.approved is False

    def test_risk_update(self):
        inst = PTCAInstance()
        inst.update_risk(0.3, factor="test")
        assert inst.risk_score == pytest.approx(0.3)
        inst.reset_risk()
        assert inst.risk_score == 0.0

    def test_route(self):
        inst = PTCAInstance()
        result = inst.route(node=0, phase=0, slot=0, s1=1.0)
        assert isinstance(result, ExchangeResult)

    def test_memory(self):
        inst = PTCAInstance()
        inst.remember("key", "value")
        assert inst.recall("key") == "value"
        assert inst.recall("missing", "default") == "default"

    def test_snapshot_keys(self):
        inst = PTCAInstance()
        snap = inst.snapshot()
        assert "session_id" in snap
        assert "S5_CONTEXT" in snap
        assert "S9_AUDIT" in snap

    def test_repr(self):
        inst = PTCAInstance(model_id="m")
        assert "PTCAInstance" in repr(inst)
        assert "m" in repr(inst)
