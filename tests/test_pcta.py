"""Tests for interdependent_lib.pcta."""

import pytest

from interdependent_lib.pcna.network import PCNANetwork
from interdependent_lib.pcta.circle import CIRCLE_SIZE, PCTACircle


def _make_circle(in_size=2, hidden=4, out_size=1, circle_id=0) -> PCTACircle:
    members = [PCNANetwork([in_size, hidden, out_size]) for _ in range(CIRCLE_SIZE)]
    return PCTACircle(members, circle_id=circle_id)


class TestPCTACircle:
    def test_circle_size_constant(self):
        assert CIRCLE_SIZE == 7

    def test_construction(self):
        c = _make_circle()
        assert len(c.members) == CIRCLE_SIZE

    def test_wrong_member_count_raises(self):
        members = [PCNANetwork([2, 2]) for _ in range(6)]
        with pytest.raises(ValueError, match="exactly 7"):
            PCTACircle(members)

    def test_len(self):
        c = _make_circle()
        assert len(c) == CIRCLE_SIZE

    def test_as_tensor_length(self):
        c = _make_circle(in_size=2, hidden=3, out_size=1)
        t = c.as_tensor()
        assert len(t) == c.tensor_size

    def test_tensor_size_matches_member_params(self):
        c = _make_circle()
        expected = sum(net.parameter_count for net in c.members)
        assert c.tensor_size == expected

    def test_forward_output_shape(self):
        c = _make_circle(in_size=3, out_size=2)
        inputs = [[0.1, 0.2, 0.3]] * CIRCLE_SIZE
        outputs = c.forward(inputs)
        assert len(outputs) == CIRCLE_SIZE
        assert all(len(o) == 2 for o in outputs)

    def test_forward_wrong_input_count(self):
        c = _make_circle()
        with pytest.raises(ValueError):
            c.forward([[0.1, 0.2]] * 5)

    def test_audit_returns_dict(self):
        c = _make_circle()
        result = c.audit()
        assert isinstance(result, dict)

    def test_audit_keys(self):
        c = _make_circle()
        result = c.audit()
        assert "weight_norms" in result
        assert "weight_mean" in result
        assert "weight_spread" in result
        assert "param_counts" in result
        assert "total_params" in result
        assert "circle_id" in result

    def test_audit_norm_count(self):
        c = _make_circle()
        result = c.audit()
        assert len(result["weight_norms"]) == CIRCLE_SIZE

    def test_audit_norms_nonnegative(self):
        c = _make_circle()
        result = c.audit()
        assert all(n >= 0.0 for n in result["weight_norms"])

    def test_audit_spread_nonnegative(self):
        c = _make_circle()
        result = c.audit()
        assert result["weight_spread"] >= 0.0

    def test_audit_cached(self):
        c = _make_circle()
        result = c.audit()
        assert c._last_audit is result

    def test_audit_circle_id(self):
        c = _make_circle(circle_id=3)
        result = c.audit()
        assert result["circle_id"] == 3

    def test_audit_total_params(self):
        c = _make_circle()
        result = c.audit()
        assert result["total_params"] == c.tensor_size

    def test_repr(self):
        c = _make_circle(circle_id=1)
        assert "PCTACircle" in repr(c)
        assert "1" in repr(c)
