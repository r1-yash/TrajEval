"""Tests for Step 3: result types (Verdict, Evidence, GraderResult).

Each test proves exactly one behavior from the step plan.
"""

from dataclasses import FrozenInstanceError

import pytest

from trajecteval.results import Dimension, Evidence, GraderResult, Verdict

PRICE_EVIDENCE = Evidence(
    kind="price_exceeds_limit",
    step=7,
    field="itinerary.total_price",
    value=1400,
)


# ---------------------------------------------------------------- 1. verdicts


def test_every_verdict_constructs():
    # Evidence: optional for PASS/ERROR, required for FAIL/WARN.
    ok = GraderResult(dimension="final_state", verdict=Verdict.PASS, reason="state matches spec")
    err = GraderResult(dimension="bounds", verdict=Verdict.ERROR, reason="spec field missing")
    fail = GraderResult(
        dimension="final_state", verdict=Verdict.FAIL, reason="too expensive",
        evidence=(PRICE_EVIDENCE,),
    )
    warn = GraderResult(
        dimension="bounds", verdict=Verdict.WARN, reason="nothing checked",
        evidence=(Evidence(kind="empty_trajectory", step=None),),
    )
    assert [r.verdict for r in (ok, err, fail, warn)] == [
        Verdict.PASS, Verdict.ERROR, Verdict.FAIL, Verdict.WARN,
    ]


# ------------------------------------------------------------ 2. bad verdicts


def test_bad_verdict_rejected():
    with pytest.raises(ValueError, match="invalid verdict 'MAYBE'"):
        GraderResult(dimension="bounds", verdict="MAYBE", reason="x")


def test_wrong_case_verdict_rejected():
    # Verdict is a closed set: 'fail' is not a spelling of FAIL.
    with pytest.raises(ValueError, match="invalid verdict 'fail'"):
        GraderResult(dimension="bounds", verdict="fail", reason="x")


def test_plain_string_verdict_coerces_to_member():
    # "FAIL" (correct spelling) is accepted and becomes Verdict.FAIL.
    result = GraderResult(dimension="bounds", verdict="FAIL", reason="x",
                           evidence=(PRICE_EVIDENCE,))
    assert result.verdict is Verdict.FAIL


# ----------------------------------------------------------- 3. missing reason


def test_empty_reason_rejected_for_every_verdict():
    with pytest.raises(ValueError, match="result 'reason' must be a non-empty string"):
        GraderResult(dimension="bounds", verdict=Verdict.FAIL, reason="",
                     evidence=(PRICE_EVIDENCE,))
    with pytest.raises(ValueError, match="result 'reason' must be a non-empty string"):
        GraderResult(dimension="bounds", verdict=Verdict.PASS, reason="")


def test_empty_dimension_rejected():
    with pytest.raises(ValueError, match="result 'dimension' must be a non-empty string"):
        GraderResult(dimension="", verdict=Verdict.PASS, reason="x")


def test_bad_dimension_rejected():
    # "bound" must fail here, not become a fifth column in the comparison table.
    with pytest.raises(ValueError, match="invalid dimension 'bound'"):
        GraderResult(dimension="bound", verdict=Verdict.PASS, reason="x")


def test_plain_string_dimension_coerces_to_member():
    result = GraderResult(dimension="bounds", verdict=Verdict.PASS, reason="x")
    assert result.dimension is Dimension.BOUNDS


def test_to_dict_dimension_is_plain_string():
    result = GraderResult(dimension="bounds", verdict=Verdict.PASS, reason="x")
    assert result.to_dict()["dimension"] == "bounds"


# --------------------------------------------------- 4. evidence step or none


def test_evidence_with_step():
    assert PRICE_EVIDENCE.step == 7
    assert PRICE_EVIDENCE.field == "itinerary.total_price"
    assert PRICE_EVIDENCE.value == 1400


def test_evidence_without_step_is_whole_trajectory():
    ev = Evidence(kind="empty_trajectory")
    assert ev.step is None  # meaningfully whole-trajectory, not missing


# ----------------------------------------------------------- 5. immutability


def test_result_is_immutable():
    result = GraderResult(dimension="bounds", verdict=Verdict.PASS, reason="x")
    with pytest.raises(FrozenInstanceError):
        result.verdict = Verdict.FAIL


def test_evidence_is_immutable():
    with pytest.raises(FrozenInstanceError):
        PRICE_EVIDENCE.step = 99


def test_evidence_stored_as_tuple_even_when_given_a_list():
    result = GraderResult(
        dimension="bounds", verdict=Verdict.FAIL, reason="x",
        evidence=[PRICE_EVIDENCE],  # list at construction...
    )
    assert isinstance(result.evidence, tuple)  # ...tuple at rest
    with pytest.raises(FrozenInstanceError):
        result.evidence = ()


# -------------------------------------------------------- 6. dict round trip


def test_dict_round_trip_equality():
    original = GraderResult(
        dimension="final_state",
        verdict=Verdict.FAIL,
        reason="booked price 1400 exceeds spec limit 1200",
        evidence=(PRICE_EVIDENCE, Evidence(kind="empty_trajectory")),
    )
    restored = GraderResult.from_dict(original.to_dict())
    assert restored == original  # object equality, not dict equality


def test_to_dict_verdict_is_plain_string():
    result = GraderResult(dimension="bounds", verdict=Verdict.ERROR, reason="cannot judge")
    data = result.to_dict()
    assert data["verdict"] == "ERROR" and isinstance(data["verdict"], str)
    assert data["evidence"] == []


# ------------------------------------------------------------- 7. ERROR != FAIL


def test_error_never_equals_fail():
    assert Verdict.ERROR != Verdict.FAIL
    error = GraderResult(dimension="bounds", verdict=Verdict.ERROR, reason="cannot judge")
    fail = GraderResult(dimension="bounds", verdict=Verdict.FAIL, reason="cannot judge",
                         evidence=(PRICE_EVIDENCE,))
    assert error != fail


# --------------------------------------------------- 8 & 9. evidence rules


def test_fail_and_warn_require_evidence():
    with pytest.raises(ValueError, match="FAIL results must cite at least one evidence"):
        GraderResult(dimension="bounds", verdict=Verdict.FAIL, reason="x")
    with pytest.raises(ValueError, match="WARN results must cite at least one evidence"):
        GraderResult(dimension="bounds", verdict=Verdict.WARN, reason="x")


def test_error_may_carry_diagnostic_evidence():
    # ERROR evidence points at *why judging was blocked* (here: a malformed
    # spec field, step=None) -- the most useful thing an ERROR can say.
    # It is not a judgment about the trajectory, so ERROR stays != FAIL.
    result = GraderResult(
        dimension="final_state", verdict=Verdict.ERROR,
        reason="spec field 'expected_final_state' is malformed",
        evidence=[Evidence(kind="spec_field_malformed",
                           field="expected_final_state", value="not-a-dict")],
    )
    assert result.verdict is Verdict.ERROR
    assert result.evidence[0].kind == "spec_field_malformed"


# ------------------------------------------------------- 10. evidence details


def test_evidence_kind_must_be_non_empty():
    with pytest.raises(ValueError, match="evidence 'kind' must be a non-empty string"):
        Evidence(kind="")


def test_evidence_step_must_be_one_based():
    with pytest.raises(ValueError, match="evidence 'step' must be a 1-based int or None"):
        Evidence(kind="x", step=0)
    with pytest.raises(ValueError, match="evidence 'step' must be a 1-based int or None"):
        Evidence(kind="x", step=True)  # bool must not sneak in as an int


def test_evidence_item_must_be_evidence():
    with pytest.raises(ValueError, match="items must be Evidence"):
        GraderResult(dimension="bounds", verdict=Verdict.FAIL, reason="x",
                     evidence=[{"kind": "not-typed"}])
