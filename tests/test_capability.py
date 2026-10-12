"""The capability router client: what it asks the endpoint and how it reads the two answers."""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from routing_lab import capability
from routing_lab.backends import SKILL_CRITERIA, CapabilityDecision, Decision, questions
from routing_lab.capability import MAX_REQUEST_CHARS, TIERS, CapabilityResult, CapabilityRouter, capability_state
from routing_lab.data import load_contracts
from routing_lab.taxonomy import SKILLS, taxonomy_for_routes

TIER_PROBS = {"low": 0.1, "medium": 0.25, "high": 0.65}
SKILL_PROBS = {"coding": 0.7, "reasoning": 0.2, "tools": 0.04, "writing": 0.03, "extraction": 0.03}


def decision(**kwargs) -> CapabilityDecision:
    base = {"taxonomy_id": "capability-v1", "route": "high", "probabilities": TIER_PROBS, "skill": "coding",
            "skill_probabilities": SKILL_PROBS, "input_tokens": 120, "output_tokens": 0, "state_tokens_dropped": 0}
    return CapabilityDecision(**{**base, **kwargs})


def router_with(answer) -> CapabilityRouter:
    backend = Mock()
    backend.predict_state.side_effect = answer if isinstance(answer, list) else [answer]
    return CapabilityRouter("https://x.eastus2.inference.ml.azure.com/score", "flash-gpu", backend=backend)


def test_the_catalog_is_three_ordered_tiers_with_a_fixed_skill_list():
    contracts = load_contracts(capability.CONTRACTS, "capability-v1")
    assert TIERS == ("low", "medium", "high") and list(contracts) == list(TIERS)
    assert taxonomy_for_routes(contracts) == "capability-v1" and SKILLS == tuple(SKILL_CRITERIA)
    built = questions(contracts, taxonomy_id="capability-v1", skills=True)
    assert list(built) == ["route", "skill"] and list(built["skill"]["criteria"]) == list(SKILLS)
    assert "smallest tier" in built["route"]["instructions"]
    assert "skill" not in questions(contracts, taxonomy_id="capability-v1")
    for tier in contracts.values():
        assert not tier.can_write and tier.tools == []


def test_the_state_is_only_the_request_and_optional_context():
    assert capability_state("  Reformat this date.  ") == "User request:\nReformat this date."
    assert capability_state("Do it", context="order A-1") == "User request:\nDo it\nAvailable context:\norder A-1"
    for bad in ("", "   ", "x" * (MAX_REQUEST_CHARS + 1)):
        with pytest.raises(ValueError):
            capability_state(bad)


def test_the_result_keeps_full_distributions_and_derives_the_summary_from_them():
    result = router_with(decision()).classify("Review this cache for race conditions.")
    assert result.tier == "high" and result.skill == "coding" and result.confidence == 0.65
    assert result.tier_probabilities == TIER_PROBS and result.skill_probabilities == SKILL_PROBS
    assert result.expected_tier == pytest.approx(0.25 + 2 * 0.65)
    assert result.input_tokens == 120 and result.deployment == "flash-gpu" and result.latency_ms >= 0
    out = result.to_dict()
    assert out["tier"] == "high" and out["expected_tier"] == 1.55 and set(out["skill_probabilities"]) == set(SKILLS)
    uncertain = CapabilityResult({"low": 0.34, "medium": 0.33, "high": 0.33}, SKILL_PROBS, 1.0)
    assert uncertain.confidence == 0.34 and uncertain.expected_tier == pytest.approx(0.33 + 0.66)


def test_the_request_text_reaches_the_endpoint_in_the_trained_format():
    router = router_with(decision())
    router.classify("Reformat this date.", context="no tools")
    router.backend.predict_state.assert_called_once_with("User request:\nReformat this date.\nAvailable context:\nno tools")


@pytest.mark.parametrize("answer", [
    Decision(taxonomy_id="capability-v1", route="low", probabilities={"low": 1.0, "medium": 0.0, "high": 0.0}),
    decision(skill_probabilities={"coding": 1.0, "reasoning": 0.0, "tools": 0.0, "writing": 0.0, "extraction": 0.0, "magic": 0.0}),
])
def test_an_answer_missing_either_head_is_refused(answer):
    with pytest.raises(ValueError):
        router_with(answer).classify("x")


def test_the_client_is_built_for_both_heads_and_a_short_timeout():
    router = CapabilityRouter("https://x.eastus2.inference.ml.azure.com/score", "flash-gpu")
    assert list(router.backend.question) == ["route", "skill"] and router.backend.taxonomy_id == "capability-v1"
    assert router.backend._timeout == (5, 30) and router.deployment == "flash-gpu"


def test_throttling_and_dropped_connections_are_retried_but_real_errors_are_not(monkeypatch):
    import requests

    monkeypatch.setattr(capability.time, "sleep", lambda s: None)
    router = router_with([RuntimeError("Clef endpoint returned HTTP 429"), requests.ConnectionError("closed"), decision()])
    assert router.classify("x").tier == "high" and router.backend.predict_state.call_count == 3
    router.backend.reset_connection.assert_called_once()
    broken = router_with(RuntimeError("Clef endpoint returned HTTP 500"))
    with pytest.raises(RuntimeError, match="500"):
        broken.classify("x")
    assert broken.backend.predict_state.call_count == 1
    busy = router_with(RuntimeError("Clef endpoint returned HTTP 429"))
    busy.backend.predict_state.side_effect = RuntimeError("Clef endpoint returned HTTP 429")
    with pytest.raises(RuntimeError):
        busy.classify("x")
    assert busy.backend.predict_state.call_count == CapabilityRouter.RETRIES + 1
