"""Probability validation and native Clef adapter tests without model imports."""

from __future__ import annotations

from collections.abc import Callable
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from routing_lab import clef
from routing_lab.backends import INSTRUCTIONS, ClefBackend, Decision, decision_from_native, state
from routing_lab.clef import DEFAULT_MAX_LENGTH
from routing_lab.data import ROUTES, Contract, Record


def distribution(route: str = "debugging", confidence: float = 1.0) -> dict[str, float]:
    return {label: confidence if label == route else (1 - confidence) / 5 for label in ROUTES}


@pytest.mark.parametrize(
    "probabilities",
    [
        {"debugging": 1.0},
        {**distribution(), "unknown": 0.0},
        {**distribution(), "debugging": 0.8},
        {**distribution(), "debugging": 1.1, "fallback": -0.1},
        {**distribution(), "debugging": float("nan")},
        {**distribution(), "debugging": float("inf")},
        {**distribution(), "debugging": float("-inf")},
        distribution("fallback"),
    ],
)
def test_decision_rejects_invalid_distribution(probabilities: dict[str, float]) -> None:
    with pytest.raises(ValidationError):
        Decision(route="debugging", probabilities=probabilities)


def test_decision_rejects_invalid_route() -> None:
    with pytest.raises(ValidationError):
        Decision.model_validate({"route": "not-a-route"})


def test_decision_accepts_absent_probabilities_and_tied_maximum() -> None:
    assert Decision(route="fallback").probabilities is None
    tied = {route: 1 / len(ROUTES) for route in ROUTES}
    assert Decision(route="debugging", probabilities=tied).probabilities == tied


@pytest.mark.parametrize("field", ["input_tokens", "output_tokens", "state_tokens_dropped"])
def test_decision_rejects_negative_usage(field: str) -> None:
    with pytest.raises(ValidationError):
        Decision.model_validate({"route": "fallback", field: -1})


@pytest.mark.parametrize(("variant", "context_mode"), [("flash", "context"), ("clef", "query")])
def test_clef_uses_native_contract_and_preserves_usage(
    monkeypatch: pytest.MonkeyPatch, make_record: Callable[..., Record],
    contracts: dict[str, Contract], variant: str, context_mode: str,
) -> None:
    probabilities = distribution("debugging", 0.8)
    runtime = clef.ClefRuntime(None, None, None, variant)
    loader = Mock(return_value=runtime)
    decide = Mock(return_value={
        "answers": {"route": {"choice": "debugging", "probabilities": probabilities}},
        "usage": {"input_tokens": 71, "output_tokens": 0, "state_tokens_dropped": 9},
    })
    monkeypatch.setattr(clef, "load", loader)
    monkeypatch.setattr(clef, "decide", decide)
    backend = ClefBackend(contracts, variant=variant, device="cuda", context_mode=context_mode)
    loader.assert_called_once_with(variant, device="cuda", snapshot=None, calibrated=False)
    record = make_record(query="Fix the synthetic crash.", context="Trace: synthetic failure.")
    decision = backend.predict(record)
    expected_state = state(record, context_mode)
    question = {"route": {
        "type": "choice", "instructions": INSTRUCTIONS,
        "criteria": {key: contracts[key].description for key in ROUTES},
    }}
    decide.assert_called_once_with(runtime, expected_state, question, max_length=DEFAULT_MAX_LENGTH)
    assert decision.model_dump() == {
        "taxonomy_id": "coding-v1",
        "route": "debugging", "probabilities": probabilities,
        "input_tokens": 71, "output_tokens": 0, "state_tokens_dropped": 9,
    }
    metadata = backend.metadata()
    assert metadata["checkpoint"] == clef.RELEASES[variant].model_id
    assert metadata["revision"] == clef.RELEASES[variant].revision
    assert metadata["questions"] == question
    assert metadata["max_length"] == DEFAULT_MAX_LENGTH
    assert metadata["variant"] == variant


def test_clef_snapshot_and_max_length_are_forwarded(
    monkeypatch: pytest.MonkeyPatch, make_record: Callable[..., Record],
    contracts: dict[str, Contract],
) -> None:
    loader = Mock(return_value=clef.ClefRuntime(None, None, None, "flash", 2.0))
    decide = Mock(return_value={
        "answers": {"route": {"choice": "debugging", "probabilities": distribution("debugging", 0.8)}},
        "usage": {"input_tokens": 2048, "output_tokens": 0, "state_tokens_dropped": 0},
    })
    monkeypatch.setattr(clef, "load", loader)
    monkeypatch.setattr(clef, "decide", decide)
    backend = ClefBackend(contracts, max_length=4096, snapshot="outputs/tuned", calibrated=True)
    backend.predict(make_record())
    loader.assert_called_once_with("flash", device="cuda", snapshot="outputs/tuned", calibrated=True)
    assert decide.call_args.kwargs == {"max_length": 4096}
    assert backend.metadata()["snapshot"] == "outputs/tuned"
    assert backend.metadata()["temperature"] == 2.0


@pytest.mark.parametrize("result", [
    {}, {"answers": {"route": {"choice": "debugging"}}, "usage": {}},
    {"answers": {"route": {"choice": "debugging", "probabilities": None}},
     "usage": {"input_tokens": 1, "output_tokens": 0, "state_tokens_dropped": 0}},
])
def test_decision_from_native_rejects_incomplete_responses(result: dict) -> None:
    with pytest.raises(ValueError):
        decision_from_native(result, "coding-v1")


def test_state_exposes_only_decision_time_fields(make_record: Callable[..., Record]) -> None:
    record = make_record(
        id="DO_NOT_EXPOSE_ID", group_id="DO_NOT_EXPOSE_GROUP",
        repository="DO_NOT_EXPOSE_REPOSITORY", provenance="DO_NOT_EXPOSE_PROVENANCE",
        reviewer="DO_NOT_EXPOSE_REVIEWER", tags=["DO_NOT_EXPOSE_TAG"],
    )
    assert state(record, "query") == record.query
    assert state(record, "context") == (
        f"User request:\n{record.query}\nAvailable context:\n{record.context}"
    )
    with pytest.raises(ValueError, match="Context mode"):
        state(record, "unknown")
