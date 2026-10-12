from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from capability_lab.clients import ModelSpec
from capability_lab.registry import build_registry
from routing_lab.selector import (
    PRESET_SETTINGS,
    Constraints,
    Endpoint,
    MeasuredRegistry,
    Settings,
    expected_success,
    select,
)

ROOT = Path(__file__).resolve().parents[1]


def ep(name, price_out, *, tools=True, allowed=True, healthy=True, mean_out=100.0):
    return Endpoint(name, name, 0.0, price_out, tools, True, mean_out, 0.0, 0.0, allowed=allowed, healthy=healthy)


RATES = {  # (endpoint, tier) -> pass rate, the same for every skill
    ("cheap", "low"): 0.95, ("cheap", "medium"): 0.5, ("cheap", "high"): 0.1,
    ("mid", "low"): 0.97, ("mid", "medium"): 0.9, ("mid", "high"): 0.4,
    ("big", "low"): 0.98, ("big", "medium"): 0.97, ("big", "high"): 0.9,
}


def rate(endpoint, skill, tier):
    return RATES.get((endpoint, tier))


POOL = [ep("cheap", 1.0), ep("mid", 5.0), ep("big", 20.0)]
BALANCED = Settings(floor=0.7, cost_weight=0.6, latency_weight=0.0)
ONE_TOKEN = Constraints(input_tokens=0, output_tokens=1_000_000)  # cost equals price_out


def test_expected_success_weights_each_tier_by_its_probability():
    value = expected_success(rate, "mid", {"low": 0.5, "medium": 0.5}, {"coding": 1.0})
    assert value == pytest.approx((0.97 + 0.9) / 2)


def test_an_easy_request_goes_to_the_cheapest_endpoint_above_the_floor():
    result = select(POOL, rate, {"low": 0.9, "medium": 0.08, "high": 0.02}, {"coding": 1.0}, BALANCED, ONE_TOKEN)
    assert result.choice == "cheap" and not result.below_floor and not result.uncertain
    assert [c.status for c in result.candidates].count("selected") == 1


def test_a_harder_request_moves_up_because_the_cheap_endpoint_falls_below_the_floor():
    result = select(POOL, rate, {"low": 0.05, "medium": 0.9, "high": 0.05}, None, BALANCED, ONE_TOKEN)
    assert result.choice == "mid"
    assert next(c for c in result.candidates if c.endpoint == "cheap").status == "below_floor"


def test_unsure_clef_raises_the_floor_instead_of_guessing():
    unsure = {"low": 0.5, "medium": 0.3, "high": 0.2}
    plain = select(POOL, rate, {"low": 0.8, "medium": 0.15, "high": 0.05}, None, Settings(0.74, 0.6, 0.0), ONE_TOKEN)
    careful = select(POOL, rate, unsure, None, Settings(0.7, 0.6, 0.0), ONE_TOKEN)
    assert plain.choice == "cheap" and not plain.uncertain
    assert careful.uncertain and careful.effective_floor == pytest.approx(0.75)
    assert careful.choice == "mid" and "probability" in careful.reason


def test_high_tier_probability_is_a_warning_even_when_the_top_tier_is_confident():
    result = select(POOL, rate, {"low": 0.7, "medium": 0.0, "high": 0.3}, None, BALANCED, ONE_TOKEN)
    assert result.uncertain and "high tier" in result.uncertainty_reasons[0]


def test_hard_filters_remove_endpoints_before_any_scoring():
    pool = [ep("cheap", 1.0, tools=False), ep("mid", 5.0, healthy=False), ep("big", 20.0)]
    result = select(pool, rate, {"low": 1.0}, None, BALANCED, Constraints(needs_tools=True, output_tokens=1_000_000))
    reasons = {c.endpoint: c.reasons for c in result.candidates}
    assert result.choice == "big" and "cannot call tools" in reasons["cheap"][0] and "unhealthy" in reasons["mid"][0]
    allowed = select(POOL, rate, {"low": 1.0}, None, BALANCED, Constraints(allow=frozenset({"mid"})))
    assert allowed.choice == "mid"


def test_an_unmeasured_endpoint_is_never_selectable():
    pool = [*POOL, ep("new", 0.01)]
    result = select(pool, rate, {"low": 1.0}, None, BALANCED, ONE_TOKEN)
    assert result.choice == "cheap"
    assert "not measured" in next(c for c in result.candidates if c.endpoint == "new").reasons[0]
    only_new = select([ep("new", 0.01)], rate, {"low": 1.0}, None, BALANCED)
    assert only_new.choice is None and "No endpoint" in only_new.reason


def test_when_nothing_reaches_the_floor_the_most_reliable_endpoint_is_chosen_and_flagged():
    result = select(POOL, rate, {"low": 0.0, "medium": 0.0, "high": 1.0}, None, Settings(0.95, 0.6, 0.0), ONE_TOKEN)
    assert result.choice == "big" and result.below_floor and "No endpoint reaches" in result.reason


def test_cost_weight_trades_success_for_price_among_endpoints_above_the_floor():
    tiers = {"low": 0.5, "medium": 0.5, "high": 0.0}
    frugal = select(POOL, rate, tiers, None, Settings(0.5, 1.2, 0.0), ONE_TOKEN)
    lavish = select(POOL, rate, tiers, None, Settings(0.5, 0.0, 0.0), ONE_TOKEN)
    assert frugal.choice == "cheap" and lavish.choice == "big"


def test_the_decision_is_deterministic_and_never_depends_on_input_order():
    tiers, skills = {"low": 0.4, "medium": 0.4, "high": 0.2}, {"coding": 0.5, "writing": 0.5}
    first = select(POOL, rate, tiers, skills, BALANCED, ONE_TOKEN)
    again = select(list(reversed(POOL)), rate, tiers, skills, BALANCED, ONE_TOKEN)
    assert first.choice == again.choice and first.to_dict()["candidates"] == again.to_dict()["candidates"]


def test_a_verbose_endpoint_is_priced_for_the_extra_tokens_it_writes():
    pool = [ep("terse", 10.0, mean_out=50.0), ep("wordy", 10.0, mean_out=300.0), ep("typical", 10.0, mean_out=100.0)]
    result = select(pool, lambda *_: 0.95, {"low": 1.0}, None, Settings(0.5, 1.0, 0.0), Constraints(input_tokens=0, output_tokens=1000))
    costs = {c.endpoint: c.est_cost_usd for c in result.candidates}
    assert costs["wordy"] > costs["typical"] > costs["terse"] and result.choice == "terse"


def test_presets_match_the_agent_screen_policy():
    policy = json.loads((ROOT / "agent_demo" / "data" / "policy.json").read_text(encoding="utf-8-sig"))["preferences"]
    for name, level in (("lean", ("average", "low", "low")), ("balanced", ("good", "medium", "medium")),
                        ("premium", ("very_good", "high", "high"))):
        quality, latency, cost = policy["quality"][level[0]], policy["latency"][level[1]], policy["cost"][level[2]]
        expected = Settings(floor=quality["floor"], cost_weight=cost["weight"], latency_weight=latency["weight"])
        assert PRESET_SETTINGS[name] == expected


def test_the_registry_falls_back_to_the_tier_rate_and_refuses_thin_data():
    data = {"version": "v", "endpoints": [{**ep("a", 1.0).__dict__, "cells": {"coding": {"low": [9, 10], "high": [1, 2]}, "writing": {"low": [3, 4]}}}]}
    registry = MeasuredRegistry(data)
    assert registry.rate("a", "coding", "low") == 0.9
    assert registry.rate("a", "coding", "high") is None  # two samples are not enough on their own or pooled
    assert registry.rate("a", "tools", "low") == pytest.approx(12 / 14)  # pooled over skills
    assert registry.rate("ghost", "coding", "low") is None
    shares = registry.evidence("a", {"low": 0.5, "high": 0.5}, {"coding": 0.5, "tools": 0.5})
    assert shares == {"exact_share": 0.25, "pooled_share": 0.25, "missing_share": 0.5}
    assert [e.allowed for e in registry.with_changes(allowed={"a": False})] == [False]


def test_registry_builder_counts_passes_by_skill_and_tier_and_skips_failed_calls():
    models = {"m": ModelSpec("m", "m", "F", "responses", True, 1.0, 2.0, True, "test")}
    labels = [{"task_id": f"t{i}", "skill": "coding", "tier": "low"} for i in range(4)]
    runs = [{"task_id": f"t{i}", "model": "m", "error": None, "passed": i < 3, "tokens_out": 100 + 10 * i, "latency_ms": 1000 + 50 * i}
            for i in range(4)]
    runs.append({"task_id": "t0", "model": "m", "error": "HTTP 429", "passed": None, "tokens_out": 0, "latency_ms": 0})
    registry = build_registry(models, runs, labels, tasks_hash="abc", built=date(2026, 10, 8))
    item = registry["endpoints"][0]
    assert item["cells"] == {"coding": {"low": [3.0, 4]}} and registry["version"] == "measured-abc"
    assert item["lat_per_token_ms"] == pytest.approx(5.0) and item["lat_base_ms"] == pytest.approx(500.0)
    assert item["mean_out_tokens"] == 115.0 and item["verified_price"] is True


def test_the_shipped_measured_registry_covers_all_ten_endpoints():
    path = ROOT / "data" / "registry" / "measured.json"
    if not path.exists():
        pytest.skip("run python -m capability_lab registry")
    registry = MeasuredRegistry.load(path)
    assert len(registry.endpoints) == 10
    assert registry.rate("gpt-5.6-sol", "reasoning", "low") > 0.9
    assert not next(e for e in registry.endpoints if e.id == "phi-4").supports_tools
