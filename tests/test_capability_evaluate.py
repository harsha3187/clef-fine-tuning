"""Clef as a router: how its answers are cached, scored and turned into routing choices."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from test_capability_runner import MODEL_COST, row, synthetic

from capability_lab import analysis, evaluate


def pred(tier_probs, skill="reasoning", **extra):
    top = max(tier_probs, key=tier_probs.get)
    return {"tier": top, "tier_probabilities": tier_probs, "skill": skill, "skill_probabilities": {skill: 1.0}, "confidence": tier_probs[top], **extra}


ONE = {"low": {"low": 0.9, "medium": 0.05, "high": 0.05}, "medium": {"low": 0.1, "medium": 0.8, "high": 0.1}, "high": {"low": 0.05, "medium": 0.05, "high": 0.9}}


def world(n=12):
    patterns = {f"easy{i}": row("a", "b", "c", "d", "e") for i in range(n)} | {f"hard{i}": row("e") for i in range(n)}
    tasks, runs = synthetic(patterns, MODEL_COST)
    return tasks, runs, analysis.tier_labels(tasks, runs)


def test_clef_is_asked_once_per_task_and_answers_are_cached(tmp_path):
    tasks, _, _ = world(3)

    class Router:
        calls = 0

        def classify(self, text):
            Router.calls += 1
            return SimpleNamespace(to_dict=lambda: pred(ONE["low"], latency_ms=5.0))
    out = tmp_path / "clef.jsonl"
    first = evaluate.ask_clef(Router(), tasks, out, workers=2)
    assert Router.calls == len(tasks) == len(first)
    again = evaluate.ask_clef(Router(), tasks, out, workers=2)
    assert Router.calls == len(tasks) and again.keys() == first.keys()
    assert len(out.read_text(encoding="utf-8").splitlines()) == len(tasks)


def test_tier_metrics_count_direction_of_error_and_compare_with_the_majority_class():
    tasks, runs, labels = world(10)
    perfect = {label["task_id"]: pred(ONE[label["tier"]]) for label in labels}
    m = evaluate.tier_metrics(labels, perfect)
    assert m["tier_accuracy"] == 1.0 and m["macro_f1"] == pytest.approx(2 / 3, abs=0.001) and m["underestimates"] == m["overestimates"] == 0
    assert m["majority_class_accuracy"] == 0.5 and m["confusion"]["low"]["low"] == 10 and m["skill_accuracy"] == 1.0
    assert m["brier"] < 0.05 and 0 <= m["ece"] < 0.15
    always_low = {label["task_id"]: pred(ONE["low"]) for label in labels}
    wrong = evaluate.tier_metrics(labels, always_low)
    assert wrong["tier_accuracy"] == 0.5 and wrong["underestimates"] == 10 and wrong["overestimates"] == 0
    assert wrong["brier"] > m["brier"] and wrong["ece"] > m["ece"]


def test_measured_rates_leave_the_task_being_routed_out():
    tasks, runs, labels = world(6)
    cells = evaluate.Cells(labels, analysis.pass_matrix(runs))
    assert cells.rate("a", "reasoning", "low") == 1.0 and cells.rate("a", "reasoning", "high") == 0.0
    # five easy tasks remain once one is left out; a task alone in its cell falls back to nothing measured
    assert cells.rate("a", "reasoning", "low", leave_out="easy0") == 1.0
    sparse = evaluate.Cells(labels[:7] + labels[-1:], analysis.pass_matrix(runs))
    assert sparse.rate("e", "reasoning", "high", leave_out=labels[-1]["task_id"]) is None
    assert cells.rate("a", None, "low") == 1.0


def test_a_router_with_the_right_tiers_matches_the_oracle_and_a_wrong_one_loses_passes():
    tasks, runs, labels = world(12)
    good = {label["task_id"]: pred(ONE[label["tier"]]) for label in labels}
    bad = {label["task_id"]: pred(ONE["low"]) for label in labels}
    result = evaluate.evaluate_arms(tasks, labels, runs, good, floor=0.85)
    arms = result["arms"]
    assert arms["always e"]["pass_rate"] == 1.0 and arms["always a"]["pass_rate"] == 0.5 and result["best_value_endpoint"] == "e"
    assert arms["Clef (tier and skill)"]["pass_rate"] == 1.0 and arms["oracle cheapest passing"]["pass_rate"] == 1.0
    assert arms["true tier and skill (upper bound)"]["models_used"] == {"a": 12, "e": 12}
    saved = arms["always e"]["mean_cost_usd"] - arms["Clef (tier and skill)"]["mean_cost_usd"]
    assert saved > 0.02 and arms["Clef (tier and skill)"]["mean_cost_usd"] == pytest.approx((0.001 + 0.05) / 2 + evaluate.ROUTER_COST_USD, abs=1e-6)
    assert arms["true tier and skill (upper bound)"]["mean_cost_usd"] == pytest.approx((0.001 + 0.05) / 2, abs=1e-6)
    wrong = evaluate.evaluate_arms(tasks, labels, runs, bad, floor=0.85)["arms"]["Clef (tier only)"]
    assert wrong["pass_rate"] == 0.5 and wrong["models_used"] == {"a": 24}
    free = evaluate.evaluate_arms(tasks, labels, runs, good, router_cost=0.0)["arms"]["Clef (tier only)"]
    assert free["mean_cost_usd"] == pytest.approx(0.0255, abs=1e-6)
    json.dumps(result)


def test_uncertainty_is_handled_by_the_probabilities_not_by_the_argmax():
    tasks, runs, labels = world(12)
    unsure = {label["task_id"]: pred({"low": 0.5, "medium": 0.0, "high": 0.5}) for label in labels}
    arms = evaluate.evaluate_arms(tasks, labels, runs, unsure, floor=0.85)["arms"]
    # half the probability mass sits on "high", where only "e" passes, so the expected pass rate of cheap endpoints is 0.5
    assert arms["Clef (tier only)"]["models_used"] == {"e": 24} and arms["Clef (tier only)"]["pass_rate"] == 1.0
    relaxed = evaluate.evaluate_arms(tasks, labels, runs, unsure, floor=0.45)["arms"]["Clef (tier only)"]
    assert relaxed["models_used"] == {"a": 24} and relaxed["pass_rate"] == 0.5


def test_the_length_heuristic_is_a_real_baseline_that_ignores_difficulty():
    assert evaluate.heuristic_probs("x" * 100)["low"] == 1.0 and evaluate.heuristic_probs("x" * 500)["medium"] == 1.0
    assert evaluate.heuristic_probs("x" * 2000)["high"] == 1.0
    tasks, runs, labels = world(12)
    arms = evaluate.evaluate_arms(tasks, labels, runs, {l["task_id"]: pred(ONE[l["tier"]]) for l in labels})["arms"]  # noqa: E741
    assert arms["length heuristic"]["pass_rate"] == 0.5  # every synthetic prompt is short, so it always says "low"
