"""Paired downstream outcomes include failed attempts and routing costs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from routing_lab.outcomes import Outcome, compare_outcomes


def outcome(task: str, policy: str, **changes: Any) -> dict[str, Any]:
    return {
        "task_id": task, "policy": policy, "success": True,
        "total_cost_usd": 1.0, "total_latency_seconds": 2.0, "attempts": 1,
        "evidence": "synthetic harness assertion", "harness_version": "synthetic-v1",
        **changes,
    }


def write_outcomes(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    path = tmp_path / "outcomes.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "same nonempty task set"),
        ([outcome("a", "base")], "same nonempty task set"),
        ([outcome("a", "base"), outcome("b", "candidate")], "same nonempty task set"),
        ([outcome("a", "base"), outcome("a", "candidate"), outcome("a", "base")], "Duplicate"),
        ([outcome("a", "base"), outcome("a", "candidate"), outcome("b", "other")], "Unexpected policy"),
        ([outcome("a", "base"), outcome("a", "candidate", harness_version="v2")], "same grading"),
    ],
)
def test_outcomes_reject_unpaired_duplicate_or_incompatible_results(
    tmp_path: Path, rows: list[dict[str, Any]], message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        compare_outcomes(write_outcomes(tmp_path, rows), "base", "candidate")


def test_outcomes_require_distinct_policy_names(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct"):
        compare_outcomes(write_outcomes(tmp_path, []), "base", "base")


def test_pairing_uses_task_ids_not_row_order_and_includes_all_costs(tmp_path: Path) -> None:
    rows = [
        outcome("a", "base", success=False, total_cost_usd=2.0, attempts=2),
        outcome("b", "candidate", success=False, total_cost_usd=4.0, attempts=3),
        outcome("b", "base", success=False, total_cost_usd=3.0, attempts=4),
        outcome("a", "candidate", total_cost_usd=1.5, total_latency_seconds=8.0, attempts=2),
    ]
    path = write_outcomes(tmp_path, rows)
    result = compare_outcomes(path, "base", "candidate")
    assert result == compare_outcomes(path, "base", "candidate")
    baseline, candidate = result["policies"]["base"], result["policies"]["candidate"]
    assert baseline["tasks"] == candidate["tasks"] == 2
    assert baseline["success_rate"] == 0
    assert baseline["cost_per_success_usd"] is None
    assert baseline["total_cost_usd"] == 5.0
    assert baseline["total_attempts"] == 6
    assert candidate["success_rate"] == 0.5
    assert candidate["total_cost_usd"] == 5.5
    assert candidate["cost_per_success_usd"] == 5.5
    assert candidate["total_attempts"] == 5
    assert candidate["mean_latency_seconds"] == 5.0
    assert result["success_rate_delta"] == 0.5
    assert result["paired_task_bootstrap_95ci"] == [0.0, 1.0]
    assert result["fine_tuning_decision"] == "manual_review_required"


def test_zero_successes_in_both_policies_has_defined_delta_not_cost_ratio(tmp_path: Path) -> None:
    path = write_outcomes(tmp_path, [
        outcome("a", policy, success=False) for policy in ("base", "candidate")
    ])
    result = compare_outcomes(path, "base", "candidate")
    assert result["success_rate_delta"] == 0
    assert result["paired_task_bootstrap_95ci"] == [0.0, 0.0]
    assert all(row["cost_per_success_usd"] is None for row in result["policies"].values())


@pytest.mark.parametrize(
    "changes",
    [
        {"success": "true"}, {"attempts": 0}, {"attempts": 1.5},
        {"total_cost_usd": -1.0}, {"total_cost_usd": float("nan")},
        {"total_latency_seconds": -1.0}, {"total_latency_seconds": float("inf")},
        {"task_id": ""}, {"evidence": ""}, {"harness_version": ""}, {"extra": "field"},
    ],
)
def test_outcome_validation_is_strict(changes: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        Outcome.model_validate(outcome("a", "base", **changes))
