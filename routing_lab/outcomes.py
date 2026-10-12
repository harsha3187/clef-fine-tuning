"""Compare independently graded downstream runs without executing untrusted code."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Outcome(BaseModel):
    """One real harness result, with costs including routing and all retries."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    task_id: str = Field(min_length=1)
    policy: str = Field(min_length=1)
    success: bool
    total_cost_usd: float = Field(ge=0)
    total_latency_seconds: float = Field(ge=0)
    attempts: int = Field(ge=1)
    evidence: str = Field(min_length=1)
    harness_version: str = Field(min_length=1)


def compare_outcomes(path: Path, baseline: str, candidate: str) -> dict:
    """Require paired task IDs and a common grader before computing deltas."""
    if baseline == candidate:
        raise ValueError("Choose distinct policies")
    rows = [Outcome.model_validate_json(line) for line in
            path.read_text(encoding="utf-8").splitlines() if line.strip()]
    groups: dict[str, dict[str, Outcome]] = {baseline: {}, candidate: {}}
    for row in rows:
        if row.policy not in groups:
            raise ValueError(f"Unexpected policy: {row.policy}")
        if row.task_id in groups[row.policy]:
            raise ValueError("Duplicate task/policy pair")
        groups[row.policy][row.task_id] = row
    if not groups[baseline] or groups[baseline].keys() != groups[candidate].keys():
        raise ValueError("Policies must contain the same nonempty task set")
    if len({r.harness_version for r in rows}) != 1:
        raise ValueError("Use the same grading harness version")
    results = {}
    for name, tasks in groups.items():
        success = sum(r.success for r in tasks.values())
        cost = sum(r.total_cost_usd for r in tasks.values())
        results[name] = {
            "tasks": len(tasks), "success_rate": success / len(tasks),
            "total_cost_usd": cost,
            "cost_per_success_usd": cost / success if success else None,
            "mean_latency_seconds": sum(r.total_latency_seconds for r in tasks.values()) / len(tasks),
            "total_attempts": sum(r.attempts for r in tasks.values()),
        }
    deltas = [int(groups[candidate][key].success) - int(groups[baseline][key].success)
              for key in groups[baseline]]
    rng = random.Random(42)
    bootstrap = [sum(rng.choices(deltas, k=len(deltas))) / len(deltas) for _ in range(1000)]
    return {
        "policies": results, "success_rate_delta": sum(deltas) / len(deltas),
        "paired_task_bootstrap_95ci": np.percentile(bootstrap, [2.5, 97.5]).tolist(),
        "caution": "Descriptive estimate; tasks may be correlated. Predeclare tolerances and inspect evidence.",
        "fine_tuning_decision": "manual_review_required",
    }
