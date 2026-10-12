"""Turn raw benchmark results into measured pass rates, tier labels and the savings a capability router could reach.

Tier labels come from the endpoints themselves: the share of endpoints that solve a task. A task nearly everyone
solves is "low" (a cheap model is enough); one almost nobody solves is "high". Nothing here assumes which model is
small or large.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from capability_lab.clients import ModelSpec
from capability_lab.tasks import Task

TIERS = ("low", "medium", "high")
LOW_AT_LEAST = 0.7
HIGH_BELOW = 0.35
MIN_ENDPOINTS = 5


def wilson(passes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = passes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)


def pass_matrix(runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    """task -> model -> {pass rate over valid samples, mean cost, mean latency}. Errors are missing data, not failures."""
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        if r["error"] is None and r["passed"] is not None:
            grouped[r["task_id"]][r["model"]].append(r)
    return {task: {model: {"rate": sum(x["passed"] for x in rs) / len(rs), "cost": sum(x["cost_usd"] for x in rs) / len(rs),
                           "latency_ms": sum(x["latency_ms"] for x in rs) / len(rs), "n": len(rs)}
                   for model, rs in by_model.items()} for task, by_model in grouped.items()}


def classify(fraction: float, low_at_least: float = LOW_AT_LEAST, high_below: float = HIGH_BELOW) -> str:
    return "low" if fraction >= low_at_least else "high" if fraction < high_below else "medium"


def tier_labels(tasks: list[Task], runs: list[dict[str, Any]], *, low_at_least: float = LOW_AT_LEAST,
                high_below: float = HIGH_BELOW, min_endpoints: int = MIN_ENDPOINTS) -> list[dict[str, Any]]:
    matrix = pass_matrix(runs)
    labels = []
    for task in tasks:
        row = matrix.get(task.id, {})
        if len(row) < min_endpoints:
            continue  # too few endpoints answered for the share to mean anything
        fraction = sum(v["rate"] for v in row.values()) / len(row)
        passing = [m for m, v in row.items() if v["rate"] >= 0.5]
        cheapest = min(passing, key=lambda m: row[m]["cost"]) if passing else None
        labels.append({"task_id": task.id, "family": task.family, "skill": task.skill, "level": task.level,
                       "pass_fraction": round(fraction, 4), "tier": classify(fraction, low_at_least, high_below),
                       "endpoints": len(row), "passing": sorted(passing), "cheapest_passing": cheapest,
                       "cheapest_cost": round(row[cheapest]["cost"], 8) if cheapest else None, "unsolved": not passing})
    return labels


def profiles(labels: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Measured pass rate per endpoint, skill and tier, with a 95% interval and the sample size."""
    tier_of = {label["task_id"]: (label["skill"], label["tier"]) for label in labels}
    counts: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    for r in runs:
        if r["error"] is None and r["passed"] is not None and r["task_id"] in tier_of:
            skill, tier = tier_of[r["task_id"]]
            cell = counts[(r["model"], skill, tier)]
            cell[0] += float(r["passed"])
            cell[1] += 1
    out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for (model, skill, tier), (passes, n) in counts.items():
        low, high = wilson(passes, n)
        out.setdefault(model, {}).setdefault(skill, {})[tier] = {"pass_rate": round(passes / n, 4), "n": n, "ci_low": low, "ci_high": high}
    return out


def model_summary(runs: list[dict[str, Any]], models: dict[str, ModelSpec]) -> list[dict[str, Any]]:
    rows = []
    for model_id, spec in models.items():
        mine = [r for r in runs if r["model"] == model_id]
        valid = [r for r in mine if r["error"] is None and r["passed"] is not None]
        if not mine:
            continue
        passes = sum(r["passed"] for r in valid)
        low, high = wilson(passes, len(valid))
        rows.append({"model": model_id, "family": spec.family, "attempts": len(mine), "errors": len(mine) - len(valid),
                     "pass_rate": round(passes / len(valid), 4) if valid else None, "ci_low": low, "ci_high": high,
                     "mean_cost_usd": round(sum(r["cost_usd"] for r in valid) / len(valid), 6) if valid else None,
                     "mean_latency_s": round(sum(r["latency_ms"] for r in valid) / len(valid) / 1000, 2) if valid else None,
                     "mean_output_tokens": round(sum(r["tokens_out"] for r in valid) / len(valid)) if valid else None,
                     "price_verified": spec.verified})
    return sorted(rows, key=lambda r: -(r["pass_rate"] or 0))


def simulate_selector(tasks: list[Task], labels: list[dict[str, Any]], runs: list[dict[str, Any]], *, floor: float = 0.85,
                      min_n: int = 3) -> dict[str, Any]:
    """What a router could reach if it knew each task's true tier.

    For each task the profile is rebuilt without that task (leave-one-out), then the cheapest endpoint whose
    measured pass rate for the task's skill and tier meets the floor is chosen. This is an upper bound for the
    capability approach: it assumes perfect tier knowledge.
    """
    matrix = pass_matrix(runs)
    label_of = {label["task_id"]: label for label in labels}
    cell: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0])
    for task_id, row in matrix.items():
        label = label_of.get(task_id)
        if label:
            for model, v in row.items():
                c = cell[(model, label["skill"], label["tier"])]
                c[0] += v["rate"]
                c[1] += 1
    mean_cost: dict[str, float] = {}
    for model in {m for row in matrix.values() for m in row}:
        costs = [row[model]["cost"] for row in matrix.values() if model in row]
        mean_cost[model] = sum(costs) / len(costs)
    picked = passed = 0
    cost = 0.0
    chosen_models: dict[str, int] = defaultdict(int)
    for task in tasks:
        label, row = label_of.get(task.id), matrix.get(task.id)
        if not label or not row:
            continue
        options = []
        for model, v in row.items():
            passes, n = cell[(model, label["skill"], label["tier"])]
            passes, n = passes - v["rate"], n - 1  # leave this task out
            if n >= min_n:
                options.append((model, passes / n))
        if not options:
            continue
        ok = [m for m, rate in options if rate >= floor]
        model = min(ok, key=lambda m: mean_cost[m]) if ok else max(options, key=lambda o: o[1])[0]
        picked += 1
        passed += row[model]["rate"]
        cost += row[model]["cost"]
        chosen_models[model] += 1
    return {"tasks": picked, "pass_rate": round(passed / picked, 4) if picked else None,
            "mean_cost_usd": round(cost / picked, 6) if picked else None, "floor": floor, "models_used": dict(chosen_models)}


def strategies(tasks: list[Task], labels: list[dict[str, Any]], runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Always-one-endpoint arms, the oracle that picks the cheapest passing endpoint, and the perfect-tier selector."""
    matrix = pass_matrix(runs)
    ids = {label["task_id"] for label in labels}
    out: dict[str, Any] = {"always": {}}
    models = {m for t in ids for m in matrix.get(t, {})}
    for model in sorted(models):
        rows = [matrix[t][model] for t in ids if model in matrix.get(t, {})]
        if rows:
            out["always"][model] = {"pass_rate": round(sum(r["rate"] for r in rows) / len(rows), 4),
                                    "mean_cost_usd": round(sum(r["cost"] for r in rows) / len(rows), 6), "tasks": len(rows)}
    oracle = [label for label in labels if label["cheapest_passing"]]
    out["oracle"] = {"pass_rate": round(len(oracle) / len(labels), 4) if labels else None,
                     "mean_cost_usd": round(sum(label["cheapest_cost"] for label in oracle) / len(oracle), 6) if oracle else None,
                     "unsolved_tasks": sum(label["unsolved"] for label in labels)}
    out["perfect_tier_selector"] = simulate_selector(tasks, labels, runs)
    return out
