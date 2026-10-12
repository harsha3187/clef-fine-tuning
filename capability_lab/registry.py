"""Build the measured endpoint registry the selector reads, from benchmark runs and tier labels.

Every number comes from the benchmark: pass counts per skill and tier (with sample sizes), mean output length, and a
straight-line latency fit. Prices come from the model list and keep their verified flag.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any

from capability_lab.clients import ModelSpec


def _fit_latency(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Latency in ms as a base plus a per-output-token slope; never negative, flat when the data cannot say more."""
    if not points:
        return 0.0, 0.0
    n = len(points)
    mean_x, mean_y = sum(x for x, _ in points) / n, sum(y for _, y in points) / n
    var = sum((x - mean_x) ** 2 for x, _ in points)
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / var if var > 0 else 0.0
    if slope < 0:
        return mean_y, 0.0
    return max(0.0, mean_y - slope * mean_x), slope


def build_registry(models: dict[str, ModelSpec], runs: list[dict[str, Any]], labels: list[dict[str, Any]], *,
                   tasks_hash: str, built: date | None = None) -> dict[str, Any]:
    label_of = {label["task_id"]: label for label in labels}
    cells: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: [0.0, 0.0])))
    latency: dict[str, list[tuple[float, float]]] = defaultdict(list)
    out_tokens: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        label = label_of.get(run["task_id"])
        if run["error"] is not None or run["passed"] is None or label is None or run["model"] not in models:
            continue
        cell = cells[run["model"]][label["skill"]][label["tier"]]
        cell[0] += float(run["passed"])
        cell[1] += 1
        latency[run["model"]].append((float(run["tokens_out"]), float(run["latency_ms"])))
        out_tokens[run["model"]].append(run["tokens_out"])
    endpoints = []
    for model_id, spec in models.items():
        if model_id not in cells:
            continue
        base, slope = _fit_latency(latency[model_id])
        endpoints.append({
            "id": model_id, "deployment": spec.deployment, "price_in": spec.price_in, "price_out": spec.price_out,
            "supports_tools": spec.tools, "verified_price": spec.verified, "api": spec.api,
            "mean_out_tokens": round(sum(out_tokens[model_id]) / len(out_tokens[model_id]), 1),
            "lat_base_ms": round(base, 1), "lat_per_token_ms": round(slope, 3),
            "cells": {skill: {tier: [cell[0], int(cell[1])] for tier, cell in tiers.items()} for skill, tiers in cells[model_id].items()},
        })
    return {"schema": 1, "version": f"measured-{tasks_hash}", "built": (built or date.today()).isoformat(),
            "source": {"tasks_hash": tasks_hash, "attempts": len(runs), "labelled_tasks": len(labels),
                       "note": "Pass counts by skill and tier from the capability benchmark; latency is a fit on short non-streaming calls."},
            "endpoints": endpoints}
