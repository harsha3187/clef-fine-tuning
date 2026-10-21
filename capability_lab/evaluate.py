"""Evaluate Clef as a capability router against measured outcomes.

For every labelled task, Clef is asked for tier and skill probabilities from the request text alone. A selector
then picks an endpoint with the production selector (routing_lab.selector), using measured pass rates that exclude
the task being routed (leave-one-out). The same selector runs with other sources of tier knowledge, so the only
thing that changes between arms is who tells it how hard the request is:

* always one fixed endpoint (the largest, the best value, the cheapest)
* a naive heuristic (request length) as the tier source
* Clef's probabilities
* the true measured tier (an upper bound: perfect knowledge)
* the oracle that picks the cheapest passing endpoint per task (cannot be deployed)
"""

from __future__ import annotations

import json
import math
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from capability_lab.analysis import TIERS, pass_matrix
from capability_lab.tasks import Task
from routing_lab.selector import Constraints, Endpoint, Settings, select

ROUTER_COST_USD = 0.0004  # estimated GPU time per Clef decision at full utilization; see docs/router.md
MIN_CELL = 4


def ask_clef(router: Any, tasks: list[Task], out: Path, *, workers: int = 4) -> dict[str, dict[str, Any]]:
    """Clef's answer for each task, cached on disk so a rerun only asks for what is missing."""
    cached: dict[str, dict[str, Any]] = {}
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cached[row["task_id"]] = row
    todo = [t for t in tasks if t.id not in cached]
    lock = threading.Lock()

    def one(task: Task) -> dict[str, Any]:
        result = router.classify(task.prompt)
        return {"task_id": task.id, **result.to_dict()}

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(one, todo):
            with lock:
                handle.write(json.dumps(row) + "\n")
                cached[row["task_id"]] = row
    return cached


def tier_metrics(labels: list[dict[str, Any]], predictions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pairs = [(label, predictions[label["task_id"]]) for label in labels if label["task_id"] in predictions]
    n = len(pairs)
    confusion = {t: {p: 0 for p in TIERS} for t in TIERS}
    brier = log_loss = 0.0
    bins: list[list[tuple[float, bool]]] = [[] for _ in range(10)]
    skill_hits = 0
    for label, pred in pairs:
        confusion[label["tier"]][pred["tier"]] += 1
        probs = pred["tier_probabilities"]
        brier += sum((probs[t] - (t == label["tier"])) ** 2 for t in TIERS)
        log_loss += -math.log(max(probs[label["tier"]], 1e-9))
        bins[min(9, int(pred["confidence"] * 10))].append((pred["confidence"], pred["tier"] == label["tier"]))
        skill_hits += pred["skill"] == label["skill"]
    accuracy = sum(confusion[t][t] for t in TIERS) / n if n else None
    f1s = []
    for t in TIERS:
        tp = confusion[t][t]
        precision = tp / max(1, sum(confusion[o][t] for o in TIERS))
        recall = tp / max(1, sum(confusion[t].values()))
        f1s.append(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall))
    ece = sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(ok for _, ok in b) / len(b)) for b in bins if b) if n else None
    # being one tier off is much less harmful than two: report how often Clef is within one tier, and under-estimates
    order = {t: i for i, t in enumerate(TIERS)}
    under = sum(order[p["tier"]] < order[label["tier"]] for label, p in pairs)
    over = sum(order[p["tier"]] > order[label["tier"]] for label, p in pairs)
    return {"tasks": n, "tier_accuracy": round(accuracy, 4) if accuracy is not None else None, "macro_f1": round(sum(f1s) / 3, 4),
            "confusion": confusion, "brier": round(brier / n, 4) if n else None, "log_loss": round(log_loss / n, 4) if n else None,
            "ece": round(ece, 4) if ece is not None else None, "skill_accuracy": round(skill_hits / n, 4) if n else None,
            "underestimates": under, "overestimates": over,
            "majority_class_accuracy": round(max(sum(confusion[t].values()) for t in TIERS) / n, 4) if n else None}


def heuristic_probs(prompt: str) -> dict[str, float]:
    """A deliberately naive tier guess from request length alone, as the baseline Clef must beat."""
    tier = "low" if len(prompt) < 350 else "medium" if len(prompt) < 900 else "high"
    return {t: float(t == tier) for t in TIERS}


class Cells:
    """Measured pass rates by (endpoint, skill, tier), with pooled fallbacks and the ability to leave one task out."""

    def __init__(self, labels: list[dict[str, Any]], matrix: dict[str, dict[str, dict[str, float]]]) -> None:
        self.by_cell: dict[tuple[str, str, str], list[float]] = defaultdict(lambda: [0.0, 0])
        self.by_tier: dict[tuple[str, str], list[float]] = defaultdict(lambda: [0.0, 0])
        self.label_of = {label["task_id"]: label for label in labels}
        for task_id, row in matrix.items():
            label = self.label_of.get(task_id)
            if label:
                for model, v in row.items():
                    for key, store in (((model, label["skill"], label["tier"]), self.by_cell), ((model, label["tier"]), self.by_tier)):
                        store[key][0] += v["rate"]
                        store[key][1] += 1
        self.matrix = matrix

    def rate(self, model: str, skill: str | None, tier: str, leave_out: str | None = None) -> float | None:
        own = self.matrix.get(leave_out, {}).get(model) if leave_out else None
        own_label = self.label_of.get(leave_out) if leave_out else None

        def read(store: dict, key: tuple, belongs: bool) -> tuple[float, int]:
            passes, n = store[key]
            if own is not None and belongs:
                passes, n = passes - own["rate"], n - 1
            return passes, n

        if skill is not None:
            passes, n = read(self.by_cell, (model, skill, tier), bool(own_label and own_label["skill"] == skill and own_label["tier"] == tier))
            if n >= MIN_CELL:
                return passes / n
        passes, n = read(self.by_tier, (model, tier), bool(own_label and own_label["tier"] == tier))
        return passes / n if n >= MIN_CELL else None


def proxy_endpoints(matrix: dict[str, dict[str, dict[str, float]]], models: list[str]) -> list[Endpoint]:
    """Endpoints whose only price is their measured mean cost per task, so offline cost equals what the runs billed."""
    out = []
    for m in models:
        cost = sum(matrix[t][m]["cost"] for t in matrix if m in matrix[t]) / sum(m in matrix[t] for t in matrix)
        out.append(Endpoint(m, m, 0.0, cost * 1e6, True, True, 1.0, 0.0, 0.0))
    return out


PROXY_CONSTRAINTS = Constraints(input_tokens=0, output_tokens=1)


def choose(endpoints: list[Endpoint], cells: Cells, task_id: str, tier_probs: dict[str, float],
           skill_probs: dict[str, float] | None, candidates: list[str], settings: Settings,
           constraints: Constraints = PROXY_CONSTRAINTS) -> str:
    """The production selector, fed measured rates that leave the routed task out."""
    pool = [e for e in endpoints if e.id in candidates]
    selection = select(pool, lambda model, skill, tier: cells.rate(model, skill, tier, leave_out=task_id),
                       tier_probs, skill_probs, settings, constraints)
    if selection.choice is not None:
        return selection.choice
    return max(pool, key=lambda e: e.price_out).id  # nothing measured: be safe and take the costliest


def evaluate_arms(tasks: list[Task], labels: list[dict[str, Any]], runs: list[dict[str, Any]],
                  predictions: dict[str, dict[str, Any]], *, floor: float = 0.85, router_cost: float = ROUTER_COST_USD,
                  settings: Settings | None = None, endpoints: list[Endpoint] | None = None,
                  constraints: Constraints = PROXY_CONSTRAINTS) -> dict[str, Any]:
    matrix = pass_matrix(runs)
    cells = Cells(labels, matrix)
    prompt_of = {t.id: t.prompt for t in tasks}
    rows = [label for label in labels if label["task_id"] in predictions and label["task_id"] in matrix]
    models = sorted({m for label in rows for m in matrix[label["task_id"]]})
    mean_cost = {m: sum(matrix[t][m]["cost"] for t in matrix if m in matrix[t]) / sum(m in matrix[t] for t in matrix) for m in models}
    settings = settings or Settings(floor=floor, cost_weight=1.0, latency_weight=0.0)
    pool = endpoints or proxy_endpoints(matrix, models)
    if endpoints is not None:
        constraints = Constraints(needs_tools=False, input_tokens=constraints.input_tokens, output_tokens=constraints.output_tokens)
    arms: dict[str, dict[str, Any]] = {}

    def tally(name: str, pick, extra_cost: float = 0.0) -> None:
        passed = cost = 0.0
        used: dict[str, int] = defaultdict(int)
        n = 0
        for label in rows:
            tid = label["task_id"]
            candidates = list(matrix[tid])
            model = pick(label, candidates)
            if model not in matrix[tid]:
                continue
            n += 1
            passed += matrix[tid][model]["rate"]
            cost += matrix[tid][model]["cost"] + extra_cost
            used[model] += 1
        if n:
            arms[name] = {"tasks": n, "pass_rate": round(passed / n, 4), "mean_cost_usd": round(cost / n, 6), "models_used": dict(used)}

    def route(label: dict[str, Any], candidates: list[str], tiers: dict[str, float], skills: dict[str, float] | None) -> str:
        return choose(pool, cells, label["task_id"], tiers, skills, candidates, settings, constraints)

    overall = {m: sum(matrix[tid][m]["rate"] for tid in matrix if m in matrix[tid]) / sum(m in matrix[tid] for tid in matrix) for m in models}
    # "best value": the cheapest endpoint that is within three points of the best overall pass rate
    best_value = min((m for m in models if overall[m] >= max(overall.values()) - 0.03), key=lambda m: mean_cost[m])
    for fixed in (max(models, key=lambda m: mean_cost[m]), best_value, min(models, key=lambda m: mean_cost[m])):
        tally(f"always {fixed}", lambda label, cands, f=fixed: f if f in cands else cands[0])
    tally("length heuristic", lambda label, cands: route(label, cands, heuristic_probs(prompt_of[label["task_id"]]), None))
    tally("Clef (tier and skill)", lambda label, cands: route(label, cands, predictions[label["task_id"]]["tier_probabilities"],
                                                              predictions[label["task_id"]]["skill_probabilities"]), router_cost)
    tally("Clef (tier only)", lambda label, cands: route(label, cands, predictions[label["task_id"]]["tier_probabilities"], None), router_cost)
    tally("true tier and skill (upper bound)", lambda label, cands: route(label, cands, {label["tier"]: 1.0}, {label["skill"]: 1.0}))
    tally("oracle cheapest passing", lambda label, cands: label["cheapest_passing"] or max(cands, key=lambda m: mean_cost[m]))
    return {"floor": settings.floor, "settings": {"floor": settings.floor, "cost_weight": settings.cost_weight,
                                                    "latency_weight": settings.latency_weight},
            "router_cost_usd": router_cost, "arms": arms, "best_value_endpoint": best_value, "tasks": len(rows)}
