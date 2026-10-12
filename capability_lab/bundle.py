"""Turn measured tier labels into a fine-tuning bundle and fixed evaluation splits.

Two held-out sets exist and neither enters the bundle:
- test_id: tasks from training families, never trained on (templates are shared, so this is the optimistic number).
- test_ood: whole families the router never sees, the honest check that it learned difficulty, not templates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from routing_lab.backends import questions
from routing_lab.capability import CONTRACTS, capability_state
from routing_lab.data import load_contracts
from routing_lab.evaluation import sha256

# One family per skill at least, and a mix of easy and hard, so the out-of-distribution score means something.
HELD_OUT_FAMILIES = ("schedule_csp", "summarize_facts", "code_bug_line", "tool_selection", "json_transform", "bit_ops")
FRACTIONS = (("train", 0.70), ("validation", 0.10), ("calibration", 0.10), ("test_id", 0.10))
TRAINING_ONLY = (("train", 0.70), ("validation", 0.15), ("calibration", 0.15))  # extra sets whose evaluation lives elsewhere
KIND = "capability-v1"


def _bucket(task_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()


def assign_splits(labels: list[dict[str, Any]], held_out: tuple[str, ...] = HELD_OUT_FAMILIES, seed: int = 7,
                  fractions: tuple[tuple[str, float], ...] = FRACTIONS) -> dict[str, str]:
    """Deterministic split, stratified by family and tier so every partition sees every kind of task."""
    splits: dict[str, str] = {}
    groups: dict[tuple[str, str], list[str]] = {}
    for label in labels:
        if label["family"] in held_out:
            splits[label["task_id"]] = "test_ood"
        else:
            groups.setdefault((label["family"], label["tier"]), []).append(label["task_id"])
    for ids in groups.values():
        ids.sort(key=lambda task_id: _bucket(task_id, seed))
        for index, task_id in enumerate(ids):
            position, edge = (index + 0.5) / len(ids), 0.0
            for name, share in fractions:
                edge += share
                if position < edge:
                    splits[task_id] = name
                    break
    return splits


def build_bundle(tasks: list, labels: list[dict[str, Any]], out: Path, *, tasks_hash: str, thresholds: dict[str, float],
                 endpoints: list[str], held_out: tuple[str, ...] = HELD_OUT_FAMILIES, seed: int = 7,
                 splits: dict[str, str] | None = None, forbidden_prompts: set[str] | None = None,
                 extra_sets: list[dict[str, str]] | None = None, test_sets: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Write train, validation and calibration rows plus a hashed manifest; test rows are never written."""
    splits = splits if splits is not None else assign_splits(labels, held_out, seed)
    by_id = {t.id: t for t in tasks}
    catalog = questions(load_contracts(CONTRACTS, "capability-v1"), taxonomy_id="capability-v1", skills=True)
    if forbidden_prompts and any(by_id[t].prompt in forbidden_prompts for t, s in splits.items() if s in ("train", "validation", "calibration")):
        raise ValueError("A bundle task also appears in a held-out evaluation set")
    rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "calibration": []}
    for label in labels:
        split = splits[label["task_id"]]
        if split not in rows:
            continue
        task = by_id[label["task_id"]]
        rows[split].append({"id": task.id, "family": task.family, "level": task.level, "state": capability_state(task.prompt),
                            "questions": catalog, "answers": {"route": label["tier"], "skill": task.skill}})
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        stale.unlink()
    files = {}
    for name, items in rows.items():
        path = out / f"{name}.jsonl"
        path.write_text("".join(json.dumps(item) + "\n" for item in items), encoding="utf-8")
        files[f"{name}.jsonl"] = {"sha256": sha256(path), "count": len(items)}
    tier_counts = {name: {tier: sum(r["answers"]["route"] == tier for r in items) for tier in ("low", "medium", "high")}
                   for name, items in rows.items()}
    held = {"test_id": sum(s == "test_id" for s in splits.values()), "test_ood": sum(s == "test_ood" for s in splits.values())}
    manifest = {"kind": KIND, "status": "complete", "files": files, "tier_counts": tier_counts, "held_out_counts": held,
                "held_out_families": list(held_out), "test_data_exported": False, "split_seed": seed,
                "extra_training_sets": extra_sets or [], "evaluation_only_sets": test_sets or [],
                "label_source": {"tasks_hash": tasks_hash, "thresholds": thresholds, "endpoints": endpoints,
                                "rule": "tier from the share of endpoints that solve the task"}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
