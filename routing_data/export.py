"""Canonical records to a fine-tuning bundle in the format `routing_lab.finetune_capability` trains on.

Pipeline: rank the models inside each source and sub-benchmark, give every query a tier, drop queries that appear in more
than one source, cap each sub-benchmark so no single one dominates, hold out whole sub-benchmarks for an out-of-distribution
test, and split the rest. Test rows are written next to the bundle, never inside it.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

from capability_lab.bundle import FRACTIONS, assign_splits
from routing_data.clean import CleanRules, exclude_reason
from routing_data.labels import TIERS, LabelRules, label_record, rank_pools
from routing_data.schema import SOURCES, read_jsonl
from routing_data.skills import dataset_key, skill_hint
from routing_lab.backends import CAPABILITY_PROMPT_VERSION, questions
from routing_lab.capability import CONTRACTS, capability_state
from routing_lab.data import load_contracts
from routing_lab.evaluation import sha256

# Sub-benchmarks never trained on: one or two per kind of work, from different sources, mixed difficulty.
HELD_OUT = ("gsm8k", "livecodebench", "tau2", "finqa", "bbh", "arenahardmath", "history", "boolq")
KIND = "capability-v1"


def labelled_queries(canonical: Path, rules: LabelRules, sources: tuple[str, ...] = SOURCES,
                     clean: CleanRules | None = None, dropped: Counter | None = None) -> list[dict[str, Any]]:
    """Every query that can be labelled, lightweight (no observations), in source priority order.

    With `clean`, queries failing a cleaning rule are skipped and counted in `dropped` by reason.
    """
    out: list[dict[str, Any]] = []
    for source in sources:
        path = canonical / f"{source}.jsonl"
        if not path.exists():
            continue
        records = list(read_jsonl(path))
        groups = rank_pools(records, rules)
        for record in records:
            if clean is not None:
                reason = exclude_reason(record, groups.get((record.source, record.dataset)), rules, clean)
                if reason:
                    if dropped is not None:
                        dropped[f"{source}: {reason}"] += 1
                    continue
            label = label_record(record, groups, rules)
            if label is not None:
                out.append({"id": record.id, "source": source, "dataset": record.dataset, "family": dataset_key(record.dataset),
                            "query": record.query, "sha": record.sha, "label": asdict(label),
                            "skill_hint": skill_hint(record.dataset), "license": record.license})
        del records
    return out


def dedupe(queries: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the first copy of each query across sources (priority order); count what was dropped."""
    seen: set[str] = set()
    kept = []
    for q in queries:
        if q["sha"] in seen:
            continue
        seen.add(q["sha"])
        kept.append(q)
    return kept, len(queries) - len(kept)


# R2-Bench is one mixed collection (RouterArena), not a single task, so it gets a larger share than a sub-benchmark.
MIXTURE_CAPS = {"routerarena": 10}


def cap_per_dataset(queries: list[dict[str, Any]], cap: int, seed: int) -> list[dict[str, Any]]:
    """At most `cap` queries per sub-benchmark (a multiple of it for mixed collections), chosen by a seeded hash."""
    by: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for q in queries:
        by[(q["source"], q["family"])].append(q)
    out = []
    for items in by.values():
        items.sort(key=lambda q: hashlib.sha256(f"{seed}:{q['sha']}".encode()).hexdigest())
        out.extend(items[:cap * MIXTURE_CAPS.get(items[0]["family"], 1)])
    return out


def to_row(q: dict[str, Any], catalog: dict, *, max_chars: int, use_skill: bool) -> dict[str, Any]:
    query = q["query"] if len(q["query"]) <= max_chars else q["query"][:max_chars] + "\n[request truncated]"
    return {"id": q["id"], "family": q["family"], "level": 0, "state": capability_state(query), "questions": catalog,
            "answers": {"route": q["label"]["tier"], "skill": q["skill_hint"] if use_skill else None},
            "source": q["source"], "pass_share": round(q["label"]["pass_share"], 3)}


def _write(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return {"sha256": sha256(path), "count": len(rows)}


def export(canonical: Path, out: Path, *, rules: LabelRules = LabelRules(), cap: int = 150, max_chars: int = 3000,  # noqa: B008
           held_out: tuple[str, ...] = HELD_OUT, seed: int = 7, use_skill: bool = False,
           sources: tuple[str, ...] = SOURCES, clean: CleanRules | None = None) -> dict[str, Any]:
    cleaned: Counter = Counter()
    queries = labelled_queries(canonical, rules, sources, clean, cleaned)
    unique, dropped = dedupe(queries)
    capped = cap_per_dataset(unique, cap, seed)
    catalog = questions(load_contracts(CONTRACTS, "capability-v1"), taxonomy_id="capability-v1", skills=True)
    seen_states: set[str] = set()  # two long queries can share the same first max_chars; keep one so no text repeats across splits
    distinct = []
    for q in capped:
        state = to_row(q, catalog, max_chars=max_chars, use_skill=use_skill)["state"]
        if state not in seen_states:
            seen_states.add(state)
            distinct.append(q)
    truncated_duplicates = len(capped) - len(distinct)
    capped = distinct
    pseudo = [{"task_id": q["id"], "family": q["family"], "tier": q["label"]["tier"]} for q in capped]
    splits = assign_splits(pseudo, held_out, seed, FRACTIONS)
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in capped:
        rows[splits[q["id"]]].append(to_row(q, catalog, max_chars=max_chars, use_skill=use_skill))
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.iterdir():
        stale.unlink()
    files = {f"{name}.jsonl": _write(out / f"{name}.jsonl", rows[name]) for name in ("train", "validation", "calibration")}
    heldout_dir = out.parent / f"{out.name}-heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    heldout = {name: _write(heldout_dir / f"{name}.jsonl", rows[name]) for name in ("test_id", "test_ood")}
    tier_counts = {name: dict(Counter(r["answers"]["route"] for r in rows[name])) for name in (*files_names(), "test_id", "test_ood")}
    manifest = {
        "kind": KIND, "status": "complete", "files": files, "tier_counts": tier_counts,
        "held_out_families": sorted(held_out), "held_out_counts": {k: v["count"] for k, v in heldout.items()},
        "test_data_exported": False, "split_seed": seed, "extra_training_sets": [], "evaluation_only_sets": [],
        "label_source": {
            "kind": "external public routing datasets", "sources": list(sources), "rules": asdict(rules),
            "prompt_version": CAPABILITY_PROMPT_VERSION, "per_dataset_cap": cap, "max_state_chars": max_chars,
            "skill_labels": "dataset-level hint" if use_skill else "none (skill loss is skipped for these rows)",
            "cleaning": asdict(clean) if clean else None, "dropped_by_cleaning": dict(cleaned), "duplicates_dropped": dropped, "same_text_after_truncation_dropped": truncated_duplicates,
            "licenses": {s: next((q["license"] for q in unique if q["source"] == s), "") for s in sources},
            "note": "Rights to the original prompts remain with the upstream benchmark providers; use for research only.",
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    report = {"dropped_by_cleaning": dict(cleaned), "labelled": len(queries), "after_dedupe": len(unique), "after_cap": len(capped), "duplicates_dropped": dropped,
              "tier_counts": tier_counts, "files": {k: v["count"] for k, v in {**files, **heldout}.items()},
              "by_source": dict(Counter(q["source"] for q in capped)), "held_out_families": sorted(held_out)}
    return report


def files_names() -> tuple[str, str, str]:
    return ("train", "validation", "calibration")


def tier_table(canonical: Path, rules: LabelRules = LabelRules(), sources: tuple[str, ...] = SOURCES) -> dict[str, Any]:  # noqa: B008
    """Tier mix per source and sub-benchmark, before dedupe and capping, for a sanity check of the labels."""
    table: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for q in labelled_queries(canonical, rules, sources):
        table[q["source"]][q["dataset"]][q["label"]["tier"]] += 1
    return {s: {d: {t: c.get(t, 0) for t in TIERS} for d, c in sorted(ds.items())} for s, ds in table.items()}
