"""Tier labels from model outcomes, and a skill hint from the dataset name.

A query is not "hard" in the abstract; it is hard for a class of model. Within each source and sub-benchmark, the models are
ranked by their own mean score and split into thirds: weak, middle and strong. A query is then:

- low    if most of the weak third solve it (a small model is enough),
- medium if most of the middle or the strong third solve it but the weak third does not,
- high   if even the strong third mostly fails it.

"Solve" means a score at or above `pass_at`, or, on benchmarks whose metric is strict (exact match, token F1) so that even the strongest models rarely score high, at or above `relative_pass` of what the strongest third scores on average. This stops a strict metric from making every query look unsolvable.
"Solve" is judged per query on the observed models. This is the same idea as the benchmark labels built on this project's own ten
endpoints (how much model it takes), made comparable across pools of different strength by ranking inside each pool.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from routing_data.schema import CanonicalRecord, Observation

TIERS = ("low", "medium", "high")


@dataclass(frozen=True)
class LabelRules:
    pass_at: float = 0.7
    relative_pass: float = 0.8  # on strict metrics: this share of the strongest third's mean score counts as solved
    low_share: float = 0.5  # share of the weak third that must pass for "low"
    other_share: float = 0.5  # share of the middle or strong third that must pass for "medium"
    min_models: int = 6  # a pool needs this many ranked models, or its queries stay unlabelled
    min_per_group: int = 2  # observations a query needs from each third
    r2_budget: int = 2000  # R2-Bench: label from the generous output budget


@dataclass
class Pool:
    groups: dict[str, str]  # model -> weak, middle or strong
    threshold: float  # score that counts as solved in this sub-benchmark


@dataclass
class Label:
    tier: str
    pass_share: float  # share of all observed models that pass, for transparency and alternative rules
    weak: float
    middle: float
    strong: float
    models: int


def usable(record: CanonicalRecord, rules: LabelRules) -> list[Observation]:
    """Observations that count: scored, and for R2-Bench only the labelling budget."""
    out = []
    for o in record.observations:
        if o.quality is None:
            continue
        if record.source == "r2bench" and o.budget_tokens != rules.r2_budget:
            continue
        out.append(o)
    return out


def rank_pools(records: Iterable[CanonicalRecord], rules: LabelRules) -> dict[tuple[str, str], Pool]:
    """For each (source, dataset), the group of every model with enough observations and the score that counts as solved."""
    sums: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0.0]))
    sizes: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        key = (record.source, record.dataset)
        sizes[key] += 1
        for o in usable(record, rules):
            cell = sums[key][o.model]
            cell[0] += o.quality
            cell[1] += 1
    groups: dict[tuple[str, str], Pool] = {}
    for key, per_model in sums.items():
        ranked = sorted(((total / n, model) for model, (total, n) in per_model.items() if n >= 0.5 * sizes[key]))
        if len(ranked) < rules.min_models:
            continue
        third = math.ceil(len(ranked) / 3)
        members = {model: "weak" if i < third else "strong" if i >= len(ranked) - third else "middle"
                   for i, (_, model) in enumerate(ranked)}
        strong_mean = sum(score for score, _ in ranked[-third:]) / third
        groups[key] = Pool(members, max(0.05, min(rules.pass_at, rules.relative_pass * strong_mean)))
    return groups


def label_record(record: CanonicalRecord, groups: dict[tuple[str, str], Pool], rules: LabelRules) -> Label | None:
    pool = groups.get((record.source, record.dataset))
    if pool is None:
        return None
    passed: dict[str, list[bool]] = defaultdict(list)
    for o in usable(record, rules):
        if o.model in pool.groups:
            passed[pool.groups[o.model]].append(o.quality >= pool.threshold)
    if any(len(passed[g]) < rules.min_per_group for g in ("weak", "middle", "strong")):
        return None
    share = {g: sum(v) / len(v) for g, v in passed.items()}
    everyone = [x for v in passed.values() for x in v]
    if share["weak"] >= rules.low_share:
        tier = "low"
    elif max(share["middle"], share["strong"]) >= rules.other_share:
        tier = "medium"
    else:
        tier = "high"
    return Label(tier, sum(everyone) / len(everyone), share["weak"], share["middle"], share["strong"], len(everyone))
