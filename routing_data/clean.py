"""Cleaning rules applied before tier labelling, each with a stated reason so the drops can be counted and audited.

A label that says "models failed this" is only worth training on when failing means the request was hard, not that the
score was meaningless. These rules remove the cases where the score is not a reliable difficulty signal:

- subjective datasets: pairwise preference (arena-hard) and emotion labels with no clear answer (emorynlp, meld);
- judge-scored queries with partial credit from most models (open-ended requests, where there is no one right answer);
- queries every model fails, in a sub-benchmark where even the strongest models average very little: there a zero is as likely to
  be an answer-format miss as a hard problem.
"""

from __future__ import annotations

from dataclasses import dataclass

from routing_data.labels import LabelRules, Pool, usable
from routing_data.schema import CanonicalRecord
from routing_data.skills import dataset_key

SUBJECTIVE_DATASETS = frozenset({"arenahard", "arenahardcoding", "arenahardcreativewriting", "arenahardmath", "emorynlp", "meld"})
JUDGED_SOURCES = frozenset({"r2bench"})  # scores come from an LLM judge, not an exact answer


@dataclass(frozen=True)
class CleanRules:
    drop_subjective: bool = True
    partial_credit_share: float = 0.4  # judged queries where this share of scores are partial are open-ended
    unmeasurable_threshold: float = 0.25  # a pool whose solve threshold is below this is too hard to read a zero from
    drop_all_fail_in_hard_pools: bool = True


def exclude_reason(record: CanonicalRecord, pool: Pool | None, labels: LabelRules, rules: CleanRules) -> str | None:
    """The reason a query should not be trained on, or None to keep it."""
    if rules.drop_subjective and dataset_key(record.dataset) in SUBJECTIVE_DATASETS:
        return "subjective dataset"
    observed = usable(record, labels)
    if not observed:
        return None
    if record.source in JUDGED_SOURCES:
        partial = sum(0.02 < o.quality < 0.98 for o in observed) / len(observed)
        if partial >= rules.partial_credit_share:
            return "judge gave partial credit (open-ended)"
    if (rules.drop_all_fail_in_hard_pools and pool is not None and pool.threshold < rules.unmeasurable_threshold
            and max(o.quality for o in observed) <= 0.0):
        return "every model scored 0 in a pool where the best models barely score"
    return None
