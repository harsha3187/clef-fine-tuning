"""The canonical record every source dataset is converted to.

A routing dataset says, for one query, how each candidate model did. The four public sources store that in four
different shapes (a row per query and model, a JSON file per model, a CSV per model and token budget). They all reduce
to this: one query, plus a list of observations. Response text is deliberately left out: it is large, and tier labels
need only the score, cost and token counts.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SOURCES = ("llmrouterbench", "xroutebench", "scope60k", "r2bench")
LICENSES = {
    "xroutebench": "not stated on the dataset card; prompts come from public benchmarks",
    "llmrouterbench": "not stated on the dataset page; prompts come from public benchmarks",
    "scope60k": "apache-2.0",
    "r2bench": "mit (rights to upstream prompts stay with their providers)",
}
_SPACE = re.compile(r"\s+")


def query_sha(text: str) -> str:
    """Hash of the query with whitespace and case folded, so the same question from two sources is recognised."""
    return hashlib.sha256(_SPACE.sub(" ", text.strip().lower()).encode("utf-8")).hexdigest()[:20]


@dataclass
class Observation:
    """One candidate model on one query. Only `model` and `quality` are always present."""

    model: str
    quality: float | None  # 0 to 1, from the source's own metric; None when the source gave no usable score
    cost_usd: float | None = None
    tokens_in: int | None = None
    tokens_out: int | None = None
    latency_s: float | None = None
    budget_tokens: int | None = None  # output-length budget the model was held to (R2-Bench); None means unconstrained
    runs: int = 1  # how many identical runs were averaged into this observation


@dataclass
class CanonicalRecord:
    id: str
    source: str
    dataset: str  # the sub-benchmark: gsm8k, tau2, swe-bench, ...
    query: str
    observations: list[Observation] = field(default_factory=list)
    split_origin: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    license: str = ""

    @property
    def sha(self) -> str:
        return query_sha(self.query)

    def to_json(self) -> str:
        data = asdict(self)
        data["sha"] = self.sha
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> CanonicalRecord:
        data = json.loads(line)
        data.pop("sha", None)
        data["observations"] = [Observation(**o) for o in data["observations"]]
        return cls(**data)

    def add(self, observation: Observation) -> None:
        """Add an observation; a repeat of the same model and budget is averaged in rather than listed twice."""
        for existing in self.observations:
            if existing.model == observation.model and existing.budget_tokens == observation.budget_tokens:
                total = existing.runs + observation.runs
                for name in ("quality", "cost_usd", "tokens_in", "tokens_out", "latency_s"):
                    old, new = getattr(existing, name), getattr(observation, name)
                    if old is None:
                        setattr(existing, name, new)
                    elif new is not None:
                        merged = (old * existing.runs + new * observation.runs) / total
                        setattr(existing, name, round(merged) if name.startswith("tokens") else merged)
                existing.runs = total
                return
        self.observations.append(observation)


def clean_quality(value: Any) -> float | None:
    """A score as 0 to 1, or None for missing values and the -1 some judges write for 'no score'."""
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if not isinstance(value, (int, float)) or value != value or value < 0:
        return None
    return min(1.0, float(value))


class Accumulator:
    """Collects records by query within one source and merges the observations that belong to the same query."""

    def __init__(self, source: str) -> None:
        self.source = source
        self._by_sha: dict[str, CanonicalRecord] = {}

    def add(self, dataset: str, query: str, observation: Observation, *, split: str | None = None,
            metadata: dict[str, Any] | None = None, identity: str | None = None) -> None:
        """`identity` groups rows the source itself says are one query, when the text differs between runs."""
        text = query.strip()
        if not text:
            return
        key = f"{dataset}:{identity or query_sha(text)}"
        record = self._by_sha.get(key)
        if record is None:
            record = CanonicalRecord(id=f"{self.source}/{dataset}/{(identity or query_sha(text))[:12]}", source=self.source, dataset=dataset,
                                     query=text, metadata=dict(metadata or {}), license=LICENSES[self.source])
            self._by_sha[key] = record
        if split and split not in record.split_origin:
            record.split_origin.append(split)
        record.add(observation)

    def records(self) -> Iterator[CanonicalRecord]:
        yield from self._by_sha.values()


def write_jsonl(records: Iterable[CanonicalRecord], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(record.to_json() + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> Iterator[CanonicalRecord]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield CanonicalRecord.from_json(line)
