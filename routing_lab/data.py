"""Validate routing records and create leakage-resistant review splits."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from routing_lab.taxonomy import TaxonomyId, taxonomy_for_routes, taxonomy_routes

ROUTES = taxonomy_routes("coding-v1")
Route = Literal[
    "debugging", "code_generation", "test_generation",
    "code_review", "repository_qa", "fallback",
]


class Record(BaseModel):
    """One proposed or reviewed initial-owner decision."""

    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(min_length=1)
    group_id: str = Field(min_length=1)
    repository: str = Field(min_length=1)
    query: str = Field(min_length=1)
    context: str
    taxonomy_id: TaxonomyId = "coding-v1"
    expected_route: str
    acceptable_routes: list[str] = Field(min_length=1)
    tags: list[str]
    provenance: str = Field(min_length=1)
    review_status: Literal["unreviewed", "reviewed"]
    reviewer: str

    @field_validator("id", "group_id", "repository", "provenance")
    @classmethod
    def nonblank_identifier(cls, value: str) -> str:
        if not value.strip() or value != value.strip():
            raise ValueError("Identifiers must be nonblank without surrounding whitespace")
        return value

    @model_validator(mode="after")
    def check_review(self) -> Record:
        routes = taxonomy_routes(self.taxonomy_id)
        if self.expected_route not in routes or any(r not in routes for r in self.acceptable_routes):
            raise ValueError(f"Routes must belong to taxonomy {self.taxonomy_id}")
        if self.expected_route not in self.acceptable_routes:
            raise ValueError("Expected route must be an acceptable route")
        if len(set(self.acceptable_routes)) != len(self.acceptable_routes):
            raise ValueError("Acceptable routes must be unique")
        if self.review_status == "reviewed" and not self.reviewer.strip():
            raise ValueError("Reviewed records require a named reviewer")
        if not self.query.strip():
            raise ValueError("Query must not be whitespace")
        return self


class Contract(BaseModel):
    """Capabilities describe an agent; they never grant runtime permissions."""

    model_config = ConfigDict(extra="forbid", strict=True)
    description: str = Field(min_length=1)
    tools: list[str]
    can_write: bool
    success_criteria: str = Field(min_length=1)


def load_contracts(path: Path, taxonomy_id: str = "coding-v1") -> dict[str, Contract]:
    """Load a complete contract catalog for the explicitly selected taxonomy."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Contracts must be a route-to-contract object")
    taxonomy_for_routes(raw, taxonomy_id)
    return {key: Contract.model_validate(value) for key, value in raw.items()}


def records_taxonomy(records: list[Record], taxonomy_id: str | None = None) -> TaxonomyId:
    """Reject mixed datasets and explicit selections that disagree with records."""
    if not records:
        raise ValueError("Need nonempty records to determine taxonomy")
    selected = records[0].taxonomy_id
    if any(record.taxonomy_id != selected for record in records):
        raise ValueError("Mixed taxonomies are not supported in one dataset or experiment")
    if taxonomy_id is not None:
        taxonomy_routes(taxonomy_id)
        if selected != taxonomy_id:
            raise ValueError(f"Record taxonomy {selected} does not match selected taxonomy {taxonomy_id}")
    return selected


def load_records(
    path: Path, *, reviewed_only: bool = False, taxonomy_id: str | None = None,
) -> list[Record]:
    """Reject malformed records and duplicate identities, not entire lines silently."""
    records = [
        Record.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").split("\n") if line.strip()
    ]
    if not records:
        raise ValueError(f"Empty dataset: {path}")
    if len({r.id for r in records}) != len(records):
        raise ValueError("Duplicate record IDs")
    records_taxonomy(records, taxonomy_id)
    if reviewed_only and any(r.review_status != "reviewed" for r in records):
        raise ValueError("Benchmark requires reviewed records; use probe mode for drafts")
    return records


def write_records(path: Path, records: list[Record]) -> None:
    """Write JSONL without replacing an existing dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        for record in records:
            stream.write(record.model_dump_json() + "\n")


def fingerprint(record: Record) -> str:
    """Find exact normalized duplicates even when IDs differ."""
    text = " ".join((record.query + "\n" + record.context).lower().split())
    return hashlib.sha256(text.encode()).hexdigest()


def assert_disjoint(left: list[Record], right: list[Record]) -> None:
    """Reject identity, task, repository, and exact-text overlap."""
    if left and right:
        records_taxonomy(left + right)
    for name, getter in (
        ("id", lambda r: r.id),
        ("group", lambda r: r.group_id),
        ("repository", lambda r: r.repository),
        ("normalized input", fingerprint),
    ):
        if {getter(r) for r in left} & {getter(r) for r in right}:
            raise ValueError(f"Data leakage: overlapping {name}")


def split_reviewed(records: list[Record], seed: int = 42) -> dict[str, list[Record]]:
    """Split connected repositories/tasks/duplicates, never their paraphrases.

    Uses approximately 60/20/20 of connected groups. Class balance is reported by
    callers rather than fabricated by breaking group boundaries.
    """
    if any(r.review_status != "reviewed" for r in records):
        raise ValueError("Review and adjudicate records before freezing splits")
    if records:
        records_taxonomy(records)
    parent = list(range(len(records)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, record in enumerate(records):
        for key in (
            ("repository", record.repository),
            ("group", record.group_id),
            ("text", fingerprint(record)),
        ):
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups: dict[int, list[Record]] = {}
    for i, record in enumerate(records):
        groups.setdefault(root(i), []).append(record)
    blocks = list(groups.values())
    if len(blocks) < 5:
        raise ValueError("Need at least five independent repository/task groups")
    random.Random(seed).shuffle(blocks)
    test_count = max(1, round(len(blocks) * 0.2))
    dev_count = max(1, round(len(blocks) * 0.2))
    boundaries = {
        "test": blocks[:test_count],
        "dev": blocks[test_count:test_count + dev_count],
        "train": blocks[test_count + dev_count:],
    }
    return {name: [r for block in subset for r in block]
            for name, subset in boundaries.items()}
