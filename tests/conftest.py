"""Synthetic fixtures for offline routing tests; no bundled labels are promoted."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from routing_lab.backends import questions
from routing_lab.data import ROUTES, Contract, Record, write_records
from routing_lab.evaluation import sha256
from routing_lab.finetune_data import PARTITIONS, POLICY_VERSION, TRACKS, native_example
from routing_lab.taxonomy import taxonomy_routes


@pytest.fixture
def record_payload() -> dict[str, Any]:
    return {
        "id": "synthetic-1",
        "group_id": "task-1",
        "repository": "repo-1",
        "query": "Explain the synthetic scheduler.",
        "context": "The scheduler dispatches a synthetic job.",
        "expected_route": "repository_qa",
        "acceptable_routes": ["repository_qa"],
        "tags": ["synthetic"],
        "provenance": "original pytest fixture",
        "review_status": "reviewed",
        "reviewer": "synthetic test adjudicator",
    }


@pytest.fixture
def make_record(record_payload: dict[str, Any]) -> Callable[..., Record]:
    sequence = itertools.count()

    def make(**changes: Any) -> Record:
        number = next(sequence)
        payload = {
            **record_payload,
            "id": f"synthetic-{number}",
            "group_id": f"task-{number}",
            "repository": f"repo-{number}",
            "query": f"Explain synthetic component {number}.",
            **changes,
        }
        if "expected_route" in changes and "acceptable_routes" not in changes:
            payload["acceptable_routes"] = [changes["expected_route"]]
        return Record.model_validate(payload)

    return make


@pytest.fixture
def contracts() -> dict[str, Contract]:
    return {
        route: Contract(
            description=f"Synthetic owner for {route}",
            tools=["read"],
            can_write=False,
            success_criteria=f"Resolve a synthetic {route} task",
        )
        for route in ROUTES
    }


@pytest.fixture
def write_dataset(tmp_path: Path) -> Callable[..., Path]:
    def write(records: list[Record], name: str = "records.jsonl") -> Path:
        path = tmp_path / name
        write_records(path, records)
        return path

    return write


@pytest.fixture
def bundle(tmp_path, make_record):
    root = tmp_path / "bundle"
    root.mkdir()
    manifest = {
        "status": "complete", "policy_version": POLICY_VERSION, "context_mode": "context",
        "files": {}, "catalogs": {}, "tracks": {}, "partition_counts": {},
        "source": {"manifest_sha256": "a" * 64, "files": {"manifest.json": {"sha256": "a" * 64}}},
        "frozen_test10_exported": False, "frozen_test10": {},
    }
    mixed = {partition: [] for partition in PARTITIONS}
    for track in TRACKS:
        catalog = {route: Contract(description=f"Select {route}", tools=[], can_write=False,
                                   success_criteria="Expected intent") for route in taxonomy_routes(track)}
        name = f"{track}.contracts.json"
        (root / name).write_text(json.dumps({k: v.model_dump() for k, v in catalog.items()}),
                                 encoding="utf-8")
        manifest["files"][name] = {"sha256": sha256(root / name)}
        manifest["catalogs"][track] = {"file": name, **manifest["files"][name]}
        manifest["tracks"][track] = {"partitions": {}}
        frozen_name = f"{track}-test10.jsonl"
        manifest["frozen_test10"][track] = {"file": frozen_name, "sha256": "b" * 64}
        manifest["source"]["files"][frozen_name] = {"sha256": "b" * 64}
        for partition in PARTITIONS:
            rows = [make_record(taxonomy_id=track, expected_route=taxonomy_routes(track)[0])]
            name = f"{track}-{partition}.jsonl"
            write_records(root / name, rows)
            info = {"sha256": sha256(root / name), "count": len(rows)}
            manifest["files"][name] = info
            manifest["tracks"][track]["partitions"][partition] = info
            mixed[partition].extend(native_example(row, questions(catalog, taxonomy_id=track))
                                    for row in rows)
    for partition, rows in mixed.items():
        name = f"mixed-{partition}.jsonl"
        (root / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        manifest["files"][name] = {"sha256": sha256(root / name), "count": len(rows)}
        manifest["partition_counts"][partition] = len(rows)
    (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return root
