"""Strict dataset validation and transitive leakage boundaries."""

from __future__ import annotations

import json
from collections.abc import Callable
from itertools import combinations
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from routing_lab.data import (
    ROUTES, Contract, Record, assert_disjoint, fingerprint, load_contracts,
    load_records, split_reviewed, write_records,
)


@pytest.mark.parametrize(
    "changes",
    [
        {"unexpected": "extra"},
        {"id": 1},
        {"id": ""},
        {"id": " padded"},
        {"group_id": " "},
        {"repository": "repo "},
        {"provenance": "\t"},
        {"query": "\n "},
        {"context": None},
        {"expected_route": "unknown"},
        {"acceptable_routes": []},
        {"acceptable_routes": ["debugging"]},
        {"acceptable_routes": ["repository_qa", "repository_qa"]},
        {"acceptable_routes": ["repository_qa", "unknown"]},
        {"tags": "synthetic"},
        {"tags": [1]},
        {"review_status": "approved"},
        {"reviewer": " "},
    ],
)
def test_record_rejects_invalid_payload(
    record_payload: dict[str, Any], changes: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        Record.model_validate({**record_payload, **changes})


def test_missing_required_field(record_payload: dict[str, Any]) -> None:
    del record_payload["provenance"]
    with pytest.raises(ValidationError):
        Record.model_validate(record_payload)


def test_draft_loads_only_outside_reviewed_gate(
    make_record: Callable[..., Record], write_dataset: Callable[..., Path],
) -> None:
    draft = make_record(review_status="unreviewed", reviewer="")
    path = write_dataset([draft])
    assert load_records(path) == [draft]
    with pytest.raises(ValueError, match="reviewed"):
        load_records(path, reviewed_only=True)
    with pytest.raises(ValueError, match="Review"):
        split_reviewed([draft])


@pytest.mark.parametrize("text", ["", " \n\t\n", '{"id":', "null"])
def test_empty_or_malformed_jsonl_is_rejected(tmp_path: Path, text: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_records(path)


def test_duplicate_ids_and_nonoverwriting_writer(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    first, second = make_record(id="same"), make_record(id="same")
    path = tmp_path / "dataset.jsonl"
    write_records(path, [first, second])
    original = path.read_bytes()
    with pytest.raises(ValueError, match="Duplicate"):
        load_records(path)
    with pytest.raises(FileExistsError):
        write_records(path, [first])
    assert path.read_bytes() == original


def test_jsonl_round_trip_and_blank_lines(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    record = make_record(query="Explain café scheduler.")
    path = tmp_path / "unicode.jsonl"
    path.write_text("\n" + record.model_dump_json() + "\n \n", encoding="utf-8")
    assert load_records(path, reviewed_only=True) == [record]


@pytest.mark.parametrize("mutation", ["missing", "extra", "wrong_type", "coercion"])
def test_contract_taxonomy_and_strict_fields(
    contracts: dict[str, Contract], tmp_path: Path, mutation: str,
) -> None:
    payload: Any = {key: value.model_dump() for key, value in contracts.items()}
    if mutation == "missing":
        payload.pop(ROUTES[0])
    elif mutation == "extra":
        payload["unknown"] = payload[ROUTES[0]]
    elif mutation == "wrong_type":
        payload = list(payload)
    else:
        payload[ROUTES[0]]["can_write"] = "false"
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        load_contracts(path)


def test_valid_contracts_round_trip(contracts: dict[str, Contract], tmp_path: Path) -> None:
    path = tmp_path / "contracts.json"
    path.write_text(
        json.dumps({key: value.model_dump() for key, value in contracts.items()}),
        encoding="utf-8",
    )
    assert load_contracts(path) == contracts


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"id": "shared-id"}, "id"),
        ({"group_id": "shared-group"}, "group"),
        ({"repository": "shared-repo"}, "repository"),
        ({"query": "  SHARED   query ", "context": "\n SHARED context "}, "normalized input"),
    ],
)
def test_assert_disjoint_rejects_each_leakage_channel(
    make_record: Callable[..., Record], changes: dict[str, str], message: str,
) -> None:
    left = make_record(
        id="shared-id", group_id="shared-group", repository="shared-repo",
        query="Shared query", context="Shared context",
    )
    right = make_record(**changes)
    with pytest.raises(ValueError, match=message):
        assert_disjoint([left], [right])
    with pytest.raises(ValueError, match=message):
        assert_disjoint([right], [left])


def test_assert_disjoint_accepts_independent_records(make_record: Callable[..., Record]) -> None:
    assert_disjoint([make_record()], [make_record()])


def test_split_keeps_transitive_group_repository_and_duplicate_chain_together(
    make_record: Callable[..., Record],
) -> None:
    chain = [
        make_record(group_id="bridge-task"),
        make_record(group_id="bridge-task", repository="bridge-repo"),
        make_record(repository="bridge-repo", query="Bridge text", context="A B"),
        make_record(query=" BRIDGE   TEXT ", context="a\nb"),
    ]
    records = chain + [make_record() for _ in range(7)]
    assert fingerprint(chain[2]) == fingerprint(chain[3])
    splits = split_reviewed(records, seed=7)
    assert splits == split_reviewed(records, seed=7)
    assert set(splits) == {"train", "dev", "test"}
    assert all(splits.values())
    assert sorted(r.id for rows in splits.values() for r in rows) == sorted(r.id for r in records)
    chain_ids = {r.id for r in chain}
    assert sum(chain_ids <= {r.id for r in rows} for rows in splits.values()) == 1
    for left, right in combinations(splits.values(), 2):
        assert_disjoint(left, right)


@pytest.mark.parametrize("count", [0, 1, 2, 3, 4])
def test_split_rejects_fewer_than_five_independent_groups(
    make_record: Callable[..., Record], count: int,
) -> None:
    with pytest.raises(ValueError, match="five independent"):
        split_reviewed([make_record() for _ in range(count)])


def test_split_counts_connected_components_not_rows(make_record: Callable[..., Record]) -> None:
    records = [make_record(repository="one-repository") for _ in range(12)]
    with pytest.raises(ValueError, match="five independent"):
        split_reviewed(records)
