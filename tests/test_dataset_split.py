"""Group-safety, rare-class protection, development-exposed pinning and manifest integrity."""

import json
from collections import Counter

import pytest

from routing_lab.data import write_records
from routing_lab.dataset_split import create_parser, split_dataset, stratified_group_split
from routing_lab.evaluation import sha256


def test_invalid_fraction_and_empty_records_are_rejected(make_record):
    records = [make_record()]
    with pytest.raises(ValueError, match="between 0 and 1"):
        stratified_group_split(records, test_fraction=0)
    with pytest.raises(ValueError, match="between 0 and 1"):
        stratified_group_split(records, test_fraction=1)
    with pytest.raises(ValueError, match="at least one record"):
        stratified_group_split([])


def test_best_effort_stratification_on_independent_singleton_blocks(make_record):
    # Every record below gets a unique repository/group/query from make_record,
    # so each is its own block; stratification should be close to exact.
    counts = {"debugging": 50, "code_generation": 20, "test_generation": 10,
             "code_review": 10, "repository_qa": 5, "fallback": 5}
    records = [make_record(expected_route=route) for route, n in counts.items() for _ in range(n)]
    train, test = stratified_group_split(records, test_fraction=0.10, seed=42)
    assert len(train) + len(test) == 100
    assert {record.id for record in train} & {record.id for record in test} == set()
    assert 8 <= len(test) <= 12
    test_counts = Counter(record.expected_route for record in test)
    assert 3 <= test_counts["debugging"] <= 8
    assert 1 <= test_counts["code_generation"] <= 5


def test_connected_block_never_crosses_the_split(make_record):
    shared = [
        make_record(id="a", group_id="shared-group", repository="shared-repo", expected_route="debugging"),
        make_record(id="b", group_id="shared-group", repository="other-repo", expected_route="code_generation"),
        make_record(id="c", group_id="other-group", repository="shared-repo", expected_route="test_generation"),
    ]
    singletons = [make_record(expected_route="repository_qa") for _ in range(20)]
    train, test = stratified_group_split(shared + singletons, test_fraction=0.30, seed=7)
    train_ids, test_ids = {r.id for r in train}, {r.id for r in test}
    assert {"a", "b", "c"} <= train_ids or {"a", "b", "c"} <= test_ids


def test_constant_repository_does_not_collapse_into_one_unsplittable_block(make_record):
    # Mirrors BFCL: every row shares one placeholder repository name (unlike
    # SWE-bench's genuine per-task repository), but group_id still varies.
    records = [
        make_record(repository="bfcl-tool-environment", group_id=f"group-{i}", expected_route=route)
        for i, route in enumerate(["debugging"] * 60 + ["code_generation"] * 30 + ["fallback"] * 10)
    ]
    train, test = stratified_group_split(records, test_fraction=0.10, seed=11)
    assert len(train) + len(test) == 100
    assert test  # previously raised ValueError: nothing was eligible for the slice
    assert 7 <= len(test) <= 13


def test_development_exposed_rows_never_enter_the_held_out_slice(make_record):
    exposed = [make_record(expected_route="debugging") for _ in range(10)]
    other = [make_record(expected_route="code_generation") for _ in range(10)]
    exposed_ids = {record.id for record in exposed}
    train, test = stratified_group_split(exposed + other, test_fraction=0.50, seed=1,
                                         development_exposed_ids=exposed_ids)
    assert exposed_ids.isdisjoint({record.id for record in test})
    assert exposed_ids <= {record.id for record in train}


def test_a_singleton_route_is_never_fully_drained_from_train(make_record):
    records = [make_record(expected_route="debugging") for _ in range(19)]
    records.append(make_record(expected_route="fallback"))
    train, test = stratified_group_split(records, test_fraction=0.5, seed=3)
    assert "fallback" in {record.expected_route for record in train}
    assert "fallback" not in {record.expected_route for record in test}


def build_fake_final_export(tmp_path, records, taxonomy="coding-v1", *, status="complete", name="final_export"):
    export_dir = tmp_path / name
    export_dir.mkdir()
    data_path = export_dir / f"{taxonomy}.jsonl"
    write_records(data_path, records)
    catalog_path = export_dir / f"{taxonomy}.contracts.json"
    catalog_path.write_text("{}", encoding="utf-8")
    manifest = {
        "status": status,
        "files": {f"{taxonomy}.jsonl": {"sha256": sha256(data_path), "count": len(records)}},
        "catalogs": {taxonomy: {"file": catalog_path.name, "sha256": sha256(catalog_path)}},
    }
    (export_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return export_dir


def test_split_dataset_writes_train_test_and_manifest(make_record, tmp_path):
    records = [make_record(expected_route="debugging") for _ in range(40)]
    records += [make_record(expected_route="fallback") for _ in range(10)]
    export_dir = build_fake_final_export(tmp_path, records)
    output = tmp_path / "splits"
    result = split_dataset(export_dir, output, test_fraction=0.10, seed=5)
    assert result["status"] == "complete"
    train_path, test_path = output / "coding-v1-train90.jsonl", output / "coding-v1-test10.jsonl"
    assert train_path.exists() and test_path.exists()
    assert (output / "coding-v1.contracts.json").exists()
    track = result["tracks"]["coding-v1"]
    assert track["total"] == 50
    assert track["train_count"] + track["test_count"] == 50
    assert result["files"]["coding-v1-train90.jsonl"]["count"] == track["train_count"]


def test_split_dataset_rejects_incomplete_or_tampered_export(make_record, tmp_path):
    records = [make_record(expected_route="debugging") for _ in range(10)]
    incomplete = build_fake_final_export(tmp_path, records, status="incomplete", name="incomplete_export")
    with pytest.raises(ValueError, match="must be complete"):
        split_dataset(incomplete, tmp_path / "s1")

    export_dir = build_fake_final_export(tmp_path, records, taxonomy="coding-v1", name="complete_export")
    path = export_dir / "coding-v1.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="changed since the final export"):
        split_dataset(export_dir, tmp_path / "s2")


def test_split_dataset_rejects_an_all_empty_export(tmp_path):
    export_dir = tmp_path / "empty_export"
    export_dir.mkdir()
    (export_dir / "manifest.json").write_text(json.dumps({
        "status": "complete", "files": {"coding-v1.jsonl": {"sha256": "x", "count": 0}},
        "catalogs": {},
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="No non-empty taxonomy files"):
        split_dataset(export_dir, tmp_path / "s3")


def test_cli_defaults_are_registered():
    args = create_parser().parse_args(["--final-export", "f", "--output", "o"])
    assert args.test_fraction == 0.10
    assert args.seed == 42
