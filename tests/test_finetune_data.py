"""Offline preparation tests. All labels and catalogs here are synthetic silver."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from itertools import combinations
from pathlib import Path

import pytest

import routing_lab.finetune_data as preparation
from routing_lab.backends import questions, state
from routing_lab.data import fingerprint, load_contracts, write_records
from routing_lab.dataset_split import SPLIT_POLICY_VERSION, _assert_group_safe, _discriminating_fields
from routing_lab.evaluation import sha256
from routing_lab.finetune_data import (
    BUNDLE_FILES, PARTITIONS, TRACKS, create_parser, main, prepare_finetune_data, verify_bundle,
)
from routing_lab.taxonomy import taxonomy_routes


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.fixture
def source_factory(tmp_path, make_record):
    def create(*, transform=None, count=100):
        source = tmp_path / "source"
        source.mkdir()
        records = {}
        for track in TRACKS:
            major, secondary, minority = (
                ("debugging", "code_generation", "fallback") if track == "coding-v1"
                else ("tool_lookup", "tool_action", "memory_retrieval")
            )
            records[track] = {}
            for partition, size in (("train90", count), ("test10", 10)):
                rows = []
                for index in range(size):
                    route = major if index < size * 0.8 else secondary
                    if partition == "train90" and index == size - 1:
                        route = minority
                    record = make_record(
                        taxonomy_id=track, expected_route=route,
                        tags=["synthetic", "development_exposed"] if partition == "train90" and index == 0
                        else ["synthetic", "ai_labeled"],
                        review_status="unreviewed", reviewer="",
                        provenance=f"synthetic silver source; {track}; {partition}; {index}",
                    )
                    if track == "tool-api-v1":
                        record = record.model_copy(update={"repository": "constant-tool-environment"})
                    rows.append(record)
                records[track][partition] = rows
        if transform:
            transform(records)
        manifest = {
            "status": "complete", "policy_version": SPLIT_POLICY_VERSION,
            "source_export_manifest_sha256": "a" * 64, "files": {}, "catalogs": {},
        }
        for track in TRACKS:
            for partition, rows in records[track].items():
                path = source / f"{track}-{partition}.jsonl"
                write_records(path, rows)
                manifest["files"][path.name] = {
                    "sha256": sha256(path), "count": len(rows),
                    "route_counts": dict(Counter(record.expected_route for record in rows)),
                }
            catalog = source / f"{track}.contracts.json"
            catalog.write_text(json.dumps({
                route: {
                    "description": f"Offered synthetic {route} capability",
                    "tools": ["read"], "can_write": False, "success_criteria": "Synthetic success",
                } for route in taxonomy_routes(track)
            }), encoding="utf-8")
            manifest["catalogs"][track] = {"file": catalog.name, "sha256": sha256(catalog)}
        (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        return source, records
    return create


def test_native_bundle_preserves_provenance_and_never_exports_test(source_factory, tmp_path):
    source, originals = source_factory()
    original_hashes = {path.name: sha256(path) for path in source.iterdir()}
    output = tmp_path / "prepared"
    result = prepare_finetune_data(source, output)
    assert result["status"] == "complete"
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8")) == result
    assert result["label_quality"] == "unreviewed_silver_not_human_gold"
    assert result["model_strategy"] == "one-shared-clef-flash-catalog-conditioned"
    assert result["training_launched"] is False
    assert result["source"]["manifest_sha256"] == original_hashes["manifest.json"]
    assert result["source"]["source_export_manifest_sha256"] == "a" * 64
    assert result["requested_fractions"] == {"train": .8, "validation": .1, "calibration": .1}
    assert result["partition_counts"] == {"train": 160, "validation": 20, "calibration": 20}
    assert {path.name for path in output.iterdir()} == (
        {"manifest.json", *(f"{track}.contracts.json" for track in TRACKS)}
        | {f"{track}-{part}.jsonl" for track in TRACKS for part in PARTITIONS}
        | {f"mixed-{part}.jsonl" for part in PARTITIONS}
    )
    assert {path.name: sha256(path) for path in source.iterdir()} == original_hashes
    exported = {}
    test_ids = {record.id for track in TRACKS for record in originals[track]["test10"]}
    for track in TRACKS:
        for partition in PARTITIONS:
            rows = read_jsonl(output / f"{track}-{partition}.jsonl")
            exported.update({row["id"]: row for row in rows})
        assert result["frozen_test10"][track]["sha256"] == original_hashes[f"{track}-test10.jsonl"]
        assert sha256(output / f"{track}.contracts.json") == original_hashes[f"{track}.contracts.json"]
    expected = {
        record.id: record.model_dump()
        for track in TRACKS for record in originals[track]["train90"]
    }
    assert exported == expected
    assert set(exported).isdisjoint(test_ids)
    for name, info in result["files"].items():
        assert sha256(output / name) == info["sha256"]
        if name.endswith(".jsonl"):
            assert len(read_jsonl(output / name)) == info["count"]
    for partition in PARTITIONS:
        examples = read_jsonl(output / f"mixed-{partition}.jsonl")
        assert len(examples) == result["partition_counts"][partition]
        record_ids = {
            row["id"] for track in TRACKS for row in read_jsonl(output / f"{track}-{partition}.jsonl")
        }
        assert {example["id"] for example in examples} == record_ids
        for example in examples:
            assert set(example) == {"id", "taxonomy_id", "state", "questions", "answers"}
            original = next(
                row for row in originals[example["taxonomy_id"]]["train90"] if row.id == example["id"]
            )
            assert example["state"] == state(original, "context")
            assert example["questions"] == questions(
                load_contracts(source / f"{original.taxonomy_id}.contracts.json", original.taxonomy_id),
                taxonomy_id=original.taxonomy_id,
            )
            catalog = load_contracts(source / f"{original.taxonomy_id}.contracts.json", original.taxonomy_id)
            assert list(example["questions"]["route"]["criteria"]) == list(catalog)
            assert example["answers"] == {"route": original.expected_route}
            assert original.provenance not in example["state"]


def test_reproducible_bytes_and_train_only_majority(source_factory, tmp_path):
    source, _ = source_factory()
    first, second = tmp_path / "first", tmp_path / "second"
    result = prepare_finetune_data(source, first, seed=13)
    assert prepare_finetune_data(source, second, seed=13) == result
    assert {p.name: p.read_bytes() for p in first.iterdir()} == {
        p.name: p.read_bytes() for p in second.iterdir()
    }
    third = tmp_path / "different"
    other = prepare_finetune_data(source, third, seed=14)
    assert other["files"]["mixed-train.jsonl"]["sha256"] != result["files"]["mixed-train.jsonl"]["sha256"]
    for track in TRACKS:
        counts = Counter(row["expected_route"] for row in read_jsonl(first / f"{track}-train.jsonl"))
        expected = min(counts, key=lambda route: (-counts[route], route))
        assert result["training_majority_routes"][track] == expected


def test_group_and_cross_track_duplicate_inputs_are_pinned_and_disjoint(source_factory, tmp_path):
    def transform(records):
        coding = records["coding-v1"]["train90"]
        tool = records["tool-api-v1"]["train90"]
        coding[1] = coding[1].model_copy(update={"group_id": coding[0].group_id})
        coding[2] = coding[2].model_copy(update={"repository": coding[1].repository})
        tool[5] = tool[5].model_copy(update={
            "query": coding[2].query.upper() + "  ", "context": coding[2].context.lower(),
        })
        tool[6] = tool[6].model_copy(update={"group_id": tool[5].group_id})
    source, originals = source_factory(transform=transform)
    output = tmp_path / "prepared"
    result = prepare_finetune_data(source, output)
    partition_ids = {}
    partition_fingerprints = {}
    for partition in PARTITIONS:
        ids = {row["id"] for row in read_jsonl(output / f"mixed-{partition}.jsonl")}
        partition_ids[partition] = ids
        partition_fingerprints[partition] = {
            fingerprint(record) for track in TRACKS for record in originals[track]["train90"] if record.id in ids
        }
    for left, right in combinations(PARTITIONS, 2):
        assert partition_ids[left].isdisjoint(partition_ids[right])
        assert partition_fingerprints[left].isdisjoint(partition_fingerprints[right])
        for track in TRACKS:
            rows = originals[track]["train90"]
            _assert_group_safe(
                [record for record in rows if record.id in partition_ids[left]],
                [record for record in rows if record.id in partition_ids[right]],
                _discriminating_fields(rows),
            )
    expected_train = {
        *(record.id for record in originals["coding-v1"]["train90"][:3]),
        *(record.id for record in originals["tool-api-v1"]["train90"][5:7]),
    }
    assert expected_train <= partition_ids["train"]
    assert result["audit"]["cross_track_shared_rows_pinned_to_train"] == 2
    assert result["audit"]["grouping_fields"]["tool-api-v1"] == ["group"]


def test_minority_classes_stay_in_training_and_coverage_warnings_are_explicit(source_factory, tmp_path):
    source, original = source_factory()
    output = tmp_path / "prepared"
    result = prepare_finetune_data(source, output)
    for track in TRACKS:
        singleton = original[track]["train90"][-1]
        assert singleton.id in {row["id"] for row in read_jsonl(output / f"{track}-train.jsonl")}
        for partition in ("validation", "calibration"):
            info = result["tracks"][track]["partitions"][partition]
            assert singleton.expected_route in info["routes_absent"]
            assert any(
                warning["taxonomy_id"] == track and warning["partition"] == partition
                and singleton.expected_route in warning["routes_absent"]
                for warning in result["route_coverage_warnings"]
            )
    assert any("code_review" in warning["routes_absent"] for warning in result["route_coverage_warnings"])


def test_all_pinned_is_best_effort_not_fabricated_heldout_coverage(source_factory, tmp_path):
    def transform(records):
        for track in TRACKS:
            records[track]["train90"] = [
                record.model_copy(update={"tags": ["development_exposed"]})
                for record in records[track]["train90"]
            ]
    source, _ = source_factory(transform=transform, count=3)
    result = prepare_finetune_data(source, tmp_path / "prepared")
    assert result["partition_counts"] == {"train": 6, "validation": 0, "calibration": 0}
    assert result["route_coverage_warnings"]


@pytest.mark.parametrize("kind", ["cross-track-input", "cross-track-id", "same-track-input", "test-test-id"])
def test_source_leakage_is_rejected_before_export(source_factory, tmp_path, kind):
    def transform(records):
        coding = records["coding-v1"]
        tool = records["tool-api-v1"]
        if kind == "cross-track-input":
            tool["test10"][0] = tool["test10"][0].model_copy(update={
                "query": coding["train90"][1].query, "context": coding["train90"][1].context,
            })
        elif kind == "cross-track-id":
            tool["train90"][0] = tool["train90"][0].model_copy(update={"id": coding["train90"][1].id})
        elif kind == "same-track-input":
            coding["test10"][0] = coding["test10"][0].model_copy(update={
                "query": coding["train90"][1].query, "context": coding["train90"][1].context,
            })
        else:
            tool["test10"][0] = tool["test10"][0].model_copy(update={"id": coding["test10"][0].id})
    source, _ = source_factory(transform=transform)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Data leakage|Duplicate record IDs"):
        prepare_finetune_data(source, output)
    assert not output.exists()


@pytest.mark.parametrize("name", [
    "coding-v1-train90.jsonl", "tool-api-v1-test10.jsonl", "coding-v1.contracts.json",
])
def test_tampered_source_hash_rejected(source_factory, tmp_path, name):
    source, _ = source_factory()
    path = source / name
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="Source hash mismatch"):
        prepare_finetune_data(source, tmp_path / "prepared")


@pytest.mark.parametrize("change", ["incomplete", "count", "routes", "catalog-path", "missing-track"])
def test_invalid_source_manifest_rejected(source_factory, tmp_path, change):
    source, _ = source_factory()
    path = source / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if change == "incomplete":
        manifest["status"] = "incomplete"
    elif change == "count":
        manifest["files"]["coding-v1-train90.jsonl"]["count"] += 1
    elif change == "routes":
        manifest["files"]["coding-v1-train90.jsonl"]["route_counts"]["debugging"] += 1
    elif change == "catalog-path":
        manifest["catalogs"]["coding-v1"]["file"] = "..\\outside.json"
    else:
        del manifest["catalogs"]["tool-api-v1"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        prepare_finetune_data(source, tmp_path / "prepared")


@pytest.mark.parametrize("fractions", [
    (0, .5, .5), (.8, .1, .2), (.8, -.1, .3),
    (float("nan"), .1, .1), (float("inf"), .1, .1),
])
def test_invalid_fractions_fail_before_reading_source(tmp_path, fractions):
    with pytest.raises(ValueError, match="Fractions"):
        prepare_finetune_data(
            tmp_path / "absent", tmp_path / "prepared", train_fraction=fractions[0],
            validation_fraction=fractions[1], calibration_fraction=fractions[2],
        )


def test_refuses_overwrite(source_factory, tmp_path):
    source, _ = source_factory()
    output = tmp_path / "prepared"
    prepare_finetune_data(source, output)
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError):
        prepare_finetune_data(source, output)
    assert before == {path.name: path.read_bytes() for path in output.iterdir()}


def test_failed_export_never_publishes_complete_manifest(source_factory, tmp_path, monkeypatch):
    source, _ = source_factory()
    output = tmp_path / "prepared"
    def fail(*args, **kwargs):
        raise OSError("synthetic disk failure")
    monkeypatch.setattr(preparation, "write_records", fail)
    with pytest.raises(OSError, match="disk failure"):
        prepare_finetune_data(source, output)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "incomplete"


def test_source_change_during_export_is_detected(source_factory, tmp_path, monkeypatch):
    source, _ = source_factory()
    original_write = preparation.write_records
    def mutate_source(path, records):
        original_write(path, records)
        manifest_path = source / "manifest.json"
        manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
    monkeypatch.setattr(preparation, "write_records", mutate_source)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="Source hash mismatch: manifest.json"):
        prepare_finetune_data(source, output)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "incomplete"


@pytest.mark.parametrize("kind", ["reviewed", "exposed-test"])
def test_source_silver_and_frozen_test_policy_enforced(source_factory, tmp_path, kind):
    def transform(records):
        partition = "train90" if kind == "reviewed" else "test10"
        update = {"review_status": "reviewed", "reviewer": "synthetic reviewer"} if kind == "reviewed" else {
            "tags": ["development_exposed"],
        }
        rows = records["coding-v1"][partition]
        rows[0] = rows[0].model_copy(update=update)
    source, _ = source_factory(transform=transform)
    with pytest.raises(ValueError, match="unreviewed silver|development_exposed"):
        prepare_finetune_data(source, tmp_path / "prepared")


def test_cli_defaults_and_offline_entrypoint(source_factory, tmp_path, capsys):
    args = create_parser().parse_args(["--source", "s", "--output", "o"])
    assert (args.train_fraction, args.validation_fraction, args.calibration_fraction) == (.8, .1, .1)
    assert args.seed == 42
    source, _ = source_factory()
    output = tmp_path / "prepared"
    main(["--source", str(source), "--output", str(output), "--context-mode", "query"])
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    assert result["context_mode"] == "query"
    assert all(
        not row["state"].startswith("User request:")
        for row in read_jsonl(output / "mixed-train.jsonl")
    )


def test_custom_fractions_and_query_only_duplicates(source_factory, tmp_path):
    def transform(records):
        rows = records["coding-v1"]["train90"]
        rows[5] = rows[5].model_copy(update={"query": rows[4].query, "context": "Different context."})
    source, originals = source_factory(transform=transform)
    output = tmp_path / "prepared"
    result = prepare_finetune_data(
        source, output, train_fraction=.6, validation_fraction=.2, calibration_fraction=.2,
        context_mode="query",
    )
    assert result["partition_counts"] == {"train": 120, "validation": 40, "calibration": 40}
    assert result["audit"]["query_only_duplicate_rows_pinned_to_train"] == 2
    training_ids = {row["id"] for row in read_jsonl(output / "mixed-train.jsonl")}
    assert {record.id for record in originals["coding-v1"]["train90"][4:6]} <= training_ids


def test_query_only_frozen_input_leakage_rejected(source_factory, tmp_path):
    def transform(records):
        coding = records["coding-v1"]
        tool = records["tool-api-v1"]
        tool["test10"][0] = tool["test10"][0].model_copy(update={
            "query": coding["train90"][5].query, "context": "Distinct tool context.",
        })
    source, _ = source_factory(transform=transform)
    with pytest.raises(ValueError, match="query-only input"):
        prepare_finetune_data(source, tmp_path / "prepared", context_mode="query")


def test_corrupted_written_output_never_publishes_complete_manifest(source_factory, tmp_path, monkeypatch):
    source, _ = source_factory()
    output = tmp_path / "prepared"
    original_write = preparation.write_records
    def corrupt_earlier_file(path, records):
        original_write(path, records)
        if path.name == "coding-v1-validation.jsonl":
            earlier = output / "coding-v1-train.jsonl"
            earlier.write_bytes(earlier.read_bytes() + b"\n")
    monkeypatch.setattr(preparation, "write_records", corrupt_earlier_file)
    with pytest.raises(ValueError, match="Output hash mismatch"):
        prepare_finetune_data(source, output)
    assert json.loads((output / "manifest.json").read_text(encoding="utf-8"))["status"] == "incomplete"


def test_verify_bundle_works_on_staged_copy_without_source(source_factory, tmp_path):
    source, _ = source_factory()
    output = tmp_path / "prepared"
    expected = prepare_finetune_data(source, output)
    staged = tmp_path / "staged"
    shutil.copytree(output, staged)
    shutil.rmtree(source)
    shutil.rmtree(output)
    assert verify_bundle(staged) == expected
    assert set(expected["files"]) == BUNDLE_FILES
    assert len(expected["files"]) == 11


@pytest.mark.parametrize("filename", [
    "mixed-train.jsonl", "coding-v1-validation.jsonl", "tool-api-v1.contracts.json",
])
def test_verify_bundle_rejects_payload_corruption(source_factory, tmp_path, filename):
    source, _ = source_factory()
    output = tmp_path / "prepared"
    prepare_finetune_data(source, output)
    path = output / filename
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="Output hash mismatch"):
        verify_bundle(output)


@pytest.mark.parametrize("change", [
    "incomplete", "policy", "missing-file", "omitted-hash", "invalid-hash", "catalog-hash",
    "partition-hash", "source-hash", "test-hash", "path-traversal", "unexpected-test", "unexpected-directory",
    "manifest-array", "manifest-invalid-json",
])
def test_verify_bundle_rejects_incomplete_or_inconsistent_manifests(source_factory, tmp_path, change):
    source, _ = source_factory()
    output = tmp_path / "prepared"
    manifest = prepare_finetune_data(source, output)
    if change == "incomplete":
        manifest["status"] = "incomplete"
    elif change == "policy":
        manifest["policy_version"] = "unknown"
    elif change == "missing-file":
        (output / "mixed-train.jsonl").unlink()
    elif change == "omitted-hash":
        del manifest["files"]["mixed-train.jsonl"]
    elif change == "invalid-hash":
        manifest["files"]["mixed-train.jsonl"]["sha256"] = "not-a-sha256"
    elif change == "catalog-hash":
        manifest["catalogs"]["coding-v1"]["sha256"] = "0" * 64
    elif change == "partition-hash":
        manifest["tracks"]["coding-v1"]["partitions"]["train"]["sha256"] = "0" * 64
    elif change == "source-hash":
        manifest["source"]["manifest_sha256"] = "0" * 64
    elif change == "test-hash":
        manifest["frozen_test10"]["coding-v1"]["sha256"] = "0" * 64
    elif change == "path-traversal":
        manifest["files"]["..\\outside.jsonl"] = {"sha256": "0" * 64}
    elif change == "unexpected-test":
        (output / "coding-v1-test10.jsonl").write_text("Do not upload", encoding="utf-8")
    elif change == "unexpected-directory":
        (output / "unexpected").mkdir()
    payload = json.dumps(manifest)
    if change == "manifest-array":
        payload = "[]"
    elif change == "manifest-invalid-json":
        payload = "{invalid"
    (output / "manifest.json").write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError):
        verify_bundle(output)


def test_verify_bundle_missing_manifest_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="manifest.json"):
        verify_bundle(tmp_path / "nonexistent")
