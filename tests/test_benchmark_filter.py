"""Source-filtered bundle creation: exact row preservation, integrity, and rejections."""

import json
from pathlib import Path

import pytest

from routing_lab.benchmark_filter import filter_bundle_by_source
from routing_lab.evaluation import sha256


def make_bundle(tmp_path: Path, rows: list[dict], *, status: str = "complete_unreviewed_probes",
               name: str = "bundle") -> Path:
    bundle = tmp_path / name
    bundle.mkdir()
    candidates = bundle / "candidates.jsonl"
    candidates.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    (bundle / "manifest.json").write_text(json.dumps({
        "status": status, "revisions": {"fixture": "pinned"},
        "files": {"candidates.jsonl": {"sha256": sha256(candidates)}},
    }), encoding="utf-8")
    return bundle


def test_keeps_only_matching_sources_verbatim(tmp_path):
    rows = [
        {"id": "a", "source": "swe-bench", "query": "fix the bug"},
        {"id": "b", "source": "bfcl", "query": "call a tool"},
        {"id": "c", "source": "swe-bench", "query": "add a feature"},
    ]
    bundle = make_bundle(tmp_path, rows)
    output = tmp_path / "filtered"
    manifest = filter_bundle_by_source(bundle, {"swe-bench"}, output)
    assert manifest["status"] == "complete_unreviewed_probes"
    assert manifest["total_rows"] == 2
    assert manifest["sources"] == ["swe-bench"]
    kept = [json.loads(line) for line in (output / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
    assert kept == [rows[0], rows[2]]
    assert sha256(output / "candidates.jsonl") == manifest["files"]["candidates.jsonl"]["sha256"]
    assert manifest["filtered_from"]["manifest_sha256"] == sha256(bundle / "manifest.json")


def test_rejects_incomplete_or_tampered_or_empty_result(tmp_path):
    rows = [{"id": "a", "source": "bfcl", "query": "call a tool"}]
    incomplete = make_bundle(tmp_path, rows, status="incomplete", name="incomplete-bundle")
    with pytest.raises(ValueError, match="incomplete"):
        filter_bundle_by_source(incomplete, {"bfcl"}, tmp_path / "out1")

    tampered = make_bundle(tmp_path, rows, name="tampered-bundle")
    (tampered / "candidates.jsonl").write_text(
        (tampered / "candidates.jsonl").read_text(encoding="utf-8") + "\n", encoding="utf-8",
    )
    with pytest.raises(ValueError, match="content changed"):
        filter_bundle_by_source(tampered, {"bfcl"}, tmp_path / "out2")

    clean = make_bundle(tmp_path, rows, name="clean-bundle")
    with pytest.raises(ValueError, match="No candidate rows matched"):
        filter_bundle_by_source(clean, {"swe-bench"}, tmp_path / "out3")


def test_rejects_empty_source_set(tmp_path):
    bundle = make_bundle(tmp_path, [{"id": "a", "source": "bfcl", "query": "x"}])
    with pytest.raises(ValueError, match="at least one source"):
        filter_bundle_by_source(bundle, set(), tmp_path / "out")


def test_caps_only_oversized_queries_and_logs_each_change():
    from routing_lab.benchmark_filter import cap_oversized_queries_for_transport

    rows = [
        {"id": "short", "query": "x" * 10, "expected_route": "debugging"},
        {"id": "long", "query": "y" * 100, "expected_route": "code_generation"},
    ]
    adjusted, log = cap_oversized_queries_for_transport(rows, max_query_chars=50)
    assert adjusted[0] == rows[0]
    assert adjusted[0] is not rows[0] or adjusted[0]["query"] == rows[0]["query"]
    assert len(adjusted[1]["query"]) == 50
    assert adjusted[1]["query"] == "y" * 50
    assert adjusted[1]["expected_route"] == "code_generation"
    assert adjusted[1]["id"] == "long"
    assert log == [{"id": "long", "original_query_chars": 100, "kept_query_chars": 50}]
    # Original rows are untouched.
    assert rows[1]["query"] == "y" * 100


def test_cap_oversized_queries_rejects_non_positive_limit():
    from routing_lab.benchmark_filter import cap_oversized_queries_for_transport

    with pytest.raises(ValueError, match="positive"):
        cap_oversized_queries_for_transport([{"id": "a", "query": "x"}], max_query_chars=0)

