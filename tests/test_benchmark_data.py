"""Public-data provenance, first-decision boundaries and explicitly non-gold labels."""

import hashlib
import json
from dataclasses import replace

import pytest
import requests

from routing_lab.backends import state
from routing_lab.benchmark_data import (
    adapt_bfcl, adapt_swe, adapt_toolbench, build_dataset, first_request,
    jsonl, proposed_route, verify_snapshot,
)
from routing_lab.benchmark_sources import (
    REVISIONS, Resource, destination, download_resource, probe_toolbench_archives,
)
from routing_lab.data import load_records
from routing_lab.evaluation import sha256
from collections.abc import Callable
from pathlib import Path
from routing_lab.backends import ConstantBackend, Decision
from routing_lab.benchmark_filter import filter_bundle_by_source
from routing_lab.benchmark_resume import close_interrupted_run, completed_ids, create_parser, merge_runs, read_predictions, remaining_records
from routing_lab.data import Record
from routing_lab.evaluation import run_experiment


def swe_row(identity="org__repo-1"):
    return {
        "instance_id": identity, "repo": "org/repo", "base_commit": "abc",
        "problem_statement": "A regression causes an exception.",
        "patch": "PATCH_MUST_NOT_REACH_ROUTER", "test_patch": "TEST_PATCH_MUST_NOT_REACH_ROUTER",
        "FAIL_TO_PASS": ["FAIL_TEST_MUST_NOT_REACH_ROUTER"],
        "PASS_TO_PASS": ["PASS_TEST_MUST_NOT_REACH_ROUTER"],
        "eval_script": "EVAL_MUST_NOT_REACH_ROUTER", "hints_text": "HINT_MUST_NOT_REACH_ROUTER",
    }


def bfcl_row(identity="simple_1"):
    return {"id": identity, "question": [[{"role": "user", "content": "Find tomorrow's weather."}]],
            "function": [{"name": "weather", "description": "Forecast weather"}]}


def toolbench_row():
    return {
        "query_id": 1, "query": "Find a flight.",
        "api_list": [{"api_name": "flight", "api_description": "Flight search",
                      "template_response": "TEMPLATE_NOT_AN_INPUT"}],
        "relevant APIs": [["GOLD_TOOL_SELECTION", "flight"]],
    }


def test_swe_native_answers_are_separate():
    candidate, targets = adapt_swe(swe_row(), "test")
    text = candidate["query"] + candidate["context"]
    assert "MUST_NOT_REACH_ROUTER" not in text
    assert targets["native_targets"]["patch"] == swe_row()["patch"]
    assert candidate["native_split"] == "test"
    assert candidate["review_status"] == "unreviewed"


def test_only_first_bfcl_user_input_and_preceding_instructions():
    row = bfcl_row()
    row["question"] = [
        [{"role": "system", "content": "Available capability context"},
         {"role": "user", "content": "Find tomorrow's weather."},
         {"role": "assistant", "content": "POST_DECISION_SECRET"}],
        [{"role": "user", "content": "FUTURE_TURN_SECRET"}],
    ]
    row["path"] = ["GOLD_PATH_SECRET"]
    candidate, target = adapt_bfcl(row, "multi_turn_base", {"ground_truth": "GOLD_CALL_SECRET"})
    text = candidate["query"] + candidate["context"]
    assert "SECRET" not in text
    assert "Available capability context" in text
    assert "later_turns_excluded" in candidate["flags"]
    assert target["native_solution_path"] == ["GOLD_PATH_SECRET"]
    assert candidate["native_turn_count"] == 2


@pytest.mark.parametrize("question", [[], ["wrong"], [[{"role": "user", "content": []}]],
                                    [[{"role": "system", "content": "No user"}]]])
def test_bad_first_turn_is_rejected(question):
    with pytest.raises(ValueError):
        first_request({"question": question})


def test_environment_dependent_inputs_are_flagged():
    row = bfcl_row()
    del row["function"]
    row["initial_config"] = {"not_visible_to_router": "ENVIRONMENT_SECRET"}
    candidate, _ = adapt_bfcl(row, "memory", None)
    assert "native_environment_not_reconstructed" in candidate["flags"]
    assert "ENVIRONMENT_SECRET" not in candidate["context"]


def test_toolbench_keeps_offered_tools_not_gold_selection():
    row = toolbench_row()
    row["api_list"].append({"api_name": "unused", "api_description": "Also offered"})
    candidate, target = adapt_toolbench(row, "G1_query")
    assert "unused" in candidate["context"]
    assert "GOLD_TOOL_SELECTION" not in candidate["context"]
    assert "TEMPLATE_NOT_AN_INPUT" not in candidate["context"]
    assert target["relevant_apis"] == row["relevant APIs"]
    assert candidate["native_split"] == "official_example"


def test_related_queries_group_together_across_categories():
    left, _ = adapt_bfcl(bfcl_row("x"), "simple_python", None)
    right, _ = adapt_bfcl(bfcl_row("y"), "live_simple", None)
    assert left["id"] != right["id"]
    assert left["group_id"] == right["group_id"]


@pytest.mark.parametrize(("query", "source", "route"), [
    ("Regression raises an exception", "swe-bench", "debugging"),
    ("Add support for a new format", "swe-bench", "code_generation"),
    ("Some observations", "swe-bench", None),
    ("Book a flight", "bfcl", "fallback"),
    ("Write Python code", "bfcl", None),
])
def test_label_policy_proposes_or_abstains(query, source, route):
    assert proposed_route(query, source)[0] == route


def test_unicode_jsonl_is_not_split_inside_strings(tmp_path):
    path = tmp_path / "input.jsonl"
    path.write_text('{"query":"before\u2028after"}\n', encoding="utf-8")
    assert len(jsonl(path)) == 1


class Response:
    def __init__(self, body=b"benchmark", content_type="application/octet-stream", status=200):
        self.body = body
        self.headers = {"Content-Type": content_type}
        self.status_code = status

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        yield self.body


class Session:
    def __init__(self, response):
        self.response = response

    def get(self, *args, **kwargs):
        return self.response


def test_download_checks_both_upstream_hashes_and_refuses_overwrite(tmp_path):
    body = b"benchmark"
    resource = Resource(
        "bfcl/input.json", "https://example.test/input", "bfcl", "input", len(body),
        hashlib.sha1(f"blob {len(body)}\0".encode() + body).hexdigest(),
        hashlib.sha256(body).hexdigest(),
    )
    result = download_resource(Session(Response(body)), resource, tmp_path)
    assert result["content_sha256"] == resource.sha256
    with pytest.raises(FileExistsError):
        download_resource(Session(Response(body)), resource, tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        download_resource(Session(Response(body)), replace(resource, path="bad.json", sha256="0" * 64), tmp_path)
    assert not (tmp_path / "bad.json").exists()
    assert not list(tmp_path.rglob("*.part"))


@pytest.mark.parametrize("name", ["../bad", "C:/bad", "/bad", "a\\..\\bad", ""])
def test_download_paths_cannot_escape(tmp_path, name):
    with pytest.raises(ValueError):
        destination(tmp_path, name)


def test_html_and_http_failures_never_become_dataset_files(tmp_path):
    resource = Resource("data.json", "https://example.test/input", "bfcl", "input")
    with pytest.raises(ValueError, match="HTML"):
        download_resource(Session(Response(b"<html>", "text/html")), resource, tmp_path)
    with pytest.raises(requests.HTTPError):
        download_resource(Session(Response(status=404)), resource, tmp_path)
    assert not (tmp_path / "data.json").exists()


def test_toolbench_probe_records_network_failure_explicitly():
    class Offline:
        def get(self, *args, **kwargs):
            raise requests.Timeout("network unavailable")

    results = probe_toolbench_archives(Offline())
    assert len(results) == 3
    assert all(row["status"] == "network_error" and "error" in row for row in results)


@pytest.fixture
def raw_snapshot(tmp_path):
    arrow = pytest.importorskip("pyarrow")
    parquet = pytest.importorskip("pyarrow.parquet")
    raw = tmp_path / "raw"
    raw.mkdir()
    resources = []

    def add(name, source, role, rows, *, parquet_format=False, array=False):
        path = raw / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if parquet_format:
            parquet.write_table(arrow.Table.from_pylist(rows), path)
        else:
            text = json.dumps(rows) if array else "".join(json.dumps(row) + "\n" for row in rows)
            path.write_text(text, encoding="utf-8")
        resources.append({"path": name, "source": source, "role": role, "content_sha256": sha256(path)})

    add("swe-bench/data/dev-00000.parquet", "swe-bench", "input", [swe_row()], parquet_format=True)
    add("swe-bench/data/test-00000.parquet", "swe-bench", "input", [swe_row("org__repo-2")], parquet_format=True)
    coding = bfcl_row("coding")
    coding["question"][0][0]["content"] = "Write Python code"
    missing = bfcl_row("memory")
    del missing["function"]
    large = bfcl_row("long")
    large["function"][0]["description"] = "x" * 33000
    add("bfcl/BFCL_v4_simple_python.json", "bfcl", "input", [bfcl_row(), coding, missing, large])
    add("bfcl/possible_answer/simple.json", "bfcl", "answer", [{"id": "simple_1", "ground_truth": "SECRET"}])
    add("toolbench/G1_query.json", "toolbench", "sample", [toolbench_row()], array=True)
    manifest = {"status": "complete_available_sources_toolbench_full_missing",
                "revisions": REVISIONS, "plan": resources, "files": resources}
    (raw / "download-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return raw


def test_build_preserves_all_candidates_and_separates_draft_endpoint_subsets(raw_snapshot, tmp_path):
    output = tmp_path / "routing"
    manifest = build_dataset(raw_snapshot, output)
    assert manifest["candidate_count"] == 7
    assert manifest["draft_probe_count"] == 4
    assert manifest["endpoint_compatible_count"] == 3
    assert manifest["swe_native_dev_test_repository_overlap"] == ["org/repo"]
    assert len(jsonl(output / "native-targets.jsonl")) == 7
    assert len(jsonl(output / "excluded-from-probes.jsonl")) == 3
    records = load_records(output / "endpoint-probes.jsonl")
    assert all(record.review_status == "unreviewed" and not record.reviewer for record in records)
    assert all("SECRET" not in state(record, "context") for record in records)
    with pytest.raises(ValueError, match="reviewed"):
        load_records(output / "endpoint-probes.jsonl", reviewed_only=True)
    with pytest.raises(FileExistsError):
        build_dataset(raw_snapshot, output)
    for name, entry in manifest["files"].items():
        assert sha256(output / name) == entry["sha256"]


def test_changed_sources_are_rejected_before_adaptation(raw_snapshot):
    path = raw_snapshot / "bfcl" / "BFCL_v4_simple_python.json"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Source content changed"):
        verify_snapshot(raw_snapshot)


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


class FlakyBackend(ConstantBackend):
    """Fails on a specific call index, mirroring a remote mid-run interruption."""

    def __init__(self, fail_at: int, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.calls = 0
        self.fail_at = fail_at

    def predict(self, record: Record) -> Decision:
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("synthetic remote interruption")
        return super().predict(record)


def run_two_parts(records: list[Record], tmp_path: Path, *, fail_at: int) -> tuple[Path, Path]:
    """First run fails partway (mirrors a real interrupted benchmark); second finishes the rest."""
    interrupted = tmp_path / "part-1-interrupted"
    with pytest.raises(RuntimeError, match="synthetic remote interruption"):
        run_experiment(FlakyBackend(fail_at=fail_at), records, interrupted, probe=True)
    done = set(completed_ids(interrupted / "predictions.jsonl"))
    first = tmp_path / "part-1-closed"
    close_interrupted_run(interrupted, records, first)
    remaining = remaining_records(records, done)
    second = tmp_path / "part-2"
    run_experiment(ConstantBackend(), remaining, second, probe=True)
    return first, second


def test_merge_reconstructs_one_summary_from_two_interrupted_runs(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    records = [make_record(expected_route="debugging") for _ in range(5)]
    first, second = run_two_parts(records, tmp_path, fail_at=3)
    output = tmp_path / "merged"
    summary = merge_runs(records, [first, second], output)
    assert summary["count"] == 5
    assert summary["mode"] == "probe_unreviewed_not_a_benchmark"
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["total_rows"] == 5
    assert [part["rows"] for part in manifest["merged_from"]] == [2, 3]
    merged_rows = read_predictions(output / "predictions.jsonl")
    assert [row["id"] for row in merged_rows] == [
        *[row["id"] for row in read_predictions(first / "predictions.jsonl")],
        *[row["id"] for row in read_predictions(second / "predictions.jsonl")],
    ]
    assert json.loads((output / "summary.json").read_text(encoding="utf-8")) == summary


def test_first_request_latency_reflects_true_chronological_order(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    records = [make_record(expected_route="debugging") for _ in range(4)]
    first, second = run_two_parts(records, tmp_path, fail_at=2)
    summary = merge_runs(records, [first, second], tmp_path / "merged")
    first_row = read_predictions(first / "predictions.jsonl")[0]
    assert summary["latency_ms"]["first_request"] == first_row["latency_ms"]


def test_rejects_an_incomplete_contributing_run(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    records = [make_record(expected_route="debugging") for _ in range(3)]
    incomplete = tmp_path / "incomplete"
    with pytest.raises(RuntimeError):
        run_experiment(FlakyBackend(fail_at=1), records, incomplete, probe=True)
    with pytest.raises(ValueError, match="not a complete run"):
        merge_runs(records, [incomplete], tmp_path / "merged")


def test_close_interrupted_run_covers_only_the_truly_completed_subset(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    records = [make_record(expected_route="debugging") for _ in range(4)]
    interrupted = tmp_path / "interrupted"
    with pytest.raises(RuntimeError, match="synthetic remote interruption"):
        run_experiment(FlakyBackend(fail_at=3), records, interrupted, probe=True)
    closed = tmp_path / "closed"
    summary = close_interrupted_run(interrupted, records, closed)
    assert summary["count"] == 2
    manifest = json.loads((closed / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "complete"
    assert manifest["closed_from_interrupted_run"]["rows_completed"] == 2
    assert json.loads((closed / "summary.json").read_text(encoding="utf-8")) == summary
    # The closed sub-run is now a normal, valid merge_runs contributor.
    remaining = remaining_records(records, set(completed_ids(interrupted / "predictions.jsonl")))
    second = tmp_path / "rest"
    run_experiment(ConstantBackend(), remaining, second, probe=True)
    merged = merge_runs(records, [closed, second], tmp_path / "merged")
    assert merged["count"] == 4


def test_close_interrupted_run_rejects_a_run_that_is_not_interrupted(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    records = [make_record(expected_route="debugging")]
    complete_run = tmp_path / "already-complete"
    run_experiment(ConstantBackend(), records, complete_run, probe=True)
    with pytest.raises(ValueError, match="not an interrupted run"):
        close_interrupted_run(complete_run, records, tmp_path / "closed")


def test_rejects_overlapping_rows_between_runs(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    records = [make_record(expected_route="debugging") for _ in range(3)]
    first = tmp_path / "all"
    run_experiment(ConstantBackend(), records, first, probe=True)
    second = tmp_path / "overlap"
    run_experiment(ConstantBackend(), records[:1], second, probe=True)
    with pytest.raises(ValueError, match="more than one run"):
        merge_runs(records, [first, second], tmp_path / "merged")


def test_rejects_missing_or_extra_rows(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    records = [make_record(expected_route="debugging") for _ in range(3)]
    partial = tmp_path / "partial"
    run_experiment(ConstantBackend(), records[:2], partial, probe=True)
    with pytest.raises(ValueError, match="incomplete or mismatched"):
        merge_runs(records, [partial], tmp_path / "merged-missing")

    other_record = make_record(expected_route="debugging")
    foreign = tmp_path / "foreign"
    run_experiment(ConstantBackend(), [*records, other_record], foreign, probe=True)
    with pytest.raises(ValueError, match="outside the target dataset"):
        merge_runs(records, [foreign], tmp_path / "merged-extra")


def test_rejects_when_target_dataset_content_differs_from_what_actually_ran(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    """dataset_digest only covers input records, not decisions; this is its real guarantee:
    merge_runs refuses a target dataset whose content was edited after the run completed,
    even if IDs are unchanged, so a stale/edited file is never silently reattributed."""
    records = [make_record(expected_route="debugging") for _ in range(2)]
    run_dir = tmp_path / "edited-after"
    run_experiment(ConstantBackend(), records, run_dir, probe=True)
    edited = [records[0].model_copy(update={"context": "A different context entirely."}), records[1]]
    with pytest.raises(ValueError, match="predictions do not match its own recorded dataset digest"):
        merge_runs(edited, [run_dir], tmp_path / "merged")


def test_rejects_mismatched_taxonomy_or_probe_mode(
    make_record: Callable[..., Record], tmp_path: Path,
) -> None:
    reviewed = [make_record(expected_route="debugging", review_status="reviewed", reviewer="x")]
    benchmark_run = tmp_path / "benchmark"
    run_experiment(ConstantBackend(), reviewed, benchmark_run)
    probe_records = [make_record(expected_route="debugging")]
    probe_run = tmp_path / "probe"
    run_experiment(ConstantBackend(), probe_records, probe_run, probe=True)
    with pytest.raises(ValueError, match="same probe/benchmark mode"):
        merge_runs([*reviewed, *probe_records], [benchmark_run, probe_run], tmp_path / "merged")


def test_duplicate_ids_within_a_single_run_are_rejected(tmp_path: Path) -> None:
    run_dir = tmp_path / "dup"
    run_dir.mkdir()
    (run_dir / "predictions.jsonl").write_text(
        json.dumps({"id": "a", "decision": {"route": "debugging"}, "latency_ms": 1.0}) + "\n"
        + json.dumps({"id": "a", "decision": {"route": "debugging"}, "latency_ms": 1.0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate row IDs"):
        completed_ids(run_dir / "predictions.jsonl")


def test_cli_accepts_repeated_run_dir_flags():
    args = create_parser().parse_args([
        "--data", "d.jsonl", "--taxonomy", "coding-v1",
        "--run-dir", "part-1", "--run-dir", "part-2", "--output", "o",
    ])
    assert [str(path) for path in args.run_dirs] == ["part-1", "part-2"]
