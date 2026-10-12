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
