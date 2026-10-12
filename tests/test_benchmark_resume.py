"""Merge disjoint physical probe runs after a remote mid-run interruption."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from routing_lab.backends import ConstantBackend, Decision
from routing_lab.data import Record
from routing_lab.evaluation import run_experiment
from routing_lab.benchmark_resume import (
    close_interrupted_run, completed_ids, create_parser, merge_runs, read_predictions,
    remaining_records,
)


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
