"""Resume a probe-mode routing_lab.evaluation run after a remote mid-run failure.

A single physical run never invents a retried-away prediction or skips a row
silently; routing_lab.evaluation.run_experiment already persists every attempt
before computing a decision and marks an interrupted run's manifest status as
"incomplete". This module closes an interrupted run's completed prefix into
its own self-consistent "complete" sub-run -- recomputing its dataset digest
over only the rows actually completed, never over the larger input the
interrupted attempt was originally given -- then merges one or more disjoint
complete runs covering exactly the same dataset into one reconstructed
summary, for convenience and audit. It never re-derives a prediction, and it
never silently tolerates a missing, duplicate, or digest-mismatched row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from routing_lab.backends import Decision
from routing_lab.data import Record, load_records, records_taxonomy
from routing_lab.evaluation import sha256, summarize


def read_predictions(path: Path) -> list[dict]:
    """Read one run's predictions.jsonl in its physical write (chronological) order."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = [row["id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate row IDs within a single run: {path}")
    return rows


def completed_ids(path: Path) -> list[str]:
    """IDs this run completed, in physical request order."""
    return [row["id"] for row in read_predictions(path)]


def remaining_records(records: list[Record], done_ids: set[str]) -> list[Record]:
    """Rows not yet completed, preserving the dataset's original order."""
    return [record for record in records if record.id not in done_ids]


def _dataset_digest(records: list[Record]) -> str:
    return hashlib.sha256("\n".join(record.model_dump_json() for record in records).encode()).hexdigest()


def close_interrupted_run(interrupted_dir: Path, attempted_records: list[Record], output: Path) -> dict:
    """Close the completed prefix of an interrupted run into its own complete sub-run.

    ``attempted_records`` must be the exact list originally given to the
    interrupted run_experiment call (the same list determines row order and is
    used to validate that every completed ID genuinely belongs to it). No
    prediction is altered, invented, or re-ordered; only the manifest/summary
    are regenerated so the digest matches the true completed subset.
    """
    manifest = json.loads((interrupted_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "incomplete":
        raise ValueError(f"{interrupted_dir} is not an interrupted run")
    by_id = {record.id: record for record in attempted_records}
    if len(by_id) != len(attempted_records):
        raise ValueError("Duplicate record IDs in attempted_records")
    rows = read_predictions(interrupted_dir / "predictions.jsonl")
    ids = [row["id"] for row in rows]
    missing = set(ids) - set(by_id)
    if missing:
        raise ValueError(f"Interrupted run completed rows outside attempted_records: {sorted(missing)[:5]}")
    completed_records = [by_id[row_id] for row_id in ids]
    decisions = [Decision.model_validate(row["decision"]) for row in rows]
    latencies = [row["latency_ms"] for row in rows]
    summary = summarize(completed_records, decisions, latencies)
    summary["mode"] = manifest["mode"]
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("xb") as stream:
        stream.write((interrupted_dir / "predictions.jsonl").read_bytes())
    closed_manifest = {
        **manifest, "status": "complete",
        "dataset_digest": _dataset_digest(completed_records),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "closed_from_interrupted_run": {
            "run_dir": str(interrupted_dir),
            "manifest_sha256": sha256(interrupted_dir / "manifest.json"),
            "predictions_sha256": sha256(interrupted_dir / "predictions.jsonl"),
            "rows_completed": len(rows),
        },
        "note": (
            "Closed from an interrupted run: contains exactly the rows that run "
            "completed before failing, unaltered, with a digest recomputed over "
            "only that completed subset rather than the larger original input."
        ),
    }
    (output / "manifest.json").write_text(json.dumps(closed_manifest, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def merge_runs(records: list[Record], run_dirs: list[Path], output: Path) -> dict:
    """Merge disjoint, complete physical runs covering exactly this record set.

    Rejects a run that is not "complete", a row completed by more than one run,
    a run whose predictions do not exactly match its own recomputed dataset
    digest (tamper/mismatch detection), and any gap or extra row versus the
    full target record set. Preserves true physical request order across runs
    so latency `first_request`/`p50`/`p95` reflect the real session, not the
    dataset's row order.
    """
    taxonomy_id = records_taxonomy(records)
    by_id = {record.id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Duplicate record IDs in the target dataset")
    all_rows: list[dict] = []
    seen_ids: set[str] = set()
    contributions = []
    modes: set[str] = set()
    for run_dir in run_dirs:
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "complete":
            raise ValueError(f"{run_dir} is not a complete run")
        if manifest.get("taxonomy_id") != taxonomy_id:
            raise ValueError(f"{run_dir} taxonomy does not match the target dataset")
        rows = read_predictions(run_dir / "predictions.jsonl")
        ids_here = [row["id"] for row in rows]
        overlap = seen_ids & set(ids_here)
        if overlap:
            raise ValueError(f"Row completed by more than one run: {sorted(overlap)[:5]}")
        missing = set(ids_here) - set(by_id)
        if missing:
            raise ValueError(f"{run_dir} completed rows outside the target dataset: {sorted(missing)[:5]}")
        run_records = [by_id[row_id] for row_id in ids_here]
        expected_digest = _dataset_digest(run_records)
        if expected_digest != manifest.get("dataset_digest"):
            raise ValueError(f"{run_dir} predictions do not match its own recorded dataset digest")
        all_rows.extend(rows)
        seen_ids.update(ids_here)
        modes.add(manifest["mode"])
        contributions.append({
            "run_dir": str(run_dir), "manifest_sha256": sha256(run_dir / "manifest.json"),
            "predictions_sha256": sha256(run_dir / "predictions.jsonl"), "rows": len(ids_here),
        })
    target_ids = set(by_id)
    if seen_ids != target_ids:
        missing, extra = sorted(target_ids - seen_ids), sorted(seen_ids - target_ids)
        raise ValueError(f"Merge incomplete or mismatched: missing={missing[:5]} extra={extra[:5]}")
    if len(modes) != 1:
        raise ValueError("Contributing runs do not share the same probe/benchmark mode")
    ordered_records = [by_id[row["id"]] for row in all_rows]
    decisions = [Decision.model_validate(row["decision"]) for row in all_rows]
    latencies = [row["latency_ms"] for row in all_rows]
    summary = summarize(ordered_records, decisions, latencies)
    summary["mode"] = modes.pop()
    output.mkdir(parents=True, exist_ok=False)
    with (output / "predictions.jsonl").open("xb") as stream:
        for run_dir in run_dirs:
            stream.write((run_dir / "predictions.jsonl").read_bytes())
    merged_manifest = {
        "status": "complete", "mode": summary["mode"], "taxonomy_id": taxonomy_id,
        "total_rows": len(records),
        "note": (
            "Reconstructed from multiple physical probe runs after an interrupted "
            "remote benchmark; see merged_from for each contributing run's exact "
            "provenance. Latency first_request/p50/p95 reflect the full chronological "
            "session across all contributing runs, in physical request order."
        ),
        "merged_from": contributions,
    }
    (output / "manifest.json").write_text(json.dumps(merged_manifest, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--taxonomy", required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True, dest="run_dirs")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    records = load_records(args.data, taxonomy_id=args.taxonomy)
    result = merge_runs(records, args.run_dirs, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
