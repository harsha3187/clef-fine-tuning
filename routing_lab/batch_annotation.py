"""Audited, resumable two-pass batch labeling; never human-reviewed gold.

Remote execution requires an explicit cumulative physical-request budget. A
serial import is a read-only recovery operation, not a relabeling operation.
The `retry` subcommand adds explicit, bounded retries for locally rejected
batch annotations only; it never relaxes evidence validation or touches a
successful pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import shutil
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from routing_lab.annotation import (
    POLICY_VERSION, AnnotationItem, Judge, _completed, _results, digest, judge_prompt, load_plan,
)
from routing_lab.annotation_judge import JudgeResponse, _finite_float, _reject_constant, _strict_object
from routing_lab.batch_format import Batch, batch_payload, batch_prompt, batch_schema, build_batches, validate_batch
from routing_lab.data import Contract, Record, write_records
from routing_lab.evaluation import sha256

BATCH_POLICY_VERSION = "blind-double-batch-annotation-v1"
RETRY_POLICY_VERSION = "strict-batch-validation-retry-v1"
_BUDGET_METADATA = {"timeout_seconds", "max_output_tokens", "config_sha256"}
_CODE_FILES = (
    "batch_annotation.py", "batch_format.py", "annotation.py", "annotation_judge.py",
    "data.py", "taxonomy.py", "backends.py",
)
LOGGER = logging.getLogger(__name__)


def _loads(text: str) -> dict:
    value = json.loads(text, object_pairs_hook=_strict_object, parse_constant=_reject_constant,
                       parse_float=_finite_float)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _read_json(path: Path) -> dict:
    return _loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if text and not text.endswith("\n"):
        raise ValueError(f"Incomplete journal/artifact line: {path.name}")
    if any(not line.strip() for line in text.splitlines()):
        raise ValueError(f"Blank journal/artifact line: {path.name}")
    return [_loads(line) for line in text.splitlines()]


def _encoded_jsonl(rows: list[dict]) -> bytes:
    return "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows).encode("utf-8")


def _atomic_bytes(path: Path, data: bytes) -> None:
    pending = path.with_name(path.name + ".pending")
    with pending.open("wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)


def _atomic_json(path: Path, value: dict) -> None:
    _atomic_bytes(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode("utf-8"))


@contextmanager
def _lock(output: Path) -> Iterator[None]:
    path = output / ".batch-annotation.lock"
    with path.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        yield
    finally:
        path.unlink()


def _append(path: Path, events: list[dict], event: dict) -> None:
    value = {
        **event, "utc": datetime.now(timezone.utc).isoformat(),
        "previous_sha256": events[-1]["event_sha256"] if events else None,
    }
    value["event_sha256"] = digest(value)
    with path.open("ab") as stream:
        stream.write(_encoded_jsonl([value]))
        stream.flush()
        os.fsync(stream.fileno())
    events.append(value)


def _positive_int(value: object) -> bool:
    return type(value) is int and value > 0


def _usage(event: dict) -> bool:
    fields = ("input_tokens", "output_tokens")
    if not any(key in event for key in fields):
        return False
    if not all(type(event.get(key)) is int and event[key] >= 0 for key in fields):
        raise ValueError("Invalid physical-response token usage")
    return True


def _response_metadata(response: JudgeResponse) -> dict:
    value = {
        "model": response.model, "response_id": response.response_id,
        "input_tokens": response.input_tokens, "output_tokens": response.output_tokens,
        "elapsed_ms": response.elapsed_ms,
    }
    _validate_response_metadata(value)
    return value


def _validate_response_metadata(value: dict) -> None:
    if not _usage(value):
        raise ValueError("Completed physical response has no token usage")
    if any(not isinstance(value.get(key), str) or not value[key].strip()
           for key in ("model", "response_id")):
        raise ValueError("Physical response lacks an actual model or response identity")
    if "elapsed_ms" in value and (
        type(value["elapsed_ms"]) not in (int, float)
        or not math.isfinite(value["elapsed_ms"]) or value["elapsed_ms"] < 0
    ):
        raise ValueError("Invalid physical-response duration")


def _code_hashes() -> dict[str, str]:
    return {name: sha256(Path(__file__).with_name(name)) for name in _CODE_FILES}


def _identity(plan: Path, plan_manifest: dict, items: list[AnnotationItem],
              catalogs: dict[str, dict[str, Contract]], metadata: dict, batches: list[Batch],
              imported: dict, batch_size: int, max_input_chars: int, imported_batch: dict | None = None) -> dict:
    return {
        "policy_version": BATCH_POLICY_VERSION,
        "plan_sha256": sha256(plan / "manifest.json"),
        "plan_files": plan_manifest["files"],
        "judge": metadata, "implementation_sha256": _code_hashes(),
        "batch_size": batch_size, "max_input_chars": max_input_chars,
        "selected_ids_sha256": digest([item.id for item in items]),
        "selected_inputs_sha256": digest([item.model_dump() for item in items]),
        "selected_count": len(items),
        "prompt_sha256": {
            taxonomy: [digest(batch_prompt(contracts, index)) for index in (0, 1)]
            for taxonomy, contracts in catalogs.items()
        },
        "batches": [{
            "id": batch.id, "taxonomy_id": batch.taxonomy_id,
            "ids": [item.id for item in batch.items],
            "schema_sha256": digest(batch_schema(batch)),
            "payload_sha256": [digest(batch_payload(batch, index)) for index in (0, 1)],
        } for batch in batches],
        "serial_import": imported,
        "batch_import": imported_batch or {},
    }


def _legacy(source: Path, plan: Path, items: list[AnnotationItem], catalogs: dict,
            metadata: dict) -> tuple[dict, list[dict], dict[tuple[str, int], dict]]:
    """Recover complete rows from the journal, not the legacy failure's stale counters."""
    if (source / ".annotation.lock").exists():
        raise ValueError("Serial run is still locked; stop it before importing")
    manifest = _read_json(source / "manifest.json")
    if manifest.get("policy_version") != POLICY_VERSION or manifest.get("plan_sha256") != sha256(
        plan / "manifest.json"
    ):
        raise ValueError("Serial import requires the same annotation plan and policy")
    old_metadata = manifest["judge"]
    if {k: v for k, v in old_metadata.items() if k not in _BUDGET_METADATA} != {
        k: v for k, v in metadata.items() if k not in _BUDGET_METADATA
    }:
        raise ValueError("Serial import requires identical provider, endpoint, deployment, model and auth policy")
    count = manifest["selected_count"]
    if not _positive_int(count) or count > len(items):
        raise ValueError("Invalid serial selected-row inventory")
    selected = items[:count]
    if digest([item.id for item in selected]) != manifest["selected_ids_sha256"]:
        raise ValueError("Serial selected IDs differ from the frozen plan")
    prompts = {taxonomy: [judge_prompt(contracts, index) for index in (0, 1)]
               for taxonomy, contracts in catalogs.items()}
    if manifest["prompt_sha256"] != digest(prompts):
        raise ValueError("Serial prompts differ from the frozen annotation policy")
    if manifest["annotation_code_sha256"] != sha256(Path(__file__).with_name("annotation.py")):
        raise ValueError("Serial annotation implementation differs from the validated importer")
    events = _read_jsonl(source / "events.jsonl")
    try:
        completed = _completed(events, selected)
    except (KeyError, TypeError) as error:
        raise ValueError("Malformed serial audit journal") from error
    started = [event for event in events if event["event"] == "started"]
    if not started or any(type(event["request_number"]) is not int
                          or type(event["pass_index"]) is not int for event in events):
        raise ValueError("Invalid serial physical-request identities")
    for event in events:
        if event["event"] in {"completed", "failed"}:
            _usage(event)
        if event["event"] == "completed":
            _validate_response_metadata(event)
    complete_ids = {item.id for item in selected if all((item.id, index) in completed for index in (0, 1))}
    if {key[0] for key in completed} != complete_ids:
        raise ValueError("Serial import has a partially completed row; partial-row import is unsupported")
    if manifest["status"] in {"complete_selected_rows", "request_budget_reached"}:
        if sha256(source / "events.jsonl") != manifest["events_sha256"]:
            raise ValueError("Completed serial audit artifacts changed")
        if sha256(source / "results.jsonl") != manifest["results_sha256"]:
            raise ValueError("Completed serial result artifacts changed")
        if _read_jsonl(source / "results.jsonl") != _results(selected, completed):
            raise ValueError("Serial results differ from independently validated assessments")
    elif manifest["status"] not in {"failed", "incomplete"}:
        raise ValueError("Unsupported serial run status")
    descriptor = {
        "manifest_sha256": sha256(source / "manifest.json"),
        "events_sha256": sha256(source / "events.jsonl"),
        "selected_count": count, "requests_attempted": len(started),
        "completed_rows": len(complete_ids), "completed_assessments": len(completed),
        "ids": [item.id for item in selected if item.id in complete_ids],
        "prompt_sha256": manifest["prompt_sha256"],
        "annotation_code_sha256": manifest["annotation_code_sha256"],
    }
    votes = {}
    for key, event in completed.items():
        votes[key] = {
            "id": event["id"], "pass_index": event["pass_index"], "input_sha256": event["input_sha256"],
            "annotation": event["annotation"], "model": event["model"], "response_id": event["response_id"],
            "origin": {
                "kind": "serial", "request_number": event["request_number"],
                "source_events_sha256": descriptor["events_sha256"],
                "source_manifest_sha256": descriptor["manifest_sha256"],
                "prompt_sha256": manifest["prompt_sha256"],
                "annotation_code_sha256": manifest["annotation_code_sha256"],
            },
        }
    return descriptor, events, votes


def _imported(run: Path, identity: dict, plan: Path, items: list[AnnotationItem],
              catalogs: dict) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    descriptor = identity["serial_import"]
    if not descriptor:
        return [], {}
    source = run / "serial_import"
    for name in ("manifest", "events"):
        extension = ".jsonl" if name == "events" else ".json"
        if sha256(source / (name + extension)) != descriptor[name + "_sha256"]:
            raise ValueError("Imported serial provenance artifacts changed")
    actual, events, votes = _legacy(source, plan, items, catalogs, identity["judge"])
    if actual != descriptor:
        raise ValueError("Imported serial provenance inventory changed")
    return events, votes


def _import_batch(source: Path, plan: Path, items: list[AnnotationItem],
                  catalogs: dict[str, dict[str, Contract]],
                  metadata: dict) -> tuple[dict, list[dict], dict[tuple[str, int], dict]]:
    """One-off import of a PRIOR BATCHED run's completed rows across a deliberate,
    backward-compatible relaxation of validate_annotation (for example, allowing a
    clarification-routed label to carry missing_information). Unlike _legacy(), this
    does NOT require the source run's implementation hash to match current code:
    every imported vote is independently re-validated under CURRENT code via
    _replay(), which is what actually decides acceptance. Only rows with both
    passes complete are imported; a differing observed model version, a locked
    source run, or any row outside the current plan is rejected.
    """
    if (source / ".batch-annotation.lock").exists():
        raise ValueError("Source batch run is still locked; stop it before importing")
    source_manifest = _read_json(source / "manifest.json")
    source_identity = _read_json(source / "identity.json")
    if source_manifest.get("policy_version") != BATCH_POLICY_VERSION or source_identity.get(
        "plan_sha256"
    ) != sha256(plan / "manifest.json"):
        raise ValueError("Batch import requires the identical annotation plan and policy")
    source_judge = source_identity["judge"]
    if {k: v for k, v in source_judge.items() if k not in _BUDGET_METADATA} != {
        k: v for k, v in metadata.items() if k not in _BUDGET_METADATA
    }:
        raise ValueError("Batch import requires identical provider, endpoint, deployment, model and auth policy")
    if source_identity.get("serial_import") or source_identity.get("batch_import"):
        raise ValueError("Importing a batch run that itself chained another import is not supported")
    by_id = {item.id: item for item in items}
    batches = []
    for descriptor in source_identity["batches"]:
        try:
            batch_items = tuple(by_id[row_id] for row_id in descriptor["ids"])
        except KeyError as error:
            raise ValueError("Source batch references a row outside the current plan") from error
        batches.append(Batch(batch_items))
    events = _read_jsonl(source / "batch_events.jsonl")
    votes, _ = _replay(events, batches, {}, 0, source_identity)
    all_batch_ids = {row_id for descriptor in source_identity["batches"] for row_id in descriptor["ids"]}
    complete_ids = {row_id for row_id in all_batch_ids if all((row_id, index) in votes for index in (0, 1))}
    if {key[0] for key in votes} != complete_ids:
        raise ValueError("Source batch run has a partially completed row; partial-row import is unsupported")
    requests_attempted = sum(event["event"] == "started" for event in events)
    descriptor_out = {
        "manifest_sha256": sha256(source / "manifest.json"),
        "identity_sha256": sha256(source / "identity.json"),
        "events_sha256": sha256(source / "batch_events.jsonl"),
        "selected_count": len(items), "requests_attempted": requests_attempted,
        "completed_rows": len(complete_ids), "completed_assessments": len(votes),
        "ids": sorted(complete_ids),
        "source_implementation_sha256": source_identity["implementation_sha256"],
        "revalidated_under_implementation_sha256": _code_hashes(),
    }
    renamed_votes = {
        key: {**vote, "origin": {
            "kind": "imported_batch", "source_run_manifest_sha256": descriptor_out["manifest_sha256"],
            "source_run_events_sha256": descriptor_out["events_sha256"], "original_origin": vote["origin"],
        }}
        for key, vote in votes.items()
    }
    return descriptor_out, events, renamed_votes


def _imported_batch(run: Path, identity: dict, plan: Path, items: list[AnnotationItem],
                    catalogs: dict) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    descriptor = identity["batch_import"]
    if not descriptor:
        return [], {}
    source = run / "batch_import"
    for name, extension in (("manifest", ".json"), ("identity", ".json"), ("events", ".jsonl")):
        key = "events_sha256" if name == "events" else f"{name}_sha256"
        filename = "batch_events.jsonl" if name == "events" else name + extension
        if sha256(source / filename) != descriptor[key]:
            raise ValueError("Imported batch provenance artifacts changed")
    actual, events, votes = _import_batch(source, plan, items, catalogs, identity["judge"])
    if actual != descriptor:
        raise ValueError("Imported batch provenance inventory changed")
    return events, votes


def _replay(events: list[dict], batches: list[Batch], imported_votes: dict,
            imported_requests: int, identity: dict) -> tuple[dict, set[tuple[str, int]]]:
    votes = dict(imported_votes)
    batch_map = {batch.id: batch for batch in batches}
    descriptions = {batch["id"]: batch for batch in identity["batches"]}
    starts: dict[int, dict] = {}
    finished: set[int] = set()
    completed: set[tuple[str, int]] = set()
    models = {vote["model"] for vote in votes.values()}
    if len(models) > 1:
        raise ValueError("Imported assessments have differing actual model versions")
    previous = None
    for event in events:
        unhashed = {key: value for key, value in event.items() if key != "event_sha256"}
        if event.get("previous_sha256") != previous or event.get("event_sha256") != digest(unhashed):
            raise ValueError("Batch journal hash chain changed")
        previous = event["event_sha256"]
        number, index, batch_id = event["request_number"], event["pass_index"], event["batch_id"]
        if type(number) is not int or type(index) is not int or index not in (0, 1) or batch_id not in batch_map:
            raise ValueError("Invalid batch request identity")
        key = (batch_id, index)
        description = descriptions[batch_id]
        if event["event"] == "started":
            if number != imported_requests + len(starts) + 1 or key in completed:
                raise ValueError("Invalid physical-request sequence or duplicate successful pass")
            expected = {
                "schema_sha256": description["schema_sha256"],
                "payload_sha256": description["payload_sha256"][index],
                "prompt_sha256": identity["prompt_sha256"][description["taxonomy_id"]][index],
            }
            if any(event.get(field) != value for field, value in expected.items()):
                raise ValueError("Journal request differs from frozen prompt, schema or input")
            starts[number] = event
        elif event["event"] in {"completed", "failed"}:
            if number not in starts or number in finished:
                raise ValueError("Missing request start or duplicate terminal response")
            start = starts[number]
            if any(event[field] != start[field] for field in ("batch_id", "pass_index")):
                raise ValueError("Batch response differs from its physical request")
            finished.add(number)
            _usage(event)
            if event["event"] == "failed":
                continue
            if key in completed:
                raise ValueError("Duplicate completed batch pass")
            _validate_response_metadata(event)
            if models and event["model"] not in models:
                raise ValueError("Observed judge model version changed")
            models.add(event["model"])
            validated = validate_batch(event["payload"], batch_map[batch_id])
            for item in batch_map[batch_id].items:
                vote_key = (item.id, index)
                if vote_key in votes:
                    raise ValueError("A row was assessed by multiple successful requests")
                votes[vote_key] = {
                    "id": item.id, "pass_index": index, "input_sha256": item.input_sha256,
                    "annotation": validated[item.id].model_dump(),
                    "model": event["model"], "response_id": event["response_id"],
                    "origin": {
                        "kind": "batch", "request_number": number, "batch_id": batch_id,
                        "prompt_sha256": start["prompt_sha256"], "schema_sha256": start["schema_sha256"],
                        "payload_sha256": start["payload_sha256"],
                    },
                }
            completed.add(key)
        else:
            raise ValueError("Unknown batch journal event")
    return votes, completed


def _vote_rows(items: list[AnnotationItem], votes: dict) -> list[dict]:
    return [votes[(item.id, index)] for item in items for index in (0, 1) if (item.id, index) in votes]


def _result_rows(items: list[AnnotationItem], votes: dict) -> list[dict]:
    rows = _results(items, votes)
    for row in rows:
        row["assessment_origins"] = [votes[(row["id"], index)]["origin"] for index in (0, 1)]
        row["assessment_models"] = [votes[(row["id"], index)]["model"] for index in (0, 1)]
    return rows


def _summary(items: list[AnnotationItem], votes: dict, events: list[dict], legacy_events: list[dict],
             batches: list[Batch], completed: set) -> dict:
    physical_events = [*legacy_events, *events]
    attempted = sum(event["event"] == "started" for event in physical_events)
    metered = [event for event in physical_events
               if event["event"] in {"completed", "failed"} and _usage(event)]
    results = _result_rows(items, votes)
    imported_attempts = sum(event["event"] == "started" for event in legacy_events)
    return {
        "requests_attempted": attempted,
        "new_requests_attempted": attempted - imported_attempts,
        "imported_requests_attempted": imported_attempts,
        "imported_completed_rows": len({key[0] for key, vote in votes.items()
                                       if vote["origin"]["kind"] in {"serial", "imported_batch"}}),
        "new_batch_count": len(batches),
        "required_successful_new_requests": 2 * len(batches),
        "remaining_successful_requests": 2 * len(batches) - len(completed),
        "estimated_total_requests_without_new_failures": attempted + 2 * len(batches) - len(completed),
        "completed_batches": sum(all((batch.id, index) in completed for index in (0, 1)) for batch in batches),
        "completed_assessments": len(votes), "completed_rows": len(results),
        "pending_rows": len(items) - len(results),
        "result_status_counts": dict(Counter(row["status"] for row in results)),
        "observed_judge_models": sorted({vote["model"] for vote in votes.values()}),
        "known_input_tokens": sum(event["input_tokens"] for event in metered),
        "known_output_tokens": sum(event["output_tokens"] for event in metered),
        "requests_without_valid_usage": attempted - len(metered),
    }


def _checkpoint(output: Path, manifest: dict, status: str, items: list[AnnotationItem],
                votes: dict, events: list[dict], legacy_events: list[dict],
                batches: list[Batch], completed: set) -> None:
    _atomic_bytes(output / "votes.jsonl", _encoded_jsonl(_vote_rows(items, votes)))
    _atomic_bytes(output / "results.jsonl", _encoded_jsonl(_result_rows(items, votes)))
    manifest.update(
        **_summary(items, votes, events, legacy_events, batches, completed),
        status=status, event_count=len(events),
        events_sha256=sha256(output / "batch_events.jsonl"),
        journal_head_sha256=events[-1]["event_sha256"] if events else None,
        votes_sha256=sha256(output / "votes.jsonl"), results_sha256=sha256(output / "results.jsonl"),
    )
    _atomic_json(output / "manifest.json", manifest)


def _verify_checkpoint(output: Path, manifest: dict, items: list[AnnotationItem],
                       events: list[dict], legacy_events: list[dict], batches: list[Batch],
                       imported_votes: dict, identity: dict) -> tuple[dict, set]:
    count = manifest["event_count"]
    if type(count) is not int or count < 0 or count > len(events):
        raise ValueError("Batch journal was truncated")
    prefix = events[:count]
    if hashlib.sha256(_encoded_jsonl(prefix)).hexdigest() != manifest["events_sha256"]:
        raise ValueError("Checkpointed batch journal artifacts changed")
    if (prefix[-1]["event_sha256"] if prefix else None) != manifest["journal_head_sha256"]:
        raise ValueError("Checkpoint journal head changed")
    if len(events) != count and manifest["status"] != "in_progress":
        raise ValueError("Completed batch journal artifacts changed")
    imported_requests = sum(event["event"] == "started" for event in legacy_events)
    votes, completed = _replay(prefix, batches, imported_votes, imported_requests, identity)
    current_votes, current_completed = (
        _replay(events, batches, imported_votes, imported_requests, identity)
        if len(events) > count else (votes, completed)
    )
    for name, expected in (
        ("votes", _vote_rows(items, votes)), ("results", _result_rows(items, votes)),
    ):
        path = output / f"{name}.jsonl"
        actual = _read_jsonl(path)
        matches_checkpoint = sha256(path) == manifest[f"{name}_sha256"] and actual == expected
        # A process can stop after writing derived artifacts but before committing
        # their manifest. Only exact replayed content, never arbitrary edits, recovers.
        current = (_vote_rows(items, current_votes) if name == "votes"
                   else _result_rows(items, current_votes))
        interrupted_checkpoint = (
            manifest["status"] == "in_progress" and len(events) > count and actual == current
        )
        if not matches_checkpoint and not interrupted_checkpoint:
            raise ValueError("Stored votes/results artifacts changed or differ from the audit journal")
    summary = _summary(items, votes, prefix, legacy_events, batches, completed)
    if any(manifest.get(key) != value for key, value in summary.items()):
        raise ValueError("Progress counters differ from the physical audit journal")
    if manifest["status"] == "complete_selected_rows" and len(votes) != 2 * len(items):
        raise ValueError("Run falsely claims completion")
    return current_votes, current_completed


def _load_run(plan: Path, output: Path, metadata: dict | None = None) -> tuple:
    plan_manifest, all_items, catalogs = load_plan(plan)
    items = [item for item in all_items if item.eligibility == "ready"]
    manifest, identity = _read_json(output / "manifest.json"), _read_json(output / "identity.json")
    if sha256(output / "identity.json") != manifest["identity_sha256"]:
        raise ValueError("Frozen run identity changed")
    identity.setdefault("batch_import", {})  # absent in runs created before this import path existed
    if manifest["policy_version"] != identity["policy_version"] or manifest["plan_sha256"] != identity["plan_sha256"]:
        raise ValueError("Manifest differs from the frozen run identity")
    if manifest["status"] not in {"in_progress", "failed", "request_budget_reached", "complete_selected_rows"}:
        raise ValueError("Unsupported annotation checkpoint status")
    approvals = manifest["budget_approvals"]
    if not isinstance(approvals, list) or not approvals or any(
        not _positive_int(approval.get("request_limit")) or not isinstance(approval.get("utc"), str)
        for approval in approvals
    ) or approvals[-1]["request_limit"] != manifest["approved_request_limit"]:
        raise ValueError("Invalid cumulative request budget approval history")
    if metadata is not None and metadata != identity["judge"]:
        raise ValueError("Resume requires the identical judge configuration")
    if identity["policy_version"] != BATCH_POLICY_VERSION:
        raise ValueError("Unsupported batch annotation policy")
    for name, expected in manifest["snapshot_sha256"].items():
        if Path(name).name != name or sha256(output / "snapshot" / name) != expected:
            raise ValueError("Frozen input/catalog snapshot changed")
    expected_snapshots = {
        "manifest.json": sha256(plan / "manifest.json"),
        **{name: expected for name, expected in plan_manifest["files"].items()},
    }
    if manifest["snapshot_sha256"] != expected_snapshots:
        raise ValueError("Frozen snapshot inventory differs from the annotation plan")
    legacy_events, imported_votes = _imported(output, identity, plan, items, catalogs)
    batch_import_events, batch_imported_votes = _imported_batch(output, identity, plan, items, catalogs)
    legacy_events = legacy_events + batch_import_events
    imported_votes = {**imported_votes, **batch_imported_votes}
    skipped = set(identity["serial_import"].get("ids", [])) | set(identity["batch_import"].get("ids", []))
    batches = build_batches([item for item in items if item.id not in skipped], catalogs,
                            batch_size=identity["batch_size"], max_input_chars=identity["max_input_chars"])
    expected = _identity(plan, plan_manifest, items, catalogs, identity["judge"], batches,
                         identity["serial_import"], identity["batch_size"], identity["max_input_chars"],
                         identity["batch_import"])
    if identity != expected:
        raise ValueError("Resume/export requires identical plan, selection, prompts, schemas and implementation")
    if _read_jsonl(output / "selected.jsonl") != [item.model_dump() for item in items]:
        raise ValueError("Frozen selected input inventory changed")
    if sha256(output / "selected.jsonl") != manifest["selected_sha256"]:
        raise ValueError("Frozen selected input artifact changed")
    events = _read_jsonl(output / "batch_events.jsonl")
    votes, completed = _verify_checkpoint(
        output, manifest, items, events, legacy_events, batches, imported_votes, identity,
    )
    if not _positive_int(manifest["approved_request_limit"]) or manifest["approved_request_limit"] < sum(
        event["event"] == "started" for event in [*legacy_events, *events]
    ):
        raise ValueError("Recorded request budget is below attempted requests")
    return (manifest, identity, plan_manifest, items, catalogs, batches, events, legacy_events, votes, completed)


def run_annotations(plan: Path, judge: Judge, output: Path, *, max_requests: int,
                    batch_size: int = 100, max_input_chars: int = 128000,
                    allow_remote: bool = False, resume: bool = False,
                    import_serial_run: Path | None = None, import_batch_run: Path | None = None) -> dict:
    """Make at most the cumulative approved number of physical attempts, without retries."""
    if not allow_remote:
        raise PermissionError("Remote annotation requires explicit --allow-remote consent")
    if not _positive_int(max_requests):
        raise ValueError("An explicit positive physical-request budget is required")
    if resume and (import_serial_run is not None or import_batch_run is not None):
        raise ValueError("--import-serial-run/--import-batch-run are supported only for a NEW run")
    if import_serial_run is not None and import_batch_run is not None:
        raise ValueError("Only one of --import-serial-run or --import-batch-run may be used at a time")
    if type(batch_size) is not int or not 1 <= batch_size <= 100 or type(max_input_chars) is not int or not (
        1024 <= max_input_chars <= 1_000_000
    ):
        raise ValueError("Invalid batch size or input character bound")
    from azure.core.exceptions import AzureError
    from openai import APIError

    from routing_lab.annotation_judge import JudgeError

    metadata = judge.metadata()
    if metadata.get("max_retries", 0) != 0 or metadata.get("follow_redirects", False) is not False:
        raise ValueError("Physical-request accounting requires disabled SDK retries and redirects")
    if not resume:
        plan_manifest, all_items, catalogs = load_plan(plan)
        items = [item for item in all_items if item.eligibility == "ready"]
        if not items:
            raise ValueError("No annotation-ready rows")
        imported, legacy_events, votes = {}, [], {}
        if import_serial_run is not None:
            imported, legacy_events, votes = _legacy(import_serial_run, plan, items, catalogs, metadata)
        imported_batch, batch_legacy_events, batch_votes = {}, [], {}
        if import_batch_run is not None:
            imported_batch, batch_legacy_events, batch_votes = _import_batch(
                import_batch_run, plan, items, catalogs, metadata,
            )
        legacy_events = legacy_events + batch_legacy_events
        votes = {**votes, **batch_votes}
        if imported.get("requests_attempted", 0) + imported_batch.get("requests_attempted", 0) > max_requests:
            raise ValueError("Request budget is below imported physical requests already attempted")
        skipped = set(imported.get("ids", [])) | set(imported_batch.get("ids", []))
        batches = build_batches([item for item in items if item.id not in skipped], catalogs,
                                batch_size=batch_size, max_input_chars=max_input_chars)
        identity = _identity(plan, plan_manifest, items, catalogs, metadata, batches,
                             imported, batch_size, max_input_chars, imported_batch)
        output.mkdir(parents=True, exist_ok=False)
    with _lock(output):
        if not resume:
            snapshot = output / "snapshot"
            snapshot.mkdir()
            for name in ("manifest.json", *plan_manifest["files"]):
                shutil.copyfile(plan / name, snapshot / name)
            if import_serial_run is not None:
                destination = output / "serial_import"
                destination.mkdir()
                for name in ("manifest.json", "events.jsonl", "results.jsonl"):
                    if (import_serial_run / name).exists():
                        shutil.copyfile(import_serial_run / name, destination / name)
                if any(sha256(destination / name) != imported[key] for name, key in (
                    ("manifest.json", "manifest_sha256"), ("events.jsonl", "events_sha256"),
                )):
                    raise ValueError("Serial source changed during the import")
            if import_batch_run is not None:
                destination = output / "batch_import"
                destination.mkdir()
                shutil.copyfile(import_batch_run / "manifest.json", destination / "manifest.json")
                shutil.copyfile(import_batch_run / "identity.json", destination / "identity.json")
                shutil.copyfile(import_batch_run / "batch_events.jsonl", destination / "batch_events.jsonl")
                if any(sha256(destination / name) != imported_batch[key] for name, key in (
                    ("manifest.json", "manifest_sha256"), ("identity.json", "identity_sha256"),
                    ("batch_events.jsonl", "events_sha256"),
                )):
                    raise ValueError("Batch source changed during the import")
            _atomic_json(output / "identity.json", identity)
            _atomic_bytes(output / "selected.jsonl", _encoded_jsonl([item.model_dump() for item in items]))
            _atomic_bytes(output / "batch_events.jsonl", b"")
            events, completed = [], set()
            manifest = {
                "policy_version": BATCH_POLICY_VERSION, "plan_sha256": identity["plan_sha256"],
                "identity_sha256": sha256(output / "identity.json"),
                "selected_sha256": sha256(output / "selected.jsonl"),
                "snapshot_sha256": {path.name: sha256(path) for path in snapshot.iterdir()},
                "approved_request_limit": max_requests, "budget_approvals": [{
                    "utc": datetime.now(timezone.utc).isoformat(), "request_limit": max_requests,
                }],
            }
            _checkpoint(output, manifest, "in_progress", items, votes, events, legacy_events, batches, completed)
        loaded = _load_run(plan, output, metadata)
        manifest, identity, _, items, catalogs, batches, events, legacy_events, votes, completed = loaded
        if identity["batch_size"] != batch_size or identity["max_input_chars"] != max_input_chars:
            raise ValueError("Resume requires identical batching limits")
        spent = sum(event["event"] == "started" for event in [*legacy_events, *events])
        if max_requests < spent:
            raise ValueError("Request budget is below cumulative physical requests already attempted")
        if not manifest["budget_approvals"] or max_requests != manifest["approved_request_limit"]:
            manifest["budget_approvals"].append({
                "utc": datetime.now(timezone.utc).isoformat(), "request_limit": max_requests,
            })
        manifest["approved_request_limit"] = max_requests
        _checkpoint(output, manifest, "in_progress", items, votes, events,
                    legacy_events, batches, completed)
        for batch in batches:
            for index in (0, 1):
                if (batch.id, index) in completed:
                    continue
                if spent >= max_requests:
                    _checkpoint(output, manifest, "request_budget_reached", items, votes, events,
                                legacy_events, batches, completed)
                    return manifest
                spent += 1
                prompt = batch_prompt(catalogs[batch.taxonomy_id], index)
                payload, schema = batch_payload(batch, index), batch_schema(batch)
                base = {"request_number": spent, "batch_id": batch.id, "pass_index": index}
                _append(output / "batch_events.jsonl", events, {
                    **base, "event": "started", "prompt_sha256": digest(prompt),
                    "schema_sha256": digest(schema), "payload_sha256": digest(payload),
                })
                _checkpoint(output, manifest, "in_progress", items, votes, events, legacy_events, batches, completed)
                response = None
                accepted = False
                try:
                    response = judge.annotate(system_prompt=prompt, input_payload=payload, schema=schema)
                    response_metadata = _response_metadata(response)
                    annotations = validate_batch(response.payload, batch)
                    models = {vote["model"] for vote in votes.values()}
                    if models and response.model not in models:
                        raise ValueError("Observed judge model version changed during the annotation run")
                    _append(output / "batch_events.jsonl", events, {
                        **base, **response_metadata, "event": "completed", "payload": response.payload,
                    })
                    start = events[-2]
                    for item in batch.items:
                        votes[(item.id, index)] = {
                            "id": item.id, "pass_index": index, "input_sha256": item.input_sha256,
                            "annotation": annotations[item.id].model_dump(),
                            "model": response.model, "response_id": response.response_id,
                            "origin": {
                                "kind": "batch", "request_number": spent, "batch_id": batch.id,
                                "prompt_sha256": start["prompt_sha256"], "schema_sha256": start["schema_sha256"],
                                "payload_sha256": start["payload_sha256"],
                            },
                        }
                    completed.add((batch.id, index))
                    accepted = True
                except (ValueError, APIError, AzureError, JudgeError) as error:
                    failure = {**base, "event": "failed", "error_type": type(error).__name__}
                    if response is not None:
                        failure.update(_response_metadata(response))
                        failure["payload"] = response.payload
                    _append(output / "batch_events.jsonl", events, failure)
                    raise
                finally:
                    # Also checkpoint KeyboardInterrupt or an unexpected programmer error
                    # without suppressing it or fabricating a service response.
                    _checkpoint(output, manifest, "in_progress" if accepted else "failed", items, votes,
                                events, legacy_events, batches, completed)
        _checkpoint(output, manifest, "complete_selected_rows", items, votes, events,
                    legacy_events, batches, completed)
        return manifest


def export_consensus(plan: Path, run: Path, output: Path, *, allow_partial: bool = False) -> dict:
    """Revalidate all journaled votes and export separate unreviewed consensus tracks."""
    with _lock(run):
        loaded = _load_run(plan, run)
        manifest, identity, plan_manifest, items, _, _, _, _, votes, _ = loaded
        if manifest["status"] not in {"complete_selected_rows", "request_budget_reached", "failed"}:
            raise ValueError("Interrupted run must be resumed/checkpointed before export")
        results = _result_rows(items, votes)
        accepted = {row["id"]: row for row in results if row["status"] == "silver_consensus"}
        if not allow_partial and len(accepted) != len(items):
            raise ValueError("Not all ready rows have consensus; use explicit --allow-partial")
        if not accepted:
            raise ValueError("No consensus labels to export")
        run_hash = sha256(run / "manifest.json")
        records: dict[str, list[Record]] = {taxonomy: [] for taxonomy in plan_manifest["catalogs"]}
        for item in items:
            if item.id not in accepted:
                continue
            result = accepted[item.id]
            records[item.taxonomy_id].append(Record.model_validate({
                "id": item.id, "taxonomy_id": item.taxonomy_id, "group_id": item.group_id,
                "repository": item.repository, "query": item.query, "context": item.context,
                "expected_route": result["route"], "acceptable_routes": result["acceptable_routes"],
                "tags": [
                    item.source, item.category, item.native_split, BATCH_POLICY_VERSION,
                    f"source:{item.source}", f"native_split:{item.native_split}", f"group:{item.group_id}",
                    "development_exposed" if item.development_exposed else "holdout_candidate_not_frozen",
                ],
                "provenance": json.dumps({
                    "label_origin": "AI blind two-pass consensus; not human gold",
                    "plan_sha256": identity["plan_sha256"], "run_manifest_sha256": run_hash,
                    "run_identity_sha256": manifest["identity_sha256"], "input_sha256": item.input_sha256,
                    "native_id": item.native_id, "source": item.source, "native_split": item.native_split,
                    "development_exposed": item.development_exposed,
                    "assessment_origins": result["assessment_origins"],
                    "assessment_models": result["assessment_models"],
                }, sort_keys=True),
                "review_status": "unreviewed", "reviewer": "",
            }))
        audit_rows = []
        result_by_id = {row["id"]: row for row in results}
        for item in items:
            if item.id in accepted:
                continue
            audit_rows.append({
                "id": item.id, "taxonomy_id": item.taxonomy_id, "input_sha256": item.input_sha256,
                "status": result_by_id.get(item.id, {}).get("status", "pending_assessments"),
                "review_status": "unreviewed", "reviewer": "",
                "assessments": [votes[(item.id, index)] for index in (0, 1) if (item.id, index) in votes],
            })
        output.mkdir(parents=True, exist_ok=False)
        exported = {
            "status": "incomplete", "mode": "silver_consensus_probe_only", "allow_partial": allow_partial,
            "partial_consensus": len(accepted) != len(items),
            "accepted_count": len(accepted), "ready_count": len(items),
            "unexported_ready_count": len(items) - len(accepted),
            "pending_count": len(items) - len(results),
            "disagreement_or_uncertain_count": sum(row["status"] == "needs_adjudication" for row in results),
            "eligibility_counts": plan_manifest["eligibility_counts"],
            "run_manifest_sha256": run_hash, "run_identity_sha256": manifest["identity_sha256"],
            "plan_sha256": identity["plan_sha256"], "files": {}, "catalogs": {},
            "coverage": {},
        }
        _atomic_json(output / "manifest.json", exported)
        for taxonomy, rows in records.items():
            path = output / f"{taxonomy}.jsonl"
            write_records(path, rows)
            exported["files"][path.name] = {
                "sha256": sha256(path), "count": len(rows),
                "route_counts": dict(Counter(row.expected_route for row in rows)),
            }
            catalog = output / f"{taxonomy}.contracts.json"
            shutil.copyfile(run / "snapshot" / plan_manifest["catalogs"][taxonomy], catalog)
            exported["catalogs"][taxonomy] = {"file": catalog.name, "sha256": sha256(catalog)}
            track_items = [item for item in items if item.taxonomy_id == taxonomy]
            exported["coverage"][taxonomy] = {
                "ready": len(track_items), "consensus": len(rows),
                "development_exposed_consensus": sum(
                    item.development_exposed for item in track_items if item.id in accepted
                ),
                "unexported": len(track_items) - len(rows),
            }
        _atomic_bytes(output / "adjudication.jsonl", _encoded_jsonl(audit_rows))
        exported["files"]["adjudication.jsonl"] = {
            "sha256": sha256(output / "adjudication.jsonl"), "count": len(audit_rows),
        }
        exported["status"] = "complete"
        _atomic_json(output / "manifest.json", exported)
        return exported


def _retry_candidate(loaded: tuple, max_retry_requests: int, max_requests: int) -> dict | None:
    manifest, _, _, _, _, batches, events, _, _, _ = loaded
    starts = Counter(
        (event["batch_id"], event["pass_index"])
        for event in events if event["event"] == "started"
    )
    retries_used = sum(count - 1 for count in starts.values())
    if any(count > 2 for count in starts.values()) or retries_used > max_retry_requests:
        raise ValueError("Historical attempts exceed the approved retry policy")
    if manifest["status"] in {"request_budget_reached", "complete_selected_rows"}:
        return None
    if manifest["status"] != "failed" or not events or events[-1]["event"] != "failed":
        raise ValueError("Interrupted or uncheckpointed requests require inspection, not automatic retries")
    failed = events[-1]
    if failed.get("error_type") not in {"ValueError", "ValidationError"} or "payload" not in failed:
        raise ValueError("Only saved, locally rejected annotation responses are retryable")
    observed = manifest["observed_judge_models"]
    if observed and failed["model"] not in observed:
        raise ValueError("Model-version changes must not be retried automatically")
    batch = next(batch for batch in batches if batch.id == failed["batch_id"])
    try:
        validate_batch(failed["payload"], batch)
    except ValueError:
        pass
    else:
        raise ValueError("The failed response passes annotation validation; investigate its other error")
    key = (failed["batch_id"], failed["pass_index"])
    if starts[key] >= 2:
        raise ValueError("This batch pass failed its single permitted retry; the run remains stopped")
    if retries_used >= max_retry_requests:
        raise ValueError("The cumulative retry allowance is exhausted")
    if manifest["requests_attempted"] >= max_requests:
        raise ValueError("The cumulative physical-request cap leaves no room for a retry")
    return failed


def run_with_retries(plan: Path, judge: Judge, output: Path, *, max_requests: int,
                     max_retry_requests: int, allow_remote: bool = False) -> dict:
    """Resume without changing prompts, successful votes, evidence rules or SDK retries."""
    if not allow_remote:
        raise PermissionError("Remote annotation requires explicit --allow-remote consent")
    if type(max_requests) is not int or max_requests <= 0 or type(max_retry_requests) is not int or (
        max_retry_requests <= 0
    ):
        raise ValueError("Explicit positive cumulative request and retry limits are required")
    lock = output / ".batch-worker.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        loaded = _load_run(plan, output, judge.metadata())
        manifest, identity = loaded[:2]
        baseline = manifest["imported_requests_attempted"] + manifest["required_successful_new_requests"]
        if max_requests > baseline + max_retry_requests:
            raise ValueError("The request cap exceeds the original work plus the approved retry allowance")
        policy = {
            "version": RETRY_POLICY_VERSION,
            "identity_sha256": manifest["identity_sha256"],
            "worker_sha256": sha256(Path(__file__)),
            "max_retries_per_pass": 1,
            "max_retry_requests": max_retry_requests,
            "maximum_cumulative_requests": baseline + max_retry_requests,
        }
        policy_path = output / "retry_policy.json"
        if policy_path.exists():
            if _read_json(policy_path) != policy:
                raise ValueError("Retry policy or worker implementation changed")
        else:
            _atomic_json(policy_path, policy)
        while True:
            candidate = _retry_candidate(loaded, max_retry_requests, max_requests)
            if candidate is not None:
                LOGGER.warning(
                    "Retrying rejected request %s, batch %s pass %s once; original prompt and strict validation retained",
                    candidate["request_number"], candidate["batch_id"], candidate["pass_index"],
                )
            before = loaded[0]["requests_attempted"]
            try:
                return run_annotations(
                    plan, judge, output, max_requests=max_requests,
                    batch_size=identity["batch_size"], max_input_chars=identity["max_input_chars"],
                    allow_remote=True, resume=True,
                )
            except ValueError:
                updated = _load_run(plan, output, judge.metadata())
                if updated[0]["requests_attempted"] <= before:
                    raise
                # Retry only a new, audited annotation rejection, never an integrity/configuration error.
                _retry_candidate(updated, max_retry_requests, max_requests)
                loaded = updated
    finally:
        lock.unlink()


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--plan", type=Path, required=True)
    run.add_argument("--judge-config", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--batch-size", type=int, default=100)
    run.add_argument("--max-input-chars", type=int, default=128000)
    run.add_argument("--max-requests", type=int, required=True,
                     help="Cumulative physical requests INCLUDING every imported serial attempt")
    run.add_argument("--allow-remote", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--import-serial-run", type=Path)
    run.add_argument("--import-batch-run", type=Path,
                     help="One-off import of a prior BATCHED run's completed rows, re-validated under current code")
    export = commands.add_parser("export")
    export.add_argument("--plan", type=Path, required=True)
    export.add_argument("--run", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--allow-partial", action="store_true")
    retry = commands.add_parser("retry", help="Retry one locally rejected batch pass, within a bounded allowance")
    retry.add_argument("--plan", type=Path, required=True)
    retry.add_argument("--judge-config", type=Path, required=True)
    retry.add_argument("--output", type=Path, required=True)
    retry.add_argument("--max-requests", type=int, required=True)
    retry.add_argument("--max-retry-requests", type=int, required=True)
    retry.add_argument("--allow-remote", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    if args.command in {"run", "retry"}:
        from routing_lab.annotation_judge import AzureResponsesJudge, load_judge_config
        with AzureResponsesJudge(load_judge_config(args.judge_config)) as judge:
            if args.command == "run":
                result = run_annotations(
                    args.plan, judge, args.output, batch_size=args.batch_size, max_requests=args.max_requests,
                    max_input_chars=args.max_input_chars, allow_remote=args.allow_remote, resume=args.resume,
                    import_serial_run=args.import_serial_run, import_batch_run=args.import_batch_run,
                )
            else:
                logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
                result = run_with_retries(
                    args.plan, judge, args.output, max_requests=args.max_requests,
                    max_retry_requests=args.max_retry_requests, allow_remote=args.allow_remote,
                )
    else:
        result = export_consensus(args.plan, args.run, args.output, allow_partial=args.allow_partial)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
