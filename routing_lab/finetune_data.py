"""Prepare one catalog-conditioned model's silver data; never launch training.

Only train90 is partitioned. Frozen test10 is read for integrity/leakage audits
and referenced by hash, never exported into the upload bundle.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

from routing_lab.backends import questions, state
from routing_lab.data import Record, fingerprint, load_contracts, load_records, write_records
from routing_lab.dataset_split import (
    DEVELOPMENT_EXPOSED_TAG,
    SPLIT_POLICY_VERSION,
    _assert_group_safe,
    _discriminating_fields,
    stratified_group_split,
)
from routing_lab.evaluation import sha256
from routing_lab.taxonomy import taxonomy_routes

POLICY_VERSION = "shared-silver-train90-80-10-10-v2"
TRACKS = ("coding-v1", "tool-api-v1")
PARTITIONS = ("train", "validation", "calibration")
BUNDLE_FILES = frozenset(
    [f"{track}-{partition}.jsonl" for track in TRACKS for partition in PARTITIONS]
    + [f"mixed-{partition}.jsonl" for partition in PARTITIONS]
    + [f"{track}.contracts.json" for track in TRACKS]
)


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _verify_manifest_payload(bundle: Path, manifest: dict) -> None:
    if manifest.get("status") != "complete" or manifest.get("policy_version") != POLICY_VERSION:
        raise ValueError("Bundle manifest must be complete and use the supported preparation policy")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != BUNDLE_FILES:
        raise ValueError("Bundle files must list exactly the eleven expected payload basenames")
    if {path.name for path in bundle.iterdir()} != BUNDLE_FILES | {"manifest.json"}:
        raise ValueError("Bundle contains missing or unexpected entries; frozen test data must not be uploaded")
    for name, info in files.items():
        if not isinstance(info, dict) or not _valid_digest(info.get("sha256")):
            raise ValueError(f"Invalid bundle sha256 metadata: {name}")
        path = bundle / name
        if path.is_symlink() or not path.is_file() or path.resolve().parent != bundle.resolve():
            raise ValueError(f"Bundle payload must be a regular local file: {name}")
        if sha256(path) != info["sha256"]:
            raise ValueError(f"Output hash mismatch: {name}")
    catalogs = manifest.get("catalogs")
    if not isinstance(catalogs, dict) or set(catalogs) != set(TRACKS):
        raise ValueError("Bundle must declare both frozen catalogs")
    tracks = manifest.get("tracks")
    if not isinstance(tracks, dict) or set(tracks) != set(TRACKS):
        raise ValueError("Bundle must declare both tracks")
    for track in TRACKS:
        name = f"{track}.contracts.json"
        catalog = catalogs[track]
        if (
            not isinstance(catalog, dict) or catalog.get("file") != name
            or catalog.get("sha256") != files[name]["sha256"]
        ):
            raise ValueError(f"Catalog metadata disagrees with bundle files: {track}")
        track_info = tracks[track]
        partitions = track_info.get("partitions") if isinstance(track_info, dict) else None
        if not isinstance(partitions, dict) or set(partitions) != set(PARTITIONS):
            raise ValueError(f"Missing track partition metadata: {track}")
        for partition in PARTITIONS:
            info = partitions[partition]
            if (
                not isinstance(info, dict)
                or info.get("sha256") != files[f"{track}-{partition}.jsonl"]["sha256"]
            ):
                raise ValueError(f"Track partition hash disagrees with bundle files: {track}/{partition}")
    source = manifest.get("source")
    if not isinstance(source, dict) or not _valid_digest(source.get("manifest_sha256")):
        raise ValueError("Bundle must reference a valid source manifest hash")
    source_files = source.get("files")
    if not isinstance(source_files, dict) or not source_files:
        raise ValueError("Bundle must reference source file hashes")
    for name, info in source_files.items():
        if not isinstance(info, dict) or not _valid_digest(info.get("sha256")):
            raise ValueError(f"Invalid source reference hash: {name}")
    if source_files.get("manifest.json", {}).get("sha256") != source["manifest_sha256"]:
        raise ValueError("Source manifest hash references disagree")
    frozen = manifest.get("frozen_test10")
    if manifest.get("frozen_test10_exported") is not False or not isinstance(frozen, dict):
        raise ValueError("Frozen test10 must remain reference-only")
    if set(frozen) != set(TRACKS):
        raise ValueError("Both frozen test10 references are required")
    for track in TRACKS:
        name = f"{track}-test10.jsonl"
        info = frozen[track]
        if (
            not isinstance(info, dict) or info.get("file") != name
            or not _valid_digest(info.get("sha256"))
            or info["sha256"] != source_files.get(name, {}).get("sha256")
        ):
            raise ValueError(f"Frozen test10 reference hashes disagree: {track}")


def verify_bundle(bundle: Path) -> dict:
    """Return a complete manifest after verifying every local payload SHA-256.

    ``files`` maps all eleven payload basenames (including both catalogs) to
    objects containing ``sha256``. No other files or directories are allowed
    beside these payloads and ``manifest.json``. Source/test hashes are checked
    for valid, consistent references only: verification never opens source or
    test data and therefore works on an Azure-staged copy.

    Hashes detect corruption, not malicious replacement of the manifest itself;
    consumers needing authenticity must independently pin the manifest hash.
    """
    path = bundle / "manifest.json"
    if path.is_symlink() or not path.is_file() or path.resolve().parent != bundle.resolve():
        raise ValueError("Bundle requires a regular local manifest.json")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeError) as error:
        raise ValueError("Bundle manifest is not valid JSON") from error
    if not isinstance(manifest, dict):
        raise ValueError("Bundle manifest must be an object")
    _verify_manifest_payload(bundle, manifest)
    return manifest


def native_example(record: Record, catalog_questions: dict, *, context_mode: str = "context") -> dict:
    """Reuse inference inputs verbatim; keep supervision outside the input state."""
    return {
        "id": record.id,
        "taxonomy_id": record.taxonomy_id,
        "state": state(record, context_mode),
        "questions": catalog_questions,
        "answers": {"route": record.expected_route},
    }


def _keys(
    record: Record, fields: dict[str, list[str]], context_mode: str = "context",
) -> set[tuple[str, str]]:
    keys = {("id", record.id), ("normalized input", fingerprint(record))}
    if context_mode == "query":
        keys.add(("query-only input", " ".join(record.query.lower().split())))
    if "repository" in fields[record.taxonomy_id]:
        keys.add(("repository", record.repository))
    if "group" in fields[record.taxonomy_id]:
        keys.add(("group", record.group_id))
    return keys


def _audit_pair(
    left: list[Record], right: list[Record], fields: dict[str, list[str]], context_mode: str = "context",
) -> None:
    left_keys = {key for record in left for key in _keys(record, fields, context_mode)}
    right_keys = {key for record in right for key in _keys(record, fields, context_mode)}
    overlap = left_keys & right_keys
    if overlap:
        kinds = ", ".join(sorted({kind for kind, _ in overlap}))
        raise ValueError(f"Data leakage: overlapping {kinds}, including across tracks")


def _holdout(
    records: list[Record], fraction: float, seed: int, pinned: set[str], fields: list[str],
) -> tuple[list[Record], list[Record]]:
    # A field informative before the first holdout must not become an ignored
    # constant in the second split, thereby allowing its remaining block to split.
    if set(fields) - set(_discriminating_fields(records)):
        return records, []
    try:
        return stratified_group_split(
            records, test_fraction=fraction, seed=seed, development_exposed_ids=pinned,
        )
    except ValueError as error:
        if str(error) != (
            "Group-safety and rare-class protections left no rows eligible for the held-out slice"
        ):
            raise
        return records, []


def _source_file(source: Path, name: str, digest: str) -> Path:
    path = source / name
    # A backslash is a path separator on Windows only, so refuse it by name rather than rely on the OS to resolve it.
    if "\\" in name or path.resolve().parent != source.resolve():
        raise ValueError("Source manifest files must be directly inside the source directory")
    if sha256(path) != digest:
        raise ValueError(f"Source hash mismatch: {name}")
    return path


def prepare_finetune_data(
    source: Path,
    output: Path,
    *,
    train_fraction: float = 0.80,
    validation_fraction: float = 0.10,
    calibration_fraction: float = 0.10,
    seed: int = 42,
    context_mode: str = "context",
) -> dict:
    """Freeze deterministic, group-safe mixed-model inputs from a verified split.

    Cross-track connected inputs are conservatively pinned to training, rather
    than independently split or deduplicated away. Minority routes are never
    drained from training; impossible held-out coverage is explicitly reported.
    """
    fractions = dict(zip(PARTITIONS, (train_fraction, validation_fraction, calibration_fraction)))
    if any(not math.isfinite(value) or not 0 < value < 1 for value in fractions.values()):
        raise ValueError("Fractions must be finite and strictly between 0 and 1")
    if not math.isclose(sum(fractions.values()), 1.0, rel_tol=0, abs_tol=1e-9):
        raise ValueError("Fractions must sum to 1")
    if context_mode not in {"query", "context"}:
        raise ValueError("Context mode must be query or context")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("Seed must be an integer")
    if output.exists():
        raise FileExistsError(output)
    manifest_path = source / "manifest.json"
    manifest_hash = sha256(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("policy_version") != SPLIT_POLICY_VERSION:
        raise ValueError("Source must be a complete group-safe train90/test10 split")
    if set(manifest.get("catalogs", {})) != set(TRACKS):
        raise ValueError("Source must include exactly coding-v1 and tool-api-v1 catalogs")

    source_hashes = {"manifest.json": manifest_hash}
    inputs: dict[str, dict[str, list[Record]]] = {}
    catalogs: dict[str, dict] = {}
    catalog_paths: dict[str, Path] = {}
    frozen_test: dict[str, dict] = {}
    for track in TRACKS:
        inputs[track] = {}
        for partition in ("train90", "test10"):
            name = f"{track}-{partition}.jsonl"
            info = manifest["files"][name]
            path = _source_file(source, name, info["sha256"])
            records = load_records(path, taxonomy_id=track)
            counts = dict(Counter(record.expected_route for record in records))
            if len(records) != info["count"] or counts != info["route_counts"]:
                raise ValueError(f"Source counts mismatch: {name}")
            if any(record.review_status != "unreviewed" for record in records):
                raise ValueError("This preparation requires unreviewed silver labels")
            inputs[track][partition] = records
            source_hashes[name] = info["sha256"]
            if partition == "test10":
                frozen_test[track] = {"file": name, "sha256": info["sha256"], "count": len(records)}
        catalog_info = manifest["catalogs"][track]
        catalog_path = _source_file(source, catalog_info["file"], catalog_info["sha256"])
        catalogs[track] = questions(load_contracts(catalog_path, track), taxonomy_id=track)
        catalog_paths[track] = catalog_path
        source_hashes[catalog_info["file"]] = catalog_info["sha256"]

    train90 = [record for track in TRACKS for record in inputs[track]["train90"]]
    test10 = [record for track in TRACKS for record in inputs[track]["test10"]]
    all_ids = [record.id for record in train90 + test10]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate record IDs, including across tracks or frozen test10")
    if any(DEVELOPMENT_EXPOSED_TAG in record.tags for record in test10):
        raise ValueError("Frozen test10 contains development_exposed records")
    source_fields = {
        track: _discriminating_fields(inputs[track]["train90"] + inputs[track]["test10"])
        for track in TRACKS
    }
    _audit_pair(train90, test10, source_fields, context_mode)
    fields = {track: _discriminating_fields(inputs[track]["train90"]) for track in TRACKS}
    key_tracks: dict[tuple[str, str], set[str]] = defaultdict(set)
    key_counts: Counter = Counter()
    for record in train90:
        for key in _keys(record, fields, context_mode):
            key_tracks[key].add(record.taxonomy_id)
            key_counts[key] += 1
    shared_keys = {key for key, tracks in key_tracks.items() if len(tracks) > 1}
    cross_track_ids = {
        record.id for record in train90 if _keys(record, fields, context_mode) & shared_keys
    }
    query_duplicates = {key for key, count in key_counts.items() if key[0] == "query-only input" and count > 1}
    query_duplicate_ids = {
        record.id for record in train90 if _keys(record, fields, context_mode) & query_duplicates
    }
    exposed_ids = {record.id for record in train90 if DEVELOPMENT_EXPOSED_TAG in record.tags}
    pinned = cross_track_ids | exposed_ids | query_duplicate_ids
    partitions: dict[str, dict[str, list[Record]]] = {}
    for track in TRACKS:
        remaining, calibration = _holdout(
            inputs[track]["train90"], calibration_fraction, seed, pinned, fields[track],
        )
        # Use the achieved remainder, not an assumed exact first-stage fraction.
        validation_share = min(
            validation_fraction * len(inputs[track]["train90"]) / len(remaining),
            math.nextafter(1.0, 0.0),
        )
        train, validation = _holdout(remaining, validation_share, seed + 1, pinned, fields[track])
        partitions[track] = dict(zip(PARTITIONS, (train, validation, calibration)))
        for left, right in combinations(PARTITIONS, 2):
            _assert_group_safe(partitions[track][left], partitions[track][right], fields[track])

    mixed = {
        partition: [record for track in TRACKS for record in partitions[track][partition]]
        for partition in PARTITIONS
    }
    for left, right in combinations(PARTITIONS, 2):
        _audit_pair(mixed[left], mixed[right], fields, context_mode)
    for records in mixed.values():
        _audit_pair(records, test10, source_fields, context_mode)
    exported_ids = [record.id for records in mixed.values() for record in records]
    if len(exported_ids) != len(train90) or set(exported_ids) != {record.id for record in train90}:
        raise ValueError("Partitions must preserve every train90 record exactly once")
    if not pinned <= {record.id for record in mixed["train"]}:
        raise ValueError("Pinned records escaped training")

    result: dict = {
        "status": "incomplete",
        "policy_version": POLICY_VERSION,
        "model_strategy": "one-shared-clef-flash-catalog-conditioned",
        "label_quality": "unreviewed_silver_not_human_gold",
        "training_launched": False,
        "context_mode": context_mode,
        "seed": seed,
        "stage_seeds": {"calibration": seed, "validation": seed + 1, "mixed_shuffle": seed + 2},
        "requested_fractions": fractions,
        "source": {
            "directory": str(source.resolve()),
            "manifest_sha256": manifest_hash,
            "policy_version": manifest["policy_version"],
            "source_export_manifest_sha256": manifest.get("source_export_manifest_sha256"),
            "files": {name: {"sha256": digest} for name, digest in sorted(source_hashes.items())},
        },
        "frozen_test10": frozen_test,
        "frozen_test10_exported": False,
        "files": {},
        "catalogs": {},
        "tracks": {},
        "partition_counts": {partition: len(records) for partition, records in mixed.items()},
        "training_majority_routes": {},
        "route_coverage_warnings": [],
        "audit": {
            "all_source_ids_unique": True,
            "partition_overlap": False,
            "frozen_test10_overlap": False,
            "cross_track_inputs_checked": True,
            "cross_track_shared_rows_pinned_to_train": len(cross_track_ids),
            "query_only_duplicate_rows_pinned_to_train": len(query_duplicate_ids),
            "development_exposed_rows_pinned_to_train": len(exposed_ids),
            "grouping_fields": fields,
            "source_grouping_fields": source_fields,
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    output_manifest = output / "manifest.json"
    output_manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    for track in TRACKS:
        catalog_output = output / f"{track}.contracts.json"
        shutil.copyfile(catalog_paths[track], catalog_output)
        digest = sha256(catalog_output)
        if digest != manifest["catalogs"][track]["sha256"]:
            raise ValueError(f"Copied catalog hash mismatch: {track}")
        result["catalogs"][track] = {"file": catalog_output.name, "sha256": digest}
        result["files"][catalog_output.name] = {"sha256": digest, "format": "routing-contract-catalog"}
        track_result: dict = {"source_train90_count": len(inputs[track]["train90"]), "partitions": {}}
        for partition, records in partitions[track].items():
            path = output / f"{track}-{partition}.jsonl"
            write_records(path, records)
            counts = dict(sorted(Counter(record.expected_route for record in records).items()))
            absent = sorted(set(taxonomy_routes(track)) - set(counts))
            info = {"sha256": sha256(path), "count": len(records), "route_counts": counts}
            result["files"][path.name] = info
            track_result["partitions"][partition] = {
                **info, "actual_fraction": len(records) / len(inputs[track]["train90"]),
                "routes_absent": absent,
            }
            if absent:
                result["route_coverage_warnings"].append(
                    {"taxonomy_id": track, "partition": partition, "routes_absent": absent},
                )
        train_counts = Counter(record.expected_route for record in partitions[track]["train"])
        majority = min(train_counts, key=lambda route: (-train_counts[route], route))
        result["training_majority_routes"][track] = majority
        track_result["training_majority_route"] = majority
        track_result["development_exposed_pinned_to_train"] = sum(
            record.id in exposed_ids for record in partitions[track]["train"]
        )
        result["tracks"][track] = track_result
    for partition, records in mixed.items():
        shuffled = list(records)
        random.Random(seed + 2).shuffle(shuffled)
        path = output / f"mixed-{partition}.jsonl"
        with path.open("x", encoding="utf-8") as stream:
            for record in shuffled:
                example = native_example(record, catalogs[record.taxonomy_id], context_mode=context_mode)
                stream.write(json.dumps(example, ensure_ascii=False) + "\n")
        result["files"][path.name] = {
            "sha256": sha256(path), "count": len(records), "format": "systemone-choice",
        }
    # Detect concurrent source changes before publishing a successful manifest.
    for name, digest in source_hashes.items():
        _source_file(source, name, digest)
    result["status"] = "complete"
    _verify_manifest_payload(output, result)
    pending_manifest = output / "manifest.pending.json"
    pending_manifest.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    pending_manifest.replace(output_manifest)
    return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-fraction", type=float, default=0.80)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--calibration-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context-mode", choices=("query", "context"), default="context")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    result = prepare_finetune_data(
        args.source, args.output, train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction, calibration_fraction=args.calibration_fraction,
        seed=args.seed, context_mode=args.context_mode,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
