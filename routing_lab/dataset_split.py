"""Group-safe, best-effort stratified 90/10 split of the finalized AI-labeled dataset.

This freezes an untouched benchmark slice before any fine-tuning begins. It is
NOT a substitute for routing_lab.data.split_reviewed, which requires
human-reviewed records for an actual trusted benchmark; these labels remain
AI-derived, not human gold. Development-exposed rows are never eligible for
the held-out slice, per the existing annotation policy. Connected repository/
group/near-duplicate blocks never cross the train/test boundary, so
stratification by route is best-effort within that constraint, not exact --
especially for routes with very few total rows. A repository or group field
that is constant across the whole dataset (for example BFCL's single
placeholder repository name, unlike SWE-bench's genuine per-task repository)
carries no discriminating information and is excluded from grouping, rather
than being allowed to merge the entire dataset into one unsplittable block.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from routing_lab.data import Record, fingerprint, load_records, records_taxonomy, write_records
from routing_lab.evaluation import sha256

SPLIT_POLICY_VERSION = "group-safe-stratified-90-10-v1"
DEVELOPMENT_EXPOSED_TAG = "development_exposed"


def _discriminating_fields(records: list[Record]) -> list[str]:
    """Drop a grouping signal that is constant across the whole set.

    Some sources (for example BFCL) use a single placeholder repository name
    for every row, unlike SWE-bench's genuine per-task repository. Unioning by
    a universal constant merges the entire dataset into one block and provides
    no actual leakage protection, so such a field is excluded rather than kept
    for superficial consistency.
    """
    fields = []
    if len({record.repository for record in records}) > 1:
        fields.append("repository")
    if len({record.group_id for record in records}) > 1:
        fields.append("group")
    return fields


def _blocks(records: list[Record]) -> list[list[Record]]:
    """Union rows sharing an informative repository, group, or near-duplicate text."""
    fields = _discriminating_fields(records)
    parent = list(range(len(records)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, record in enumerate(records):
        keys = [("text", fingerprint(record))]
        if "repository" in fields:
            keys.append(("repository", record.repository))
        if "group" in fields:
            keys.append(("group", record.group_id))
        for key in keys:
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    groups: dict[int, list[Record]] = defaultdict(list)
    for i, record in enumerate(records):
        groups[root(i)].append(record)
    return list(groups.values())


def _assert_group_safe(train: list[Record], test: list[Record], fields: list[str]) -> None:
    """Reject id, near-duplicate text, and any *discriminating* group/repository overlap.

    Deliberately narrower than routing_lab.data.assert_disjoint, which also
    checks repository unconditionally: a repository field that is constant
    across the whole dataset (excluded from ``fields``; see
    _discriminating_fields) would always "overlap" between any two nonempty
    splits without indicating any real leakage, since it carries no
    information to begin with.
    """
    if train and test:
        records_taxonomy(train + test)
    checks = [("id", lambda record: record.id), ("normalized input", fingerprint)]
    if "repository" in fields:
        checks.append(("repository", lambda record: record.repository))
    if "group" in fields:
        checks.append(("group", lambda record: record.group_id))
    for name, getter in checks:
        if {getter(record) for record in train} & {getter(record) for record in test}:
            raise ValueError(f"Data leakage: overlapping {name}")


def stratified_group_split(records: list[Record], *, test_fraction: float = 0.10, seed: int = 42,
                           development_exposed_ids: set[str] | None = None) -> tuple[list[Record], list[Record]]:
    """Greedy, seeded, route-balanced assignment of connected blocks to a held-out slice.

    A block is never split. Development-exposed rows are always pinned to the
    larger partition. A route is never fully drained out of the larger
    partition, even if that means under-representing it in the held-out slice.
    """
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if not records:
        raise ValueError("Need at least one record to split")
    records_taxonomy(records)
    development_exposed_ids = development_exposed_ids or set()
    fields = _discriminating_fields(records)
    blocks = _blocks(records)
    overall_counts = Counter(record.expected_route for record in records)
    target_test_rows = round(len(records) * test_fraction)
    target_per_route = {route: count * test_fraction for route, count in overall_counts.items()}

    rng = random.Random(seed)
    rng.shuffle(blocks)
    blocks.sort(key=len)  # smallest-first after the seeded shuffle breaks ties; see overshoots_target below

    eligible: list[list[Record]] = []
    for block in blocks:
        if not any(record.id in development_exposed_ids for record in block):
            eligible.append(block)

    test_route_counts: Counter = Counter()
    test_rows = 0
    test_ids: set[str] = set()

    def zero_out_larger_partition(route_counts: Counter) -> bool:
        return any(overall_counts[route] - (test_route_counts[route] + count) < 1
                  for route, count in route_counts.items())

    def overshoots_target(block: list[Record], route_counts: Counter) -> bool:
        return any(test_route_counts[route] + count > target_per_route.get(route, 0) + len(block)
                  for route, count in route_counts.items())

    for strict in (True, False):
        for block in list(eligible):
            if test_rows >= target_test_rows:
                break
            route_counts = Counter(record.expected_route for record in block)
            if zero_out_larger_partition(route_counts):
                continue
            if strict and test_rows > 0 and overshoots_target(block, route_counts):
                continue
            test_ids.update(record.id for record in block)
            test_route_counts.update(route_counts)
            test_rows += len(block)
            eligible.remove(block)
        if test_rows >= target_test_rows:
            break

    test_records = [record for record in records if record.id in test_ids]
    train_records = [record for record in records if record.id not in test_ids]
    if not test_records:
        raise ValueError("Group-safety and rare-class protections left no rows eligible for the held-out slice")
    _assert_group_safe(train_records, test_records, fields)
    return train_records, test_records


def split_dataset(final_export: Path, output: Path, *, test_fraction: float = 0.10, seed: int = 42) -> dict:
    """Split every non-empty taxonomy file in a finalized export into train90/test10."""
    manifest = json.loads((final_export / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("Finalized dataset export must be complete before splitting")
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    output.mkdir(parents=True, exist_ok=False)
    result: dict = {
        "status": "incomplete", "policy_version": SPLIT_POLICY_VERSION,
        "source_export_manifest_sha256": sha256(final_export / "manifest.json"),
        "test_fraction": test_fraction, "seed": seed, "files": {}, "catalogs": {}, "tracks": {},
    }
    for taxonomy in sorted(manifest["catalogs"]):
        info = manifest["files"].get(f"{taxonomy}.jsonl")
        if info is None or info["count"] == 0:
            continue
        path = final_export / f"{taxonomy}.jsonl"
        if sha256(path) != info["sha256"]:
            raise ValueError(f"{taxonomy}.jsonl changed since the final export was written")
        records = load_records(path, taxonomy_id=taxonomy)
        development_exposed_ids = {record.id for record in records if DEVELOPMENT_EXPOSED_TAG in record.tags}
        train, test = stratified_group_split(records, test_fraction=test_fraction, seed=seed,
                                             development_exposed_ids=development_exposed_ids)
        train_path, test_path = output / f"{taxonomy}-train90.jsonl", output / f"{taxonomy}-test10.jsonl"
        write_records(train_path, train)
        write_records(test_path, test)
        overall_counts = Counter(record.expected_route for record in records)
        train_counts = Counter(record.expected_route for record in train)
        test_counts = Counter(record.expected_route for record in test)
        for split_path, split_records, split_counts in (
            (train_path, train, train_counts), (test_path, test, test_counts),
        ):
            result["files"][split_path.name] = {
                "sha256": sha256(split_path), "count": len(split_records), "route_counts": dict(split_counts),
            }
        result["tracks"][taxonomy] = {
            "total": len(records), "train_count": len(train), "test_count": len(test),
            "development_exposed_pinned_to_train": sum(
                1 for record in train if DEVELOPMENT_EXPOSED_TAG in record.tags
            ),
            "overall_route_proportions": {
                route: count / len(records) for route, count in overall_counts.items()
            },
            "test_route_proportions": {
                route: count / len(test) for route, count in test_counts.items()
            },
            "routes_absent_from_test": sorted(set(overall_counts) - set(test_counts)),
        }
        catalog_info = manifest["catalogs"][taxonomy]
        catalog_src = final_export / catalog_info["file"]
        if sha256(catalog_src) != catalog_info["sha256"]:
            raise ValueError(f"{taxonomy} contract catalog changed since the final export was written")
        catalog_out = output / f"{taxonomy}.contracts.json"
        shutil.copyfile(catalog_src, catalog_out)
        result["catalogs"][taxonomy] = {"file": catalog_out.name, "sha256": sha256(catalog_out)}
    if not result["tracks"]:
        raise ValueError("No non-empty taxonomy files found to split")
    result["status"] = "complete"
    (output / "manifest.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-export", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--test-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = create_parser().parse_args(argv)
    result = split_dataset(args.final_export, args.output, test_fraction=args.test_fraction, seed=args.seed)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
