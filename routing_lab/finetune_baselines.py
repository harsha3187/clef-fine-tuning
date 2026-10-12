"""Training-only majority and lexical comparators on frozen validation partitions."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from routing_lab.backends import ConstantBackend, RulesBackend
from routing_lab.data import load_records, records_taxonomy
from routing_lab.evaluation import run_experiment
from routing_lab.finetune_data import TRACKS, verify_bundle


def evaluate_baselines(bundle: Path, output: Path) -> dict:
    """Evaluate inexpensive comparators without changing silver-label review status."""
    verify_bundle(bundle)
    output.mkdir(parents=True, exist_ok=False)
    summaries = {}
    for track in TRACKS:
        training_path = bundle / f"{track}-train.jsonl"
        validation_path = bundle / f"{track}-validation.jsonl"
        training = load_records(training_path, taxonomy_id=track)
        validation = load_records(validation_path, taxonomy_id=track)
        majority = Counter(record.expected_route for record in training).most_common(1)[0][0]
        backend = ConstantBackend(majority, taxonomy_id=records_taxonomy(training))
        sources = {"training": training_path, "validation": validation_path,
                   "bundle_manifest": bundle / "manifest.json"}
        summaries[track] = {
            "majority": run_experiment(
                backend, validation, output / f"{track}-majority", probe=True, source_paths=sources,
            ),
        }
        if track == "coding-v1":
            summaries[track]["rules"] = run_experiment(
                RulesBackend(), validation, output / f"{track}-rules", probe=True, source_paths=sources,
            )
    return summaries


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = create_parser().parse_args()
    reports = evaluate_baselines(args.bundle, args.output)
    for track, comparators in reports.items():
        for name, summary in comparators.items():
            print(f"{track}/{name}: accuracy={summary['strict_accuracy']:.4f}, "
                  f"macro_f1={summary['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
