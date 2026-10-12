"""Run with python -m routing_lab.cli; no model loads without explicit selection."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from routing_lab.backends import (
    AzureLLMBackend, ClefBackend, ConstantBackend, RulesBackend, TfidfBackend,
)
from routing_lab.clef import DEFAULT_MAX_LENGTH, RELEASES
from routing_lab.data import assert_disjoint, load_contracts, load_records, split_reviewed, write_records
from routing_lab.evaluation import run_experiment
from routing_lab.endpoint import ClefEndpointBackend
from routing_lab.outcomes import compare_outcomes
from routing_lab.taxonomy import TAXONOMIES


def create_parser() -> argparse.ArgumentParser:
    """Define explicit local and cloud actions."""
    parser = argparse.ArgumentParser(description="Coding-agent routing experiments")
    subs = parser.add_subparsers(dest="command", required=True)
    validate = subs.add_parser("validate")
    validate.add_argument("--data", type=Path, required=True)
    validate.add_argument("--reviewed-only", action="store_true")
    validate.add_argument("--taxonomy", choices=TAXONOMIES, default="coding-v1")
    split = subs.add_parser("split")
    split.add_argument("--data", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--seed", type=int, default=42)
    split.add_argument("--taxonomy", choices=TAXONOMIES, default="coding-v1")
    run = subs.add_parser("run")
    run.add_argument("--data", type=Path, default=Path("data/probes.jsonl"))
    run.add_argument("--contracts", type=Path)
    run.add_argument("--taxonomy", choices=TAXONOMIES, default="coding-v1")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--backend", choices=["rules", "constant", "majority", "tfidf", "clef", "clef-endpoint", "azure"],
                     default="rules")
    run.add_argument("--train", type=Path)
    run.add_argument("--probe", action="store_true")
    run.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    run.add_argument("--context-mode", choices=["query", "context"], default="context")
    run.add_argument("--variant", choices=sorted(RELEASES), default="flash")
    run.add_argument("--reverse-options", action="store_true")
    run.add_argument("--max-length", type=int,
                     help="Total input token budget (clef/clef-endpoint only; default 16384)")
    run.add_argument("--snapshot", type=Path,
                     help="Local Clef directory: a fine-tuned export or a release layout (clef backend only)")
    run.add_argument("--deployment",
                     help="Target this endpoint deployment by name (clef-endpoint only)")
    run.add_argument("--adapter-sha256",
                     help="Require responses from a fine-tuned deployment with this adapter hash (clef-endpoint only)")
    run.add_argument("--calibrated", action="store_true",
                     help="Apply the fine-tuned export's fitted temperature (needs --snapshot)")
    run.add_argument("--allow-model-download", action="store_true")
    run.add_argument("--allow-remote", action="store_true")
    outcomes = subs.add_parser("outcomes")
    outcomes.add_argument("--data", type=Path, required=True)
    outcomes.add_argument("--baseline", default="always_general")
    outcomes.add_argument("--candidate", default="routed")
    outcomes.add_argument("--output", type=Path, required=True)
    return parser


def execute(args: argparse.Namespace) -> None:
    """Execute an action; operational errors propagate to an explicit CLI error."""
    if args.command == "outcomes":
        result = compare_outcomes(args.data, args.baseline, args.candidate)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        print(json.dumps(result, indent=2))
        return
    records = load_records(args.data, reviewed_only=(
        args.command == "split" or
        (args.command == "validate" and args.reviewed_only) or
        (args.command == "run" and not args.probe)
    ), taxonomy_id=args.taxonomy)
    if args.command == "validate":
        print(json.dumps({"records": len(records), "taxonomy_id": args.taxonomy,
                          "routes": dict(Counter(r.expected_route for r in records)),
                          "review": dict(Counter(r.review_status for r in records))}, indent=2))
        return
    if args.command == "split":
        splits = split_reviewed(records, args.seed)
        args.output.mkdir(parents=True, exist_ok=False)
        for name, subset in splits.items():
            write_records(args.output / f"{name}.jsonl", subset)
        print(json.dumps({name: dict(Counter(r.expected_route for r in subset))
                          for name, subset in splits.items()}, indent=2))
        return
    if args.output.exists():
        raise ValueError("Run output already exists; choose a new run directory")
    if args.backend == "clef" and not args.allow_model_download:
        raise ValueError("Clef requires explicit --allow-model-download (run on an Azure GPU)")
    if args.backend in {"azure", "clef-endpoint"} and not args.allow_remote:
        raise ValueError("Remote endpoint requires --allow-remote and approved input data")
    if args.reverse_options and args.backend != "azure":
        raise ValueError("--reverse-options applies only to the azure backend; Clef sorts options itself")
    if args.max_length is not None and args.backend not in {"clef", "clef-endpoint"}:
        raise ValueError("--max-length is supported only by the clef/clef-endpoint backends")
    if args.snapshot is not None and args.backend != "clef":
        raise ValueError("--snapshot is supported only by the clef backend")
    if (args.deployment or args.adapter_sha256) and args.backend != "clef-endpoint":
        raise ValueError("--deployment/--adapter-sha256 are supported only by the clef-endpoint backend")
    if args.calibrated and args.snapshot is None:
        raise ValueError("--calibrated needs a fine-tuned export passed with --snapshot")
    contracts_path = args.contracts or Path("contracts") / (
        "agents.json" if args.taxonomy == "coding-v1" else f"{args.taxonomy}.json"
    )
    contracts = load_contracts(contracts_path, args.taxonomy)
    sources = {"dataset": args.data, "contracts": contracts_path}
    started = time.perf_counter()
    if args.backend in {"majority", "tfidf"}:
        if args.train is None:
            raise ValueError("Supervised baselines require --train on disjoint reviewed data")
        training = load_records(args.train, reviewed_only=True, taxonomy_id=args.taxonomy)
        assert_disjoint(training, records)
        sources["training"] = args.train
        backend = (
            TfidfBackend(training, args.context_mode, taxonomy_id=args.taxonomy)
            if args.backend == "tfidf" else
            ConstantBackend(Counter(r.expected_route for r in training).most_common(1)[0][0],
                            taxonomy_id=args.taxonomy)
        )
    elif args.backend == "clef":
        backend = ClefBackend(contracts, variant=args.variant, device=args.device,
                              context_mode=args.context_mode, taxonomy_id=args.taxonomy,
                              max_length=args.max_length or DEFAULT_MAX_LENGTH,
                              snapshot=str(args.snapshot) if args.snapshot else None,
                              calibrated=args.calibrated)
    elif args.backend == "azure":
        backend = AzureLLMBackend(contracts, args.context_mode, taxonomy_id=args.taxonomy,
                                  reverse=args.reverse_options)
    elif args.backend == "clef-endpoint":
        backend = ClefEndpointBackend(contracts, context_mode=args.context_mode,
                                      variant=args.variant, taxonomy_id=args.taxonomy,
                                      max_length=args.max_length, deployment=args.deployment,
                                      adapter_sha256=args.adapter_sha256)
    elif args.backend == "constant":
        backend = ConstantBackend(taxonomy_id=args.taxonomy)
    else:
        backend = RulesBackend(taxonomy_id=args.taxonomy)
    result = run_experiment(
        backend, records, args.output, probe=args.probe, source_paths=sources,
        initialization_seconds=time.perf_counter() - started,
    )
    print(json.dumps(result, indent=2))


def main() -> int:
    """Return standard process exit codes without hiding runtime failures."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = create_parser().parse_args()
    try:
        execute(args)
    except (ValueError, OSError) as exc:
        logging.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
