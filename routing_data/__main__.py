"""Command line: convert, label and export.

    python -m routing_data convert            # data/external -> data/canonical/<source>.jsonl
    python -m routing_data stats              # what the canonical files hold
    python -m routing_data export             # canonical + tier labels -> a fine-tuning bundle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from routing_data import export as exporter
from routing_data import sources
from routing_data.clean import CleanRules
from routing_data.labels import LabelRules
from routing_data.schema import SOURCES, read_jsonl, write_jsonl

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
CANONICAL = ROOT / "data" / "canonical"
CONVERTERS = {
    "xroutebench": lambda: sources.convert_xroutebench(EXTERNAL / "xroutebench"),
    "scope60k": lambda: sources.convert_scope60k(EXTERNAL / "scope60k"),
    "r2bench": lambda: sources.convert_r2bench(EXTERNAL / "r2bench"),
    "llmrouterbench": lambda: sources.convert_llmrouterbench(EXTERNAL / "llmrouterbench" / "bench-release"),
}


def cmd_convert(args: argparse.Namespace) -> int:
    for name in args.sources or SOURCES:
        started = time.perf_counter()
        count = write_jsonl(CONVERTERS[name](), CANONICAL / f"{name}.jsonl")
        print(f"{name}: {count:,} queries in {time.perf_counter() - started:.0f}s", flush=True)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    for name in args.sources or SOURCES:
        path = CANONICAL / f"{name}.jsonl"
        if not path.exists():
            continue
        datasets: Counter[str] = Counter()
        models: Counter[str] = Counter()
        observed = chars = n = 0
        for record in read_jsonl(path):
            n += 1
            datasets[record.dataset] += 1
            chars += len(record.query)
            observed += len(record.observations)
            models.update(o.model for o in record.observations)
        print(f"== {name}: {n:,} queries, {len(models)} models, {observed / max(n, 1):.1f} observations per query, "
              f"mean {chars // max(n, 1)} chars")
        print("   datasets:", dict(datasets.most_common(args.top)))
    return 0


def cmd_labels(args: argparse.Namespace) -> int:
    rules = LabelRules(pass_at=args.pass_at)
    table = exporter.tier_table(CANONICAL, rules, tuple(args.sources or SOURCES))
    for source, datasets in table.items():
        total = Counter()
        for counts in datasets.values():
            total.update(counts)
        print(f"== {source}: {dict(total)}")
        for dataset, counts in datasets.items():
            print(f"   {dataset:28} {counts}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    report = exporter.export(CANONICAL, ROOT / "data" / "finetuning" / args.name, rules=LabelRules(pass_at=args.pass_at),
                             cap=args.cap, max_chars=args.max_chars, use_skill=args.use_skill, seed=args.seed,
                             clean=None if args.no_clean else CleanRules())
    print(json.dumps(report, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    convert = sub.add_parser("convert")
    convert.add_argument("sources", nargs="*")
    stats = sub.add_parser("stats")
    stats.add_argument("sources", nargs="*")
    stats.add_argument("--top", type=int, default=12)
    labels = sub.add_parser("labels", help="Tier mix per source and sub-benchmark")
    labels.add_argument("sources", nargs="*")
    labels.add_argument("--pass-at", type=float, default=LabelRules.pass_at)
    export = sub.add_parser("export", help="Canonical records plus tier labels to a fine-tuning bundle")
    export.add_argument("--name", default="routing-ext-v1")
    export.add_argument("--cap", type=int, default=150, help="Most queries kept per source and sub-benchmark")
    export.add_argument("--max-chars", type=int, default=3000)
    export.add_argument("--pass-at", type=float, default=LabelRules.pass_at)
    export.add_argument("--seed", type=int, default=7)
    export.add_argument("--no-clean", action="store_true", help="Skip the cleaning rules (routing-ext-v1 was made this way)")
    export.add_argument("--use-skill", action="store_true", help="Train the skill head on dataset-level hints (off by default)")
    args = parser.parse_args(argv)
    unknown = [s for s in getattr(args, "sources", []) if s not in SOURCES]
    if unknown:
        parser.error(f"unknown source {unknown}; choose from {list(SOURCES)}")
    return {"convert": cmd_convert, "stats": cmd_stats, "labels": cmd_labels, "export": cmd_export}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
