"""Command line: build the task set, run the benchmark, and write the report.

    python -m capability_lab build
    python -m capability_lab run --pilot 5 --approve-cost
    python -m capability_lab report
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from capability_lab import analysis, bundle, evaluate
from capability_lab.registry import build_registry
from routing_lab.backends import CAPABILITY_PROMPT_VERSION
from routing_lab.selector import PRESET_SETTINGS
from capability_lab.clients import BenchClient, load_models
from capability_lab.runner import load_runs, pending, pilot_subset, run_benchmark
from capability_lab.toolchains import tool_tasks
from capability_lab.tasks import DATA_DIR, build_tasks, load_tasks, save_tasks, tasks_hash

ROOT = Path(__file__).resolve().parents[1]
TASK_SETS = {"main": DATA_DIR / "tasks-v2.jsonl", "tools": DATA_DIR / "tasks-tools-v1.jsonl",
             "tools2": DATA_DIR / "tasks-tools-v2.jsonl"}


def tasks_path(args: argparse.Namespace) -> Path:
    return TASK_SETS[getattr(args, "set", "main")]
OUT_DIR = ROOT / "outputs" / "capability"


def runs_path(pilot: bool, digest: str) -> Path:
    """Results belong to one exact task set; a rebuilt set starts a new file rather than mixing answers."""
    return OUT_DIR / f"runs-{'pilot-' if pilot else ''}{digest}.jsonl"


def cmd_build(args: argparse.Namespace) -> int:
    path = tasks_path(args)
    if args.set == "tools":
        tasks = tool_tasks(args.count, args.seed)
    elif args.set == "tools2":  # a training set for the second fine-tune: other seed, no Elena, no message from the evaluation set
        held_apart = {t.prompt for t in load_tasks(TASK_SETS["tools"])}
        tasks = tool_tasks(args.count, args.seed, family="tool_chain2", customers=("priya", "marcus"), avoid=held_apart)
    else:
        tasks = build_tasks(per_family=args.per_family, seed=args.seed, support_count=args.support)
    digest = save_tasks(tasks, path)
    by_skill: dict[str, int] = {}
    for t in tasks:
        by_skill[t.skill] = by_skill.get(t.skill, 0) + 1
    print(json.dumps({"tasks": len(tasks), "hash": digest, "by_skill": by_skill, "path": str(path)}, indent=2))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.approve_cost:
        raise SystemExit("This sends model calls that cost money. Re-run with --approve-cost.")
    tasks = load_tasks(tasks_path(args))
    if args.pilot:
        tasks = pilot_subset(tasks, args.pilot)
    models = load_models()
    chosen = [models[m] for m in args.models] if args.models else list(models.values())
    out = runs_path(bool(args.pilot), tasks_hash(load_tasks(tasks_path(args))))
    todo = pending(tasks, chosen, {(r["task_id"], r["model"], r["sample"]) for r in load_runs(out) if r["error"] is None}, args.samples)
    print(f"{len(tasks)} tasks, {len(chosen)} endpoints, {len(todo)} attempts to run -> {out}", flush=True)

    def progress(done: int, total: int, record: dict[str, Any]) -> None:
        if done % 25 == 0 or done == total:
            print(f"  {done}/{total}  last: {record['model']} {record['task_id']} passed={record['passed']} error={record['error']}", flush=True)

    summary = run_benchmark(BenchClient(timeout=180), tasks, chosen, out, workers=args.workers, samples=args.samples, on_progress=progress)
    print(json.dumps(summary, indent=2))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    tasks = load_tasks(tasks_path(args))
    path = runs_path(args.pilot, tasks_hash(tasks))
    runs = load_runs(path)
    if not runs:
        raise SystemExit(f"No results in {path}")
    ran = {r["task_id"] for r in runs}
    digest = tasks_hash(tasks)
    tasks = [t for t in tasks if t.id in ran]
    models = load_models()
    labels = analysis.tier_labels(tasks, runs, low_at_least=args.low_at_least, high_below=args.high_below)
    report = {
        "tasks_hash": digest, "tasks": len(tasks), "attempts": len(runs), "labelled_tasks": len(labels),
        "tier_counts": {t: sum(label["tier"] == t for label in labels) for t in analysis.TIERS},
        "unsolved": sum(label["unsolved"] for label in labels),
        "thresholds": {"low_at_least": args.low_at_least, "high_below": args.high_below},
        "models": analysis.model_summary(runs, models), "profiles": analysis.profiles(labels, runs),
        "strategies": analysis.strategies(tasks, labels, runs),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = ("pilot-" if args.pilot else "full-") + digest
    (OUT_DIR / f"report-{stem}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / f"tier_labels-{stem}.jsonl").write_text("".join(json.dumps(label) + "\n" for label in labels), encoding="utf-8")
    print_report(report, labels)
    return 0


def full_labels(args: argparse.Namespace) -> tuple[list, list, list, str]:
    tasks_all = load_tasks(tasks_path(args))
    digest = tasks_hash(tasks_all)
    runs = load_runs(runs_path(False, digest))
    if not runs:
        raise SystemExit("No benchmark results for this task set. Run the benchmark first.")
    ran = {r["task_id"] for r in runs}
    tasks = [t for t in tasks_all if t.id in ran]
    labels = analysis.tier_labels(tasks, runs, low_at_least=args.low_at_least, high_below=args.high_below)
    return tasks, labels, runs, digest


def splits_path(digest: str) -> Path:
    return DATA_DIR / f"splits-{digest}.json"


def cmd_bundle(args: argparse.Namespace) -> int:
    tasks, labels, runs, digest = full_labels(args)
    held_out = tuple(args.held_out)
    thresholds = {"low_at_least": args.low_at_least, "high_below": args.high_below}
    main_splits = bundle.assign_splits(labels, held_out)
    extra = {}
    if args.with_tools:  # version 2: add the hard tool-use training tasks; the first tool set stays an evaluation-only set
        extra_tasks, extra_labels, extra_runs, extra_digest = full_labels(argparse.Namespace(**{**vars(args), "set": "tools2"}))
        _, _, _, eval_digest = full_labels(argparse.Namespace(**{**vars(args), "set": "tools"}))
        tasks, labels, runs = tasks + extra_tasks, labels + extra_labels, runs + extra_runs
        extra = {"splits": {**main_splits, **bundle.assign_splits(extra_labels, (), fractions=bundle.TRAINING_ONLY)},
                 "forbidden_prompts": {t.prompt for t in load_tasks(TASK_SETS["tools"])},
                 "extra_sets": [{"set": "tools2", "tasks_hash": extra_digest}], "test_sets": [{"set": "tools", "tasks_hash": eval_digest}]}
    out = ROOT / "data" / "finetuning" / (args.name or ("capability-v2" if args.with_tools else "capability-v1"))
    manifest = bundle.build_bundle(tasks, labels, out, tasks_hash=digest, thresholds=thresholds,
                                   endpoints=sorted({r["model"] for r in runs}), held_out=held_out, **extra)
    splits_path(digest).write_text(json.dumps(main_splits, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps({"bundle": str(out), "files": manifest["files"], "tier_counts": manifest["tier_counts"],
                      "held_out": manifest["held_out_counts"], "splits": str(splits_path(digest))}, indent=2))
    return 0

REGISTRY_PATH = ROOT / "data" / "registry" / "measured.json"


def cmd_registry(args: argparse.Namespace) -> int:
    labels: list = []
    runs: list = []
    digests: list[str] = []
    for path in TASK_SETS.values():  # every task set that has results contributes its measurements
        if not path.exists():
            continue
        tasks = load_tasks(path)
        digest = tasks_hash(tasks)
        set_runs = load_runs(runs_path(False, digest))
        if not set_runs:
            continue
        ran = {r["task_id"] for r in set_runs}
        labels += analysis.tier_labels([t for t in tasks if t.id in ran], set_runs, low_at_least=args.low_at_least, high_below=args.high_below)
        runs += set_runs
        digests.append(digest)
    registry = build_registry(load_models(), runs, labels, tasks_hash="+".join(digests))
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=1) + "\n", encoding="utf-8")
    for e in registry["endpoints"]:
        measured = sum(n for tiers in e["cells"].values() for _, n in tiers.values())
        print(f"  {e['id']:20} {measured:5} measured attempts, mean out {e['mean_out_tokens']:6} tok, "
              f"latency {e['lat_base_ms']:.0f} ms + {e['lat_per_token_ms']:.1f} ms/tok, price verified: {e['verified_price']}")
    print(f"Wrote {REGISTRY_PATH} ({registry['version']})")
    return 0

def cmd_evaluate(args: argparse.Namespace) -> int:
    from routing_lab.capability import CapabilityRouter

    tasks_all = load_tasks(tasks_path(args))
    digest = tasks_hash(tasks_all)
    runs = load_runs(runs_path(args.pilot, digest))
    if not runs:
        raise SystemExit("No benchmark results for this task set. Run the benchmark first.")
    ran = {r["task_id"] for r in runs}
    tasks = [t for t in tasks_all if t.id in ran]
    labels = analysis.tier_labels(tasks, runs, low_at_least=args.low_at_least, high_below=args.high_below)
    url = json.loads((ROOT / "azure" / "online" / "connection.json").read_text(encoding="utf-8-sig"))["scoring_uri"]
    router = CapabilityRouter(url, args.deployment, adapter_sha256=args.adapter_sha256)
    labelled = {label["task_id"] for label in labels}
    if args.split != "all":
        assigned = json.loads(splits_path(digest).read_text(encoding="utf-8-sig"))
        wanted = {"test": ("test_id", "test_ood")}.get(args.split, (args.split,))
        labelled = {task_id for task_id, name in assigned.items() if name in wanted and task_id in labelled}
    stem = f"{args.deployment}-{args.split}-{digest}" if args.split != "all" else f"{args.deployment}-{digest}"
    if CAPABILITY_PROMPT_VERSION != "p1":  # answers depend on the prompt wording, so cached ones are kept apart
        stem += f"-{CAPABILITY_PROMPT_VERSION}"
    out_stem = stem + (f"-{args.preset}" if args.preset else "")
    predictions = evaluate.ask_clef(router, [t for t in tasks if t.id in labelled], OUT_DIR / f"clef-{stem}.jsonl", workers=args.workers)
    metrics = evaluate.tier_metrics([label for label in labels if label["task_id"] in labelled], predictions)
    settings = PRESET_SETTINGS[args.preset] if args.preset else None
    arms = evaluate.evaluate_arms(tasks, labels, runs, predictions, floor=args.floor, settings=settings)
    result = {"tasks_hash": digest, "deployment": args.deployment, "tier_metrics": metrics, "routing": arms}
    (OUT_DIR / f"evaluation-{out_stem}{'-pilot' if args.pilot else ''}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_evaluation(result)
    return 0


def print_evaluation(result: dict[str, Any]) -> None:
    m = result["tier_metrics"]
    print(f"\nClef tier accuracy {m['tier_accuracy']} (always-the-majority-class would score {m['majority_class_accuracy']}), "
          f"macro F1 {m['macro_f1']}, Brier {m['brier']}, ECE {m['ece']}, skill accuracy {m['skill_accuracy']}")
    print(f"  underestimates {m['underestimates']}, overestimates {m['overestimates']}; confusion (rows true, columns Clef):")
    for tier, row in m["confusion"].items():
        print(f"    {tier:7} {row}")
    r = result["routing"]
    print(f"\nRouting on {r['tasks']} tasks, pass-rate floor {r['floor']}, Clef cost per decision ${r['router_cost_usd']} included for Clef arms:")
    for name, arm in r["arms"].items():
        print(f"  {name:38} pass {arm['pass_rate']:.3f}   ${arm['mean_cost_usd']:.6f}/task   {arm['models_used']}")


def print_report(report: dict[str, Any], labels: list[dict[str, Any]]) -> None:
    print(f"\nTasks {report['tasks']}, attempts {report['attempts']}, labelled {report['labelled_tasks']}, "
          f"tiers {report['tier_counts']}, unsolved by every endpoint: {report['unsolved']}")
    print("\nEndpoint results (pass rate with 95% interval, mean cost per task):")
    for m in report["models"]:
        print(f"  {m['model']:20} pass {m['pass_rate']:.2f} [{m['ci_low']:.2f}-{m['ci_high']:.2f}]  ${m['mean_cost_usd']:.5f}/task  "
              f"{m['mean_latency_s']}s  out={m['mean_output_tokens']}tok  errors={m['errors']}{'' if m['price_verified'] else '  (price unverified)'}")
    s = report["strategies"]
    print(f"\nOracle (cheapest passing endpoint per task): pass {s['oracle']['pass_rate']}, ${s['oracle']['mean_cost_usd']}/task")
    sel = s["perfect_tier_selector"]
    print(f"Selector with perfect tier knowledge (leave-one-out, floor {sel['floor']}): pass {sel['pass_rate']}, ${sel['mean_cost_usd']}/task, used {sel['models_used']}")
    best = max(s["always"].items(), key=lambda kv: kv[1]["pass_rate"])
    print(f"Always the strongest measured endpoint ({best[0]}): pass {best[1]['pass_rate']}, ${best[1]['mean_cost_usd']}/task")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--set", choices=sorted(TASK_SETS), default="main", help="Which task set: the main benchmark or the harder tool-use tasks")
    build = sub.add_parser("build", parents=[shared])
    build.add_argument("--per-family", type=int, default=80)
    build.add_argument("--support", type=int, default=12)
    build.add_argument("--seed", type=int, default=7)
    build.add_argument("--count", type=int, default=100, help="Tasks to generate (tools set only)")
    run = sub.add_parser("run", parents=[shared])
    run.add_argument("--pilot", type=int, default=0, help="Run only this many tasks per family")
    run.add_argument("--models", nargs="*")
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--samples", type=int, default=1)
    run.add_argument("--approve-cost", action="store_true")
    report = sub.add_parser("report", parents=[shared])
    report.add_argument("--pilot", action="store_true")
    report.add_argument("--low-at-least", type=float, default=analysis.LOW_AT_LEAST)
    report.add_argument("--high-below", type=float, default=analysis.HIGH_BELOW)
    ev = sub.add_parser("evaluate", parents=[shared], help="Ask Clef about every labelled task and compare routing strategies")
    ev.add_argument("--pilot", action="store_true")
    ev.add_argument("--deployment", default="flash-gpu")
    ev.add_argument("--workers", type=int, default=4)
    ev.add_argument("--adapter-sha256", help="Pin the adapter hash when the deployment is fine-tuned")
    ev.add_argument("--split", choices=["all", "test", "test_id", "test_ood"], default="all",
                    help="Score only the held-out tasks (test = both) instead of every labelled task")
    ev.add_argument("--floor", type=float, default=0.85)
    ev.add_argument("--preset", choices=sorted(PRESET_SETTINGS), help="Use a screen preset instead of --floor")
    ev.add_argument("--low-at-least", type=float, default=analysis.LOW_AT_LEAST)
    ev.add_argument("--high-below", type=float, default=analysis.HIGH_BELOW)
    bd = sub.add_parser("bundle", help="Write the fine-tuning bundle and the fixed evaluation splits")
    bd.add_argument("--name", help="Bundle folder name under data/finetuning (default capability-v1, or capability-v2 with --with-tools)")
    bd.add_argument("--with-tools", action="store_true", help="Version 2: also train on the hard tool-use set (tools2)")
    bd.add_argument("--held-out", nargs="*", default=list(bundle.HELD_OUT_FAMILIES))
    bd.add_argument("--low-at-least", type=float, default=analysis.LOW_AT_LEAST)
    bd.add_argument("--high-below", type=float, default=analysis.HIGH_BELOW)
    rg = sub.add_parser("registry", help="Write the measured endpoint registry the selector reads")
    rg.add_argument("--low-at-least", type=float, default=analysis.LOW_AT_LEAST)
    rg.add_argument("--high-below", type=float, default=analysis.HIGH_BELOW)
    args = parser.parse_args()
    return {"build": cmd_build, "run": cmd_run, "report": cmd_report, "evaluate": cmd_evaluate, "bundle": cmd_bundle, "registry": cmd_registry}[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
