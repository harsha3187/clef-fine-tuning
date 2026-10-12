"""Stage allowlisted Azure ML packages for Clef fine-tuning, and for its one-time test evaluation.

``train`` uploads code, the verified training bundle and a config: never test data.
``eval`` uploads code and the two frozen test files and exists to run after model selection is final.
Neither command submits a job unless every consent flag is passed.
"""

from __future__ import annotations

import argparse
import os
import json
import shutil
from pathlib import Path

from routing_lab.evaluation import sha256
from routing_lab import finetune_capability
from routing_lab.finetune import TrainingConfig, validate_inputs
from routing_lab.finetune_data import TRACKS, verify_bundle

DEFAULTS = {
    "subscription": os.environ.get("AZURE_SUBSCRIPTION_ID", ""),  # not stored in the repository
    "resource_group": "rg-clef-exp-eus2",
    "workspace": "aml-clef-exp-eus2",
    "compute": "clef-gpu-train",
}
ROOT = Path(__file__).resolve().parents[2]


def _copy_code(destination: Path) -> None:
    """Code only: every top-level module of the package, never data or caches."""
    package = destination / "routing_lab"
    package.mkdir()
    for path in sorted((ROOT / "routing_lab").glob("*.py")):
        shutil.copyfile(path, package / path.name)


def _write_upload_manifest(destination: Path) -> None:
    staged = {
        path.relative_to(destination).as_posix(): sha256(path)
        for path in sorted(destination.rglob("*")) if path.is_file()
    }
    (destination / "upload-manifest.json").write_text(json.dumps(staged, indent=2) + "\n", encoding="utf-8")


def stage_training(bundle: Path, config: Path, destination: Path) -> Path:
    """Copy only verified partition files, code and training configuration."""
    manifest, _, _, _ = validate_inputs(bundle)
    TrainingConfig.model_validate_json(config.read_text(encoding="utf-8-sig"))
    files = list(manifest["files"])
    for name in files:
        if Path(name).name != name or "/" in name or "\\" in name or "test" in name.lower():
            raise ValueError(f"Refusing unsafe or test-data upload entry: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    _copy_code(destination)
    data = destination / "bundle"
    data.mkdir()
    for name in [*files, "manifest.json"]:
        shutil.copyfile(bundle / name, data / name)
    shutil.copyfile(config, destination / "config.json")
    folder = Path(__file__).resolve().parent
    for name in ("job.yml", "conda.yml"):
        shutil.copyfile(folder / name, destination / name)
    _write_upload_manifest(destination)
    return destination / "job.yml"


def stage_capability(bundle: Path, config: Path, destination: Path) -> Path:
    """Stage the capability fine-tune: verified train, validation and calibration rows, code and configuration."""
    manifest, _ = finetune_capability.validate_inputs(bundle)
    TrainingConfig.model_validate_json(config.read_text(encoding="utf-8-sig"))
    files = list(manifest["files"])
    for name in files:
        if Path(name).name != name or "test" in name.lower():
            raise ValueError(f"Refusing unsafe or test-data upload entry: {name}")
    destination.mkdir(parents=True, exist_ok=False)
    _copy_code(destination)
    data = destination / "bundle"
    data.mkdir()
    for name in [*files, "manifest.json"]:
        shutil.copyfile(bundle / name, data / name)
    shutil.copyfile(config, destination / "config.json")
    folder = Path(__file__).resolve().parent
    shutil.copyfile(folder / "capability-job.yml", destination / "job.yml")
    shutil.copyfile(folder / "conda.yml", destination / "conda.yml")
    _write_upload_manifest(destination)
    return destination / "job.yml"


def stage_evaluation(bundle: Path, source: Path, destination: Path) -> Path:
    """Copy the two frozen test files, only after checking them against the hashes the bundle recorded."""
    manifest = verify_bundle(bundle)
    destination.mkdir(parents=True, exist_ok=False)
    _copy_code(destination)
    data = destination / "data"
    data.mkdir()
    for track in TRACKS:
        reference = manifest["frozen_test10"][track]
        test = source / reference["file"]
        if sha256(test) != reference["sha256"]:
            raise ValueError(f"Frozen test file differs from the hash recorded in the bundle: {track}")
        shutil.copyfile(test, data / reference["file"])
        shutil.copyfile(bundle / f"{track}.contracts.json", data / f"{track}.contracts.json")
    folder = Path(__file__).resolve().parent
    for name in ("eval-job.yml", "conda.yml"):
        shutil.copyfile(folder / name, destination / name)
    _write_upload_manifest(destination)
    return destination / "eval-job.yml"


def _add_submission_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--approve-cost", action="store_true")
    parser.add_argument("--approve-upload", action="store_true")
    parser.add_argument("--subscription", default=DEFAULTS["subscription"])
    parser.add_argument("--resource-group", default=DEFAULTS["resource_group"])
    parser.add_argument("--workspace", default=DEFAULTS["workspace"])
    parser.add_argument("--compute", default=DEFAULTS["compute"])


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="Stage (and optionally submit) the training job")
    train.add_argument("--bundle", type=Path, required=True)
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--kind", choices=["step-types", "capability"], default="step-types")
    _add_submission_arguments(train)
    evaluate = commands.add_parser("eval", help="Stage (and optionally submit) the frozen-test evaluation")
    evaluate.add_argument("--bundle", type=Path, required=True)
    evaluate.add_argument("--source", type=Path, required=True,
                          help="Directory holding the original *-test10.jsonl files")
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--model-job", help="Name of the finished training job whose export to evaluate")
    evaluate.add_argument("--approve-test-use", action="store_true",
                          help="Confirm model selection is final and the frozen test set may be used")
    _add_submission_arguments(evaluate)
    return parser


def _require_consent(args: argparse.Namespace) -> None:
    if not args.approve_cost or not args.approve_upload:
        raise ValueError("Submission requires --approve-cost and --approve-upload")
    if args.command == "eval" and (not args.approve_test_use or not args.model_job):
        raise ValueError("Test evaluation requires --approve-test-use and --model-job")


def main() -> int:
    args = create_parser().parse_args()
    if args.submit:
        _require_consent(args)
    if args.command == "train":
        stage = stage_capability if args.kind == "capability" else stage_training
        job_path = stage(args.bundle, args.config, args.output)
    else:
        job_path = stage_evaluation(args.bundle, args.source, args.output)
    print(f"Staged package: {job_path}")
    if not args.submit:
        return 0
    from azure.ai.ml import MLClient, load_job
    from azure.identity import AzureCliCredential

    client = MLClient(AzureCliCredential(), args.subscription, args.resource_group, args.workspace)
    job = load_job(job_path)
    # The plain compute name; the YAML loader's azureml: form is not accepted by this SDK property.
    job.compute = args.compute
    if args.command == "eval":
        # Job outputs are only addressable as datastore paths: uploads land under azureml/<job>/<output>/.
        job.inputs["model"].path = f"azureml://datastores/workspaceblobstore/paths/azureml/{args.model_job}/results/"
    created = client.jobs.create_or_update(job)
    print(f"Submitted: {created.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
