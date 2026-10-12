"""LoRA fine-tuning of Clef's capability router on measured labels.

The training rows come from `capability_lab`: each task's tier (low, medium, high) was measured by how many
endpoints solved it, and its skill is known from how it was generated. One forward pass answers both questions, so
the loss is the route loss plus a down-weighted skill loss. Out-of-distribution test families and the in-distribution
test split never enter this module: it selects on validation and calibrates on calibration only.

The older `routing_lab.finetune` stays untouched for the step-type tracks; this module reuses its batching, loss,
optimizer and export helpers.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path

from routing_lab import clef
from routing_lab.calibration import fit_temperature, negative_log_likelihood
from routing_lab.evaluation import sha256
from routing_lab.finetune import (
    LORA_TARGETS,
    TrainingConfig,
    _export,
    _package_versions,
    _param_groups,
    _round_trip_check,
    _round_trip_reference,
    _write_json,
    accumulation_groups,
    choice_loss,
    lr_scale,
    micro_batches,
    planned_steps,
)
from routing_lab.taxonomy import SKILLS, taxonomy_routes

TIERS = taxonomy_routes("capability-v1")
HEADS = {"route": TIERS, "skill": SKILLS}
PARTITIONS = ("train", "validation", "calibration")
BUNDLE_FILES = frozenset(f"{name}.jsonl" for name in PARTITIONS)
KIND = "capability-v1"


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_bundle(bundle: Path) -> dict:
    """Return the manifest after checking the exact file set and every SHA-256 and row count."""
    path = bundle / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("Bundle requires a regular local manifest.json")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("kind") != KIND or manifest.get("status") != "complete":
        raise ValueError("Bundle manifest must be a complete capability-v1 manifest")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != BUNDLE_FILES:
        raise ValueError("Bundle must list exactly train, validation and calibration")
    if {p.name for p in bundle.iterdir()} != BUNDLE_FILES | {"manifest.json"}:
        raise ValueError("Bundle contains missing or unexpected entries; test data must not be uploaded")
    for name, info in files.items():
        file = bundle / name
        if file.is_symlink() or not file.is_file():
            raise ValueError(f"Bundle payload must be a regular local file: {name}")
        if sha256(file) != info.get("sha256"):
            raise ValueError(f"Output hash mismatch: {name}")
    if manifest.get("test_data_exported") is not False:
        raise ValueError("Bundle must declare that test data is not exported")
    return manifest


def validate_inputs(bundle: Path) -> tuple[dict, dict[str, list[dict]]]:
    """Check hashes, counts, labels, and that no id, prompt or held-out family leaks across partitions."""
    manifest = verify_bundle(bundle)
    held_out = set(manifest.get("held_out_families", []))
    examples: dict[str, list[dict]] = {}
    seen_ids: set[str] = set()
    seen_states: set[str] = set()
    for partition in PARTITIONS:
        rows = read_rows(bundle / f"{partition}.jsonl")
        if not rows or len(rows) != manifest["files"][f"{partition}.jsonl"]["count"]:
            raise ValueError(f"Partition count mismatch: {partition}")
        for row in rows:
            if row["id"] in seen_ids or row["state"] in seen_states:
                raise ValueError(f"Data leakage: repeated id or request text in {partition}")
            seen_ids.add(row["id"])
            seen_states.add(row["state"])
            if row["family"] in held_out:
                raise ValueError(f"Held-out family found in {partition}: {row['family']}")
            for head, options in HEADS.items():
                label = row["answers"].get(head)
                unlabelled_skill = head == "skill" and label is None  # rows from sources without a skill label train the tier only
                if (label not in options and not unlabelled_skill) or list(row["questions"][head]["criteria"]) != list(options):
                    raise ValueError(f"Bad {head} label or catalog in {row['id']}")
        examples[partition] = rows
    return manifest, examples


def encode_example(module, tokenizer, row: dict, max_length: int):
    """The exact sequence inference scores, plus each head's label index among its sorted options."""
    encoded = module.encode_record(tokenizer, {"state": row["state"], "questions": row["questions"]},
                                   max_length=max_length)
    targets = {}
    for question in encoded.questions:
        label = row["answers"].get(question.question_id)
        if label is None and question.question_id == "skill":
            continue
        if label not in question.option_ids:
            raise ValueError(f"Label is not an offered option: {row['id']}/{question.question_id}")
        targets[question.question_id] = question.option_ids.index(label)
    if "route" not in targets or {q.question_id for q in encoded.questions} != set(HEADS):
        raise ValueError(f"Expected route and skill questions: {row['id']}")
    return encoded, targets


def example_loss(torch, logits_by_question, encoded, targets: dict[str, int], config: TrainingConfig):
    """Route loss plus the down-weighted skill loss, each label-smoothed cross-entropy plus Brier."""
    total = 0.0
    for question, logits in zip(encoded.questions, logits_by_question, strict=True):
        if question.question_id not in targets:
            continue
        weight = 1.0 if question.question_id == "route" else config.skill_loss_weight
        total = total + weight * choice_loss(torch, logits, targets[question.question_id],
                                             config.label_smoothing, config.brier_weight)
    return total


def _raw_logits(torch, runtime, encoded_rows, device: str) -> list[dict[str, list[float]]]:
    pad = runtime.processor.tokenizer.pad_token_id
    target_device = next(runtime.model.parameters()).device
    rows = []
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        for encoded, _ in encoded_rows:
            batch = runtime.module.collate_records([encoded], pad, target_device)
            out = runtime.model(batch)[0]
            rows.append({q.question_id: item.float().cpu().tolist() for q, item in zip(encoded.questions, out, strict=True)})
    return rows


def _softmax(logits: list[float], temperature: float) -> list[float]:
    top = max(item / temperature for item in logits)
    exps = [math.exp(item / temperature - top) for item in logits]
    total = sum(exps)
    return [item / total for item in exps]


def head_metrics(raw: list[dict[str, list[float]]], encoded_rows, head: str, temperature: float = 1.0) -> dict:
    """Accuracy, macro F1, majority-class accuracy, Brier and 10-bin ECE for one head."""
    options = HEADS[head]
    kept = [(row, t[head]) for row, (_, t) in zip(raw, encoded_rows, strict=True) if head in t]  # rows without this label are skipped
    if not kept:
        return {"rows": 0, "accuracy": None, "macro_f1": None, "majority_accuracy": None, "brier": None, "ece_10_bins": None}
    raw, targets = [row for row, _ in kept], [t for _, t in kept]
    probabilities = [_softmax(row[head], temperature) for row in raw]
    predicted = [max(range(len(p)), key=p.__getitem__) for p in probabilities]
    n = len(targets)
    f1s = []
    for index in range(len(options)):
        tp = sum(p == index and t == index for p, t in zip(predicted, targets, strict=True))
        fp = sum(p == index and t != index for p, t in zip(predicted, targets, strict=True))
        fn = sum(p != index and t == index for p, t in zip(predicted, targets, strict=True))
        if tp + fp + fn:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    confidence = [max(p) for p in probabilities]
    correct = [p == t for p, t in zip(predicted, targets, strict=True)]
    ece = 0.0
    for bin_index in range(10):
        members = [i for i, c in enumerate(confidence) if bin_index / 10 <= c < (bin_index + 1) / 10 or (bin_index == 9 and c == 1.0)]
        if members:
            ece += len(members) / n * abs(sum(correct[i] for i in members) / len(members) - sum(confidence[i] for i in members) / len(members))
    brier = sum(sum((p - (j == t)) ** 2 for j, p in enumerate(row)) for row, t in zip(probabilities, targets, strict=True)) / n
    return {"rows": n, "accuracy": sum(correct) / n, "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
            "majority_accuracy": max(targets.count(i) for i in range(len(options))) / n, "brier": brier, "ece_10_bins": ece}


def train(bundle: Path, config: TrainingConfig, output: Path, *, allow_training: bool = False,
          allow_model_download: bool = False, device: str = "cuda") -> dict:
    if not allow_training or not allow_model_download:
        raise ValueError("Training requires --allow-training and --allow-model-download")
    manifest, examples = validate_inputs(bundle)
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Training output must not modify the verified bundle")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Training output must be new or empty")
    try:
        import peft  # noqa: F401
        import torch
    except ImportError as error:
        raise RuntimeError("Training needs the packages in azure/training/conda.yml") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This runner requires CUDA; CPU fallback is intentionally disabled")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "incomplete", "kind": KIND, "smoke": config.smoke, "selected_full_model": False,
              "config": config.model_dump(), "release": clef.model_info(config.variant),
              "bundle_sha256": sha256(bundle / "manifest.json"), "packages": _package_versions(),
              "partition_counts": {p: len(rows) for p, rows in examples.items()}, "training_steps": 0, "epochs": [],
              "label_source": manifest.get("label_source"), "frozen_test_used": False,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(output / "run.json", report)
    try:
        _train(torch, bundle, config, output, manifest, examples, report, device)
    except (RuntimeError, ValueError, OSError, ArithmeticError) as error:
        report.update(status="failed", selected_full_model=False, error=f"{type(error).__name__}: {error}")
        _write_json(output / "run.json", report)
        raise
    return report


def _train(torch, bundle, config, output, manifest, examples, report, device):
    from peft import LoraConfig, get_peft_model

    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats()
        report["gpu"] = {"name": torch.cuda.get_device_name(),
                         "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    runtime = clef.load(config.variant, device=device)
    model, module = runtime.model, runtime.module
    tokenizer = runtime.processor.tokenizer
    report["load_seconds"] = time.perf_counter() - started
    model.language_model = get_peft_model(model.language_model, LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=LORA_TARGETS, bias="none"))
    model.language_model.get_base_model().gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head.requires_grad_(True)
    model.head.float()
    report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["total_parameters"] = sum(p.numel() for p in model.parameters())

    train_rows = [encode_example(module, tokenizer, row, config.max_length) for row in examples["train"]]
    validation_source = examples["validation"][:config.validation_limit_per_track]
    calibration_source = examples["calibration"][:config.calibration_limit_per_track]
    validation = [encode_example(module, tokenizer, row, config.eval_max_length) for row in validation_source]
    calibration = [encode_example(module, tokenizer, row, config.eval_max_length) for row in calibration_source]
    report["train_tokens_per_epoch"] = sum(len(e.input_ids) for e, _ in train_rows)
    report["train_longest_sequence_tokens"] = max(len(e.input_ids) for e, _ in train_rows)

    optimizer = torch.optim.AdamW(_param_groups(torch, model, config))
    per_epoch, total_steps = planned_steps(len(train_rows), config)
    warmup = math.ceil(config.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_scale(step, total_steps, warmup))
    report.update(steps_per_epoch=per_epoch, planned_steps=total_steps, warmup_steps=warmup)
    pad = tokenizer.pad_token_id
    target_device = next(model.parameters()).device
    best, best_state, export = -math.inf, None, output / "export"

    for epoch in range(config.epochs):
        model.train()
        batches = micro_batches(train_rows, config.micro_batch_size, None if config.smoke else random.Random(config.seed + epoch))
        epoch_started, losses, tokens = time.perf_counter(), [], 0
        for group in accumulation_groups(batches, config.gradient_accumulation_steps):
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for chunk, weight in group:
                batch = module.collate_records([encoded for encoded, _ in chunk], pad, target_device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    outputs = model(batch)
                loss = sum(example_loss(torch, logits, encoded, targets, config)
                           for logits, (encoded, targets) in zip(outputs, chunk, strict=True)) / len(chunk)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                (loss * weight).backward()
                group_loss += loss.item() * weight
                tokens += sum(len(encoded.input_ids) for encoded, _ in chunk)
            if report["training_steps"] == 0:
                flow = {kind: any(p.grad is not None and bool(p.grad.abs().max() > 0)
                                  for name, p in model.named_parameters()
                                  if p.requires_grad and name.startswith("head.") == (kind == "head"))
                        for kind in ("lora", "head")}
                if not all(flow.values()):
                    raise RuntimeError(f"Missing nonzero training gradients: {flow}")
                report["gradient_flow_verified"] = flow
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config.max_grad_norm,
                                           error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            report["training_steps"] += 1
            losses.append(group_loss)
            if report["training_steps"] % config.log_every_steps == 0:
                elapsed = time.perf_counter() - epoch_started
                print(json.dumps({"epoch": epoch + 1, "step": report["training_steps"], "of": total_steps,
                                  "loss": round(group_loss, 4), "tokens_per_second": round(tokens / elapsed),
                                  "lr": scheduler.get_last_lr()}), flush=True)
            if report["training_steps"] >= total_steps:
                break
        model.eval()
        raw = _raw_logits(torch, runtime, validation, device)
        scores = {head: head_metrics(raw, validation, head) for head in HEADS}
        score = scores["route"]["macro_f1"]
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite validation score")
        report["epochs"].append({
            "epoch": epoch + 1, "mean_training_loss": sum(losses) / len(losses), "optimizer_steps": len(losses),
            "epoch_completed": len(losses) == per_epoch, "seconds": time.perf_counter() - epoch_started,
            "validation": scores, "route_macro_f1": score})
        if score > best:
            best, report["best_epoch"] = score, epoch + 1
            best_state = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}
            _export(torch, runtime, export, {"format": 1, "base": clef.model_info(config.variant), "temperature": None,
                                             "best_epoch": epoch + 1})
        _write_json(output / "run.json", report)
        if report["training_steps"] >= total_steps:
            break

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in best_state:
                parameter.copy_(best_state[name])
    model.eval()
    raw = _raw_logits(torch, runtime, calibration, device)
    route_logits = [row["route"] for row in raw]
    route_targets = [targets["route"] for _, targets in calibration]
    _write_json(output / "calibration_logits.json", {"route_logits": route_logits, "route_targets": route_targets})
    try:
        temperature, status = fit_temperature(route_logits, route_targets), "fitted"
    except ValueError:  # degenerate data (for example a smoke run); keep the adapter and fit offline from the saved logits
        temperature, status = 1.0, "fit_failed_unit_temperature"
    report["calibration"] = {
        "temperature": temperature, "status": status, "rows": len(route_targets),
        "nll_before": negative_log_likelihood(route_logits, route_targets, 1.0),
        "nll_after": negative_log_likelihood(route_logits, route_targets, temperature),
        "fit_data": "calibration partition, route head only; the same temperature is applied to the skill head"}
    validation_raw = _raw_logits(torch, runtime, validation, device)
    report["validation_calibrated"] = {head: head_metrics(validation_raw, validation, head, temperature) for head in HEADS}
    runtime.temperature = temperature
    meta = {"format": 1, "base": clef.model_info(config.variant), "temperature": temperature,
            "best_epoch": report["best_epoch"], "created_utc": datetime.now(timezone.utc).isoformat(),
            "bundle_sha256": report["bundle_sha256"], "label_source": manifest.get("label_source"),
            "heads": list(HEADS), "config": config.model_dump(), "fit_data": "calibration partition only"}
    _export(torch, runtime, export, meta)

    reference = _round_trip_reference(torch, runtime, examples["validation"], config, device)
    del runtime, model, optimizer, scheduler, best_state
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    report["round_trip"] = _round_trip_check(torch, export, reference, config, device)
    verify_bundle(bundle)
    if sha256(bundle / "manifest.json") != report["bundle_sha256"]:
        raise ValueError("Bundle manifest changed during training")
    report.update(
        status="smoke_complete" if config.smoke else "complete", selected_full_model=not config.smoke,
        selection="route-head validation macro F1; first epoch wins ties",
        stop_reason="max_train_steps" if config.max_train_steps is not None else "epochs",
        pipeline_wall_seconds=time.perf_counter() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
        artifacts={str(p.relative_to(output)): sha256(p) for p in output.rglob("*") if p.is_file() and p.name != "run.json"})
    _write_json(output / "run.json", report)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--allow-training", action="store_true")
    parser.add_argument("--allow-model-download", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = create_parser()
    args = parser.parse_args(argv)
    config = TrainingConfig.model_validate_json(args.config.read_text(encoding="utf-8-sig"))
    if args.dry_run:
        if args.output or args.allow_training or args.allow_model_download:
            parser.error("--dry-run cannot be combined with output or training/download consent")
        manifest, examples = validate_inputs(args.bundle)
        counts = {p: len(rows) for p, rows in examples.items()}
        per_epoch, total = planned_steps(counts["train"], config)
        print(json.dumps({"status": "dry_run", "training_launched": False, "bundle_sha256": sha256(args.bundle / "manifest.json"),
                          "partition_counts": counts, "held_out_families": manifest.get("held_out_families"),
                          "steps_per_epoch": per_epoch, "planned_steps": total, "config": config.model_dump()}, indent=2))
        return
    if not args.output or not args.allow_training or not args.allow_model_download:
        parser.error("Require --dry-run OR --output --allow-training --allow-model-download")
    train(args.bundle, config, args.output, allow_training=True, allow_model_download=True)


if __name__ == "__main__":
    main()
