"""Opt-in LoRA fine-tuning of Clef on frozen, catalog-conditioned splits.

Trains LoRA adapters on the frozen language backbone together with the joint schema head, using
label-smoothed cross-entropy plus a Brier term (the loss family Cloudflare describes for Clef).
Test data never enters this module: it selects on validation and calibrates on calibration only.
"""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from routing_lab import clef
from routing_lab.backends import ClefBackend, questions
from routing_lab.calibration import fit_temperature, negative_log_likelihood
from routing_lab.data import load_contracts, load_records
from routing_lab.evaluation import run_experiment, sha256
from routing_lab.finetune_data import PARTITIONS, TRACKS, native_example, verify_bundle

# Language-model projections only: the vision tower uses different module names and stays frozen.
LORA_TARGETS = (
    r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|in_proj_qkv|in_proj_z|out_proj"
    r"|gate_proj|up_proj|down_proj)"
)
ROUND_TRIP_EXAMPLES = 24
ROUND_TRIP_MIN_AGREEMENT = 0.9


class TrainingConfig(BaseModel):
    """Conservative first candidate, not a claim of tuned hyperparameters."""

    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)
    variant: Literal["flash", "clef"] = "flash"
    model_revision: str | None = None
    epochs: int = Field(default=3, ge=1, le=20)
    micro_batch_size: int = Field(default=4, ge=1, le=16)
    gradient_accumulation_steps: int = Field(default=4, ge=1, le=256)  # micro-batches per optimizer step
    lora_rank: int = Field(default=32, ge=1, le=256)
    lora_alpha: int = Field(default=64, ge=1, le=1024)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    lora_learning_rate: float = Field(default=1e-4, gt=0, le=1)
    head_learning_rate: float = Field(default=5e-5, gt=0, le=1)
    weight_decay: float = Field(default=0.01, ge=0, le=1)
    warmup_ratio: float = Field(default=0.05, ge=0, lt=1)
    max_grad_norm: float = Field(default=1.0, gt=0, le=100)
    label_smoothing: float = Field(default=0.05, ge=0, lt=0.5)
    brier_weight: float = Field(default=1.0, ge=0, le=10)
    skill_loss_weight: float = Field(default=0.5, ge=0, le=10)  # capability trainer only: weight of the skill head
    max_length: int = Field(default=4096, ge=512, le=16384)
    eval_max_length: int = Field(default=16384, ge=512, le=32768)
    log_every_steps: int = Field(default=10, ge=1, le=1000)
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    smoke: bool = False
    max_train_steps: int | None = Field(default=None, ge=1)
    validation_limit_per_track: int | None = Field(default=None, ge=1)
    calibration_limit_per_track: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def check(self) -> TrainingConfig:
        pinned = clef.release(self.variant).revision
        if self.model_revision is None:
            self.model_revision = pinned
        elif self.model_revision != pinned:
            raise ValueError("model_revision must equal the pinned release revision for this variant")
        if self.eval_max_length < self.max_length:
            raise ValueError("eval_max_length must not be below max_length")
        limits = (self.max_train_steps, self.validation_limit_per_track, self.calibration_limit_per_track)
        if not self.smoke and any(value is not None for value in limits):
            raise ValueError("Bounded training/evaluation is smoke-only, never a selected full model")
        if self.smoke and self.max_train_steps is None:
            raise ValueError("Smoke runs require max_train_steps")
        return self


def validate_inputs(bundle: Path) -> tuple[dict, dict, dict, dict]:
    """Verify hashes, counts and every native field against the frozen source records."""
    manifest = verify_bundle(bundle)
    context_mode = manifest.get("context_mode")
    if context_mode not in {"query", "context"}:
        raise ValueError("Bundle must declare its decision-time context mode")
    catalogs = {track: load_contracts(bundle / f"{track}.contracts.json", track) for track in TRACKS}
    records, examples, seen = {}, {}, set()
    for partition in PARTITIONS:
        records[partition] = {}
        expected = {}
        for track in TRACKS:
            name = f"{track}-{partition}.jsonl"
            rows = load_records(bundle / name, taxonomy_id=track)
            if len(rows) != manifest["files"][name]["count"]:
                raise ValueError(f"Partition count mismatch: {name}")
            records[partition][track] = rows
            catalog = questions(catalogs[track], taxonomy_id=track)
            for record in rows:
                if record.id in seen:
                    raise ValueError("Record identities overlap across tracks or partitions")
                seen.add(record.id)
                expected[record.id] = native_example(record, catalog, context_mode=context_mode)
        name = f"mixed-{partition}.jsonl"
        rows = [json.loads(line) for line in (bundle / name).read_text(encoding="utf-8").splitlines()
                if line.strip()]
        if (not rows or len(rows) != len(expected) or len(rows) != manifest["files"][name]["count"]
                or len(rows) != manifest["partition_counts"][partition]):
            raise ValueError(f"Incomplete mixed partition: {name}")
        matched = set()
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                raise ValueError(f"Invalid native example: {name}")
            key = row["id"]
            if key in matched or key not in expected or row != expected[key]:
                raise ValueError(f"Native labels/catalog/state mismatch: {name}/{key}")
            if list(row["questions"]["route"]["criteria"]) != list(expected[key]["questions"]["route"]["criteria"]):
                raise ValueError(f"Native catalog option order mismatch: {name}/{key}")
            matched.add(key)
        examples[partition] = rows
    return manifest, catalogs, records, examples


def micro_batches(rows: list, micro_batch_size: int, rng: random.Random | None,
                  window_batches: int = 16) -> list[list]:
    """Shuffle, then sort by length inside small windows so a batch pads little; shuffle the batches.

    Every batch is full except possibly one, so the batch count is ceil(rows / micro_batch_size) and
    the optimizer step count matches ``planned_steps``. ``rng=None`` is the smoke ordering: longest first.
    """
    rows = list(rows)
    if rng is None:
        rows.sort(key=lambda row: -len(row[0].input_ids))
        return [rows[i:i + micro_batch_size] for i in range(0, len(rows), micro_batch_size)]
    rng.shuffle(rows)
    window = micro_batch_size * window_batches
    batches = []
    for start in range(0, len(rows), window):
        part = sorted(rows[start:start + window], key=lambda row: len(row[0].input_ids))
        batches.extend(part[i:i + micro_batch_size] for i in range(0, len(part), micro_batch_size))
    rng.shuffle(batches)
    return batches


def accumulation_groups(batches: list[list], accumulation: int):
    """Weight each micro-batch by its share of the group's examples, so a partial batch counts fairly."""
    for offset in range(0, len(batches), accumulation):
        group = batches[offset:offset + accumulation]
        total = sum(len(chunk) for chunk in group)
        yield [(chunk, len(chunk) / total) for chunk in group]


def lr_scale(step: int, total_steps: int, warmup_steps: int, floor: float = 0.1) -> float:
    """Linear warmup, then cosine decay to ``floor`` of the peak rate."""
    if warmup_steps and step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def planned_steps(train_rows: int, config: TrainingConfig) -> tuple[int, int]:
    """Return (optimizer steps per epoch, total optimizer steps actually run)."""
    per_epoch = math.ceil(train_rows / (config.micro_batch_size * config.gradient_accumulation_steps))
    total = per_epoch * config.epochs
    return per_epoch, total if config.max_train_steps is None else min(total, config.max_train_steps)


def selection_score(results: dict) -> float:
    """Equal-weight mean of per-track macro F1, so the larger track cannot dominate selection."""
    score = sum(results[track]["macro_f1"] for track in TRACKS) / len(TRACKS)
    if not math.isfinite(score):
        raise FloatingPointError("Non-finite validation score")
    return score


def encode_example(module, tokenizer, row: dict, max_length: int):
    """Build the exact sequence inference scores, plus the label's index among the sorted options."""
    record = {"state": row["state"], "questions": row["questions"]}
    encoded = module.encode_record(tokenizer, record, max_length=max_length)
    if len(encoded.questions) != 1:
        raise ValueError(f"Expected exactly one route question: {row['id']}")
    options = encoded.questions[0].option_ids
    label = row["answers"]["route"]
    if label not in options:
        raise ValueError(f"Label is not an offered option: {row['id']}")
    return encoded, options.index(label)


def choice_loss(torch, logits, target: int, smoothing: float, brier_weight: float):
    """Label-smoothed cross-entropy plus Brier score over the offered options."""
    logits = logits.float()
    cross_entropy = torch.nn.functional.cross_entropy(
        logits.unsqueeze(0), torch.tensor([target], device=logits.device), label_smoothing=smoothing,
    )
    probabilities = torch.softmax(logits, dim=-1)
    onehot = torch.zeros_like(probabilities)
    onehot[target] = 1.0
    return cross_entropy + brier_weight * ((probabilities - onehot) ** 2).sum()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _package_versions() -> dict[str, str]:
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "safetensors", "huggingface-hub",
                 "numpy", "scipy", "pydantic"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def _param_groups(torch, model, config: TrainingConfig) -> list[dict]:
    """Separate LoRA and head learning rates; no weight decay on biases and norms."""
    groups = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        kind = "head" if name.startswith("head.") else "lora"
        decay = parameter.ndim >= 2
        group = groups.setdefault((kind, decay), {
            "params": [], "weight_decay": config.weight_decay if decay else 0.0,
            "lr": config.head_learning_rate if kind == "head" else config.lora_learning_rate,
        })
        group["params"].append(parameter)
    if not any(kind == "lora" for kind, _ in groups) or not any(kind == "head" for kind, _ in groups):
        raise RuntimeError("Both LoRA and joint-head parameters must be trainable")
    return list(groups.values())


def _evaluate(torch, runtime, catalogs, records, context_mode, config, output: Path, device: str,
              limit: int | None) -> dict:
    """Score a partition per track with the same backend and metrics used for the benchmarks."""
    results = {}
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        for track in TRACKS:
            backend = ClefBackend(
                catalogs[track], variant=config.variant, device=device, context_mode=context_mode,
                taxonomy_id=track, max_length=config.eval_max_length, runtime=runtime,
            )
            rows = records[track] if limit is None else records[track][:limit]
            results[track] = run_experiment(backend, rows, output / track, probe=True)
    return results


def _raw_logits(torch, runtime, encoded_rows, device: str) -> list[list[float]]:
    pad = runtime.processor.tokenizer.pad_token_id
    target_device = next(runtime.model.parameters()).device
    rows = []
    with torch.inference_mode(), torch.autocast(device_type=device, dtype=torch.bfloat16):
        for encoded, _ in encoded_rows:
            batch = runtime.module.collate_records([encoded], pad, target_device)
            rows.append(runtime.model(batch)[0][0].float().cpu().tolist())
    return rows


def _export(torch, runtime, export: Path, meta: dict) -> None:
    from safetensors.torch import save_file

    export.mkdir(parents=True, exist_ok=True)
    runtime.model.language_model.save_pretrained(str(export / clef.ADAPTER_DIR))
    save_file({name: tensor.detach().cpu().float().contiguous()
               for name, tensor in runtime.model.head.state_dict().items()},
              str(export / clef.HEAD_FILE))
    _write_json(export / clef.FINETUNE_META, meta)


def train(bundle: Path, config: TrainingConfig, output: Path, *, allow_training: bool = False,
          allow_model_download: bool = False, device: str = "cuda") -> dict:
    if not allow_training or not allow_model_download:
        raise ValueError("Training requires --allow-training and --allow-model-download")
    manifest, catalogs, records, examples = validate_inputs(bundle)
    if output.resolve().is_relative_to(bundle.resolve()):
        raise ValueError("Training output must not modify the verified bundle")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("Training output must be new or empty")
    try:
        import torch
        import peft  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Training needs the packages in azure/training/conda.yml") from error
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("This runner requires CUDA; CPU fallback is intentionally disabled")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "incomplete", "smoke": config.smoke, "selected_full_model": False,
              "config": config.model_dump(), "release": clef.model_info(config.variant),
              "bundle_sha256": sha256(bundle / "manifest.json"), "packages": _package_versions(),
              "partition_counts": manifest["partition_counts"], "training_steps": 0, "epochs": [],
              "label_quality": manifest.get("label_quality"), "frozen_test_used": False,
              "started_utc": datetime.now(timezone.utc).isoformat()}
    _write_json(output / "run.json", report)
    try:
        _train(torch, bundle, config, output, manifest, catalogs, records, examples, report, device)
    except (RuntimeError, ValueError, OSError, ArithmeticError) as error:
        report.update(status="failed", selected_full_model=False, error=f"{type(error).__name__}: {error}")
        _write_json(output / "run.json", report)
        raise
    return report


def _train(torch, bundle, config, output, manifest, catalogs, records, examples, report, device):
    from peft import LoraConfig, get_peft_model

    started = time.perf_counter()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats()
        report["gpu"] = {"name": torch.cuda.get_device_name(),
                         "total_memory_bytes": torch.cuda.get_device_properties(0).total_memory}
    context_mode = manifest["context_mode"]
    runtime = clef.load(config.variant, device=device)
    model, module = runtime.model, runtime.module
    tokenizer = runtime.processor.tokenizer
    report["load_seconds"] = time.perf_counter() - started

    model.language_model = get_peft_model(model.language_model, LoraConfig(
        r=config.lora_rank, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=LORA_TARGETS, bias="none",
    ))
    model.language_model.get_base_model().gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    model.head.requires_grad_(True)
    model.head.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["trainable_parameters"] = trainable
    report["total_parameters"] = sum(p.numel() for p in model.parameters())

    train_rows = [encode_example(module, tokenizer, row, config.max_length) for row in examples["train"]]
    for partition in ("validation", "calibration"):
        for row in examples[partition]:
            encode_example(module, tokenizer, row, config.eval_max_length)
    report["train_tokens_per_epoch"] = sum(len(encoded.input_ids) for encoded, _ in train_rows)
    report["train_longest_sequence_tokens"] = max(len(encoded.input_ids) for encoded, _ in train_rows)
    validation = {track: records["validation"][track] for track in TRACKS}
    limit = config.validation_limit_per_track
    calibration_rows = {track: [row for row in examples["calibration"] if row["taxonomy_id"] == track]
                        for track in TRACKS}
    if config.calibration_limit_per_track is not None:
        calibration_rows = {t: rows[:config.calibration_limit_per_track] for t, rows in calibration_rows.items()}
    calibration = [encode_example(module, tokenizer, row, config.eval_max_length)
                   for track in TRACKS for row in calibration_rows[track]]

    optimizer = torch.optim.AdamW(_param_groups(torch, model, config))
    per_epoch, total_steps = planned_steps(len(train_rows), config)
    warmup = math.ceil(config.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_scale(step, total_steps, warmup))
    report.update(steps_per_epoch=per_epoch, planned_steps=total_steps, warmup_steps=warmup)
    pad = tokenizer.pad_token_id
    target_device = next(model.parameters()).device
    best, best_state, export = -math.inf, None, output / "export"

    for epoch in range(config.epochs):
        model.train()
        # Smoke runs go longest first, so a few steps measure worst-case memory.
        batches = micro_batches(train_rows, config.micro_batch_size,
                                None if config.smoke else random.Random(config.seed + epoch))
        epoch_started, losses, tokens = time.perf_counter(), [], 0
        for group in accumulation_groups(batches, config.gradient_accumulation_steps):
            optimizer.zero_grad(set_to_none=True)
            group_loss = 0.0
            for chunk, weight in group:
                batch = module.collate_records([encoded for encoded, _ in chunk], pad, target_device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    outputs = model(batch)
                loss = sum(choice_loss(torch, logits[0], target, config.label_smoothing, config.brier_weight)
                           for logits, (_, target) in zip(outputs, chunk, strict=True)) / len(chunk)
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
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], config.max_grad_norm,
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
        results = _evaluate(torch, runtime, catalogs, validation, context_mode, config,
                            output / "validation" / f"epoch-{epoch + 1}", device, limit)
        score = selection_score(results)
        report["epochs"].append({
            "epoch": epoch + 1, "mean_training_loss": sum(losses) / len(losses), "optimizer_steps": len(losses),
            "epoch_completed": len(losses) == per_epoch, "seconds": time.perf_counter() - epoch_started,
            "validation_macro_f1": {t: results[t]["macro_f1"] for t in TRACKS},
            "validation_accuracy": {t: results[t]["acceptable_accuracy"] for t in TRACKS},
            "equal_track_mean_macro_f1": score})
        if score > best:
            best, report["best_epoch"] = score, epoch + 1
            best_state = {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters() if p.requires_grad}
            _export(torch, runtime, export, {"format": 1, "base": clef.model_info(config.variant),
                                             "temperature": None, "best_epoch": epoch + 1})
        _write_json(output / "run.json", report)
        if report["training_steps"] >= total_steps:
            break

    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in best_state:
                parameter.copy_(best_state[name])
    runtime.temperature = 1.0
    model.eval()
    raw = _raw_logits(torch, runtime, calibration, device)
    targets = [target for _, target in calibration]
    temperature = fit_temperature(raw, targets)
    report["calibration"] = {
        "temperature": temperature, "rows": len(targets),
        "nll_before": negative_log_likelihood(raw, targets, 1.0),
        "nll_after": negative_log_likelihood(raw, targets, temperature),
        "fit_data": "calibration partition only"}
    runtime.temperature = temperature
    calibrated = _evaluate(torch, runtime, catalogs, validation, context_mode, config,
                           output / "validation" / "calibrated", device, limit)
    report["validation_calibrated"] = {
        t: {"ece_10_bins": calibrated[t]["probability_metrics"]["ece_10_bins"],
            "multiclass_brier": calibrated[t]["probability_metrics"]["multiclass_brier"]} for t in TRACKS}
    meta = {"format": 1, "base": clef.model_info(config.variant), "temperature": temperature,
            "best_epoch": report["best_epoch"], "created_utc": datetime.now(timezone.utc).isoformat(),
            "bundle_sha256": report["bundle_sha256"], "label_quality": manifest.get("label_quality"),
            "config": config.model_dump(), "fit_data": "calibration partition only"}
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
        selection="equal mean per-track validation macro_f1; first epoch wins ties",
        stop_reason="max_train_steps" if config.max_train_steps is not None else "epochs",
        pipeline_wall_seconds=time.perf_counter() - started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device == "cuda" else None,
        artifacts={str(p.relative_to(output)): sha256(p) for p in output.rglob("*")
                   if p.is_file() and p.name != "run.json"})
    _write_json(output / "run.json", report)


def _round_trip_reference(torch, runtime, validation_rows: list[dict], config: TrainingConfig,
                          device: str) -> list[dict]:
    """Record the in-memory model's answers for a spread of validation rows."""
    stride = max(1, len(validation_rows) // ROUND_TRIP_EXAMPLES)
    rows = validation_rows[::stride][:ROUND_TRIP_EXAMPLES]
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        return [{"row": row, "answer": clef.decide(runtime, row["state"], row["questions"],
                                                   max_length=config.eval_max_length)} for row in rows]


def _round_trip_check(torch, export: Path, reference: list[dict], config: TrainingConfig, device: str) -> dict:
    """Reload the export as a fresh process would, and compare with the model that was trained."""
    reloaded = clef.load(config.variant, device=device, snapshot=export, calibrated=True)
    agree, worst = 0, 0.0
    for item in reference:
        row = item["row"]
        answer = clef.decide(reloaded, row["state"], row["questions"], max_length=config.eval_max_length)
        before, after = item["answer"]["answers"]["route"], answer["answers"]["route"]
        agree += before["choice"] == after["choice"]
        worst = max(worst, max(abs(before["probabilities"][k] - after["probabilities"][k])
                               for k in before["probabilities"]))
    agreement = agree / len(reference)
    del reloaded
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    result = {"examples": len(reference), "choice_agreement": agreement, "max_abs_probability_difference": worst}
    if agreement < ROUND_TRIP_MIN_AGREEMENT:
        raise ValueError(f"Export reload disagrees with the trained model: {result}")
    return result


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
        manifest, _, records, _ = validate_inputs(args.bundle)
        per_epoch, total = planned_steps(manifest["partition_counts"]["train"], config)
        print(json.dumps({"status": "dry_run", "training_launched": False,
                          "bundle_sha256": sha256(args.bundle / "manifest.json"),
                          "partition_counts": manifest["partition_counts"],
                          "per_track_counts": {p: {t: len(records[p][t]) for t in TRACKS} for p in PARTITIONS},
                          "steps_per_epoch": per_epoch, "planned_steps": total,
                          "config": config.model_dump()}, indent=2))
        return
    if not args.output or not args.allow_training or not args.allow_model_download:
        parser.error("Require --dry-run OR --output --allow-training --allow-model-download")
    train(args.bundle, config, args.output, allow_training=True, allow_model_download=True)


if __name__ == "__main__":
    main()
