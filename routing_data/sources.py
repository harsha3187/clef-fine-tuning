"""Converters: each source dataset to canonical records. Nothing here downloads; see fetch.py."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq

from routing_data.schema import Accumulator, CanonicalRecord, Observation, clean_quality

XROUTE_CONFIGS = ("llmrouter_generic", "memory_locomo", "memory_longmemeval")  # the text-only scenarios with the routing schema
XROUTE_COLUMNS = ["task_name", "query", "metric", "model_name", "performance", "input_tokens", "output_tokens", "response_time"]
SCOPE_COLUMNS = ["id", "prompt", "model_name", "is_correct", "category", "domain", "reasoning_mode", "usage_cost",
                 "usage_prompt_tokens", "usage_completion_tokens"]
R2_BUDGET_FOR_LABELS = 2000


def _rows(path: Path, columns: list[str]) -> Iterator[dict]:
    table = pq.ParquetFile(path)
    for batch in table.iter_batches(batch_size=20_000, columns=columns):
        yield from batch.to_pylist()


def convert_xroutebench(root: Path) -> Iterator[CanonicalRecord]:
    """One row per (query, model); every query was run on all 18 candidates. Price comes from the candidate table."""
    prices = {r["model_name"]: (r["input_price_per_1m"], r["output_price_per_1m"])
              for r in pq.read_table(root / "llm_candidates" / "train.parquet").to_pylist()}
    acc = Accumulator("xroutebench")
    for config in XROUTE_CONFIGS:
        for split in ("train", "test"):
            path = root / config / f"{split}.parquet"
            if not path.exists():
                continue
            for row in _rows(path, XROUTE_COLUMNS):
                price = prices.get(row["model_name"])
                tokens_in, tokens_out = row["input_tokens"], row["output_tokens"]
                cost = (tokens_in * price[0] + tokens_out * price[1]) / 1e6 if price and tokens_in is not None and tokens_out is not None else None
                acc.add(row["task_name"], row["query"],
                        Observation(row["model_name"], clean_quality(row["performance"]), cost, tokens_in, tokens_out, row["response_time"]),
                        split=split, metadata={"metric": row["metric"], "scenario": config})
    yield from acc.records()


def convert_scope60k(root: Path) -> Iterator[CanonicalRecord]:
    """One row per (question, model), correct or not. A model run in reasoning mode is a different candidate."""
    acc = Accumulator("scope60k")
    for name, split in (("train-00000-of-00001", "train"), ("test-00000-of-00001", "test"), ("anchor-00000-of-00001", "anchor")):
        path = root / f"{name}.parquet"
        if not path.exists():
            continue
        for row in _rows(path, SCOPE_COLUMNS):
            model = row["model_name"] + ("+reasoning" if row["reasoning_mode"] else "")
            acc.add(row["category"] or "unknown", row["prompt"],
                    Observation(model, clean_quality(row["is_correct"]), row["usage_cost"], row["usage_prompt_tokens"],
                                row["usage_completion_tokens"]),
                    split=split, metadata={"domain": row["domain"]})
    yield from acc.records()


def convert_r2bench(root: Path, budgets: tuple[int, ...] = (200, R2_BUDGET_FOR_LABELS)) -> Iterator[CanonicalRecord]:
    """One reduced parquet per (model, token budget). The same query appears at every budget as separate observations."""
    acc = Accumulator("r2bench")
    for path in sorted(root.glob("*.parquet")):
        model, _, budget = path.stem.rpartition("__")
        if not budget.isdigit() or int(budget) not in budgets:
            continue
        for row in _rows(path, ["key", "original_prompt", "actual_token_count", "correctness_score"]):
            acc.add("routerarena", row["original_prompt"] or "",
                    Observation(model.split("__")[-1], clean_quality(row["correctness_score"]), None, None,
                                row["actual_token_count"], None, int(budget)),
                    identity=row["key"])
    yield from acc.records()


def convert_llmrouterbench(root: Path) -> Iterator[CanonicalRecord]:
    """bench-release/<dataset>/[<split>/]<model>/<run>.json, each holding every query's score for one model."""
    acc = Accumulator("llmrouterbench")
    for path in sorted(root.rglob("*.json")):
        with path.open(encoding="utf-8") as handle:
            run = json.load(handle)
        dataset, model, split = run.get("dataset_name") or path.parts[-4], run.get("model_name") or path.parent.name, run.get("split")
        for record in run.get("records", []):
            query = record.get("origin_query") or record.get("prompt")
            if not isinstance(query, str):
                continue
            acc.add(dataset, query,
                    Observation(model, clean_quality(record.get("score")), record.get("cost"), record.get("prompt_tokens"),
                                record.get("completion_tokens")),
                    split=split)
        del run
    yield from acc.records()
