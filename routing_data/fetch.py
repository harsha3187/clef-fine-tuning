"""Download the four source datasets into data/external/ (never into version control).

Everything is public. xRouteBench, SCOPE-60K and R2-Bench come from Hugging Face; LLMRouterBench is one tarball. R2-Bench
is 25 GB in full (10 models x 16 token budgets, with response text), so it is streamed and reduced on the way in: only
the query, the judge score and the token count are kept, for the budgets asked for.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests

ROOT = Path(__file__).resolve().parents[1]
EXTERNAL = ROOT / "data" / "external"
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

XROUTE_FILES = [
    "llm_candidates/train.parquet",
    *[f"{config}/{split}.parquet" for config in ("llmrouter_generic", "memory_locomo", "memory_longmemeval", "personalized", "timeseries")
      for split in ("train", "test")],
]
SCOPE_FILES = ["data/train-00000-of-00001.parquet", "data/test-00000-of-00001.parquet", "data/anchor-00000-of-00001.parquet"]
R2_MODELS = [
    "Qwen/Qwen3-235B-A22B-Instruct-2507", "Qwen/Qwen3-Next-80B-A3B-Instruct", "Qwen/Qwen3-30B-A3B-Instruct-2507",
    "Qwen/Qwen2.5-Math-7B-Instruct", "Qwen/Qwen2.5-Math-1.5B-Instruct", "Qwen/Qwen3-0.6B",
    "meta-llama/Llama-3.1-70B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", "zai-org/GLM-4.5-Air", "zai-org/GLM-4.6",
]
R2_COLUMNS = ["prompts_id", "key", "original_prompt", "actual_token_count", "correctness_score"]
LLMROUTERBENCH_URL = "https://huggingface.co/datasets/NPULH/LLMRouterBench/resolve/main/bench-release.tar.gz"


def download(url: str, target: Path, *, chunk: int = 1 << 20) -> Path:
    """Download a file once; a partial file from an interrupted run is replaced, a complete one is kept."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for block in response.iter_content(chunk):
                handle.write(block)
    partial.replace(target)
    return target


def fetch_xroutebench() -> None:
    for path in XROUTE_FILES:
        download(HF.format(repo="ulab-ai/xRouteBench", path=path), EXTERNAL / "xroutebench" / path)


def fetch_scope() -> None:
    for path in SCOPE_FILES:
        download(HF.format(repo="Cooolder/SCOPE-60K", path=path), EXTERNAL / "scope60k" / Path(path).name)


def fetch_llmrouterbench() -> None:
    download(LLMROUTERBENCH_URL, EXTERNAL / "llmrouterbench" / "bench-release.tar.gz")


def reduce_r2_file(model: str, budget: int) -> Path:
    """Stream one 100 to 650 MB judge CSV and keep five columns per query as parquet (a few MB)."""
    target = EXTERNAL / "r2bench" / f"{model.replace('/', '__')}__{budget}.parquet"
    if target.exists():
        return target
    url = HF.format(repo="JiaqiXue/R2-Bench", path=f"data/{model}/{budget}_judge.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_csv = target.with_suffix(".csv.part")
    with requests.get(url, stream=True, timeout=(15, 300)) as response:  # to disk first: the reader needs a seekable file
        response.raise_for_status()
        with tmp_csv.open("wb") as handle:
            shutil.copyfileobj(response.raw, handle, 1 << 20)
    header = tmp_csv.open(encoding="utf-8").readline().strip().split(",")
    # Two of the ten models ship only the raw judge JSON; the score is inside it.
    columns = [c for c in R2_COLUMNS if c in header] + ([] if "correctness_score" in header else ["judge_raw"])
    options = pacsv.ConvertOptions(include_columns=columns, column_types={"prompts_id": pa.string(), "key": pa.string()})
    table = pacsv.read_csv(tmp_csv, convert_options=options, read_options=pacsv.ReadOptions(block_size=64 << 20),
                           parse_options=pacsv.ParseOptions(newlines_in_values=True))
    if "correctness_score" not in table.column_names:
        found = pc.extract_regex(table["judge_raw"], r'"correctness_score"\s*:\s*(?P<score>-?[0-9.]+)')
        score = pc.cast(pc.struct_field(found, "score"), pa.float64(), safe=False)
        table = table.drop_columns(["judge_raw"]).append_column("correctness_score", score)
    pq.write_table(table, target, compression="zstd")
    tmp_csv.unlink()
    return target


def fetch_r2bench(budgets: tuple[int, ...] = (200, 2000)) -> None:
    for model in R2_MODELS:
        for budget in budgets:
            reduce_r2_file(model, budget)
            print(f"r2bench {model} {budget} done", flush=True)


FETCHERS = {"xroutebench": fetch_xroutebench, "scope60k": fetch_scope, "llmrouterbench": fetch_llmrouterbench, "r2bench": fetch_r2bench}


def main(argv: list[str] | None = None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or list(FETCHERS)
    for name in names:
        print(f"fetching {name}", flush=True)
        FETCHERS[name]()
        print(f"{name} ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
