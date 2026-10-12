from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from routing_data import sources
from routing_data.clean import CleanRules, exclude_reason
from routing_data.export import cap_per_dataset, dedupe, export
from routing_data.labels import LabelRules, Pool, label_record, rank_pools
from routing_data.schema import Accumulator, CanonicalRecord, Observation, clean_quality, query_sha, read_jsonl, write_jsonl
from routing_data.skills import dataset_key, skill_hint
from routing_lab import finetune_capability as fc


def test_the_same_question_hashes_alike_across_spacing_and_case_and_scores_clean_up():
    assert query_sha("What is  2+2?\n") == query_sha("what is 2+2?")
    assert query_sha("a") != query_sha("b")
    assert (clean_quality(True), clean_quality(0.4), clean_quality(-1), clean_quality(None), clean_quality(3)) == (1.0, 0.4, None, None, 1.0)


def test_observations_of_one_query_merge_and_repeats_are_averaged():
    acc = Accumulator("xroutebench")
    acc.add("gsm8k", "Q?", Observation("a", 1.0, 0.01, 10, 20), split="train")
    acc.add("gsm8k", " q? ", Observation("a", 0.0, 0.03, 30, 40), split="test")
    acc.add("gsm8k", "Q?", Observation("b", 1.0))
    (record,) = list(acc.records())
    assert record.split_origin == ["train", "test"] and [o.model for o in record.observations] == ["a", "b"]
    a = record.observations[0]
    assert a.quality == 0.5 and a.runs == 2 and a.tokens_in == 20 and a.cost_usd == pytest.approx(0.02)
    acc.add("gsm8k", "", Observation("c", 1.0))  # an empty query is dropped
    assert len(list(acc.records())) == 1


def test_an_explicit_identity_groups_rows_whose_text_differs():
    acc = Accumulator("r2bench")
    acc.add("routerarena", "context A. question?", Observation("m1", 1.0, budget_tokens=2000), identity="k1")
    acc.add("routerarena", "context B. question?", Observation("m2", 0.0, budget_tokens=2000), identity="k1")
    assert len(list(acc.records())) == 1


def test_records_round_trip_through_jsonl(tmp_path):
    record = CanonicalRecord("s/d/1", "scope60k", "d", "Q?", [Observation("m", 1.0, 0.5, 1, 2, 0.3, 200)], ["train"], {"k": "v"}, "mit")
    path = tmp_path / "r.jsonl"
    assert write_jsonl([record], path) == 1
    (back,) = list(read_jsonl(path))
    assert back == record and json.loads(path.read_text(encoding="utf-8"))["sha"] == record.sha


def table(rows: dict[str, list]) -> pa.Table:
    return pa.table(rows)


def test_xroutebench_rows_become_one_query_with_a_priced_observation_per_model(tmp_path):
    root = tmp_path / "x"
    (root / "llm_candidates").mkdir(parents=True)
    pq.write_table(table({"model_name": ["m1", "m2"], "input_price_per_1m": [1.0, 2.0], "output_price_per_1m": [1.0, 4.0]}), root / "llm_candidates" / "train.parquet")
    (root / "llmrouter_generic").mkdir()
    cols = {"task_name": ["gsm8k"] * 2, "query": ["2+2?"] * 2, "metric": ["GSM8K"] * 2, "model_name": ["m1", "m2"],
            "performance": [1.0, 0.0], "input_tokens": [1000, 1000], "output_tokens": [500, 500], "response_time": [1.5, 2.5]}
    pq.write_table(table(cols), root / "llmrouter_generic" / "train.parquet")
    (record,) = list(sources.convert_xroutebench(root))
    assert record.dataset == "gsm8k" and record.split_origin == ["train"] and record.metadata["metric"] == "GSM8K"
    assert [(o.model, o.quality) for o in record.observations] == [("m1", 1.0), ("m2", 0.0)]
    assert record.observations[1].cost_usd == pytest.approx((1000 * 2 + 500 * 4) / 1e6)


def test_scope_marks_reasoning_runs_as_a_separate_candidate(tmp_path):
    cols = {"id": ["q1"] * 2, "prompt": ["P"] * 2, "model_name": ["m"] * 2, "is_correct": [True, False], "category": ["Physics"] * 2,
            "domain": ["/"] * 2, "reasoning_mode": [False, True], "usage_cost": [0.1, 0.2], "usage_prompt_tokens": [5, 5], "usage_completion_tokens": [6, 60]}
    pq.write_table(table(cols), tmp_path / "train-00000-of-00001.parquet")
    (record,) = list(sources.convert_scope60k(tmp_path))
    assert {o.model: o.quality for o in record.observations} == {"m": 1.0, "m+reasoning": 0.0} and record.dataset == "Physics"


def test_r2bench_keeps_budgets_apart_and_groups_by_the_sources_own_key(tmp_path):
    for model, budget, score in (("Qwen__Qwen3-0.6B", 200, 0.2), ("Qwen__Qwen3-0.6B", 2000, 1.0), ("zai-org__GLM-4.6", 2000, -1.0)):
        text = "ctx 1 q?" if model.startswith("Qwen") else "ctx 2 q?"  # the same query, worded differently per run
        cols = {"key": ["k1"], "original_prompt": [text], "actual_token_count": [50], "correctness_score": [score]}
        pq.write_table(table(cols), tmp_path / f"{model}__{budget}.parquet")
    (record,) = list(sources.convert_r2bench(tmp_path))
    got = {(o.model, o.budget_tokens): o.quality for o in record.observations}
    assert got == {("Qwen3-0.6B", 200): 0.2, ("Qwen3-0.6B", 2000): 1.0, ("GLM-4.6", 2000): None}


def test_llmrouterbench_reads_one_json_per_model_and_merges_queries(tmp_path):
    for model, score in (("a", 1.0), ("b", 0.0)):
        folder = tmp_path / "aime" / "hybrid" / model
        folder.mkdir(parents=True)
        run = {"model_name": model, "dataset_name": "aime", "split": "hybrid",
               "records": [{"origin_query": "Find x.", "score": score, "prompt_tokens": 10, "completion_tokens": 20, "cost": 0.01}]}
        (folder / "run.json").write_text(json.dumps(run), encoding="utf-8")
    (record,) = list(sources.convert_llmrouterbench(tmp_path))
    assert record.dataset == "aime" and {o.model: o.quality for o in record.observations} == {"a": 1.0, "b": 0.0}


def record_with(scores: dict[str, float], dataset="d", source="xroutebench", query="q") -> CanonicalRecord:
    return CanonicalRecord(f"{source}/{dataset}/{query}", source, dataset, query, [Observation(m, s) for m, s in scores.items()])


MODELS = [f"m{i}" for i in range(9)]  # m0 weakest ... m8 strongest


def test_tiers_depend_on_which_class_of_model_solves_the_query():
    rules = LabelRules()
    easy = record_with({m: 1.0 for m in MODELS}, query="easy")
    mid = record_with({m: 1.0 if i >= 3 else 0.0 for i, m in enumerate(MODELS)}, query="mid")
    strong_only = record_with({m: 1.0 if i >= 6 else 0.0 for i, m in enumerate(MODELS)}, query="strong")
    hard = record_with({m: 0.0 for m in MODELS}, query="hard")
    # give the pool a real ranking: strong models score higher overall
    filler = [record_with({m: 1.0 if i >= k else 0.0 for i, m in enumerate(MODELS)}, query=f"f{k}") for k in range(9)]
    groups = rank_pools([easy, mid, strong_only, hard, *filler], rules)
    pool = groups[("xroutebench", "d")]
    assert pool.groups["m0"] == "weak" and pool.groups["m4"] == "middle" and pool.groups["m8"] == "strong"
    tiers = [label_record(r, groups, rules).tier for r in (easy, mid, strong_only, hard)]
    assert tiers == ["low", "medium", "medium", "high"]
    assert label_record(easy, groups, rules).pass_share == 1.0


def test_a_strict_metric_does_not_make_every_query_unsolvable():
    rules = LabelRules()
    rows = [record_with({m: 0.5 if i >= 3 else 0.2 for i, m in enumerate(MODELS)}, query=f"s{k}") for k in range(5)]
    groups = rank_pools(rows, rules)
    assert groups[("xroutebench", "d")].threshold < 0.7
    assert label_record(rows[0], groups, rules).tier in {"low", "medium"}


def test_pools_with_too_few_models_or_observations_stay_unlabelled():
    rules = LabelRules()
    few = [record_with({"a": 1.0, "b": 0.0}, query="x")]
    assert rank_pools(few, rules) == {}
    rows = [record_with({m: 1.0 for m in MODELS}, query=f"q{k}") for k in range(3)]
    groups = rank_pools(rows, rules)
    thin = record_with({"m0": 1.0, "m8": 1.0}, query="thin")
    assert label_record(thin, groups, rules) is None


def test_r2bench_is_labelled_from_the_generous_budget_only():
    rules = LabelRules()
    rows = []
    for k in range(4):
        rec = CanonicalRecord(f"r2/{k}", "r2bench", "routerarena", f"q{k}")
        for i, m in enumerate(MODELS):
            rec.observations.append(Observation(m, 0.0, budget_tokens=200))
            rec.observations.append(Observation(m, 1.0 if i >= 2 else 0.0, budget_tokens=2000))
        rows.append(rec)
    groups = rank_pools(rows, rules)
    assert label_record(rows[0], groups, rules).tier == "medium"


def test_skill_hints_and_dataset_spellings():
    assert dataset_key("mmlu_pro") == dataset_key("mmlupro") == "mmlupro" and dataset_key("Human_Eval") == "humaneval"
    assert [skill_hint(d) for d in ("human_eval", "tau2", "gsm8k", "squad", "arenahard_creative_writing", "hle")] == [
        "coding", "tools", "reasoning", "extraction", "writing", None]


def make_canonical(root: Path) -> None:
    rng_models = MODELS
    for source in ("xroutebench", "scope60k"):
        recs = []
        for dataset in ("gsm8k", "mbpp", "squad", "tau2"):
            for k in range(30):
                shared = k < 3 and source == "scope60k" and dataset == "gsm8k"  # the same question as in xroutebench
                query = f"{dataset} question {k}" if source == "xroutebench" or shared else f"{source} {dataset} question {k}"
                cut = k % 9
                recs.append(CanonicalRecord(f"{source}/{dataset}/{k}", source, dataset, query,
                                            [Observation(m, 1.0 if i >= cut else 0.0) for i, m in enumerate(rng_models)], license="test"))
        write_jsonl(recs, root / f"{source}.jsonl")


def test_export_labels_dedupes_across_sources_holds_out_datasets_and_writes_a_valid_bundle(tmp_path):
    canonical = tmp_path / "canonical"
    make_canonical(canonical)
    out = tmp_path / "bundle"
    report = export(canonical, out, cap=25, held_out=("tau2",), sources=("xroutebench", "scope60k"))
    assert report["duplicates_dropped"] == 3 and report["after_cap"] > 0
    manifest, examples = fc.validate_inputs(out)  # the trainer's own verification
    assert manifest["held_out_families"] == ["tau2"] and manifest["test_data_exported"] is False
    assert {r["family"] for rows in examples.values() for r in rows}.isdisjoint({"tau2"})
    assert all(r["answers"]["skill"] is None and r["answers"]["route"] in ("low", "medium", "high") for r in examples["train"])
    heldout = out.parent / "bundle-heldout"
    ood = [json.loads(line) for line in (heldout / "test_ood.jsonl").read_text(encoding="utf-8").splitlines()]
    assert ood and {r["family"] for r in ood} == {"tau2"}
    states = [r["state"] for rows in examples.values() for r in rows]
    assert len(states) == len(set(states)) and not set(states) & {r["state"] for r in ood}
    labelled_skill = export(canonical, tmp_path / "b2", cap=25, held_out=("tau2",), use_skill=True, sources=("xroutebench",))
    assert labelled_skill["after_cap"] > 0
    _, with_skill = fc.validate_inputs(tmp_path / "b2")
    assert {r["answers"]["skill"] for r in with_skill["train"]} <= {"coding", "reasoning", "extraction"}


def test_dedupe_and_caps_are_deterministic():
    items = [{"sha": str(i % 5), "source": "s", "family": "f", "id": str(i)} for i in range(10)]
    kept, dropped = dedupe(items)
    assert len(kept) == 5 and dropped == 5
    many = [{"sha": str(i), "source": "s", "family": "f", "id": str(i)} for i in range(50)]
    assert cap_per_dataset(many, 7, 1) == cap_per_dataset(many, 7, 1) and len(cap_per_dataset(many, 7, 1)) == 7
    assert cap_per_dataset(many, 7, 1) != cap_per_dataset(many, 7, 2)


def test_head_metrics_skip_rows_without_a_skill_label():
    class Encoded:
        pass

    raw = [{"route": [3.0, 0.0, 0.0], "skill": [2.0, 0.0, 0.0, 0.0, 0.0]}, {"route": [0.0, 3.0, 0.0], "skill": [0.0, 2.0, 0.0, 0.0, 0.0]}]
    rows = [(Encoded(), {"route": 0, "skill": 0}), (Encoded(), {"route": 1})]
    assert fc.head_metrics(raw, rows, "skill")["rows"] == 1 and fc.head_metrics(raw, rows, "route")["rows"] == 2
    none = fc.head_metrics(raw, [(Encoded(), {"route": 0}), (Encoded(), {"route": 1})], "skill")
    assert none["rows"] == 0 and none["accuracy"] is None

def test_cleaning_removes_subjective_partial_credit_and_unreadable_zeros_but_keeps_strict_metrics():
    labels, rules = LabelRules(), CleanRules()
    hard_pool, easy_pool = Pool({}, 0.15), Pool({}, 0.6)
    arena = record_with({m: 1.0 for m in MODELS}, dataset="arenahard_coding")
    emotion = record_with({m: 1.0 for m in MODELS}, dataset="emorynlp")
    open_ended = record_with({m: 0.5 for m in MODELS}, dataset="routerarena", source="r2bench")
    judged_exact = record_with({m: float(i % 2) for i, m in enumerate(MODELS)}, dataset="routerarena", source="r2bench")
    for judged in (open_ended, judged_exact):  # R2-Bench is labelled from its 2,000-token budget
        for o in judged.observations:
            o.budget_tokens = 2000
    all_zero = record_with({m: 0.0 for m in MODELS}, dataset="hle")
    f1_partial = record_with({m: 0.4 for m in MODELS}, dataset="squad")  # a strict-F1 benchmark is objective: partial scores stay
    assert exclude_reason(arena, easy_pool, labels, rules) == "subjective dataset"
    assert exclude_reason(emotion, easy_pool, labels, rules) == "subjective dataset"
    assert exclude_reason(open_ended, easy_pool, labels, rules).startswith("judge gave partial credit")
    assert exclude_reason(judged_exact, easy_pool, labels, rules) is None
    assert exclude_reason(all_zero, hard_pool, labels, rules).startswith("every model scored 0")
    assert exclude_reason(all_zero, easy_pool, labels, rules) is None  # a zero is informative where good models do well
    assert exclude_reason(f1_partial, hard_pool, labels, rules) is None
    assert exclude_reason(arena, easy_pool, labels, CleanRules(drop_subjective=False)) is None


def test_export_reports_what_cleaning_removed(tmp_path):
    canonical = tmp_path / "canonical"
    make_canonical(canonical)
    extra = [record_with({m: 1.0 for m in MODELS}, dataset="emorynlp", source="xroutebench", query=f"e{k}") for k in range(5)]
    write_jsonl([*read_jsonl(canonical / "xroutebench.jsonl"), *extra], canonical / "xroutebench.jsonl")
    report = export(canonical, tmp_path / "b", cap=25, held_out=("tau2",), sources=("xroutebench",), clean=CleanRules())
    assert report["dropped_by_cleaning"] == {"xroutebench: subjective dataset": 5}
    manifest = json.loads((tmp_path / "b" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["label_source"]["dropped_by_cleaning"] == report["dropped_by_cleaning"] and manifest["label_source"]["cleaning"]["drop_subjective"]
