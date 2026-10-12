---
title: Off-the-shelf baseline results (before fine-tuning)
description: Full-scale benchmark of the unmodified Laya English endpoint on the AI-labeled 90/10 split, recorded for later comparison against a fine-tuned checkpoint.
---

> **Historical Laya baseline.** The Laya sections below were measured with the Laya English checkpoint before
> the project moved to Cloudflare Clef. They remain the comparison point for the Clef-Flash run on the
> same split; they are not Clef results. The Laya endpoint and client code no longer exist in this repository.

## Scope

This is the "taste the model before fine-tuning" baseline the plan called for:
the off-the-shelf Laya English checkpoint, unmodified, scored against the full
nominal 10% test split for both tracks (13.49% for coding because repository
groups stay intact). Labels are GPT-5.6 Sol AI-assisted
annotations (two-pass agreement plus third-pass majority vote on
disagreements), not human-reviewed gold — read these numbers as agreement with
an AI heuristic, not ground truth accuracy.

Endpoint: `laya-routing-exp`, deployment `english-cpu-large`
(`Standard_E16s_v3`, 16 vCPU / 128GB), 100% of live traffic. See
[docs/endpoint.md](./endpoint.md) for how this deployment and its latency
fixes were built.

Run date: 2026-10-02.

## Headline results

| Track | Rows | Strict/acceptable accuracy | Macro F1 | Truncation rate | Fallback/abstention rate |
|---|---|---|---|---|---|
| `coding-v1` (swe-bench) | 339 | 48.4% | 0.171 | 74.6% | 0.3% |
| `tool-api-v1` (bfcl + toolbench) | 348 | 33.0% | 0.181 | 57.2% | 40.8% |

Latency (client wall time, including auth + network + inference, after the
one-time token/connection setup cost on the first call):

| Track | First request | p50 | p95 | Throughput |
|---|---|---|---|---|
| `coding-v1` | 3.22s | 663ms | 718ms | 1.46 req/s |
| `tool-api-v1` | 3.63s | 655ms | 715ms | 1.50 req/s |

## Per-route breakdown

**`coding-v1`** (precision / recall / F1 / support):

| Route | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| debugging | 0.950 | 0.369 | 0.532 | 260 |
| code_generation | 0.342 | 0.907 | 0.496 | 75 |
| repository_qa | 0.000 | 0.000 | 0.000 | 3 |
| fallback | 0.000 | 0.000 | 0.000 | 1 |
| test_generation | — | — | — | 0 (absent from this test split) |
| code_review | — | — | — | 0 (absent from this test split) |

The default-budget model has high `code_generation` recall (0.907) but low
precision (0.342). It misses most true `debugging` rows (recall 0.369),
despite high precision on the rows it does label `debugging`.

**`tool-api-v1`** (precision / recall / F1 / support):

| Route | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| tool_lookup | 0.500 | 0.331 | 0.398 | 157 |
| tool_action | 0.289 | 0.611 | 0.393 | 36 |
| tool_workflow | 0.217 | 0.104 | 0.141 | 48 |
| no_matching_tool | 0.254 | 0.500 | 0.336 | 72 |
| clarification | 0.000 | 0.000 | 0.000 | 34 |
| web_research | 0.000 | 0.000 | 0.000 | 1 |
| memory_retrieval | — | — | — | 0 (absent from this test split) |

The model never predicts `clarification` at all (recall 0.0 on 34 true
examples) — a real, specific gap the off-the-shelf English checkpoint has on
this taxonomy.

## Truncation: a real, pervasive limitation, not a taxonomy artifact

Earlier in this project a 14-route coding taxonomy showed 100% truncation,
leading to a pivot to the simpler 6-route `coding-v1` taxonomy. At full
benchmark scale, truncation did **not** disappear with the smaller taxonomy:

* `coding-v1`: 74.6% of rows had state tokens dropped (vs. 100% under the old
  14-route taxonomy on a small sample, and ~97% on an early 30-row sample of
  just this 6-route data).
* `tool-api-v1`: 57.2% of rows also showed truncation at full scale, despite
  an earlier small bounded check reporting 0%. That earlier check was not
  representative of the full population.

Root cause (confirmed by inspecting individual request/response records, not
assumed): these requests used the Laya English checkpoint's default
~512-token input budget, and
real SWE-bench/BFCL/ToolBench request text is frequently long enough on its
own to exceed that budget — independent of how many routes the taxonomy
defines. Reducing route count only shrinks the instructions; it does not
shrink the user's actual request text. Accuracy above is still meaningfully
better than chance even under heavy truncation, which is itself useful
signal about the baseline model's behavior, but not evidence of beating a
training-majority baseline. Truncation rate should be
read as a first-class metric alongside accuracy, not a side note.
The budget is configurable, not a fixed encoder limit.

## Final result: max_len=2048 override, deployed to the live endpoint (2026-10-02)

Following the [max_len override experiment](./endpoint.md#max_len-override-experiment-2026-10-02-mechanically-works-costly-unvalidated),
`english-cpu-large` was redeployed with the scorer fix (optional, bounded
`max_len`/`head_max_len`, backward compatible) and re-benchmarked at
`max_len=2048, head_max_len=256` — the practical middle ground identified
earlier (meaningful truncation reduction at ~1-3s/request, well short of the
40-100+s cost of the 8192 ceiling). Same endpoint, same 100% live traffic,
same test10 split, this time using the **original, uncapped** test files —
the transport-cap workaround below was no longer needed once
`MAX_STATE_CHARS` was raised alongside `max_len`.

| Track | Accuracy @512 (baseline) | Accuracy @2048 | Truncation @512 | Truncation @2048 | p50 @2048 | p95 @2048 |
|---|---|---|---|---|---|---|
| `coding-v1` | 48.4% | **78.2%** | 74.6% | 9.1% | 1.16s | 3.49s |
| `tool-api-v1` | 33.0% | **23.6%** | 57.2% | 3.4% | 0.93s | 3.34s |

The observed accuracy change is opposite between tracks: +29.8 points for
coding and -9.4 points for tool/API. This is not a clean context-length
ablation: `head_max_len` also changed, and three coding inputs were capped
in the earlier run but uncapped in the final run. The measurements do not
establish that extra dialogue distracts the model.

The live endpoint defaults remain unchanged unless callers explicitly pass
overrides. No automatic per-track configuration was implemented. Neither
configuration is established as a production default; choose future settings
on training-derived validation data, not repeated test-set comparisons.

For perspective, always predicting the training-majority route gives 76.70%
on coding (`debugging`, 260/339) and 45.11% on tool/API (`tool_lookup`,
157/348). The 2048 coding result is only 1.47 percentage points above that
comparator; both tool/API results are below it. Macro F1, minority-route
recall, calibration, and latency matter alongside accuracy.

## Clef-Flash on the same test10 split (2026-10-06)

Same files, same silver labels, same contracts and instructions as the Laya runs above, scored through the
live Clef-Flash endpoint (`Cloudflare/clef-flash`, one H100) at the default `max_length` of 16,384 tokens, with
no prompt or setting changes. Runs: [coding](../outputs/clef-flash-test10-coding-v1-20261006/summary.json),
[tool/API](../outputs/clef-flash-test10-tool-api-v1-20261006/summary.json). All 339 and 348 requests
completed with no failures.

| Track | Training-majority | Laya @512 | Laya @2048 | **Clef-Flash** | Clef-Flash macro F1 (Laya @512) |
|---|---|---|---|---|---|
| `coding-v1` (339) | 76.7% | 48.4% | 78.2% | **80.8%** (95% CI 73.9 to 86.3) | 0.249 (0.171) |
| `tool-api-v1` (348) | 45.1% | 33.0% | 23.6% | **73.6%** | 0.540 (0.181) |

| | Truncation | p50 | p95 | Throughput | ECE | Top-2 accuracy |
|---|---|---|---|---|---|---|
| `coding-v1` | 0.3% | 359 ms | 539 ms | 2.5 req/s | 0.242 | 88.2% |
| `tool-api-v1` | 0.0% | 386 ms | 557 ms | 2.3 req/s | 0.201 | 91.7% |

Macro F1 counts routes with no test rows as zero, as in the Laya tables. Coding has two such routes of six, so it cannot exceed 0.67; tool/API has one of seven, so it cannot exceed 0.86.

**Tool/API is a large, real gain.** 73.6% is 28.5 points above the majority baseline, where every Laya
configuration was below it. `no_matching_tool` (F1 0.87) and `tool_lookup` (F1 0.80) are strong. Weak spots:
`tool_workflow` recall is 0.29 (34 of 48 called `tool_lookup`) and `clarification` recall is 0.21 (20 of 34
called `tool_lookup`).

**Coding is a modest gain.** 80.8% is 4.1 points above the majority baseline and 2.6 above Laya at 2048, with a
CI of 73.9 to 86.3 that includes the majority baseline's 76.7%. Do not claim it beats the baseline. The
`debugging` route is excellent (precision 0.96, recall 0.95). `code_generation` recall is 0.37: of 75 true rows,
28 are right, 20 go to `fallback`, 15 to `repository_qa`, 8 to `debugging` and 4 to `code_review`. Four routes
have 3 or fewer test rows, so their scores say nothing.

**Calibration is under-confident, more so than on the synthetic probe.** Selective error is excellent but
coverage is tiny: on coding, confidence of at least 0.5 keeps 65% of rows at 1.4% error, and at least 0.7 keeps 23%
at 0%. Almost nothing reaches 0.8. On tool/API the error is 16.8% even at 0.5 and 5.5% at 0.7, so confidence there
is a weaker signal. Fit a temperature on the calibration partition before choosing thresholds.

Caveats: these labels are AI-derived silver, not gold. Test10 has already informed earlier configuration choices,
so this is an exposed-test comparison. Truncation is nearly gone, which also changes the comparison with Laya, so
the gain mixes model quality with the larger context.

## Clef-Flash fine-tuned, on the same test10 split (2026-10-07)

LoRA fine-tuning of Clef-Flash on the frozen 3,966-row training partition (recipe and costs in
[fine-tuning.md](fine-tuning.md)), then scored once on exactly the same test files, labels, contracts and
instructions as every row above. Epoch 3 of 3 was selected on validation, and one temperature (0.937) was fitted
on the calibration partition. The test files were never used for training, selection or calibration, and this
is the only time they were scored. Runs:
[coding](../outputs/clef-flash-finetuned-test10-coding-v1-calibrated-20261007/summary.json),
[tool/API](../outputs/clef-flash-finetuned-test10-tool-api-v1-calibrated-20261007/summary.json), plus the
uncalibrated pair beside them.

| Track | Training-majority | Laya @512 | Laya @2048 | Clef-Flash off the shelf | **Clef-Flash fine-tuned** |
|---|---|---|---|---|---|
| `coding-v1` (339) | 76.7% | 48.4% | 78.2% | 80.8% | **95.3%** (95% CI 93.4 to 97.0) |
| `tool-api-v1` (348) | 45.1% | 33.0% | 23.6% | 73.6% | **95.7%** |

| | Coding off the shelf | Coding fine-tuned | Tool/API off the shelf | Tool/API fine-tuned |
|---|---|---|---|---|
| Macro F1 | 0.249 | 0.381 | 0.540 | 0.819 |
| Top-2 accuracy | 88.2% | 99.1% | 91.7% | 99.4% |
| ECE, uncalibrated | 0.242 | 0.034 | 0.201 | 0.034 |
| ECE, calibrated | n/a | 0.018 | n/a | 0.022 |
| Brier | 0.337 | 0.077 | 0.428 | 0.072 |

Per route (recall, off the shelf to fine-tuned):

* Coding: `code_generation` 0.37 to 0.92 (75 rows; it was the model's main weakness), `debugging` 0.95 to 0.97.
  `repository_qa` (3 rows) and `fallback` (1 row) are too small to read: 1 of 3 and 0 of 1 correct.
* Tool/API: `tool_workflow` 0.29 to 1.00, `clarification` 0.21 to 0.85, `tool_action` 0.64 to 0.94,
  `tool_lookup` 0.94 to 0.97, `no_matching_tool` 0.89 to 0.94.

The coding interval now excludes the training-majority baseline (76.7%), which the off-the-shelf run did not. Both
tracks go from "gain over a trivial baseline" to "close to the labeler". Calibration also improved a lot: the base
model was under-confident (ECE about 0.2) and fine-tuning alone brings it to 0.034, with the fitted temperature
(0.937, close to 1) trimming it to about 0.02. Temperature never changes a choice, so accuracy is identical with
and without it. With confidence of at least 0.9, coding keeps 93% of rows at 2.5% error and tool/API keeps 67% at
1.3%.

How much to trust this:

* **This measures agreement with the silver labeler on the same sources, not general routing skill.** The labels
  are GPT-5.6 Sol annotations, and training and test come from the same two datasets (SWE-bench, BFCL). BFCL in
  particular is heavily templated, so a large part of the `tool_workflow` and `clarification` gain is probably
  learning its conventions. Do not expect 95% on different requests, tools or labelers.
* **No test-set tuning in this run**, but test10 had already informed earlier work, so this is an exposed-test
  result and not an unbiased final estimate.
* **Untested routes stay untested.** `code_review` and `test_generation` have no training or test rows, and
  `fallback`, `repository_qa` and `web_research` have 1 to 3 test rows, so the model's behavior on them is unknown.
* **Context budgets differ from the Laya rows**, as before.
* Latency (about 160 ms median, in process on the H100) is not comparable to the endpoint's client-side figures.

## What fine-tuning changed row by row (2026-10-07)

Taking the 157 test rows the off-the-shelf Clef-Flash got wrong and re-running them live through both deployments.
Both reproduced their stored results exactly (the same choice on all 157 rows, probability difference 0.0000), so
the deployed fine-tuned model is the one that was evaluated. The list of rows is in `outputs/base-wrong-rows.json`.

| | Base wrong | Fixed by fine-tuning | Still wrong | New errors (base right, tuned wrong) |
|---|---|---|---|---|
| `coding-v1` (339) | 65 | 53 (82%) | 12 | 4 |
| `tool-api-v1` (348) | 92 | 81 (88%) | 11 | 4 |
| Total | 157 | 134 (85%) | 23 | 8 |

Net gain 126 rows. Where the fixes came from:

* Coding: `code_generation` rows sent to `fallback` (19 of 20 fixed), to `repository_qa` (15 of 15), to `code_review`
  (4 of 4); `debugging` rows sent to `code_review` (6 of 6). Still hard: `code_generation` called `debugging` (3 of 8 fixed).
* Tool/API: `tool_workflow` called `tool_lookup` (34 of 34 fixed), `clarification` called `tool_lookup` (19 of 20),
  `tool_action` called `clarification` (8 of 8). Still hard: `clarification` called `no_matching_tool` (1 of 5 fixed).
* Confidence: the base model averaged 0.37 (coding) and 0.43 (tool/API) on its own mistakes, so it was unsure when
  wrong. The tuned model averaged 0.92 and 0.95 on the rows it fixed.
* The 8 new errors are a mix: 4 `debugging` rows now called `code_generation`, plus 4 tool/API rows spread across
  routes.

Read the examples with care. Some fixes look like learned dataset conventions rather than understanding: a one-word
request ("sum") is labeled `clarification`, and a two-language list request is `tool_workflow` because of its BFCL
category. A fixed coding row ("Raise error when blueprint name contains a dot") is labeled `code_generation` where a
reader might say `debugging`, and a still-wrong one ("Require a non-empty name for Blueprints") is the same kind of
borderline request, called `debugging` at 0.83. Where the label is itself a judgment call, agreement with it is a
weak signal.

## Two infrastructure issues hit and fixed during this run

1. Intermittent HTTP 424 responses were handled with a bounded retry
   (`LayaEndpointBackend`, eight retries after the initial attempt).
   HTTP 424 can also wrap scorer errors; successful container access logs
   do not establish a gateway-only cause. Retry delays are included in
   client wall time, but retry counts were not separately recorded.
   See [routing_lab/endpoint.py](../routing_lab/endpoint.py).
2. **Hard rejection of oversized requests**, unrelated to the above and not
   fixed by retrying: two rows (`swe-bench:test:pydata__xarray-3637`, 44,292
   chars; `swe-bench:test:pydata__xarray-5662`, 57,256 chars) were
   deterministically rejected with HTTP 424 regardless of retry count, even
   in isolation, while a 24,770-char row succeeded. At the time, this looked
   like an opaque gateway- or container-side limit, so a 20,000-character
   cap was applied to a benchmark copy. This changed three rows, including
   the already-working 24,770-character row (the
   canonical `coding-v1-test10.jsonl` stayed untouched; the capped copy is at
   `outputs/coding-v1-test10-transport-capped-20261002/coding-v1-test10.jsonl`).
   **Update:** the real cause was found later, while adding the `max_len`
   override — the scorer's own `MAX_STATE_CHARS = 32_768` pydantic validation
   in `azure/online/code/score.py` was silently rejecting both rows (44,292
   and 57,256 both exceed 32,768; 24,770 does not). Raising that limit to
   131,072 alongside the `max_len` work (see below) fixed this for real; the
   final max_len=2048 benchmark above used the original, uncapped test files
   with no row failures.

## Dataset provenance for this run

* `data/public-routing/gpt56-sol-labels-coding-v1-merged-20261002/` — the
  finalized labeled dataset: 2,513 `coding-v1` rows (relabeled under the
  6-route taxonomy after the 14-route truncation finding) + 3,482
  `tool-api-v1` rows (unchanged, copied verbatim from the original
  `copilot-coding-v2` + `tool-api-v1` labeling run — not re-labeled, since
  that track's rows and taxonomy were never affected by the pivot).
* `data/public-routing/gpt56-sol-labels-coding-v1-split-20261002/` — the
  90/10 group-safe stratified re-split of the merged dataset. The
  `tool-api-v1` train/test counts (3,134 / 348) match the original split.
  Matching counts alone do not prove byte-identical data or ordering.
* See [docs/annotation.md](./annotation.md) for the full labeling pipeline
  and [routing_lab/benchmark_filter.py](../routing_lab/benchmark_filter.py)
  for the source-bundle filtering and transport-cap utilities used to avoid
  re-labeling the unaffected tool-api-v1 track.

## Next step

Prepare one shared, catalog-conditioned checkpoint using only `train90`.
Reserve training-derived validation and calibration partitions before fitting.
The existing test files have already informed model/configuration decisions:
freeze them for qualified before/after reporting, but do not call them an
untouched final holdout. An unbiased final generalization claim needs a fresh
independent reviewed set. Compare base and tuned models under identical
inputs, token budgets, route catalogs, and serving hardware.
