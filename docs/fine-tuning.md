---
title: Shared router fine-tuning data and Clef plan
description: Frozen, catalog-conditioned silver-data partitions for fine-tuning Clef, and what remains before training starts.
---

## Experiment boundary

Train one shared Clef-Flash checkpoint on both `coding-v1` and `tool-api-v1`. Each example
supplies its own route catalog and instructions, which is exactly the SystemOne record Clef
already consumes (`state`, `questions`, and a supervised answer). This is a typed-decision
fine-tune, not a fixed 13-label classifier and not a text-generation fine-tune. Never offer
coding routes to a tool/API-only request.

Preparation does not start a training job, create compute, or change endpoint traffic.

## Status

**Trained, selected and evaluated once on the frozen test split.** Clef-Flash reaches 95.3% on coding and
95.7% on tool/API, up from 80.8% and 73.6% off the shelf, with calibration error down from about 0.2 to 0.02.
The full comparison, per-route detail and caveats are in
[baseline-results.md](baseline-results.md#clef-flash-fine-tuned-on-the-same-test10-split-2026-10-07). Read the
caveats there: this measures agreement with a silver labeler on the same two sources, not general routing skill.

The fine-tuned model is deployed as `flash-ft-gpu` on the same endpoint at 0% traffic, so the default route still
serves the off-the-shelf model. See [endpoint.md](endpoint.md#fine-tuned-deployment-2026-10-07).

## Data and selection

Use only the frozen `train90` inputs (2,174 coding and 3,134 tool/API rows) to
derive training, validation, and calibration partitions. Target 80/10/10
within that pool, but keep connected repositories, groups, and normalized
duplicate inputs intact. Actual fractions can differ. Known development
examples stay in training.

The original test files remain outside the training upload. Record their
hashes for later checks. They have already been used to probe configuration,
so the final comparison is an exposed-test before/after result, not an
unbiased final generalization estimate.

Labels stay AI-derived silver. Two passes and a third-pass majority from one
GPT model do not turn them into human-reviewed gold. Do not change
`review_status` to bypass an evaluation gate.

The prepared [bundle](../data/finetuning/shared-router-v3/manifest.json) has:

| Track | Training | Validation | Calibration |
|---|---:|---:|---:|
| Coding | 1,458 | 414 | 302 |
| Tool/API | 2,508 | 313 | 313 |
| Combined | 3,966 | 727 | 615 |

Coding achieves approximately 67/19/14 rather than 80/10/10 because whole
repository groups stay together. Five of its six `test_generation` examples
fall in validation, leaving only one training example. This is a limitation
of these data and grouping rules, not evidence that six-route coverage is
adequate.

To reproduce preparation into a new directory:

```powershell
.\.venv\Scripts\python.exe -m routing_lab.finetune_data `
  --source data\public-routing\gpt56-sol-labels-coding-v1-split-20261002 `
  --output data\finetuning\shared-router-v3
```

The [preparation module](../routing_lab/finetune_data.py) verifies source
hashes, preserves each training-pool row exactly once, freezes catalogs,
and records missing-route warnings. The mixed examples include `state`,
`questions`, and supervised `answers`; labels and provenance never enter
the model's input state.

Use bundle `shared-router-v3`. It has the same partitions as v2 (3,966 train, 727
validation, 615 calibration) but its policy version no longer names Laya, so v1 and v2
bundles no longer verify. They are kept only as history; regenerate rather than edit them.

Report each track separately and use the equal-weight mean of their
validation macro F1 values for checkpoint selection. Do not let a larger
track dominate selection by pooling accuracy. Reserve calibration data for
confidence fitting after selecting weights; do not optimize model weights
or hyperparameters on it.

Coverage is a material limitation:

* Coding has no `code_review` training examples and very few examples for
  `test_generation`, `repository_qa`, and `fallback`.
* Tool/API has only two `memory_retrieval` rows in the original training pool.
* An absent route cannot be learned from positive examples in this data.
  Preserve the catalog, disclose missing coverage, and collect reviewed
  examples before claiming a general-purpose router.
* Training-majority baselines score 76.70% on the existing coding test and
  45.11% on tool/API. Aggregate accuracy alone is not a useful success gate.

Training-derived validation comparators from the earlier Laya work are saved in
[the baseline outputs](../outputs/shared-router-validation-baselines-v2/):

| Track/comparator | Accuracy | Macro F1 |
|---|---:|---:|
| Coding training-majority | 71.01% | 0.1384 |
| Coding lexical rules | 28.99% | 0.1652 |
| Tool/API training-majority | 45.69% | 0.0896 |

These runs fit majority choices on training labels only. They score silver
validation labels in probe mode; no review flags are promoted.

## How it trains

[routing_lab/finetune.py](../routing_lab/finetune.py) follows the recipe Cloudflare describes for Clef: a
frozen Qwen backbone, low-rank adapters and the joint schema head optimized together, with label-smoothed
cross-entropy plus a Brier term. It does not implement their RLCD reinforcement stage: the release ships no
training code or recipe for it, and silver labels do not justify a reward model.

| Choice | Default | Why |
|--------|---------|-----|
| Trainable | LoRA r=32, alpha 64 on the language model's attention, linear-attention and MLP projections, plus the whole joint head | Cloudflare used rank 256; our 3,966 rows are far fewer, so a smaller rank limits overfitting |
| Loss | Label-smoothed cross-entropy (0.05) plus Brier (weight 1) over the offered options | Same family as the release; Brier pushes probabilities toward calibration |
| Learning rates | LoRA 1e-4, head 5e-5, cosine decay with 5% warmup | Conservative starting point, not tuned |
| Batch | 4 sequences per forward pass (grouped by length), 4 passes per optimizer step, so 16 per step | Batch 1 measured only 700 tokens/s because short sequences are dominated by per-call overhead; batches of 4 reach 1,950 |
| Length | 4,096 tokens in training, 16,384 in evaluation | Only 5 of 3,966 training rows exceed 4,096 (longest 7,947) |
| Epochs | 3, keep the best | Chosen on validation, so later overfitting is discarded |
| Precision | bf16 backbone, fp32 adapters and head, bf16 autocast | Standard; fp32 master weights keep small updates from rounding away |

* The label index comes from the model's own option order (options are sorted by key), so training
  targets match exactly what inference scores.
* Selection uses the equal-weight mean of per-track validation macro F1, never pooled accuracy, so the
  larger tool/API track cannot dominate.
* After selection, one temperature is fitted on the calibration partition only. It rescales confidence and
  never changes which route is chosen. Accuracy is therefore identical with and without it; only ECE and
  Brier change.
* Padding is safe: a batch of six real examples from 486 to 2,048 tokens gave logits identical to scoring
  them one at a time (largest difference 1e-7, same choices). Batches are grouped by length so little
  padding is wasted.

## What the runs measured

| Job | What | Result | Cost |
|-----|------|--------|------|
| `great_chaconia_b6g14ryzy9` | Smoke: 4 steps on the longest sequences, reload check | Passed on a real H100 NVL: 24.5 GiB peak, 24 of 24 reload agreement | about USD 1 |
| `bold_school_7lnkbhvgcv` | Full run, batch 1 | **Cancelled after 18 minutes**: a steady 700 tokens/s projected 5.4 hours, over budget and near the 6-hour timeout | about USD 2 |
| `happy_gyro_zlhvln21jq` | Full run, batches of 4 | Completed: 744 steps in 130 minutes at about 1,950 tokens/s, peak 38.3 GiB | about USD 15 |
| `eager_tongue_9hxqpz2gx1` | One-time test10 evaluation, with and without calibration | Completed in 9 minutes | about USD 2 |

Total about USD 20. The batch-1 slowdown was per-call overhead on typical (about 1,100 token) sequences in the
torch fallback for linear attention; grouping four length-matched sequences per pass gave 2.8 times the
throughput. Padding is safe: a batch of six real examples from 486 to 2,048 tokens gave logits identical to
scoring them one at a time (largest difference 1e-7).

The full run's validation score rose every epoch, so the last epoch was selected (equal-track mean macro F1
0.601, 0.627, 0.657). Per-track validation accuracy was 93.5% for coding and 93.9% for tool/API at epoch 3.
Calibration fitted a temperature of 0.937 on the 615 calibration rows (loss 0.2301 to 0.2283), which is
close to 1 because fine-tuning had already fixed most of the under-confidence. The export reloads with 24 of 24
matching answers (largest probability difference 0.011). Training outputs are in
`outputs/clef-train-results/run/` (about 770 MB, nearly all of it the export).

Before any GPU spend, the same code also ran end to end on CPU with a tiny random model: it overfit seven real
examples (loss 1.81 to 0.001, accuracy 1/7 to 7/7), which confirmed the loss, the label-to-option mapping, gradient
flow into both the LoRA adapters and the head, and the LoRA targets (25 language-model projections, none on the
vision tower).

## Run it

Commands run from the project root in PowerShell. Nothing here runs a job unless every consent flag is
passed. Create the GPU cluster once (zero minimum nodes, so it costs nothing while idle), then stage and
submit the smoke job:

```powershell
.\.venv\Scripts\python.exe -m azure.training.provision --approve

.\.venv\Scripts\python.exe -m azure.training.stage train `
  --bundle data\finetuning\shared-router-v3 --config azure\training\smoke-config.json `
  --output outputs\clef-smoke-upload --submit --approve-cost --approve-upload
```

The smoke job trains four optimizer steps on the longest sequences, validates and calibrates on 16 rows per
track, exports, and reloads the export. Read `run.json` for `peak_cuda_allocated_bytes`, `tokens_per_second`
in the log, and `round_trip`. Only then submit the full config:

```powershell
.\.venv\Scripts\python.exe -m azure.training.stage train `
  --bundle data\finetuning\shared-router-v3 --config azure\training\config.json `
  --output outputs\clef-train-upload --submit --approve-cost --approve-upload
```

`provision.py` grants the cluster's identity Blob Data Contributor and File Data Privileged Contributor on
the experiment storage account and AcrPull on its registry, and nothing wider. It needs 40 vCPUs of H100
quota, which is shared subscription-wide with the endpoint's 40, so total use is 80 of 100.

## Evaluate on the frozen test split

Model selection is finished before the test files are staged. The evaluator uploads only the two
`*-test10.jsonl` files, after checking their hashes against the ones recorded in the bundle, and needs
explicit test approval and the finished training job's name:

```powershell
.\.venv\Scripts\python.exe -m azure.training.stage eval `
  --bundle data\finetuning\shared-router-v3 `
  --source data\public-routing\gpt56-sol-labels-coding-v1-split-20261002 `
  --output outputs\clef-eval-upload --model-job <training-job-name> `
  --submit --approve-cost --approve-upload --approve-test-use
```

It scores both tracks with and without calibration through the same `routing_lab.cli run` path and the
same metrics as the baseline, so results line up with
[baseline-results.md](baseline-results.md). The off-the-shelf Clef-Flash numbers there are the comparison.
Test10 has already informed earlier choices, so this is an exposed-test before/after result, not an
unbiased final estimate. Run it once.

To score an export locally on a GPU instead:

```text
python -m routing_lab.cli run --backend clef --snapshot <export> --calibrated --probe --allow-model-download ...
```

## Risks to settle first

* Coding has no `code_review` training rows and very few for several other routes, so fine-tuning
  cannot fix those. Collect reviewed examples before claiming a general router.
* Silver labels come from one model family. Gains over the baseline measure agreement with those labels,
  not routing correctness.
* Large gains on 3,966 rows can be overfitting to labeling quirks. Validation macro F1 per epoch and the
  final exposed-test comparison are the checks; neither proves generalization.
* Cloudflare's Decision Index shows Clef-Flash trailing Clef on some tasks (for example CLINC150), so
  compare the 27B on the same baseline before committing to one variant.
* The fine-tuned model has no live traffic yet; moving traffic to it is a separate, deliberate step.
* There is no mid-run checkpoint or resume: a failure after hours of training restarts from zero. The
  best epoch is exported as soon as it is found, so an interrupted run still leaves a usable export.
