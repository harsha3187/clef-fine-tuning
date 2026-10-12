# clef-flash-capability-ft-v3

A LoRA fine-tune of **Cloudflare Clef-Flash** that reads one request and says how much language-model capability it needs:
`low`, `medium` or `high`. It is the "System-1" router in this repository: a selector then maps its probabilities to the cheapest
endpoint that is expected to do the job (see `docs/router-architecture.md`).

This folder holds the adapter and the classification head only. The base model is not in the repo.

| | |
|---|---|
| Base model | `Cloudflare/clef-flash`, revision `17f0b0ad64efb65d273590632833508766b2aae6` (Apache-2.0, not gated) |
| Adapter | LoRA rank 32, alpha 64, dropout 0.05, on the language model's attention and MLP projections (`adapter/`, 400 tensors, 321 MB) |
| Head | the joint schema head, retrained (`joint_head.safetensors`, 122 tensors, 487 MB) |
| Calibration | one temperature, 1.9436, fitted on the route head with 770 held-out rows (`clef_finetune.json`) |
| Prompt | version **p2**. The request must be sent with the exact question text it was trained on (`routing_lab/backends.py`, `contracts/capability-v1.json`) |
| Trained | 2026-10-09, one H100 NVL, 160 minutes, 3 epochs (best epoch 3), 1,023 steps, about $19 |
| Training job | `hungry_eye_8t7gqzr69g` on Azure ML; registered as `clef-flash-capability-ft:3` |

## Provenance

These files are byte-identical to what runs on the Azure endpoint. On 2026-10-11 the live endpoint (`flash-cap-ft3`) reported the same
values as the files in this folder:

| | SHA-256 or value |
|---|---|
| `adapter/adapter_model.safetensors` | `3651988bb592eda963ccf96afe38ec89068840c26464094b1f5f09e481ab6beb` |
| `joint_head.safetensors` | `bec5425fd8cc4e1f0d086c12d7d62faade826b94eb89d00edb3693faf1665721` |
| Training bundle manifest (`routing-ext-v2`) | `ecb6aaeda93dec2a7fb1976d104696572524b2b25991e6b9be0bf806162c9e2c` |

Check a copy with `python -m routing_lab.verify_model models/clef-flash-capability-ft-v3`. It compares every file with `SHA256SUMS`,
validates the safetensors structure and the LoRA rank, confirms the base-model pin, and tells you if a file is still a Git LFS pointer
(`git lfs pull`).

## Training data

The cleaned bundle `routing-ext-v2`, built from four public routing datasets (`docs/routing-datasets.md`): LLMRouterBench, xRouteBench,
SCOPE-60K and R2-Bench. 5,444 training rows, 787 validation and 770 calibration, across 56 source and sub-benchmark groups, after dropping subjective datasets,
open-ended judge-scored queries and queries every model scored 0 on in hard pools. Eight sub-benchmarks (gsm8k, livecodebench, tau2, finqa, bbh,
arenahard_math, history, boolq) were held out entirely.

Each label comes from how a pool of models fared on the query: `low` if the weaker third of that pool solved it, `medium` if only the middle
or strong third did, `high` if even the strong third mostly failed. There were **no kind-of-work labels**, so the skill head was not trained
and its answers are meaningless. Do not use them.

## How well it works

On its own validation labels (787 rows): accuracy 63.7% against 41.7% for always guessing the majority class, macro F1 0.629, calibration error
(10-bin ECE) 0.053.

**On this project's own measured tasks it is clearly worse than the previous fine-tune** (`clef-flash-capability-ft:2`, trained on this project's
labels), because the external labels sit on a different, more cautious scale:

| Held-out set | This model | Version 2 (same prompt) |
|---|---|---|
| 542 tasks, tier accuracy (always "low" scores 64.2%) | 43.9% | 65.9% |
| over-estimates on those 542 | 236 | 116 |
| 60 hard tool-use tasks, tier accuracy (always "low" scores 51.7%) | 53.3%, calls 55 of 60 "medium" | 61.7% |

It over-estimates: easy requests are often called medium or high. Routing with it is safe in the sense that a stronger endpoint is chosen
too often, not too rarely, and it saves less money than version 2 did. It was not scored on the 1,893 external test rows.

## Use it

- **Hosted:** the Azure ML endpoint, through `routing_lab.capability.CapabilityRouter` (see `docs/docker.md`).
- **In a container:** `docker compose --profile local-clef up`. Mounts this folder and serves the same `/score` API (needs an NVIDIA GPU).
- **From Python:** `clef.load("flash", snapshot="models/clef-flash-capability-ft-v3", calibrated=True)` in `routing_lab/clef.py`. It downloads
  the base model (18 GB) from Hugging Face, merges the adapter and loads the head.
- **Another application:** call the running service's `POST /v1/select` (see `docs/docker.md`) or `POST /score` on the Clef container.

## Limits

- Trained on benchmark questions (maths, code, multiple choice, reading comprehension, some agent tasks), not on real agent traffic.
- Labels are relative to each source's models. "Low" means a weaker pool handled it, not that a particular endpoint of yours will.
- Calibrated on the external label scale, not yours. Recalibrate or retrain on your own measured outcomes before trusting the probabilities.
- Prompt-sensitive: a different wording of the instructions or tier descriptions is an input it was not trained on.

## Licences

The base model is Apache-2.0. The training data come from public benchmarks whose rights remain with their providers; SCOPE-60K is Apache-2.0,
R2-Bench is MIT, and the xRouteBench and LLMRouterBench dataset pages state no licence. Treat the weights as research material and review the
data terms before any commercial use.
