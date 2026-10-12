---
title: Prepared Azure ML online endpoint
description: Scoring contract and approval-gated GPU serving configuration for Cloudflare Clef.
---

## Deployment status

Deployed in East US 2 on one H100 and verified live. It bills continuously until the deployment is deleted.
See the [endpoint guide](../../docs/endpoint.md) for sizing, cost, deployment and verification.

This serves the off-the-shelf Clef-Flash checkpoint (`Cloudflare/clef-flash`, pinned revision).
No training, calibration or fine-tuning is involved. `init()` loads once per worker on CUDA and
verifies the SHA-256 of the release's `joint_schema_model.py` before importing it. Set the
`CLEF_VARIANT` environment variable in the deployment YAML to `clef` to serve the 27B model.

## Fine-tuned deployment

`deployment-finetuned.yml` creates `flash-ft-gpu` from the registered model `clef-flash-router-ft:1` with
`CLEF_FINETUNED=true`. The scorer then loads the pinned base, merges the LoRA adapter, loads the fine-tuned
head and applies the fitted temperature, and refuses to start if the model is missing. Responses add
`experiment_model.fine_tuned` with the adapter and head SHA-256. Clients select it with the
`azureml-model-deployment` header (`--deployment flash-ft-gpu`) and can pin the adapter hash.

## Contract

POST UTF-8 JSON with `Content-Type: application/json`. The body is the SystemOne
`state` plus `questions` contract, **not** `input_data` or an OpenAI message list:

```json
{
  "state": "Explain how this scheduler dispatches jobs.",
  "questions": {
    "route": {
      "type": "choice",
      "instructions": "Select one initial owning coding agent.",
      "criteria": {
        "debugging": "Diagnose and fix defects.",
        "code_generation": "Implement requested functionality.",
        "test_generation": "Create tests for existing behavior.",
        "code_review": "Review code changes.",
        "repository_qa": "Explain or locate repository behavior.",
        "fallback": "Unsupported request or missing essential context."
      }
    }
  },
  "max_length": 16384
}
```

Use the experiment's actual versioned instructions and descriptions; the ones above are
illustrative. Exactly one question named `route`, type `choice`, must carry the complete
catalog of one supported taxonomy (`coding-v1`, `copilot-coding-v2` or `tool-api-v1`).
`max_length` is optional, 512 to 32,768, and defaults to 16,384 tokens.

Limits: 524,288 UTF-8 bytes for the whole body; 1 to 131,072 characters of state; 1 to 4,096
for instructions; 1 to 2,048 per criterion. Whitespace-only text, duplicate JSON keys, wrong
types, unknown fields (including `images`, `videos` and `max_state_tokens`), batches, revision
overrides, additional questions and incomplete or extra criteria are rejected. Over-long state
is truncated by Clef itself; `usage.state_tokens_dropped` reports how much.

Success is HTTP 200 with the native answer and usage fields plus server-owned provenance:

```text
{
  "answers": {"route": {"type": "choice", "choice": <one route>, "confidence": <p>, "probabilities": <map>}},
  "usage": {"input_tokens": <n>, "output_tokens": 0, "state_tokens_dropped": <n>},
  "experiment_model": {"checkpoint": "Cloudflare/clef-flash", "revision": "<pinned>", "variant": "flash"}
}
```

Errors are sanitized and never echo request text: 400 `invalid_request`, 413
`request_too_large`, 500 `inference_failed`, 503 `not_ready`. Request contents and model
exception details are not logged.

## Files

* `code/score.py`: Azure ML `init`/`run` adapter. Importing it loads no model.
* `deployment.yml`: one `Standard_NC40ads_H100_v5` instance, one request at a time, generous
  readiness probes because startup downloads about 19 GB.
* `environment.yml` and `conda.yml`: Python 3.11, `torch` 2.11.0 and `torchvision` 0.26.0 (CUDA 12.6),
  `transformers` 5.10.2 and `azureml-inference-server-http`.
* `manage.py`: SDK helper to deploy, show status and fetch logs; it stages only the scorer,
  `routing_lab/__init__.py`, `taxonomy.py` and `clef.py`. Deployment needs `--approve-cost`.
