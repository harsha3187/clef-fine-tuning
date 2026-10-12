---
title: Review and evaluation protocol
description: Separate synthetic probes, reviewed benchmarks and downstream evidence.
---

## Agent ownership

Select one initial owning agent, not an execution plan. Debugging owns a combined
diagnose/fix/regression-test request. Standalone test creation goes to test generation.
Code review receives a proposed change. Repository Q&A is read-only explanation or
search. Fallback includes unsupported work and missing essential context.

Do not infer that a diagnosis request authorizes writing. Capability contracts
describe what an agent can do; a future execution policy must independently enforce
the user's permissions and approvals.

## Probe before benchmarking

The 60 bundled records are draft synthetic examples, ten per route. Run them as
`--probe`. Their labels are hypotheses to inspect, not independently verified truth.
Compare the raw mistakes and slices, not only an aggregate score.

Use fresh run directories for these conditions:

* English off-the-shelf checkpoint versus the separately pretrained typed checkpoint
* Request only versus request plus supplied context
* Rules and constant fallback versus model decisions

Read `truncation_rate`. The Clef runtime reports how many state tokens were dropped to
fit `max_length`. A run with truncation has a context-budget limitation, not clean
evidence that a model failed to understand the complete task.

## Build reviewed data

Start with 500 to 1,000 examples as a planning target, not a statistical guarantee.
Use approved real requests where possible and separately tag challenge examples.
SWE-bench supplies repair tasks, not gold routing labels; CodeSearchNet supplies
retrieval pairs, not complete Copilot conversations. Include your target languages.

Each JSONL record follows this shape:

```json
{
  "id": "unique-request-id",
  "group_id": "underlying-issue-or-task-id",
  "repository": "stable-repository-id",
  "query": "Explain what this function does without changing it.",
  "context": "def square(x): return x * x",
  "expected_route": "repository_qa",
  "acceptable_routes": ["repository_qa"],
  "tags": ["negation", "python"],
  "provenance": "approved-source:version-and-source-id",
  "review_status": "unreviewed",
  "reviewer": ""
}
```

Have reviewers correct the labels and identify ambiguous cases. Adjudicate
disagreements, then set `review_status` to `reviewed` with a reviewer identity.
Double-review a subset and retain the adjudication trail. The validator can check
fields, but cannot establish that review actually occurred.

Split before generating paraphrases. Keep parent task IDs. Never include solution
patches or future test results in a router input that would not have them at decision
time. A review task legitimately includes its proposed diff; a bug-fixing task must
not receive the hidden reference solution.

```text
uv run --no-sync python -m routing_lab.cli validate --data data/reviewed/routing.jsonl --reviewed-only
uv run --no-sync python -m routing_lab.cli split --data data/reviewed/routing.jsonl --output data/splits
```

The splitter allocates approximately 60/20/20 of connected repository/task groups,
not necessarily 60/20/20 of records. It joins shared tasks and exact normalized
duplicates even across repository names. Inspect the printed class counts; do not
break repository isolation to force class balance. Near-duplicate semantic detection
and source-license review remain human responsibilities.

Use train for the conventional classifier, dev for prompt and threshold selection,
and a frozen test set for final comparison. If fitting a calibration model, reserve
a calibration subset separate from the data used to score that calibration.
Keep selected public evaluation tasks and their derivatives out of training.
Pretraining contamination cannot be ruled out by these local split checks.

## Compare routing baselines

```text
uv run --no-sync python -m routing_lab.cli run --backend rules --data data/splits/dev.jsonl --output outputs/dev-rules-001
uv run --no-sync python -m routing_lab.cli run --backend constant --data data/splits/dev.jsonl --output outputs/dev-general-001
uv run --no-sync python -m routing_lab.cli run --backend majority --train data/splits/train.jsonl --data data/splits/dev.jsonl --output outputs/dev-majority-001
uv run --no-sync python -m routing_lab.cli run --backend tfidf --train data/splits/train.jsonl --data data/splits/dev.jsonl --output outputs/dev-tfidf-001
```

TF-IDF requires every route in training. Rules intentionally inspect query text only;
run Clef with `--context-mode query` for that controlled comparison. For a richer
context comparison, TF-IDF, Clef and the Azure LLM can all use query plus context.
The constant baseline always chooses fallback; its routing accuracy is not a
measurement of a general coding agent's ability to solve issues.

Probability metrics use the single expected label, while acceptable-route accuracy
allows explicit alternative owners. Brier score is multiclass sum-of-squared error;
ECE uses ten bins. Neither makes the probabilities calibrated. The risk/coverage
curve excludes predictions already routed to fallback. Empty covered sets have
`null` risk, not a perfect score. Thresholds are diagnostics, not a deployed policy.

Latency is serial wall time around prediction, including tokenization and the first
request but excluding model load. Load time is recorded separately by the CLI.
Serial requests/second is not a concurrent serving throughput benchmark. GPU calls
must complete their returned probabilities before the prediction returns.
Do not compare these local serial timings to a vendor's optimized serving claim.

Repository-bootstrap intervals are descriptive and unreliable with few independent
repositories. Do not infer safety or a tiny performance advantage from the starter
probe. The harness does not currently measure concurrent load, peak memory, GPU
utilization or deploy an automatic calibrated escalation policy.

## Test downstream outcomes

This project imports results; it does not implement or execute the five coding
agents. First choose their actual models, tool access, prompts, execution budgets
and an isolated grading harness. Use the same tasks and grading criteria for
`always_general` and `routed`. For repair tasks, use executable tests; explanation
and review tasks need their own quality rubric rather than a repair-only benchmark.

Store independently graded runs as JSONL:

```json
{"task_id":"issue-001","policy":"always_general","success":true,"total_cost_usd":0.25,"total_latency_seconds":32.0,"attempts":1,"evidence":"artifact://run-001/test-results","harness_version":"grader-v1"}
{"task_id":"issue-001","policy":"routed","success":true,"total_cost_usd":0.20,"total_latency_seconds":29.0,"attempts":1,"evidence":"artifact://run-002/test-results","harness_version":"grader-v1"}
```

These numbers illustrate the schema only. They are not experiment results.
Use total costs that include context building, routing, failed attempts, fallback,
tool compute, and allocated hosting/idle costs. Retain test logs or review evidence.
The importer validates evidence references but does not verify their contents.

```text
uv run --no-sync python -m routing_lab.cli outcomes --data outputs/downstream.jsonl --baseline always_general --candidate routed --output outputs/downstream-comparison.json
```

The comparison rejects mismatched task sets, duplicate task/policy pairs, unexpected
policies and different grader versions. Zero successes produce a `null` cost per
success. Repeat stochastic runs under a predeclared protocol; the initial importer
accepts one aggregate outcome per task/policy, not arbitrary duplicate attempts.

## Fine-tuning decision

Before opening the frozen test set, record acceptable task-success loss, latency,
total cost, non-fallback coverage and any mandatory-escalation requirements.
No arbitrary numerical threshold is treated as an approved business requirement.

Proceed only if repeated domain-specific routing mistakes remain after resolving
label ambiguity, missing context and contract overlap. Compare against the unchanged
off-the-shelf checkpoint. Do not train to repair broken downstream agents.

The next stages require a human-reviewed set, live Azure model measurements, actual
agent outcomes and an agreed decision threshold. This project intentionally does
not automatically approve or launch fine-tuning.
