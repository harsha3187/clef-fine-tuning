---
title: Routing criteria prompt-tuning experiment
description: Compare richer routing descriptions against the unchanged off-the-shelf baseline before considering weight fine-tuning.
---

## Question and controls

Does adding explicit scope, inclusions and exclusions improve initial-owner routing?
This is textual prompt engineering, not learned soft-prompt tuning or weight training.
Short criteria are a hypothesis about the errors, not an established cause.

The first comparison changes only the six criterion descriptions:

* `baseline`: descriptions from [agents.json](../contracts/agents.json)
* `expanded_v1`: descriptions from [criteria-expanded-v1.json](../contracts/criteria-expanded-v1.json)

Both arms use the same 60 records in the same order, state/context construction,
shared instructions, option order, taxonomy and pinned English model. Neither the
user input nor criteria include expected labels or example answers. Agent tools and
permissions are unchanged. The baseline prompt is not automatically replaced.

The expanded version describes when a role owns a task and where neighboring roles
take over. For example, debugging retains regression tests associated with a fix;
code generation excludes restoring broken behavior; repository Q&A distinguishes
neutral explanation from diagnosing an observed failure.

## Run and inspect

Use the section "Prompt tuning before fine-tuning: richer criteria" in the
[experiment notebook](../notebooks/01_routing_experiment.ipynb). The preview displays
both descriptions side by side. The results cell reads an existing local report
without calling Azure.

To rerun after code changes, restart the notebook kernel, run setup and backend
selection, and enable `RUN_PROMPT_COMPARISON` along with endpoint data approval.
One comparison makes 120 requests and creates a new folder. It never overwrites
prior results.

The reusable [comparison runner](../routing_lab/prompt_experiment.py) writes:

* A top-level `comparison.json` with the hypothesis, fixed planned prompts, file
  hashes, arm summaries, paired corrections/regressions, and completion status
* A manifest, predictions and metrics for each arm through the existing
  [evaluation runner](../routing_lab/evaluation.py)

Any error propagates and leaves the comparison incomplete. Partial predictions and
completed arms remain available for diagnosis. All six routes contribute to the
score; the motivating example is not the only measure of success.

## Metrics and interpretation

Compare proposed-label agreement, acceptable-route agreement, macro-F1, per-route
recall, fallback recall, and corrected versus regressed examples. Inspect input
tokens, client latency, dropped-state rate, question-truncation rate and collapsed
options. Unknown question-truncation telemetry is reported as `null`, not zero.
Longer descriptions can displace useful context or be truncated themselves.

Latency includes authentication, network, queueing and inference. The serial
baseline-then-expanded order is not a randomized performance benchmark. Each
variant receives one pass, so timing differences do not establish a speed effect.

These previously inspected, unreviewed synthetic examples are development probes.
Even an improved score is not an unbiased estimate of generalization. Because
wording and length change together, this experiment does not isolate length alone.
The probabilities are also not validated confidence in downstream task success.

## Recorded result on 2026-09-30

Both arms completed all 60 requests against the live Azure endpoint. Their manifests
have matching dataset digests and the expected model revision. The original
baseline was rerun, rather than compared with a different historical dataset.

| Measurement                        | Baseline        | Expanded v1 |
|------------------------------------|-----------------|-------------|
| Proposed-label agreement            | 30/60 (50%)     | 30/60 (50%) |
| Macro-F1                           | 0.4601          | 0.4723      |
| Debugging recall                   | 40%             | 50%         |
| Repository Q&A recall              | 0%              | 10%         |
| Test generation recall             | 70%             | 60%         |
| Code review recall                 | 70%             | 60%         |
| Fallback recall                    | 20%             | 20%         |
| State truncation reported          | 0%              | 0%          |
| Question truncation reported       | 0%              | 0%          |
| Collapsed options reported         | 0%              | 0%          |
| Client p50 latency                 | 4.314 seconds   | 4.143 seconds |

Nine predictions changed. Three became correct against the proposed labels, three
became incorrect, and three changed between incorrect routes. The motivating
`dbg-01` example changed from `code_generation` to `debugging`, matching its proposed
label. That individual correction did not translate into higher overall agreement.

Both arms reported 17,109 input tokens in total. These are the runtime's reported
usage values, not an independent count of all criterion text or evidence that
longer descriptions have no computational cost.

This is a mixed development result: macro-F1 increased by 0.0121, but overall
agreement did not improve. It does not establish that the initial errors were
caused by short descriptions, nor that weight fine-tuning is now necessary.
Keep both prompts and results; the default remains the original baseline.

Inspect the [complete comparison](../outputs/prompt-criteria-1702b7ca67/comparison.json)
for all paired outcomes, and the
[executed comparison notebook](../outputs/notebook-prompt-tuning-validated.ipynb)
for the criteria preview and result display. These local output artifacts are
ignored by source control; preserve them deliberately when sharing results.

## Next decision

Before fine-tuning:

1. Review ambiguous labels and routing boundaries without changing labels merely
   to agree with model outputs.
2. Use separate development examples for further prompt edits, versioning each
   candidate rather than modifying an already measured criteria file.
3. Freeze a candidate prompt before an independently reviewed held-out evaluation,
   keeping repository and task families disjoint.
4. Compare conventional and stronger-model baselines on that same held-out set.
5. Measure actual downstream task outcomes before deciding that model weight
   adaptation is warranted.
