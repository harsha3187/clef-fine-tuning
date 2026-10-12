---
title: Expanded agent routing and batched annotation
description: Prepare separate coding and tool/API routing tracks, label them with a strict-JSON batched GPT-5.6 Sol judge, and recover from failures without weakening validation.
---

## Scope

The primary experiment compares an LLM router with System 1 Clef on selecting
one initial coding-agent owner. A separate track handles general tool/API
requests. Do not combine their scores: their available capabilities differ.

The [expanded coding contracts](../contracts/copilot-coding-v2.json) define
14 roles: debugging, code generation, refactoring, test generation, code review,
security review, performance optimization, repository QA, research, planning,
documentation, build/release, clarification and fallback.

The [tool/API contracts](../contracts/tool-api-v1.json) define seven roles:
tool lookup, state-changing tool action, multi-operation workflow, web research,
memory retrieval, clarification and no matching tool.

These are proposed experiment contracts, not installed GitHub Copilot agents or
a completed downstream dispatch integration. Their tool lists describe required
capabilities, not permission grants. The original six-route contracts, source
data and previous results remain unchanged.

## What changed in the labeling approach

The old dataset used lexical proposals. Those can miss real bugs without explicit
failure keywords and misclassify a frontend implementation request as fallback.
The new preparation step copies none of those labels.

SWE-bench rows enter the coding track. BFCL rows enter the tool/API track under
their explicitly offered tool capabilities. For example, a frontend request can
have a coding implementation owner yet have no matching tool when its BFCL
environment offers unrelated business APIs. This is a scope distinction, not
permission to pick whichever label improves a score.

The [annotation policy](../contracts/annotation-policy-v1.json) records known
development-exposed examples and propagates that exposure across query groups.
Native dev records are development-only. Other records are holdout candidates;
they are not yet a frozen, reviewed test set.

Routing concerns the first owner, not proven downstream solvability. A clear
SWE issue can be assigned before checkout. Missing implementation context is not
automatically missing intent. Conversely, omitted BFCL native environments are
preparation blockers: they must not be mislabeled as user clarification requests.

## Full-dataset preparation

Preparation writes an immutable input-only snapshot and copies the exact agent
contracts and policy. It verifies source hashes, removes old routing proposals,
and starts every row unlabeled. It does not read native targets.

```powershell
uv run --no-sync python -m routing_lab.annotation prepare --bundle data\public-routing\public-benchmarks-20260930 --output data\public-routing\public-annotation-20260930-v2
```

Use a new output directory for a new snapshot. Never edit a frozen snapshot
in place to change criteria. The prepared snapshot inventories all 7,225 rows:

| Disposition | Rows | Treatment |
|-------------|------|-----------|
| Ready for annotation | 6,160 | 2,519 SWE-bench and 3,641 BFCL rows |
| Native context not reconstructed | 1,055 | Remain unlabeled until the environment is faithfully reconstructed |
| Official ToolBench examples | 10 | Retained for inspection, not treated as evaluation rows |

Open [the annotation notebook](../notebooks/05_annotation.ipynb). It shows the
routing criteria, the judge's exact input/output boundary, the current batch
plan and actual saved responses. Earlier code cells have also been executed
with remote calls disabled in the
[saved notebook](../outputs/annotation-notebook-gpt56-sol-20261001.ipynb).

## Current execution: strict-JSON batches

The labeling workflow sends up to 100 rows per request instead of repeating
the full criteria for every row. Rows share a routing taxonomy, native split and
development-exposure status. In practice the 128,000-character request bound
binds before the 100-row count does for most batches, since SWE-bench issue
text varies widely in length; batches therefore range from a couple of rows
to the full 100, not a fixed size. Oversized groups are split before sending;
no input text is truncated.

The initial one-row-per-request serial run stopped after 15 requests because
one response contained six evidence quotes while the local validator allowed
five. Fourteen valid assessments, covering seven complete two-pass rows, are
retained. Their original audit files are not rewritten or represented as
batched responses.

A first batching pass used a fixed 10-row cap and reached 97 consensus/
adjudication rows before the batch size was raised to 100 for efficiency (see
[notebook 05](../notebooks/05_annotation.ipynb) and the recomputed plan below).
Restarting at the larger size means relabeling those 97 rows is cheaper than
building a batch-to-batch import path for such a small amount of rework.

For the current frozen dataset, importing only the original seven serial rows:

| Item | Count |
|------|-------|
| Annotation-ready rows | 6,160 |
| Previously completed rows preserved (serial import) | 7 |
| Remaining rows | 6,153 |
| New batches at batch_size=100 | 126 |
| New requests for two passes | 252 |
| Total requests including the original 15 serial attempts | 267 |

For comparison, the fixed 10-row cap would have needed 618 batches and 1,251
total requests. This is roughly a further 79% reduction in physical requests
versus the 10-row plan, and about 98% fewer requests than the original
12,320-request one-row plan. Elapsed time is not guaranteed to fall by the same
factor: responses contain many more annotations per call and depend on model
latency and endpoint quotas; larger batches take noticeably longer to generate.

The largest planned request is 127,987 characters including instructions,
task data and schema, just below the existing 128,000-character bound.
The [batch judge configuration](../azure/annotation-judge-batched.json) uses the
same GPT-5.6 Sol deployment and Entra identity, with a 600-second timeout (the
client's maximum) and 32,000 output tokens (close to the client's 32,768
maximum) to leave headroom for up to 100 annotations in one response. No
static API key is used.

### Exact JSON contract

Each input receives an opaque key such as `item_000`. Native benchmark IDs,
categories, prior labels and solutions are not sent. The response schema lists
every expected key as required and forbids additional properties.

This synthetic example illustrates the wire format, not an actual model result:

```json
{
  "annotations": {
    "item_000": {
      "verdict": "label",
      "route": "debugging",
      "alternative_route": null,
      "rationale": "The request describes a functional wrong-result failure in existing code.",
      "evidence": {
        "field": "query",
        "quote": "The function returns the wrong type."
      },
      "missing_information": null
    }
  }
}
```

Each row has exactly one evidence object, not an unbounded evidence array.
A nullable alternative route normalizes to the existing acceptable-route list.
Missing information is either null or one specific explanation. This fixed-slot
shape avoids relying on array-count constraints that
[Azure Structured Outputs does not support](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs#unsupported-type-specific-keywords).

The client still checks character limits, allowed routes and abstention
semantics; missing or unexpected keys invalidate the batch, and results are
never paired with rows by response order, only by the opaque key. Evidence
quotes are **not** checked against the row's own query/context (see
[Evidence is no longer verified against source text](#evidence-is-no-longer-verified-against-source-text)
below) — row attribution is guaranteed structurally by that opaque key
mapping, independent of what the quote text says.

The first live ten-row batch passed validation on 2026-10-01. Both passes
agreed on all ten rows: five debugging and five code-generation labels.
The calls used 14,618 input tokens and 2,082 output tokens in total, with
client latencies of 14.98 and 12.76 seconds. Together with the seven recovered
rows, this checkpoint contains 17 consensus rows. This is a successful
batch-format check, not measured routing accuracy or full-dataset completion.

### Evidence is no longer verified against source text

Early runs rejected a whole batch pass whenever one row's evidence quote
wasn't an exact substring of that row's own query/context. At batch_size=10
this was rare; at batch_size=100 the odds that at least one row in a batch
hits it rose sharply, and the failures were systematic rather than random:

* GPT-5.6 Sol reliably stripped Markdown link syntax when quoting text that
  contained an inline link, e.g. quoting `requires the Pixel Representation
  attribute` from source text that actually reads
  `[requires the Pixel Representation attribute](url)`.
* It occasionally produced a subtly different paraphrase of a clause, e.g.
  quoting `fails once, but not the second time` from source text that reads
  `fails once, but works the second time`.

Both are the model's own formatting/wording choices while still correctly
identifying and labeling the right row, not a sign that the wrong row was
labeled. Row attribution was never based on the quote text: it comes entirely
from the opaque per-row key (`item_000`, ...), which the client already
validates exhaustively (every key required, no extras, no cross-row mixing
possible). After confirming this with the user, the verbatim substring check
was removed from `validate_annotation` on 2026-10-01. Evidence is still
required to be a non-empty string and is still recorded for audit context, but
it is no longer re-verified against the row's own input text. This applies to
both the batch and legacy serial validation paths, since they share the same
function.

### Two blind passes and accounting

Each batch receives two stateless calls. The second call reverses input order
and criterion order, but keeps the opaque keys stable. It never sees the first
call's labels. The model is instructed to assess each row independently.

Both assessments must agree before a row becomes an AI-generated consensus
label. Disagreements remain visible for adjudication. These are not
human-reviewed gold labels.

Network attempts and token usage are counted once per physical API request,
not once per returned row. Imported serial requests remain separately traceable
and count against the cumulative cap. Successful calls are retained on resume.
The base coordinator stops on validation, authentication and service errors
with an auditable checkpoint rather than accepting partial or guessed output.
The approved retry policy below can resume eligible annotation rejections.

## GPT-5.6 Sol judge configuration

The [judge configuration](../azure/annotation-judge.json) uses the supplied
Foundry endpoint and `gpt-5.6-sol` model/deployment. The client normalizes the supplied
Responses URL to the Azure OpenAI v1 API base. It uses the Responses API,
Structured Outputs and Azure CLI identity, not API keys embedded in files.

The environment already has the required OpenAI and Azure identity clients.
No additional compute instance, model weights or Clef deployment is needed for
annotation. The signed-in identity needs inference permission on the supplied
Foundry/OpenAI resource and network access to its endpoint. An Azure ML role on
the experiment workspace does not automatically grant access to this separate service.

On 2026-10-01, the configured `gpt-5.6-sol` deployment passed a synthetic
Structured Outputs request using Microsoft Entra authentication at the supplied
resource endpoint. It returned `{"ok": true}` with 47 input tokens and 12 output
tokens. No static API key, project-specific transport, or dataset row was used.
The [current connection check](../outputs/judge-connection-check-gpt56-sol-20261001/result.json)
records the actual model and nonsecret authentication metadata.

The previous `gpt-4.1` deployment passed a synthetic connectivity and Structured Outputs
check on 2026-09-30. It returned `{"ok": true}`, with 49 input tokens and six output
tokens. No dataset row was sent. The persisted
[connection check](../outputs/judge-connection-check-20260930/result.json)
is setup evidence, not a router benchmark.

Authentication already uses Microsoft Entra ID through `AzureCliCredential`
and the `https://ai.azure.com/.default` token scope. No static API key is loaded
from configuration or environment variables. The OpenAI SDK calls its credential
argument `api_key`, but here that argument receives a bearer-token provider.
Disabling key-based authentication does not require changing this mechanism.

The supplied `/openai/v1/responses` URL is a resource-level model endpoint.
A Foundry project endpoint instead has the form
`https://<resource>.services.ai.azure.com/api/projects/<project>`.
Project-scoped SDK access also uses an Entra identity with the appropriate project
role; the project name is not a credential. Do not invent a project path or
switch credential identities silently. If project-only access is required, obtain
the actual project endpoint and verify its RBAC scope before switching transports.

For the endpoint distinction and supported authentication, see the
[Foundry SDK and endpoint guide](https://learn.microsoft.com/en-us/azure/foundry/how-to/develop/sdk-overview)
and [Responses API guide](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses).
Reasoning output items are not used as annotation labels or persisted as rationale.
The visible structured answer and total reported usage are still validated.

The service reported the model alias, not a dated model version. Record and
control the deployment's model version and upgrade policy in Azure before a
long annotation run; alias matching alone cannot detect every deployment update.

No new Azure resources are provisioned by this workflow. If permission is denied,
grant an appropriate inference role at the model-serving resource and confirm
any firewall or private-endpoint connectivity requirements.

The transport uses bounded timeouts, disables automatic retries and redirects,
requests `store=False`, and rejects refusal, incomplete output, wrong model
identity and missing usage accounting.

## Annotation and review gates

Each ready row receives two stateless assessments of the same input. The second
pass gets reversed criterion ordering and no earlier vote. Both use the actual
routing question that the evaluated routers will see.

Each assessment must provide:

* One primary route and an acceptable-route set, or an explicit abstention.
* A short decision rationale.
* A non-empty evidence quote from the query or supplied context, kept as
  unverified audit context (see
  [Evidence is no longer verified against source text](#evidence-is-no-longer-verified-against-source-text)).
* Specific missing information when the input is insufficient.

Native patches, expected calls, old labels, future conversation turns, dataset
categories and row IDs are not sent as judge task data.

Matching primary and acceptable-route sets yield `silver_consensus`.
Disagreements and ambiguity yield `needs_adjudication`; no majority vote,
numeric self-confidence or lexical fallback silently resolves them.

Two passes of the configured judge are a consistency check, not independent human validation.
Exported records remain `unreviewed` with no human reviewer. Do not use GPT-5.6 Sol
as both the sole label source and the supposedly independent LLM router without
explicitly reporting that bias. Prefer a different evaluated router model and
independent adjudication of an audit sample.

## Approved bounded retries (first pass, batch_size=10)

After request 32, the model stripped Markdown link markup from an evidence
quote. The exact-span validator correctly rejected the batch; 87 completed rows
were retained. The user then approved one retry per failed batch pass and at
most 50 additional requests overall, raising the cumulative cap to 1,301.

The `retry` subcommand in [routing_lab/batch_annotation.py](../routing_lab/batch_annotation.py)
keeps the same prompts and strict validation. It never supplies the failed
response as feedback and never repeats a successful pass. Only a saved response
that fails local annotation validation can be retried. Authentication, service,
model-version, integrity and configuration errors still stop immediately. A
second rejection of the same pass stops the job rather than discarding that batch.

Retry counts are reconstructed from the hash-chained physical-request journal,
so restarting does not reset either limit. The retry policy is recorded
separately without changing the original run identity.

The bounded live retry succeeded: requests 33 and 34 completed the rejected
ten-row batch, bringing the checkpoint to 97 assessed rows, 93 consensus labels
and four disagreements. The original invalid response remains in the audit.
This run used a fixed 10-row batch size; it is preserved at
[its original location](../outputs/public-annotation-batched-gpt56-sol-20261001/)
but is superseded by the 100-row run below, which relabels these 90 non-serial
rows at negligible extra cost rather than building a batch-to-batch import path.

```powershell
uv run --no-sync python -m routing_lab.batch_annotation retry --plan data\public-routing\public-annotation-20260930-v2 --judge-config azure\annotation-judge-batched.json --output outputs\public-annotation-batched-gpt56-sol-20261001 --max-requests 1301 --max-retry-requests 50 --allow-remote
```

## Run, resume and export (current pass, batch_size=100)

The old serial run is preserved at
[its original location](../outputs/public-annotation-gpt56-sol-20261001/).
The new runner validates and snapshots its completed votes before importing them.
Only the original seven serial rows are imported; the 90 rows labeled by the
superseded 10-row batched run above are relabeled fresh.

```powershell
uv run --no-sync python -m routing_lab.batch_annotation run --plan data\public-routing\public-annotation-20260930-v2 --judge-config azure\annotation-judge-batched.json --output outputs\public-annotation-batched100-gpt56-sol-20261001 --import-serial-run outputs\public-annotation-gpt56-sol-20261001 --batch-size 100 --max-requests 267 --allow-remote
```

For the current frozen dataset this produces 126 batches (ranging from a
couple of rows to the full 100, since the 128,000-character bound usually
binds first) and needs 252 new requests, for 267 total including the 15
original serial attempts — about 79% fewer requests than the 10-row plan's
1,251, and about 98% fewer than the original 12,320-request one-row plan.

For a paused run, use the same plan/config/output with `--resume`, without
reimporting the serial run. The request limit is cumulative, including the
original attempts. Do not start multiple writers on the same run directory.
Every attempt is durably recorded before the request; on failure the run
remains failed with its prior results retained.

Consensus export remains separate from actual evaluation or training. By default
it requires consensus for every annotation-ready row; an explicit
`--allow-partial` permits a disclosed subset, with excluded counts remaining visible.

```powershell
uv run --no-sync python -m routing_lab.batch_annotation export --plan data\public-routing\public-annotation-20260930-v2 --run outputs\public-annotation-batched100-gpt56-sol-20261001 --output data\public-routing\gpt56-sol-labels-batched-20261001 --allow-partial
```

Only validated agreement is exported, with coverage and disagreement counts,
frozen contracts and provenance. The coding and tool/API tracks stay separate.
The 1,055 unreconstructed-context rows and ten ToolBench examples remain outside
the scored dataset. No human review is fabricated; a partial consensus subset
can introduce selection bias, so report coverage and disagreements alongside
any agreement score.

Do not use the same examples or related query/repository groups for both
fine-tuning and held-out evaluation. Native split and development-exposure
metadata are retained so those boundaries can be frozen before either stage.

Only one process may write a run. After an abrupt process termination, a lock
file can remain. Verify that the recorded process is no longer running before
manually removing that specific stale lock; never delete the run directory to
force recovery.

## Completed labeling and the third-pass majority vote

The full batch_size=100 pass ran to completion on 2026-10-02 (with three
transient Azure 500s/timeouts recovered via plain `--resume`, no data lost).
It produced consensus on 5,648 of 6,160 ready rows and left 512 unresolved
disagreements. See [routing_lab/majority_adjudication.py](../routing_lab/majority_adjudication.py)
for the independent third blind pass built to adjudicate those 512 rows, and
[routing_lab/batch_annotation.py](../routing_lab/batch_annotation.py)'s
`finalize` subcommand for merging the results into one clean dataset.

The third pass needs only one new vote per disagreement row (not two), so it
is proportionally cheap: 14 batches, 14 total requests for all 512 rows.
A row resolves only with 2-of-3 agreement across the three independent passes;
a true three-way split or a double abstention is **not** silently resolved.

```powershell
uv run --no-sync python -m routing_lab.majority_adjudication run --plan data\public-routing\public-annotation-20260930-v2 --judge-config azure\annotation-judge-batched.json --source-run outputs\public-annotation-batched100v3-gpt56-sol-20261001 --output outputs\public-annotation-majority-gpt56-sol-20261002 --max-requests 20 --allow-remote
uv run --no-sync python -m routing_lab.majority_adjudication export --plan data\public-routing\public-annotation-20260930-v2 --source-run outputs\public-annotation-batched100v3-gpt56-sol-20261001 --majority-run outputs\public-annotation-majority-gpt56-sol-20261002 --output data\public-routing\gpt56-sol-labels-majority-20261002
```

Of the 512 disagreements, 330 (64%) resolved by majority vote; 182 remain
genuinely unresolved (44 true three-way splits, 138 with two or more
abstentions) and are never silently promoted to a label.

The majority export is additive, not a replacement: it sits in its own
directory alongside the original two-pass consensus export, tagged
`majority_vote_third_pass` in `tags`/`provenance` so it can always be filtered
back out. `finalize_dataset` merges the two into one clean, deduplicated
dataset and explicitly excludes the still-unresolved rows rather than leaving
them mixed in or scattered across files you have to remember to skip:

```powershell
uv run --no-sync python -m routing_lab.majority_adjudication finalize --consensus-export data\public-routing\gpt56-sol-labels-batched100-20261001 --majority-export data\public-routing\gpt56-sol-labels-majority-20261002 --output data\public-routing\gpt56-sol-labels-final-20261002
```

It verifies both source exports are complete and derive from the exact same
source run, rejects any row-ID overlap, and copies the still-unresolved rows
verbatim into `excluded_unresolved.jsonl` inside the final directory — visible
and auditable, but never part of the dataset files used for evaluation or
training. The final counts for the current frozen plan:

| | Rows |
|---|---|
| Final labeled dataset (`copilot-coding-v2.jsonl` + `tool-api-v1.jsonl`) | 5,978 |
| Excluded, still unresolved (`excluded_unresolved.jsonl`) | 182 |
| Total annotation-ready rows | 6,160 |

These are AI-derived labels (up to three independent passes agreeing), not
human-reviewed gold. The 1,055 context-blocked rows and ten ToolBench examples
remain outside this count entirely, as before.

## Frozen 90/10 benchmark split

[routing_lab/dataset_split.py](../routing_lab/dataset_split.py) splits the
finalized dataset into a 90% pool (future fine-tuning/evaluation) and a 10%
held-out benchmark slice, per track. Connected repository/group/near-duplicate
blocks never cross the boundary, development-exposed rows are always pinned to
the 90% pool, and no route is ever fully drained out of the 90% pool even if
that under-represents it in the 10% slice. Stratification by route is
therefore best-effort, bounded by those group-safety constraints, not exact.

```powershell
uv run --no-sync python -m routing_lab.dataset_split --final-export data\public-routing\gpt56-sol-labels-final-20261002 --output data\public-routing\gpt56-sol-labels-split-20261002 --test-fraction 0.10 --seed 42
```

Actual result on 2026-10-02 (seed 42), with zero ID/group/repository/near-duplicate overlap verified between each track's two files:

| Track | 90% pool | 10% benchmark | Benchmark share |
|---|---|---|---|
| copilot-coding-v2 | 2,159 | 337 | 13.5% |
| tool-api-v1 | 3,134 | 348 | 10.0% |

The coding track's benchmark share overshoots 10% (13.5% instead) because
SWE-bench repositories are large, coarse blocks: only 11 distinct repositories
remain eligible after excluding development-exposed rows, and the packer
cannot split a repository to hit the target more precisely without risking
leakage. BFCL's `group_id` is far more granular (over 3,200 distinct values),
so the tool/API track lands almost exactly on target. This is reported
honestly in each run's `manifest.json`, not silently rounded or hidden.

Per the existing annotation policy, the 10% benchmark slice is meant to stay
untouched until both the off-the-shelf and the eventually fine-tuned model are
each measured against it once, so the two results are comparable.

## Before actual evaluation

1. Review disagreements, missing contexts, agent coverage and ambiguous boundaries.
2. Freeze group-safe splits separately by track, excluding development-exposed
   groups from untouched holdouts. Public-data pretraining contamination remains unknown.
3. Select an LLM router independently of the annotation judge.
4. Give LLM and Clef identical decision-time inputs and frozen criteria.
5. Deploy and verify expanded scorer support before sending new route sets to
   Azure. Local code changes do not update the existing live six-route deployment.
6. Measure label agreement, macro-F1, per-route results, abstentions, latency,
   usage and truncation. Report absent classes rather than fabricating balanced data.
7. Evaluate downstream agent outcomes separately before deciding on fine-tuning.

Records exceeding the Clef endpoint's character limit remain visible in export
metadata. A character-compatible request can still exceed Clef's `max_length` token
budget. Use the same explicitly selected records for both routers and report state
truncation.

Clef scores every option of the question jointly, so the 14-role coding track needs no
special bucket. Its probabilities are still uncalibrated for this roster: validate
calibration before choosing escalation thresholds. The old Laya six-route results do
not carry over. Follow the
[packaged deployment workflow](endpoint.md#recreate-or-update-deliberately)
when an expanded-catalog rollout is approved.

## Earlier one-row workflow (historical)

Before batching, the notebook defaulted to no remote execution and zero
approved records/requests. For a bounded initial connectivity-and-format
exercise using two dataset rows, the one-row CLI form was:

```powershell
uv run --no-sync python -m routing_lab.annotation run --plan data\public-routing\public-annotation-20260930-v2 --judge-config azure\annotation-judge.json --output outputs\annotation-check-001 --max-records 2 --max-requests 4 --allow-remote
```

This command does send dataset rows and incur inference charges. It is an
example, not a command run automatically during setup. Its export form is:

```powershell
uv run --no-sync python -m routing_lab.annotation export --plan data\public-routing\public-annotation-20260930-v2 --run outputs\annotation-check-001 --output data\public-routing\annotation-check-export-001 --allow-partial
```

Two successful passes over the ready rows required 12,320 row assessments,
meaning 12,320 requests in this one-row mode. Batching (above) shares the
criteria across rows and needs far fewer requests. These commands remain only
as documentation of the original workflow; the current labeling execution uses
the batched workflow above.
