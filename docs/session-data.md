---
title: Private VS Code session dataset
description: Review genuine user turns locally, or infer automatic silver labels, before evaluating an Azure-hosted router.
---

## Two workflows

This covers two ways to build a dataset from your own VS Code agent sessions:

* [Manual review](#manual-review-workflow): a human inspects and labels each
  candidate turn. Produces a smaller, higher-trust set.
* [Automatic silver labeling](#automatic-silver-labeling-workflow): a
  deterministic rule-based labeler infers labels from recorded responses and
  agent telemetry, with no per-example manual work. Produces a larger,
  unreviewed set faster.

Both start from the same local, authorized session inventory and never send
private data to Azure without explicit, separately gated approval.

## Manual review workflow

### Prepared snapshot

The user approved local preparation from all available VS Code agent sessions in
the last 30 days. The inventory window is 2026-08-31 10:38:07.051 UTC through
2026-09-30 10:38:08 UTC, with an exclusive end.

The available session inventory contains five sessions and 24 host-indexed user
turns. Nine turns are from this routing-development session and are blocked from
evaluation export. The remaining 15 candidates span four sessions and three
workspaces. They have no assigned labels and no remote-data approval.

This covers sessions exposed by the current VS Code agent-session inventory, not
deleted history, other profiles, every chat provider or unavailable workspaces.
Filtering uses session creation time and then the event timestamp within the
authorized window.

### Open the private reviewer

Open the [private review notebook](../data/private/session-review-20260930-103807-v2/02_real_session_dataset.ipynb)
with the project `.venv` kernel. Its inputs and saved widget state remain under
ignored private storage. The [shareable template](../notebooks/02_real_session_dataset.ipynb)
contains no conversation text or executed outputs.

1. Run setup and the reviewer cell.
2. Select a turn and inspect its query, earlier context and warning flags.
3. Recover any essential context from the original source session or its original
   attachment snapshot. Do not use later code or the eventual assistant answer.
4. Select the initial owning route and any acceptable alternatives. Exclude a turn
   if it cannot be labeled reliably; do not invent a label to fill the dataset.
5. Review privacy, identify the reviewer, and explicitly approve included text for
   the Azure Clef endpoint.
6. Save the review snapshot before selecting another turn.

The review is blinded to Clef predictions. Research, data collection, publishing and
approval turns may be outside the current six-role taxonomy. Review them for
fallback or exclusion rather than treating every real chat as a coding task.

Edits to query or context clear the widget's context/privacy approvals. Approval
is bound to a hash of the exact edited text, so changing a saved input invalidates
it. Each save creates a new snapshot; a private pointer restores the latest saved
review when the notebook restarts. Original extraction and source chats are not
modified.

### What was extracted

The [importer](../routing_lab/session_data.py) opens each approved host database
read-only, resolves its SDK session ID, and uses the host `turns.event_id` index
to identify genuine user turns. Filtering all SDK `user.message` events would also
include delegated-agent prompts; those are excluded here.

Each candidate contains the exact original user content, subject to heuristic
redaction, plus up to two prior user turns and their earlier final assistant
replies. Replies are matched to the original interaction ID. Both observed
`final_answer` and supported `final` phase names are accepted. Only events earlier
than the current decision can contribute to context.

System/developer messages, transformed prompt payloads, hidden reasoning, tool
calls/results, delegated requests and current/future solutions are not imported.
Attachment bodies are deliberately not loaded. File and document contents must
be reviewed separately if essential.

Earlier replies are capped at 5,000 characters and combined context at 10,000,
with explicit warning flags. These are review limits, not a model token budget.
Nineteen candidates have earlier final-reply context; 23 carry a current or prior
attachment warning. Seven have combined-context truncation warnings. Missing or
truncated context can invalidate an example, so the queue is not evaluation-ready.

The importer masks common credential patterns, private keys, URLs, email addresses
and home-directory identifiers. This is not comprehensive secret detection,
anonymization or permission to transmit company data. Manual review is mandatory.
Do not rewrite a request to make its intended route easier; limit edits to privacy
protection and faithful reconstruction of decision-time context.

### Export and evaluate

After review, enable `EXPORT_REVIEWED` in the private notebook. Export requires:

* Explicit inclusion and a named human reviewer
* A valid expected route and acceptable-route set
* Confirmed context review and hash-bound remote-data approval
* No routing-development turns

Export produces reviewed records and an approval manifest. Before evaluation, the
notebook verifies the dataset hash and review status, then requires both
`RUN_REAL_EVALUATION` and `ALLOW_PRIVATE_REMOTE`. Until those gates are enabled,
no private session data is sent to Azure.

Select the fixed original or expanded criteria. Use the same approved dataset for
both comparisons and retain all run outputs. Inspect state/question truncation:
Clef may not fit all of the context that a reviewer can see. Do not interpret an
error without checking what was dropped.

Session grouping and workspace grouping are retained using pseudonymous keys.
The existing split logic therefore keeps turns from one session together and
keeps the two chats from the same workspace in the same partition. Workspace
grouping is a local proxy for repository identity; manually reconcile cloned
copies or related workspaces before building a larger benchmark.

Fifteen candidates across three workspaces are a pilot, not a balanced six-route
benchmark or justification for fine-tuning. Review labels, check class coverage,
collect more independent tasks, and freeze repository/task-disjoint development
and test sets before comparing generalization.

### Privacy and provenance

Private inventory, candidates, review snapshots and the reviewer notebook live
under ignored `data/private/`. Exported reviewed data remains under ignored
`data/reviewed/`; evaluation outputs remain under ignored `outputs/`.
Ignored files are not encrypted or access-controlled and can still be uploaded
manually. Keep them on approved storage and do not share the entire project.

The extraction manifest records source-log hashes, importer hash, window, counts,
and any omitted incomplete final event. A malformed interior event or missing
indexed user event fails explicitly. Unicode separators within JSON strings are
preserved; JSONL records are split only on actual newline delimiters.

The current deliverable is a private, unlabeled review queue. No real-session
accuracy has been computed, no candidate has been promoted to gold automatically,
and no model weights have changed.

## Automatic silver-labeling workflow

### Automated path

Use the [private automatic-dataset notebook](../data/private/session-silver-20260930-111833/03_automatic_session_dataset.ipynb).
No per-example manual labeling is required. The optional
[manual review workflow](#manual-review-workflow) above remains available if a
gold evaluation set is needed later.

The snapshot contains 15 automatically labeled non-development requests:

| Inferred route   | Records |
|------------------|---------|
| Fallback         | 8       |
| Code generation  | 6       |
| Debugging        | 1       |

Eleven labels have corroborating weak signals and four are tentative. Nine
routing-development turns remain excluded. All 15 eligible candidates received
a label in this snapshot; the labeler can abstain on future ambiguous cases.
There are no examples of test generation, code review or repository Q&A here.

Many source requests concern research, data operations or publishing rather than
the supported coding-owner roles. Fallback refers to the current taxonomy; it
does not imply that the historical assistant failed.

### Recorded facts versus inferred labels

The [builder](../routing_lab/silver_data.py) uses the frozen input review queue
and reads original SDK event logs locally. Its outcome cutoff is
2026-09-30 11:18:33.106 UTC. This can include a response completed after the earlier
input-inventory cutoff without adding later user requests to the dataset.

For each original request it records:

* Sanitized final response text, event IDs and timestamps
* Model identifiers recorded on root assistant messages
* Delegated agent names and their recorded models
* Requested tools, tool starts and failed tool calls
* Labeling signals, ranking scores, uncertainty status and input-quality flags

Root messages are joined using the source interaction ID. Tool and delegation
events are joined using tool-call IDs requested by those root messages. Nearby
events from another turn or a delegated agent are not attributed merely because
they occurred at a similar time. Tool arguments, raw tool results and hidden
reasoning are not copied.

Observed root model identifiers are `gpt-5.6-sol` and `gpt-6-astra`. Observed
delegated agent names include `research`, `rubber-duck` and `task`. These are actual
execution metadata, not names of the six proposed routing roles. Generic or
unknown agents are not automatically mapped to a coding role.

### Labeling policy

`session-outcome-rules-v1` is a deterministic weak labeler, not a separate LLM judge:

1. Recognized request intent contributes five ranking points, respecting initial
   ownership such as fixes retaining their regression tests.
2. Matching claims in the recorded final response contribute two points per route.
3. An explicitly recognized specialist delegation contributes two points per route.
4. Missing final responses, no supported signals, ties or margins below two points
   cause abstention, not an invented fallback label.
5. A selected route with two different signal categories is called corroborated;
   otherwise it is tentative. The categories can still share the same underlying
   error, so this is not independent verification.

Ranking points are not correctness probabilities. A tool's success status does
not prove task success, and an assistant's report is not an independent grader.
Task success remains explicitly unverified.

Model identity never votes for a semantic route. A recorded model choice says
which model ran, not which model was cheapest, fastest or capable of solving the
task. Optimal-model labels require comparative outcomes across models.

### Leakage and privacy boundaries

The local [silver records](../data/private/session-silver-20260930-111833/silver-records.jsonl)
contain only the original sanitized query and earlier context as input, with
inferred targets marked `review_status="unreviewed"` and no human reviewer.
They are compatible with the existing evaluator in probe mode.

The separate [outcome evidence](../data/private/session-silver-20260930-111833/evidence.jsonl)
contains the post-decision response and execution telemetry. Do not concatenate
that evidence into the router's state or prompt. It is used to construct proxy
targets, not prediction-time features.

The [manifest](../data/private/session-silver-20260930-111833/manifest.json) records
label coverage, class counts, uncertainty, file/source hashes and the labeler
version. Original source events are checked against the candidate's user-text hash.
Existing artifacts are never overwritten.

The builder makes no Azure calls and does not grant privacy approval. Data remains
under ignored private storage. Automated redaction is incomplete by design; source
context may still be sensitive, and missing attachments remain unresolved.
Automation replaces manual labeling, not authorization to transmit private data.

### Evaluation

A local rules smoke test completed on all 15 records: 8/15 agreement with the
automatic labels, macro-F1 0.2042 across the six configured classes. This verifies
the dataset/evaluator pipeline, not model quality. Both labels and the rules
baseline use heuristics, so their errors can be correlated.

The notebook can run either fixed criterion version through the deployed Clef
endpoint after explicit batch privacy approval. Enable `RUN_CLEF_SILVER_PROBE`
and `ALLOW_PRIVATE_REMOTE` only with permission to send these sanitized inputs.
No per-row human route labels are required for this probe.

Report the result as agreement with automatic labels, including label coverage,
tentative-label count, class coverage and truncation. Do not call it verified
routing accuracy or use it alone to justify fine-tuning. If the historical
assistant chose the wrong approach, imitating it can reproduce that error.
