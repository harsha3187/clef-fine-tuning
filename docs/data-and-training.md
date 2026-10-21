---
title: Routing data, evaluation and fine-tuning
description: Build and label routing datasets, evaluate routers, run the prompt-tuning experiment, fine-tune Clef and review the recorded results.
ms.date: 2026-10-20
---

The sections follow the pipeline: acquire public and private data, convert and label it, evaluate routers, tune prompts, fine-tune Clef and compare the recorded results.

* [Public benchmark routing dataset](#public-benchmark-routing-dataset)
* [Private VS Code session dataset](#private-vs-code-session-dataset)
* [Routing datasets: four public sources in one canonical format](#routing-datasets-four-public-sources-in-one-canonical-format)
* [Expanded agent routing and batched annotation](#expanded-agent-routing-and-batched-annotation)
* [Review and evaluation protocol](#review-and-evaluation-protocol)
* [Routing criteria prompt-tuning experiment](#routing-criteria-prompt-tuning-experiment)
* [Shared router fine-tuning data and Clef plan](#shared-router-fine-tuning-data-and-clef-plan)
* [Off-the-shelf baseline results (before fine-tuning)](#off-the-shelf-baseline-results-before-fine-tuning)

## Public benchmark routing dataset

Download pinned SWE-bench and BFCL data, track the ToolBench blocker, and build input-only routing probes.

### What is available

| Source | Acquired task rows | Scope |
|--------|-------------------|-------|
| SWE-bench | 2,519 | Current full distribution: 225 dev and 2,294 test rows |
| BFCL | 4,696 | All 19 active V4 task categories, with available native answers and context resources |
| Original ToolBench | 10 | Official repository examples only, not the full dataset or evaluation split |

The acquisition retains licenses, source documentation, immutable revisions,
upstream sizes and hashes. The download manifest records exact URLs, downloaded
bytes, SHA-256 checksums and ToolBench availability checks. No native benchmark
code, environment setup script, evaluator or external tool API is executed.

Open [the public benchmark notebook](../notebooks/04_public_benchmarks.ipynb).
It verifies the snapshot, inspects source coverage, shows an exact Clef payload
and runs a small local smoke test. Azure inference is disabled by default.

The built snapshot contains 7,225 candidates, 5,489 automatically labeled draft
probes, and 5,484 probes within the endpoint's character limit. The draft labels
cover 1,559 debugging, 351 code-generation and 3,579 fallback requests. They do
not cover test generation, code review or repository QA.

All six notebook code cells have been executed and saved in the
[executed notebook](../data/public-routing/public-benchmarks-20260930/04_public_benchmarks.executed.ipynb).
The 20-record local smoke test completed without Azure calls.

### Dataset locations

These generated directories are ignored by version control:

* [Raw snapshot](../data/upstream/public-benchmarks-20260930/):
  downloaded source files and `download-manifest.json`.
* [Routing bundle](../data/public-routing/public-benchmarks-20260930/):
  normalized candidates, native targets, unreviewed probe records and `manifest.json`.

The bundle contains:

| Artifact | Purpose |
|----------|---------|
| `candidates.jsonl` | Every normalized initial request, source identity, native split, proposed label or abstention, and omission flags |
| `native-targets.jsonl` | Native patches, tests, expected calls or relevant-API labels, kept separate from router inputs |
| `draft-probes.jsonl` | Records with automatic proposed routes and supported input schemas |
| `endpoint-probes.jsonl` | Draft records whose full context-mode state fits the endpoint's 32,768-character bound |
| `excluded-from-probes.jsonl` | Candidate IDs excluded for ambiguous labels, unavailable native environments or sample-only status |
| `manifest.json` | Exact counts, artifact hashes, adapter hash, source revisions, split overlap and interpretation limits |

Exclusion-reason counts can overlap when a candidate has more than one reason.
The endpoint subset is not guaranteed to fit Clef's `max_length` budget. Inspect
`truncation_rate` alongside scores. Full inputs are retained rather than silently shortened.

### These are not six-route gold benchmarks

The source benchmarks measure issue resolution or tool use, not our six-agent
ownership decision. Their native gold answers cannot be treated as routing labels.

The versioned `public-request-intent-v1` policy proposes:

* `debugging` for SWE issues with explicit failure cues.
* `code_generation` for SWE issues with change or feature cues and no detected
  failure cue.
* `fallback` for general tool-use requests outside the coding contracts.
* An abstention when SWE intent is unclear or a tool-use request mentions coding.

These lexical proposals can be wrong. They remain `unreviewed`, with no reviewer,
and work only in probe mode. Agreement with them is not verified correctness.
The label policy overlaps lexical baselines, so a rules smoke test is not an
independent evaluation. The resulting dataset is imbalanced and does not provide
complete coverage of all six routes.

BFCL's native *irrelevance* category does not automatically imply our fallback
route. Available tool-call languages also do not imply a code-generation request.
The proposal policy examines the user request, not native expected calls or
category-based tool relevance.

Human review or an independently validated labeling process is still required
before freezing gold evaluation data or making a fine-tuning decision.

### Input and split boundaries

SWE inputs include the problem statement and repository/base-commit identity.
Patches, test patches, test names, evaluation scripts and hints are excluded.
Repository content has not been fetched: proposed labels describe issue intent,
not demonstrated downstream solvability.

BFCL inputs include the first user message, preceding system/developer messages
and explicitly supplied function schemas. Later messages, solution paths and
possible answers do not enter the router input. Multi-turn, memory and web-search
tasks without explicit function schemas are retained as candidates but excluded
from scored probes because their native environments are not reconstructed.

ToolBench examples include the query and all offered API descriptions. Gold
`relevant APIs` selections and response templates are excluded from inputs.
These examples are excluded from scored probes.

Native IDs, categories, revisions and splits remain attached to candidates.
Identical normalized queries share a group across categories. Native SWE dev/test
repository overlap and exact-input duplication are reported; no repository-disjoint
claim is made. The current SWE distribution has no train split. Legacy training
data, overlapping Lite/Verified subsets and inactive BFCL datasets are not silently
merged into this snapshot.

The notebook samples SWE native-dev records and BFCL evaluation records for
development. Any inspected or prompt-tuning records become development-exposed.
Do not later claim those records as untouched held-out tests.

### Reproduce the download and adaptation

Use the existing project environment and install the optional benchmark extra:

```powershell
uv sync --frozen --all-extras
uv run --no-sync python -m routing_lab.benchmark_sources --output data\upstream\public-benchmarks-20260930
uv run --no-sync python -m routing_lab.benchmark_data --raw data\upstream\public-benchmarks-20260930 --output data\public-routing\public-benchmarks-20260930
```

Each output directory must be new. To resume interrupted acquisition, repeat the
download command with `--resume`. Completed files are hash-verified rather than
silently replaced. Adaptation refuses incomplete or modified snapshots.

No dataset API key is needed. Do not install or provide RapidAPI, ToolBench,
OpenAI or search-provider credentials for this static routing preparation.
The existing Azure endpoint and Azure CLI sign-in are sufficient for optional
Clef inference; no additional Azure resource is needed.

### Original ToolBench archive blocker

The official Google Drive folder and archive link returned 404 during source
verification; one subsequent archive check timed out. The Tsinghua mirror
returned an HTML page saying "Link does not exist." The persisted download
manifest records the acquisition-time checks, including network failures.

The ten repository examples are not a replacement for the full archive.
A working, authorized full archive or official mirror is needed to complete
that acquisition. No community repack or similarly named benchmark was substituted.

### Sources and usage terms

* [SWE-bench dataset](https://huggingface.co/datasets/SWE-bench/SWE-bench),
  revision `c6fe717fd7a4c3ac1daa4055a4fd082c6a1d28a2`.
  The MIT repository license is retained, but the dataset card does not
  independently license every upstream issue and code fragment.
* [BFCL V4 source](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard),
  revision `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`.
  Apache-2.0 license and documentation are retained. The older V3 Hugging Face
  mirror is not mixed into V4.
* [Original ToolBench](https://github.com/OpenBMB/ToolBench/tree/d56fdd89faf8c91fa135090b212bb9057ee5cfc2),
  revision `d56fdd89faf8c91fa135090b212bb9057ee5cfc2`.
  The Apache-2.0 license is retained. Its README additionally restricts usage
  to research and education.

Keep upstream attribution and review applicable terms before redistribution,
fine-tuning or a different deployment use. Public benchmark contamination in
pretraining is unknown; these probes alone cannot establish generalization.

## Private VS Code session dataset

Review genuine user turns locally, or infer automatic silver labels, before evaluating an Azure-hosted router.

### Two workflows

This covers two ways to build a dataset from your own VS Code agent sessions:

* [Manual review](#manual-review-workflow): a human inspects and labels each
  candidate turn. Produces a smaller, higher-trust set.
* [Automatic silver labeling](#automatic-silver-labeling-workflow): a
  deterministic rule-based labeler infers labels from recorded responses and
  agent telemetry, with no per-example manual work. Produces a larger,
  unreviewed set faster.

Both start from the same local, authorized session inventory and never send
private data to Azure without explicit, separately gated approval.

### Manual review workflow

#### Prepared snapshot

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

#### Open the private reviewer

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

#### What was extracted

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

#### Export and evaluate

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

#### Privacy and provenance

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

### Automatic silver-labeling workflow

#### Automated path

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

#### Recorded facts versus inferred labels

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

#### Labeling policy

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

#### Leakage and privacy boundaries

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

#### Evaluation

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

## Routing datasets: four public sources in one canonical format

This turns four public LLM-routing datasets into one record shape, labels every query with how much model it takes
(low, medium or high), and writes a fine-tuning bundle the capability trainer can use. Code is in
[routing_data/](../routing_data).

### The four sources

| Source | What it holds | Pulled | Queries | Models | Licence on the dataset page |
|---|---|---|---|---|---|
| [xRouteBench](https://huggingface.co/datasets/ulab-ai/xRouteBench) | 13 classic NLP benchmarks plus long-conversation memory QA, every query run on 18 models with score, tokens, latency | `llmrouter_generic`, `memory_locomo`, `memory_longmemeval` (text only; video, multimodal, time series and personalised scenarios skipped) | 9,794 | 18 | not stated |
| [LLMRouterBench](https://huggingface.co/datasets/NPULH/LLMRouterBench) | 27 benchmarks, one JSON file per model, including agent tasks (tau2, SWE-bench), coding and arena-hard | all of `bench-release.tar.gz` (1.2 GB) | 25,035 | 40 | not stated |
| [R2-Bench](https://huggingface.co/datasets/JiaqiXue/R2-Bench) | 30,968 queries on 10 models at 16 output-token budgets, judge scored | budgets 200 and 2,000 only, streamed and reduced (the full set is 25 GB with response text) | 30,968 | 10 | MIT, upstream prompt rights stay with their providers |
| [SCOPE-60K](https://huggingface.co/datasets/Cooolder/SCOPE-60K) | science, maths, humanities and Chinese questions on 13 models, correct or not | train, test and anchor | 5,190 | 13 (reasoning runs counted as separate candidates) | Apache-2.0 |

None of these was written for this project. Their prompts come from public benchmarks whose rights stay with the original
providers, and two of the four do not state a licence, so treat the output as research material. Raw downloads go to
`data/external/` (8 GB) and are not meant for version control.

### The canonical record

One query and a list of observations, one per candidate model. Response text is left out on purpose (it is most of the size and
tier labels do not need it).

```json
{
  "id": "xroutebench/mbpp/afd4c3ebb650",
  "source": "xroutebench",
  "dataset": "mbpp",
  "query": "[System Instruction]\nYou are an expert Python programmer ...",
  "observations": [
    {"model": "qwen2.5-7b-instruct", "quality": 1.0, "cost_usd": 2.58e-05, "tokens_in": 98, "tokens_out": 31,
     "latency_s": 0.752, "budget_tokens": null, "runs": 1}
  ],
  "split_origin": ["train"],
  "metadata": {"metric": "code_eval", "scenario": "llmrouter_generic"},
  "license": "not stated on the dataset card; prompts come from public benchmarks",
  "sha": "afd4c3ebb65058b59dbf"
}
```

- `quality` is 0 to 1 from the source's own metric (accuracy, F1, judge score). Missing scores (a judge's `-1`) become `null`.
- `budget_tokens` is set only for R2-Bench, where the same model and query appear once per output budget.
- Repeated runs of the same model and query are averaged (`runs` says how many).
- `sha` is a hash of the query with whitespace and case folded, used to recognise the same question in two sources.
- R2-Bench groups by its own `key` column, because the query text for the same item differs between model runs.

Written to `data/canonical/<source>.jsonl` (293 MB in total).

### Tier labels

A query is hard for a class of model, not in the abstract. Inside each source and sub-benchmark, models are ranked by their own mean
score and split into thirds (weak, middle, strong). A query is:

| Tier | Rule |
|---|---|
| low | half or more of the weak third solve it |
| medium | not low, but half or more of the middle or strong third solve it |
| high | even the strong third mostly fails it |

"Solve" is a score of at least 0.7. On strict metrics (exact match, token F1) where even the strongest models rarely reach 0.7,
solving is 80% of what the strongest third scores on average, so a harsh metric does not make everything "high". A query needs at
least two observations from each third and a pool at least six models to be labelled. R2-Bench is labelled from its 2,000-token
budget. This is the same idea as the labels on this project's ten endpoints (how much model it takes), made comparable across pools of
different strength.

Of the 70,987 canonical queries, 70,958 could be labelled:

| Source | Low | Medium | High |
|---|---|---|---|
| LLMRouterBench | 8,531 | 8,616 | 7,887 |
| xRouteBench | 4,678 | 3,035 | 2,081 |
| SCOPE-60K | 1,677 | 2,131 | 1,356 |
| R2-Bench | 8,786 | 18,329 | 3,851 |

The mix behaves sensibly: ARC-Challenge and OpenBookQA are almost all low, Humanity's Last Exam is 83% high, SWE-bench is mostly high,
and strict-metric sets such as SQuAD and LoCoMo spread across all three.

### The fine-tuning bundle

`python -m routing_data export --name routing-ext-v1` writes a bundle in the format `routing_lab.finetune_capability` trains on,
using the current prompt (version p2):

- 1,105 queries that appear in more than one source are dropped; 8 sub-benchmarks are held out entirely (gsm8k, livecodebench, tau2,
  finqa, bbh, arenahard_math, history, boolq), covering coding, maths, agent tool use, finance, logic, history and reading comprehension.
- Each sub-benchmark is capped at 150 queries (R2-Bench, a mixed collection, at 1,500), long queries are cut at 3,000 characters, and
  the 2 queries that become identical after cutting keep one copy.
- Result: **5,830 train, 841 validation, 823 calibration** rows (train mix 33% low, 41% medium, 26% high), plus **843 in-distribution
  and 1,050 out-of-distribution test rows** written to `data/finetuning/routing-ext-v1-heldout/`, outside the bundle.
- Source mix of the rows: LLMRouterBench 3,481, xRouteBench 2,217, SCOPE-60K 2,190, R2-Bench 1,499.
- The skill answer is left empty for these rows (the trainer skips the skill loss on them), because the only skill signal is a guess from the
  benchmark's name. `--use-skill` switches on that dataset-level hint (see `routing_data/skills.py`).

### Cleaning (routing-ext-v2)

After reading samples from `routing-ext-v1`, some labels turned out to measure "the score was meaningless", not "this is hard":
an open-ended "invent a dessert recipe" labelled high because no answer can score well, and ambiguous emotion labels. Version 2 adds
`routing_data/clean.py`, which drops, with a counted reason each:

| Rule | Dropped |
|---|---|
| Subjective datasets: arena-hard (pairwise preference), emorynlp and meld (ambiguous emotion labels) | 3,428 |
| R2-Bench queries where 40% or more of the judge's scores are partial credit (open-ended) | 7,659 |
| LLMRouterBench and xRouteBench queries every model scored 0 on, in a sub-benchmark whose solve threshold is under 0.25 (Humanity's Last Exam, SWE-bench-like pools, strict-F1 memory sets, AIME) | 1,391 |

That leaves 58,483 labelled queries and a bundle of **5,444 train, 787 validation, 770 calibration** rows (train mix 35% low, 42% medium,
23% high), with 787 in-distribution and 1,050 out-of-distribution test rows outside the bundle. The 7 held-out sub-benchmarks present
are bbh, finqa, livecodebench, tau2, gsm8k, boolq and history (arenahard_math was one of the eight and is now removed as subjective).

Not fixed: a strict answer format can still make an easy question look medium (an R2-Bench biome question scored 20% solved), and
pool-relative labels remain. Run `python -m routing_data export --name routing-ext-v2` (cleaning is on by default;
`--no-clean` reproduces v1).
### What to keep in mind

- **Pool-relative labels.** "Low" means the weaker third of that pool handles it. An AIME problem solved by a pool's reasoning-distilled
  small models is labelled low. The tiers are consistent within a source, not identical to the tiers on this project's own endpoints.
- **Benchmark questions, not agent requests.** Almost everything here is question answering, maths and code. Only tau2 and SWE-bench
  resemble agent work, so this adds breadth for unseen task types, which was the weakness of the fine-tuned router, but not tool-use depth.
- **Size.** 5,830 training rows of about 1,300 tokens each is roughly seven times the previous run. Expect hours of GPU time rather than
  the 23 minutes of the earlier fine-tunes. Lower `--cap` to shrink it.
- **Not yet trained.** Nothing here has been fine-tuned or evaluated on the router.

### Commands

```powershell
python -m routing_data.fetch                  # download all four (R2-Bench is reduced as it streams)
python -m routing_data convert                # data/external -> data/canonical/<source>.jsonl
python -m routing_data stats                  # sizes, models, sub-benchmarks
python -m routing_data labels                 # tier mix per source and sub-benchmark
python -m routing_data export --name routing-ext-v1 [--cap 150] [--use-skill]
```

## Expanded agent routing and batched annotation

Prepare separate coding and tool/API routing tracks, label them with a strict-JSON batched GPT-5.6 Sol judge, and recover from failures without weakening validation.

### Scope

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

### What changed in the labeling approach

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

### Full-dataset preparation

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

### Current execution: strict-JSON batches

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

#### Exact JSON contract

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

#### Evidence is no longer verified against source text

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

#### Two blind passes and accounting

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

### GPT-5.6 Sol judge configuration

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

### Annotation and review gates

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

### Approved bounded retries (first pass, batch_size=10)

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

### Run, resume and export (current pass, batch_size=100)

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

### Completed labeling and the third-pass majority vote

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

### Frozen 90/10 benchmark split

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

### Before actual evaluation

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
[packaged deployment workflow](deployment.md#deploy-billable-needs-approval)
when an expanded-catalog rollout is approved.

### Earlier one-row workflow (historical)

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

## Review and evaluation protocol

Separate synthetic probes, reviewed benchmarks and downstream evidence.

### Agent ownership

Select one initial owning agent, not an execution plan. Debugging owns a combined
diagnose/fix/regression-test request. Standalone test creation goes to test generation.
Code review receives a proposed change. Repository Q&A is read-only explanation or
search. Fallback includes unsupported work and missing essential context.

Do not infer that a diagnosis request authorizes writing. Capability contracts
describe what an agent can do; a future execution policy must independently enforce
the user's permissions and approvals.

### Probe before benchmarking

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

### Build reviewed data

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

### Compare routing baselines

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

### Test downstream outcomes

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

### Fine-tuning decision

Before opening the frozen test set, record acceptable task-success loss, latency,
total cost, non-fallback coverage and any mandatory-escalation requirements.
No arbitrary numerical threshold is treated as an approved business requirement.

Proceed only if repeated domain-specific routing mistakes remain after resolving
label ambiguity, missing context and contract overlap. Compare against the unchanged
off-the-shelf checkpoint. Do not train to repair broken downstream agents.

The next stages require a human-reviewed set, live Azure model measurements, actual
agent outcomes and an agreed decision threshold. This project intentionally does
not automatically approve or launch fine-tuning.

## Routing criteria prompt-tuning experiment

Compare richer routing descriptions against the unchanged off-the-shelf baseline before considering weight fine-tuning.

### Question and controls

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

### Run and inspect

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

### Metrics and interpretation

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

### Recorded result on 2026-09-30

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

### Next decision

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

## Shared router fine-tuning data and Clef plan

Frozen, catalog-conditioned silver-data partitions for fine-tuning Clef, and what remains before training starts.

### Experiment boundary

Train one shared Clef-Flash checkpoint on both `coding-v1` and `tool-api-v1`. Each example
supplies its own route catalog and instructions, which is exactly the SystemOne record Clef
already consumes (`state`, `questions`, and a supervised answer). This is a typed-decision
fine-tune, not a fixed 13-label classifier and not a text-generation fine-tune. Never offer
coding routes to a tool/API-only request.

Preparation does not start a training job, create compute, or change endpoint traffic.

### Status

**Trained, selected and evaluated once on the frozen test split.** Clef-Flash reaches 95.3% on coding and
95.7% on tool/API, up from 80.8% and 73.6% off the shelf, with calibration error down from about 0.2 to 0.02.
The full comparison, per-route detail and caveats are in
[baseline results](#clef-flash-fine-tuned-on-the-same-test10-split-2026-10-07). Read the
caveats there: this measures agreement with a silver labeler on the same two sources, not general routing skill.

The fine-tuned model is deployed as `flash-ft-gpu` on the same endpoint at 0% traffic, so the default route still
serves the off-the-shelf model. See [endpoint guide](deployment.md#fine-tuned-deployment-2026-10-07).

### Data and selection

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

### How it trains

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

### What the runs measured

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

### Run it

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

### Evaluate on the frozen test split

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
[baseline results](#off-the-shelf-baseline-results-before-fine-tuning). The off-the-shelf Clef-Flash numbers there are the comparison.
Test10 has already informed earlier choices, so this is an exposed-test before/after result, not an
unbiased final estimate. Run it once.

To score an export locally on a GPU instead:

```text
python -m routing_lab.cli run --backend clef --snapshot <export> --calibrated --probe --allow-model-download ...
```

### Risks to settle first

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

## Off-the-shelf baseline results (before fine-tuning)

Full-scale benchmark of the unmodified Laya English endpoint on the AI-labeled 90/10 split, recorded for later comparison against a fine-tuned checkpoint.

> **Historical Laya baseline.** The Laya sections below were measured with the Laya English checkpoint before
> the project moved to Cloudflare Clef. They remain the comparison point for the Clef-Flash run on the
> same split; they are not Clef results. The Laya endpoint and client code no longer exist in this repository.

### Scope

This is the "taste the model before fine-tuning" baseline the plan called for:
the off-the-shelf Laya English checkpoint, unmodified, scored against the full
nominal 10% test split for both tracks (13.49% for coding because repository
groups stay intact). Labels are GPT-5.6 Sol AI-assisted
annotations (two-pass agreement plus third-pass majority vote on
disagreements), not human-reviewed gold — read these numbers as agreement with
an AI heuristic, not ground truth accuracy.

Endpoint: `laya-routing-exp`, deployment `english-cpu-large`
(`Standard_E16s_v3`, 16 vCPU / 128GB), 100% of live traffic. See
[endpoint guide](deployment.md#azure-hosted-clef-with-a-local-notebook) for how this deployment and its latency
fixes were built.

Run date: 2026-10-02.

### Headline results

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

### Per-route breakdown

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

### Truncation: a real, pervasive limitation, not a taxonomy artifact

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

### Final result: max_len=2048 override, deployed to the live endpoint (2026-10-02)

Following the max_len override experiment,
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

### Clef-Flash on the same test10 split (2026-10-06)

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

### Clef-Flash fine-tuned, on the same test10 split (2026-10-07)

LoRA fine-tuning of Clef-Flash on the frozen 3,966-row training partition (recipe and costs in
[fine-tuning guide](#shared-router-fine-tuning-data-and-clef-plan)), then scored once on exactly the same test files, labels, contracts and
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

### What fine-tuning changed row by row (2026-10-07)

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

### Two infrastructure issues hit and fixed during this run

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

### Dataset provenance for this run

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
* See [annotation guide](#expanded-agent-routing-and-batched-annotation) for the full labeling pipeline
  and [routing_lab/benchmark_filter.py](../routing_lab/benchmark_filter.py)
  for the source-bundle filtering and transport-cap utilities used to avoid
  re-labeling the unaffected tool-api-v1 track.

### Next step

Prepare one shared, catalog-conditioned checkpoint using only `train90`.
Reserve training-derived validation and calibration partitions before fitting.
The existing test files have already informed model/configuration decisions:
freeze them for qualified before/after reporting, but do not call them an
untouched final holdout. An unbiased final generalization claim needs a fresh
independent reviewed set. Compare base and tuned models under identical
inputs, token budgets, route catalogs, and serving hardware.
