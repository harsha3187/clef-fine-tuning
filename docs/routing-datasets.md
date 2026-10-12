# Routing datasets: four public sources in one canonical format

This turns four public LLM-routing datasets into one record shape, labels every query with how much model it takes
(low, medium or high), and writes a fine-tuning bundle the capability trainer can use. Code is in
[routing_data/](../routing_data).

## The four sources

| Source | What it holds | Pulled | Queries | Models | Licence on the dataset page |
|---|---|---|---|---|---|
| [xRouteBench](https://huggingface.co/datasets/ulab-ai/xRouteBench) | 13 classic NLP benchmarks plus long-conversation memory QA, every query run on 18 models with score, tokens, latency | `llmrouter_generic`, `memory_locomo`, `memory_longmemeval` (text only; video, multimodal, time series and personalised scenarios skipped) | 9,794 | 18 | not stated |
| [LLMRouterBench](https://huggingface.co/datasets/NPULH/LLMRouterBench) | 27 benchmarks, one JSON file per model, including agent tasks (tau2, SWE-bench), coding and arena-hard | all of `bench-release.tar.gz` (1.2 GB) | 25,035 | 40 | not stated |
| [R2-Bench](https://huggingface.co/datasets/JiaqiXue/R2-Bench) | 30,968 queries on 10 models at 16 output-token budgets, judge scored | budgets 200 and 2,000 only, streamed and reduced (the full set is 25 GB with response text) | 30,968 | 10 | MIT, upstream prompt rights stay with their providers |
| [SCOPE-60K](https://huggingface.co/datasets/Cooolder/SCOPE-60K) | science, maths, humanities and Chinese questions on 13 models, correct or not | train, test and anchor | 5,190 | 13 (reasoning runs counted as separate candidates) | Apache-2.0 |

None of these was written for this project. Their prompts come from public benchmarks whose rights stay with the original
providers, and two of the four do not state a licence, so treat the output as research material. Raw downloads go to
`data/external/` (8 GB) and are not meant for version control.

## The canonical record

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

## Tier labels

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

## The fine-tuning bundle

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

## Cleaning (routing-ext-v2)

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
## What to keep in mind

- **Pool-relative labels.** "Low" means the weaker third of that pool handles it. An AIME problem solved by a pool's reasoning-distilled
  small models is labelled low. The tiers are consistent within a source, not identical to the tiers on this project's own endpoints.
- **Benchmark questions, not agent requests.** Almost everything here is question answering, maths and code. Only tau2 and SWE-bench
  resemble agent work, so this adds breadth for unseen task types, which was the weakness of the fine-tuned router, but not tool-use depth.
- **Size.** 5,830 training rows of about 1,300 tokens each is roughly seven times the previous run. Expect hours of GPU time rather than
  the 23 minutes of the earlier fine-tunes. Lower `--cap` to shrink it.
- **Not yet trained.** Nothing here has been fine-tuned or evaluated on the router.

## Commands

```powershell
python -m routing_data.fetch                  # download all four (R2-Bench is reduced as it streams)
python -m routing_data convert                # data/external -> data/canonical/<source>.jsonl
python -m routing_data stats                  # sizes, models, sub-benchmarks
python -m routing_data labels                 # tier mix per source and sub-benchmark
python -m routing_data export --name routing-ext-v1 [--cap 150] [--use-skill]
```
