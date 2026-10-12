"""Dataset names to one spelling, and a coarse kind-of-work hint for each.

The hint maps a sub-benchmark to the five kinds of work the capability router names (coding, reasoning, tools, writing,
extraction). It is a judgement about the benchmark, not a per-query label, so it is marked as such and the exporter leaves
it out of training by default. Benchmarks that do not fit any kind cleanly have no hint.
"""

from __future__ import annotations

import re

ALIASES = {
    "mmlupro": "mmlupro", "humaneval": "humaneval", "aime": "aime", "aime20202024": "aime", "openbookqa": "openbookqa",
    "arcchallenge": "arcc", "arcc": "arcc", "commonsenseqa": "commonsenseqa",
}
SKILL_OF = {
    "coding": ["humaneval", "mbpp", "livecodebench", "swebench", "arenahardcoding", "computerscience"],
    "reasoning": ["gsm8k", "math", "math500", "aime", "mathbench", "livemathbench", "kandk", "korbench", "arcagi", "bbh", "gpqa",
                  "arenahardmath", "mathematics", "physics", "chemistry", "engineering", "biology"],
    "tools": ["tau2"],
    "writing": ["arenahardcreativewriting"],
    "extraction": ["squad", "boolq", "locomo", "longmemeval", "finqa", "emorynlp", "meld"],
}
_SKILL_BY_KEY = {key: skill for skill, keys in SKILL_OF.items() for key in keys}


def dataset_key(name: str) -> str:
    key = re.sub(r"[^a-z0-9]", "", name.lower())
    return ALIASES.get(key, key)


def skill_hint(dataset: str) -> str | None:
    return _SKILL_BY_KEY.get(dataset_key(dataset))
