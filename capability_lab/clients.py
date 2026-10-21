"""Model access for the benchmark: the Responses API where it works, Chat Completions where it does not."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from agent_demo.llm import AzureResponsesClient, LLMError, LLMResult

MODELS_FILE = Path(__file__).parent / "models.json"
BASE_URL = os.environ.get("AIE_LLM_ENDPOINT", "https://your-resource.services.ai.azure.com/openai/v1/")  # set AIE_LLM_ENDPOINT; not stored in the repository
SYSTEM = "Follow the instructions exactly and reply in exactly the format requested. Do not add explanations unless asked."
MAX_OUT_RESPONSES = 6000
MAX_OUT_CHAT = 3000
MAX_TOOL_TURNS = 6


@dataclass(frozen=True)
class ModelSpec:
    id: str
    deployment: str
    family: str
    api: str
    tools: bool
    price_in: float
    price_out: float
    verified: bool
    price_source: str
    json_api: str | None = None

    def cost(self, tokens_in: int, tokens_out: int) -> float:
        return tokens_in * self.price_in / 1e6 + tokens_out * self.price_out / 1e6


def load_models(path: Path = MODELS_FILE) -> dict[str, ModelSpec]:
    rows = json.loads(path.read_text(encoding="utf-8-sig"))["models"]
    models = {r["id"]: ModelSpec(**r) for r in rows}
    if len(models) != len(rows):
        raise ValueError("Duplicate model ids")
    return models


class BenchClient(AzureResponsesClient):
    """The shared Azure client plus a Chat Completions path for endpoints that do not speak Responses."""

    def __init__(self, base_url: str = BASE_URL, **kwargs: Any) -> None:
        super().__init__(base_url, **kwargs)

    def chat(self, deployment: str, system: str, user: str, max_tokens: int = MAX_OUT_CHAT) -> LLMResult:
        body = {"model": deployment, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "max_tokens": max_tokens}
        started = time.perf_counter()
        last = ""
        for attempt in range(self._retries + 1):
            try:
                response = self._session.post(self.base_url + "chat/completions", json=body, timeout=self._timeout,
                                              allow_redirects=False, headers={"Authorization": "Bearer " + self._bearer(),
                                                                              "Content-Type": "application/json"})
            except requests.RequestException as error:
                last = type(error).__name__
            else:
                if response.status_code == 200:
                    data = response.json()
                    usage = data.get("usage") or {}
                    text = (data["choices"][0]["message"].get("content") or "").strip()
                    return LLMResult(text, int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0)),
                                     round((time.perf_counter() - started) * 1000, 1), "completed")
                last = f"HTTP {response.status_code}"
                if response.status_code not in (429, 500, 502, 503, 504):
                    raise LLMError(f"{deployment} returned {last}: {response.text[:160]}")
            time.sleep(0.8 * (attempt + 1))
        raise LLMError(f"{deployment} failed after retries ({last})")


def ask(client: Any, model: ModelSpec, prompt: str) -> LLMResult:
    """One plain question, no tools."""
    if model.api == "chat":
        return client.chat(model.deployment, SYSTEM, prompt)
    return client.complete(model.deployment, system=SYSTEM, user=prompt, max_output_tokens=MAX_OUT_RESPONSES)


def run_with_tools(client: Any, model: ModelSpec, system: str, user: str, desk: Any, tools: list[dict[str, Any]]) -> tuple[LLMResult, int, int, float]:
    """The same tool loop the agents use: model calls tools, the desk answers, until the model stops."""
    total_in = total_out = 0
    latency = 0.0
    result = client.complete(model.deployment, system=system, user=user, tools=tools, max_output_tokens=MAX_OUT_RESPONSES)
    for turn in range(MAX_TOOL_TURNS + 1):
        total_in += result.input_tokens
        total_out += result.output_tokens
        latency += result.latency_ms
        if not result.function_calls or turn == MAX_TOOL_TURNS:
            return result, total_in, total_out, latency
        outputs = [{"type": "function_call_output", "call_id": c["call_id"],
                    "output": json.dumps(desk.execute(c["name"], c["arguments"]))} for c in result.function_calls]
        result = client.complete(model.deployment, input_items=outputs, previous_response_id=result.response_id, tools=tools,
                                 max_output_tokens=MAX_OUT_RESPONSES)
    return result, total_in, total_out, latency
