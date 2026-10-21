"""Azure OpenAI Responses API client used to execute routed steps, with real token accounting."""

from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

ENTRA_SCOPE = "https://ai.azure.com/.default"
_BASE = re.compile(r"https://[a-z0-9-]+\.(?:services\.ai\.azure\.com|openai\.azure\.com)/openai/v1/")
MODEL_FORMATS = Path(__file__).resolve().parents[1] / "capability_lab" / "models.json"


SEARCH_SYSTEM = ("Search the web and answer the query with concrete, current facts: names, neighbourhoods, typical "
                 "prices and opening details where known. Be concise (under 250 words) and stay on the query.")


class LLMError(RuntimeError):
    """A model call failed in a way the caller may handle by escalating or falling back."""


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    status: str
    response_id: str | None = None
    function_calls: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, str]] = field(default_factory=list)
    incomplete_reason: str | None = None
    refusal: str | None = None


class AzureResponsesClient:
    def __init__(self, base_url: str, *, credential: Any = None, timeout: float = 120.0,
                 session: requests.Session | None = None, retries: int = 2) -> None:
        if not _BASE.fullmatch(base_url):
            raise ValueError("base_url must be an Azure /openai/v1/ HTTPS endpoint")
        self.base_url = base_url
        self._timeout = timeout
        self._session = session or requests.Session()
        self._retries = retries
        self._credential = credential
        self._token: Any = None
        self._lock = threading.Lock()
        models = json.loads(MODEL_FORMATS.read_text(encoding="utf-8-sig"))["models"]
        self._json_apis = {m["deployment"]: m["json_api"] for m in models if m.get("json_api") is not None}
        if any(api not in ("responses", "chat") for api in self._json_apis.values()):
            raise ValueError("Model json_api must be responses, chat, or null")

    def _bearer(self) -> str:
        with self._lock:
            if self._credential is None:
                from routing_lab.azure_auth import make_credential

                self._credential = make_credential(process_timeout=30)
            if self._token is None or self._token.expires_on - time.time() < 300:
                self._token = self._credential.get_token(ENTRA_SCOPE)
            return self._token.token

    def complete(
        self, deployment: str, *, system: str | None = None, user: str | None = None,
        input_items: list[dict[str, Any]] | None = None, previous_response_id: str | None = None,
        tools: list[dict[str, Any]] | None = None, json_object: bool = False, max_output_tokens: int = 1200,
    ) -> LLMResult:
        items = input_items
        if items is None:
            items = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
        body: dict[str, Any] = {"model": deployment, "input": items, "max_output_tokens": max_output_tokens}
        if previous_response_id:
            body["previous_response_id"] = previous_response_id
        if tools:
            body["tools"] = tools
        if json_object:
            body["text"] = {"format": {"type": "json_object"}}
        path = "responses"
        if json_object and self._json_apis.get(deployment) == "chat":
            if tools or previous_response_id or input_items is not None:
                raise ValueError("Native Chat JSON supports only single-turn calls without tools or response history")
            path = "chat/completions"
            body = {"model": deployment, "messages": items, "max_tokens": max_output_tokens,
                    "response_format": {"type": "json_object"}}
        started = time.perf_counter()
        last = ""
        for attempt in range(self._retries + 1):
            try:
                response = self._session.post(
                    self.base_url + path, json=body, timeout=self._timeout, allow_redirects=False,
                    headers={"Authorization": "Bearer " + self._bearer(), "Content-Type": "application/json"})
            except requests.RequestException as error:
                last = type(error).__name__
            else:
                if response.status_code == 200:
                    parse = self._parse_chat if path == "chat/completions" else self._parse
                    return parse(response.json(), (time.perf_counter() - started) * 1000)
                last = f"HTTP {response.status_code}"
                if response.status_code not in (429, 500, 502, 503, 504):
                    raise LLMError(f"{deployment} returned {last}: {response.text[:200]}")
            time.sleep(0.8 * (attempt + 1))
        raise LLMError(f"{deployment} failed after retries ({last})")

    def search(self, deployment: str, query: str, *, max_output_tokens: int = 700) -> LLMResult:
        """Open web search through the model's built-in search tool; the result carries cited sources."""
        return self.complete(
            deployment, system=SEARCH_SYSTEM, user=query, max_output_tokens=max_output_tokens,
            tools=[{"type": "web_search_preview", "search_context_size": "low"}])

    @staticmethod
    def _parse_chat(payload: dict[str, Any], latency_ms: float) -> LLMResult:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise LLMError("Chat completion returned no choices")
        choice = choices[0]
        message, usage = choice["message"], payload.get("usage") or {}
        finish = choice.get("finish_reason")
        reason = None if finish == "stop" else "max_output_tokens" if finish == "length" else finish or "unknown"
        return LLMResult(
            (message.get("content") or "").strip(), int(usage.get("prompt_tokens", 0)),
            int(usage.get("completion_tokens", 0)), round(latency_ms, 1),
            "completed" if finish == "stop" else "incomplete", payload.get("id"),
            incomplete_reason=reason, refusal=message.get("refusal"),
        )

    @staticmethod
    def _parse(payload: dict[str, Any], latency_ms: float) -> LLMResult:
        text, calls, sources = [], [], []
        refusal = None
        for item in payload.get("output", []):
            if item.get("type") == "message":
                for part in item.get("content", []):
                    if part.get("type") == "refusal":
                        refusal = str(part.get("refusal") or "The model refused the request.")
                    if part.get("type") != "output_text":
                        continue
                    text.append(part.get("text", ""))
                    for note in part.get("annotations") or []:
                        if note.get("type") == "url_citation" and str(note.get("url", "")).startswith("https://"):
                            sources.append({"title": str(note.get("title", ""))[:140], "url": note["url"]})
            elif item.get("type") == "function_call":
                try:
                    args = json.loads(item.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                calls.append({"call_id": item.get("call_id"), "name": item.get("name"), "arguments": args})
        usage = payload.get("usage") or {}
        incomplete = payload.get("incomplete_details") or {}
        return LLMResult("".join(text).strip(), int(usage.get("input_tokens", 0)), int(usage.get("output_tokens", 0)),
                         round(latency_ms, 1), str(payload.get("status", "")), payload.get("id"), calls, sources,
                         incomplete_reason=incomplete.get("reason"), refusal=refusal)
