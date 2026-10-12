"""Stateless Azure OpenAI v1 Responses transport for independent annotations.

Construct ``AzureResponsesJudge(load_judge_config(path))`` as a context manager.
Construction and metadata access are offline; only ``annotate`` authenticates.
The coordinator owns request budgets and semantic annotation validation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from time import perf_counter
from types import TracebackType
from typing import TYPE_CHECKING, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from openai import OpenAI
    from openai.types.responses import Response

ENTRA_SCOPE = "https://ai.azure.com/.default"
MAX_OUTPUT_TOKENS = 32768
MAX_TIMEOUT_SECONDS = 600
_ENDPOINT = re.compile(
    r"https://(?P<host>[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?"
    r"\.(?:services\.ai\.azure\.com|openai\.azure\.com))"
    r"/openai/v1(?:/|/responses/?)"
)
_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"


class JudgeConfig(BaseModel):
    """Nonsecret, immutable connection configuration with bounded request limits."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)
    endpoint: str
    deployment: str = Field(pattern=_IDENTIFIER)
    model: str = Field(pattern=_IDENTIFIER)
    timeout_seconds: float = Field(default=90.0, gt=0, le=MAX_TIMEOUT_SECONDS)
    max_output_tokens: int = Field(default=1600, ge=16, le=MAX_OUTPUT_TOKENS)

    @field_validator("endpoint")
    @classmethod
    def normalize_endpoint(cls, value: str) -> str:
        match = _ENDPOINT.fullmatch(value)
        if match is None:
            raise ValueError(
                "Endpoint must be an HTTPS Azure resource URL ending in /openai/v1/ or "
                "/openai/v1/responses, without credentials, ports, query, or fragment"
            )
        return f"https://{match['host']}/openai/v1/"


def load_judge_config(path: str | Path) -> JudgeConfig:
    """Load strict connection metadata; never read authentication secrets from it."""
    return JudgeConfig.model_validate_json(Path(path).read_text(encoding="utf-8"))


@dataclass(frozen=True)
class JudgeResponse:
    """Successful structured output with the service's actual identity and usage."""

    payload: dict
    model: str
    response_id: str
    input_tokens: int
    output_tokens: int
    elapsed_ms: float


class JudgeError(RuntimeError):
    """An annotation response cannot be accepted."""


class JudgeRefusalError(JudgeError):
    """The service explicitly refused the annotation."""


class JudgeIncompleteError(JudgeError):
    """The response was truncated or otherwise incomplete."""


class JudgeResponseError(JudgeError):
    """Response identity, status, output, or usage failed validation."""


def _strict_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON numeric constant: {value}")


def _finite_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("JSON number exceeds finite float range")
    return number


def _validate_schema(schema: dict) -> None:
    """Reject non-strict object schemas before any authentication or request."""
    if schema.get("type") != "object":
        raise ValueError("Annotation schema must have an object root")

    def visit(node: object) -> None:
        if not isinstance(node, dict):
            raise ValueError("Schema nodes must be objects")
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties")
            required = node.get("required")
            if (
                not isinstance(properties, dict)
                or not isinstance(required, list)
                or any(not isinstance(key, str) for key in required)
                or len(required) != len(set(required))
                or set(required) != set(properties)
                or node.get("additionalProperties") is not False
            ):
                raise ValueError("Strict object schemas require all properties and additionalProperties=false")
        for key in ("properties", "$defs", "definitions"):
            if key in node:
                if not isinstance(node[key], dict):
                    raise ValueError(f"Schema {key} must be an object")
                for child in node[key].values():
                    visit(child)
        if "items" in node:
            visit(node["items"])
        for key in ("anyOf", "oneOf", "allOf"):
            if key in node:
                if not isinstance(node[key], list):
                    raise ValueError(f"Schema {key} must be an array")
                for child in node[key]:
                    visit(child)

    visit(schema)
    json.dumps(schema, allow_nan=False)


class AzureResponsesJudge:
    """Lazy Entra-authenticated Responses client; never retries or follows redirects."""

    def __init__(self, config: JudgeConfig) -> None:
        self.config = config
        self._client: OpenAI | None = None
        self._resources = ExitStack()
        self._closed = False

    def metadata(self) -> dict:
        """Return reproducible nonsecret configuration without creating a client."""
        config = self.config.model_dump()
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return {
            "provider": "azure-openai-v1-responses",
            "endpoint": self.config.endpoint,
            "deployment": self.config.deployment,
            "configured_model": self.config.model,
            "timeout_seconds": self.config.timeout_seconds,
            "max_output_tokens": self.config.max_output_tokens,
            "authentication": "AzureCliCredential",
            "auth_scope": ENTRA_SCOPE,
            "max_retries": 0,
            "follow_redirects": False,
            "store": False,
            "config_sha256": hashlib.sha256(encoded).hexdigest(),
        }

    def _get_client(self) -> OpenAI:
        if self._closed:
            raise RuntimeError("Annotation judge is closed")
        if self._client is None:
            from azure.identity import AzureCliCredential, get_bearer_token_provider
            from openai import DefaultHttpxClient, OpenAI

            # ExitStack also closes partially constructed resources if setup fails.
            with ExitStack() as resources:
                credential = AzureCliCredential()
                resources.callback(credential.close)
                token_provider = get_bearer_token_provider(credential, ENTRA_SCOPE)
                http_client = DefaultHttpxClient(
                    follow_redirects=False, timeout=self.config.timeout_seconds,
                )
                resources.callback(http_client.close)
                client = OpenAI(
                    base_url=self.config.endpoint,
                    # This SDK parameter accepts an Entra token provider, not just static API keys.
                    api_key=token_provider,
                    organization="",
                    project="",
                    timeout=self.config.timeout_seconds,
                    max_retries=0,
                    http_client=http_client,
                )
                resources.callback(client.close)
                self._resources = resources.pop_all()
                self._client = client
        return self._client

    def annotate(self, *, system_prompt: str, input_payload: dict, schema: dict) -> JudgeResponse:
        """Issue exactly one stateless request and reject unusable responses.

        API/authentication exceptions propagate unchanged. Explicit refusal and
        truncation use separate exceptions; no empty or guessed labels are returned.
        """
        if self._closed:
            raise RuntimeError("Annotation judge is closed")
        if not isinstance(system_prompt, str) or not system_prompt.strip():
            raise ValueError("System prompt must be a nonempty string")
        if not isinstance(input_payload, dict) or not input_payload:
            raise ValueError("Input payload must be a nonempty object")
        if not isinstance(schema, dict):
            raise ValueError("Schema must be an object")
        _validate_schema(schema)
        serialized = json.dumps(input_payload, ensure_ascii=False, allow_nan=False)
        client = self._get_client()
        started = perf_counter()
        response = client.responses.create(
            model=self.config.deployment,
            instructions=system_prompt,
            input=[{"role": "user", "content": serialized}],
            text={"format": {
                "type": "json_schema", "name": "annotation", "schema": schema, "strict": True,
            }},
            max_output_tokens=self.config.max_output_tokens,
            store=False,
            stream=False,
            background=False,
            tools=[],
            tool_choice="none",
            truncation="disabled",
        )
        elapsed_ms = (perf_counter() - started) * 1000
        return self._validate_response(response, elapsed_ms)

    def _validate_response(self, response: Response, elapsed_ms: float) -> JudgeResponse:
        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", None)
            raise JudgeIncompleteError(f"Judge response is incomplete ({reason})")
        if response.status != "completed" or response.error is not None:
            raise JudgeResponseError(f"Judge response was not successfully completed ({response.status})")
        if response.incomplete_details is not None:
            raise JudgeIncompleteError("Completed response unexpectedly contains incomplete details")
        actual_model = response.model
        configured = self.config.model
        if not isinstance(actual_model, str) or not (
            actual_model == configured
            or (
                re.search(r"-\d{4}-\d{2}-\d{2}$", configured) is None
                and re.fullmatch(re.escape(configured) + r"-\d{4}-\d{2}-\d{2}", actual_model)
            )
        ):
            raise JudgeResponseError("Actual response model does not match the configured judge model")
        if re.search(r"-\d{4}-\d{2}-\d{2}$", actual_model):
            try:
                date.fromisoformat(actual_model[-10:])
            except ValueError as exc:
                raise JudgeResponseError("Actual response model has an invalid version date") from exc
        if not isinstance(response.id, str) or not response.id.strip():
            raise JudgeResponseError("Judge response has no response ID")
        texts: list[str] = []
        for item in response.output:
            if item.type == "reasoning":
                continue
            if item.type != "message":
                raise JudgeResponseError("Judge returned an unexpected non-message output")
            for content in item.content:
                if content.type == "refusal":
                    raise JudgeRefusalError("Judge explicitly refused the annotation")
                if content.type != "output_text" or not isinstance(content.text, str):
                    raise JudgeResponseError("Judge returned unexpected message content")
                texts.append(content.text)
            if item.status != "completed" or item.role != "assistant":
                raise JudgeResponseError("Judge returned an invalid or unfinished message")
        text = "".join(texts)
        if not text.strip():
            raise JudgeResponseError("Judge returned empty output")
        try:
            payload = json.loads(
                text, object_pairs_hook=_strict_object, parse_constant=_reject_constant,
                parse_float=_finite_float,
            )
        except ValueError as exc:
            raise JudgeResponseError("Judge returned invalid JSON") from exc
        if not isinstance(payload, dict) or not payload:
            raise JudgeResponseError("Judge JSON must be a nonempty object")
        usage = response.usage
        if usage is None or any(
            type(value) is not int or value < 0
            for value in (usage.input_tokens, usage.output_tokens, usage.total_tokens)
        ):
            raise JudgeResponseError("Judge response must report actual nonnegative integer token usage")
        if usage.total_tokens != usage.input_tokens + usage.output_tokens:
            raise JudgeResponseError("Judge reported inconsistent token usage")
        if usage.output_tokens > self.config.max_output_tokens:
            raise JudgeResponseError("Judge exceeded the configured output token budget")
        return JudgeResponse(
            payload=payload, model=actual_model, response_id=response.id,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens, elapsed_ms=elapsed_ms,
        )

    def close(self) -> None:
        """Close owned resources once; a closed judge cannot issue more requests."""
        if not self._closed:
            self._closed = True
            self._resources.close()

    def __enter__(self) -> Self:
        if self._closed:
            raise RuntimeError("Annotation judge is closed")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
