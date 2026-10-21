"""Offline transport and configuration tests; no credentials or remote requests."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import azure.identity
import httpx2
import openai
import pytest
from pydantic import ValidationError

from routing_lab.annotation_judge import (
    ENTRA_SCOPE, AzureResponsesJudge, JudgeConfig, JudgeIncompleteError,
    JudgeRefusalError, JudgeResponse, JudgeResponseError, load_judge_config,
)

ENDPOINT = "https://your-resource.services.ai.azure.com/openai/v1/"
SCHEMA = {
    "type": "object", "properties": {"label": {"type": "string"}},
    "required": ["label"], "additionalProperties": False,
}
REQUEST = {"system_prompt": "Assign a label.", "input_payload": {"query": "Synthetic"}, "schema": SCHEMA}


def config(**changes: object) -> JudgeConfig:
    return JudgeConfig.model_validate({
        "endpoint": ENDPOINT, "deployment": "independent-judge", "model": "gpt-4.1", **changes,
    })


def response(**changes: object) -> SimpleNamespace:
    return SimpleNamespace(**{
        "id": "resp_synthetic", "model": "gpt-4.1-2025-04-14", "status": "completed",
        "error": None, "incomplete_details": None,
        "output": [SimpleNamespace(
            type="message", role="assistant", status="completed",
            content=[SimpleNamespace(type="output_text", text='{"label":"synthetic"}')],
        )],
        "usage": SimpleNamespace(input_tokens=25, output_tokens=7, total_tokens=32),
        **changes,
    })


@pytest.fixture
def sdk(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    credential = Mock()
    credential_factory = Mock(return_value=credential)
    token_provider = Mock(return_value="synthetic-entra-token")
    provider_factory = Mock(return_value=token_provider)
    http_client = Mock()
    http_factory = Mock(return_value=http_client)
    client = Mock()
    client.responses.create.return_value = response()
    client_factory = Mock(return_value=client)
    monkeypatch.setattr(azure.identity, "AzureCliCredential", credential_factory)
    monkeypatch.setattr(azure.identity, "get_bearer_token_provider", provider_factory)
    monkeypatch.setattr(openai, "DefaultHttpxClient", http_factory)
    monkeypatch.setattr(openai, "OpenAI", client_factory)
    return SimpleNamespace(
        credential=credential, credential_factory=credential_factory,
        token_provider=token_provider, provider_factory=provider_factory,
        http_client=http_client, http_factory=http_factory,
        client=client, client_factory=client_factory,
    )


@pytest.mark.parametrize("suffix", ["", "responses", "responses/"])
@pytest.mark.parametrize("host", ["sample.services.ai.azure.com", "sample.openai.azure.com"])
def test_endpoint_normalization(host: str, suffix: str) -> None:
    assert config(endpoint=f"https://{host}/openai/v1/{suffix}").endpoint == f"https://{host}/openai/v1/"


@pytest.mark.parametrize("endpoint", [
    ENDPOINT.replace("https:", "http:"), ENDPOINT + "?api-key=secret", ENDPOINT + "#fragment",
    ENDPOINT + "?", ENDPOINT + "#", ENDPOINT.replace("https://", "https://user:secret@"),
    ENDPOINT.replace(".azure.com", ".azure.com.evil.test"),
    ENDPOINT.replace(".azure.com", ".azure.com:443"), "https://localhost/openai/v1/",
    "https://services.ai.azure.com/openai/v1/", "https://api.openai.com/openai/v1/",
    "https://a.b.services.ai.azure.com/openai/v1/", "https://127.0.0.1/openai/v1/",
    ENDPOINT + "../responses", ENDPOINT + "%72esponses", ENDPOINT + "responses?api-version=x",
    ENDPOINT.replace("/openai/v1/", "/openai/deployments/gpt-4.1/"),
    " " + ENDPOINT, ENDPOINT + "\n", ENDPOINT.replace("your-resource", "your_\tresource"),
])
def test_rejects_unsafe_endpoints(endpoint: str) -> None:
    with pytest.raises(ValidationError):
        config(endpoint=endpoint)


@pytest.mark.parametrize("changes", [
    {"api_key": "secret"}, {"max_retries": 1}, {"timeout_seconds": 0},
    {"timeout_seconds": -1}, {"timeout_seconds": 601}, {"timeout_seconds": float("inf")},
    {"timeout_seconds": float("nan")}, {"timeout_seconds": "90"}, {"timeout_seconds": True},
    {"max_output_tokens": 0}, {"max_output_tokens": 15}, {"max_output_tokens": 32769},
    {"max_output_tokens": 1600.0}, {"max_output_tokens": "1600"}, {"max_output_tokens": True},
    {"deployment": ""}, {"deployment": " bad"}, {"deployment": "name/path"},
    {"model": ""}, {"model": "gpt-4.1\n"}, {"model": None},
])
def test_strict_config_rejects_invalid_values(changes: dict) -> None:
    with pytest.raises(ValidationError):
        config(**changes)


def test_config_file_and_bounds(tmp_path: Path) -> None:
    supplied = Path(__file__).resolve().parents[1] / "azure" / "annotation-judge.json"
    loaded = load_judge_config(supplied)
    assert loaded.endpoint == ENDPOINT
    assert loaded.deployment == loaded.model == "gpt-5.6-sol"
    assert loaded.timeout_seconds == 90 and loaded.max_output_tokens == 1600
    assert config(timeout_seconds=600, max_output_tokens=32768).max_output_tokens == 32768
    assert config(timeout_seconds=0.1, max_output_tokens=16).timeout_seconds == 0.1
    with pytest.raises(ValidationError):
        loaded.model = "gpt-4.1-mini"
    invalid = tmp_path / "judge.json"
    invalid.write_text('{"api_key":"secret"}', encoding="utf-8")
    with pytest.raises(ValidationError):
        load_judge_config(invalid)


def test_lazy_metadata_and_close(sdk: SimpleNamespace) -> None:
    judge = AzureResponsesJudge(config())
    metadata = judge.metadata()
    assert metadata["configured_model"] == "gpt-4.1"
    assert metadata["provider"] == "azure-openai-v1-responses"
    assert len(metadata["config_sha256"]) == 64
    assert metadata == AzureResponsesJudge(config(endpoint=ENDPOINT + "responses")).metadata()
    assert metadata == AzureResponsesJudge(config(timeout_seconds=90)).metadata()
    assert metadata["config_sha256"] != AzureResponsesJudge(config(max_output_tokens=20)).metadata()["config_sha256"]
    assert "token" not in json.dumps(metadata).replace("max_output_tokens", "")
    sdk.credential_factory.assert_not_called()
    sdk.client_factory.assert_not_called()
    judge.close()
    judge.close()
    sdk.credential.close.assert_not_called()
    with pytest.raises(RuntimeError, match="closed"):
        judge.annotate(**REQUEST)
    with pytest.raises(RuntimeError, match="closed"):
        judge.__enter__()


def test_outgoing_request_and_actual_response(sdk: SimpleNamespace, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "DO_NOT_USE")
    with AzureResponsesJudge(config(timeout_seconds=12, max_output_tokens=100)) as judge:
        result = judge.annotate(**REQUEST)
        judge.annotate(**REQUEST)
    assert isinstance(result, JudgeResponse)
    assert result.payload == {"label": "synthetic"}
    assert result.model == "gpt-4.1-2025-04-14" and result.response_id == "resp_synthetic"
    assert result.input_tokens == 25 and result.output_tokens == 7 and result.elapsed_ms >= 0
    sdk.credential_factory.assert_called_once_with()
    sdk.provider_factory.assert_called_once_with(sdk.credential, "https://ai.azure.com/.default")
    assert ENTRA_SCOPE == "https://ai.azure.com/.default"
    sdk.http_factory.assert_called_once_with(follow_redirects=False, timeout=12)
    sdk.client_factory.assert_called_once_with(
        base_url=ENDPOINT, api_key=sdk.token_provider, organization="", project="",
        timeout=12, max_retries=0, http_client=sdk.http_client,
    )
    expected = {
        "model": "independent-judge", "instructions": REQUEST["system_prompt"],
        "input": [{"role": "user", "content": json.dumps(REQUEST["input_payload"])}],
        "text": {"format": {"type": "json_schema", "name": "annotation", "schema": SCHEMA, "strict": True}},
        "max_output_tokens": 100, "store": False, "stream": False, "background": False,
        "tools": [], "tool_choice": "none", "truncation": "disabled",
    }
    assert sdk.client.responses.create.call_count == 2
    assert all(call.kwargs == expected for call in sdk.client.responses.create.call_args_list)
    sdk.credential.close.assert_called_once()
    sdk.client.close.assert_called_once()
    sdk.http_client.close.assert_called_once()


@pytest.mark.parametrize("model", ["gpt-4.1", "gpt-4.1-2025-04-14"])
def test_model_family_or_exact_version(sdk: SimpleNamespace, model: str) -> None:
    sdk.client.responses.create.return_value = response(model=model)
    with AzureResponsesJudge(config(model=model)) as judge:
        assert judge.annotate(**REQUEST).model == model


@pytest.mark.parametrize(("configured", "actual"), [
    ("gpt-4.1", "gpt-4.1-mini"), ("gpt-4.1", "gpt-4.1-mini-2025-04-14"),
    ("gpt-4.1", "gpt-4.10"), ("gpt-4.1", "gpt-4.1-preview"), ("gpt-4.1", None),
    ("gpt-4.1", "gpt-4.1-2025-99-99"),
    ("gpt-4.1-2025-04-14", "gpt-4.1"), ("gpt-4.1-2025-04-14", "gpt-4.1-2026-04-14"),
])
def test_rejects_wrong_model(sdk: SimpleNamespace, configured: str, actual: str | None) -> None:
    sdk.client.responses.create.return_value = response(model=actual)
    with AzureResponsesJudge(config(model=configured)) as judge:
        with pytest.raises(JudgeResponseError, match="model"):
            judge.annotate(**REQUEST)


@pytest.mark.parametrize("status", ["failed", "queued", "in_progress", "cancelled", None])
def test_requires_completed_response(sdk: SimpleNamespace, status: str | None) -> None:
    sdk.client.responses.create.return_value = response(status=status)
    with AzureResponsesJudge(config()) as judge:
        with pytest.raises(JudgeResponseError, match="completed"):
            judge.annotate(**REQUEST)


def test_refusal_is_distinct_from_truncation(sdk: SimpleNamespace) -> None:
    refused = response()
    refused.output[0].content = [SimpleNamespace(type="refusal", refusal="Do not echo private refusal text")]
    sdk.client.responses.create.return_value = refused
    with AzureResponsesJudge(config()) as judge:
        with pytest.raises(JudgeRefusalError, match="refused"):
            judge.annotate(**REQUEST)
        sdk.client.responses.create.return_value = response(
            status="incomplete", incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        )
        with pytest.raises(JudgeIncompleteError, match="max_output_tokens"):
            judge.annotate(**REQUEST)
        sdk.client.responses.create.return_value = response(
            incomplete_details=SimpleNamespace(reason="content_filter"),
        )
        with pytest.raises(JudgeIncompleteError):
            judge.annotate(**REQUEST)


@pytest.mark.parametrize("text", [
    "", "  ", "not JSON", '{"label":', "null", "[]", "{}", '"label"',
    '{"label":NaN}', '{"label":Infinity}', '{"label":1e999}', '{"label":"a","label":"b"}',
])
def test_rejects_empty_or_invalid_json(sdk: SimpleNamespace, text: str) -> None:
    result = response()
    result.output[0].content[0].text = text
    sdk.client.responses.create.return_value = result
    with AzureResponsesJudge(config()) as judge:
        with pytest.raises(JudgeResponseError):
            judge.annotate(**REQUEST)


@pytest.mark.parametrize("changes", [
    {"usage": None}, {"input_tokens": -1}, {"output_tokens": -1},
    {"input_tokens": None}, {"output_tokens": True}, {"input_tokens": "25"},
    {"total_tokens": None}, {"total_tokens": 31}, {"output_tokens": 1601, "total_tokens": 1626},
])
def test_requires_real_bounded_usage(sdk: SimpleNamespace, changes: dict) -> None:
    result = response()
    if "usage" in changes:
        result.usage = changes["usage"]
    else:
        for key, value in changes.items():
            setattr(result.usage, key, value)
    sdk.client.responses.create.return_value = result
    with AzureResponsesJudge(config()) as judge:
        with pytest.raises(JudgeResponseError, match="token"):
            judge.annotate(**REQUEST)


@pytest.mark.parametrize("case", ["no_output", "tool", "unfinished", "wrong_role", "error", "id"])
def test_rejects_unexpected_response(sdk: SimpleNamespace, case: str) -> None:
    result = response()
    if case == "no_output":
        result.output = []
    elif case == "tool":
        result.output = [SimpleNamespace(type="function_call")]
    elif case == "unfinished":
        result.output[0].status = "in_progress"
    elif case == "wrong_role":
        result.output[0].role = "user"
    elif case == "error":
        result.error = SimpleNamespace(code="server_error")
    else:
        result.id = ""
    sdk.client.responses.create.return_value = result
    with AzureResponsesJudge(config()) as judge:
        with pytest.raises(JudgeResponseError):
            judge.annotate(**REQUEST)


@pytest.mark.parametrize("model", ["gpt-5.6-sol", "gpt-5.6-sol-2026-07-09"])
def test_reasoning_metadata_is_not_an_annotation(sdk: SimpleNamespace, model: str) -> None:
    result = response(model=model)
    result.output.insert(0, SimpleNamespace(type="reasoning", summary=["DO_NOT_PERSIST"]))
    sdk.client.responses.create.return_value = result
    with AzureResponsesJudge(config(model="gpt-5.6-sol", deployment="gpt-5.6-sol")) as judge:
        actual = judge.annotate(**REQUEST)
    assert actual.payload == {"label": "synthetic"}
    assert "DO_NOT_PERSIST" not in repr(actual)
    assert actual.output_tokens == 7
    assert sdk.client.responses.create.call_args.kwargs["model"] == "gpt-5.6-sol"


def test_reasoning_alone_cannot_replace_a_visible_answer(sdk: SimpleNamespace) -> None:
    sdk.client.responses.create.return_value = response(
        model="gpt-5.6-sol", output=[SimpleNamespace(type="reasoning")],
    )
    with AzureResponsesJudge(config(model="gpt-5.6-sol")) as judge:
        with pytest.raises(JudgeResponseError, match="empty"):
            judge.annotate(**REQUEST)


@pytest.mark.parametrize("changes", [
    {"system_prompt": " "}, {"input_payload": {}}, {"input_payload": {"x": float("nan")}},
    {"schema": {}}, {"schema": {"type": "array"}},
    {"schema": {**SCHEMA, "additionalProperties": True}},
    {"schema": {**SCHEMA, "required": []}},
    {"schema": {**SCHEMA, "properties": {"label": {"type": "object"}}}},
])
def test_invalid_requests_do_not_authenticate(sdk: SimpleNamespace, changes: dict) -> None:
    with AzureResponsesJudge(config()) as judge:
        with pytest.raises(ValueError):
            judge.annotate(**{**REQUEST, **changes})
    sdk.credential_factory.assert_not_called()
    sdk.client.responses.create.assert_not_called()


def test_errors_propagate_without_retry_and_resources_close(sdk: SimpleNamespace) -> None:
    error = TimeoutError("synthetic timeout")
    sdk.client.responses.create.side_effect = error
    with pytest.raises(TimeoutError) as caught:
        with AzureResponsesJudge(config()) as judge:
            judge.annotate(**REQUEST)
    assert caught.value is error
    sdk.client.responses.create.assert_called_once()
    sdk.client.close.assert_called_once()
    sdk.credential.close.assert_called_once()


def test_partial_setup_failure_closes_resources(sdk: SimpleNamespace) -> None:
    sdk.client_factory.side_effect = ValueError("synthetic constructor failure")
    with AzureResponsesJudge(config()) as judge:
        with pytest.raises(ValueError, match="constructor"):
            judge.annotate(**REQUEST)
    sdk.credential.close.assert_called_once()
    sdk.http_client.close.assert_called_once()


@pytest.mark.parametrize("status", [200, 307, 429, 500])
@pytest.mark.parametrize("model", ["gpt-4.1", "gpt-5.6-sol"])
def test_real_sdk_transport_has_no_redirects_or_retries(
    monkeypatch: pytest.MonkeyPatch, status: int, model: str,
) -> None:
    """Exercise SDK serialization and HTTP policy with a local in-memory transport."""
    seen: list[httpx2.Request] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        seen.append(request)
        body = {
            "id": "resp_offline", "object": "response", "created_at": 0,
            "status": "completed", "model": model, "error": None,
            "incomplete_details": None, "instructions": None, "metadata": {},
            "parallel_tool_calls": False, "temperature": 1, "top_p": 1,
            "tool_choice": "none", "tools": [],
            "output": [{"id": "msg_offline", "type": "message", "role": "assistant", "status": "completed",
                        "content": [{"type": "output_text", "text": '{"label":"synthetic"}', "annotations": []}]}],
            "usage": {"input_tokens": 25, "output_tokens": 7, "total_tokens": 32,
                      "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": 0}},
        }
        return httpx2.Response(
            status, json=body if status == 200 else {"error": {"message": "synthetic"}},
            headers={"location": "https://untrusted.invalid/redirect"},
        )

    credential = Mock()
    monkeypatch.setattr(azure.identity, "AzureCliCredential", Mock(return_value=credential))
    provider = Mock(return_value="synthetic-entra-token")
    monkeypatch.setattr(azure.identity, "get_bearer_token_provider", Mock(return_value=provider))
    monkeypatch.setenv("OPENAI_API_KEY", "DO_NOT_USE")
    monkeypatch.setenv("OPENAI_ORG_ID", "DO_NOT_USE")
    monkeypatch.setenv("OPENAI_PROJECT_ID", "DO_NOT_USE")
    wrapper = openai.DefaultHttpxClient
    transports: list[openai.DefaultHttpxClient] = []

    def transport_factory(**kwargs: object) -> openai.DefaultHttpxClient:
        client = wrapper(transport=httpx2.MockTransport(handler), **kwargs)
        transports.append(client)
        return client

    monkeypatch.setattr(openai, "DefaultHttpxClient", transport_factory)
    with AzureResponsesJudge(config(model=model, deployment=model)) as judge:
        if status == 200:
            assert judge.annotate(**REQUEST).response_id == "resp_offline"
        else:
            with pytest.raises(openai.APIStatusError):
                judge.annotate(**REQUEST)
    assert len(seen) == 1
    assert str(seen[0].url) == ENDPOINT + "responses"
    assert seen[0].headers["authorization"] == "Bearer synthetic-entra-token"
    assert "api-key" not in seen[0].headers
    assert "DO_NOT_USE" not in str(seen[0].headers)
    assert not transports[0].follow_redirects and transports[0].is_closed
    body = json.loads(seen[0].content)
    assert body["model"] == model
    assert body["store"] is False and body["tools"] == []
    assert body["text"]["format"]["strict"] is True
    assert not {"previous_response_id", "conversation", "metadata"} & body.keys()
    credential.close.assert_called_once()
