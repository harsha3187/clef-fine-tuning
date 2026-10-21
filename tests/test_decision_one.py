from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import requests

from routing_lab.decision_one import API_PATH, TOKEN_SCOPE, DecisionOneRouter


def payload():
    return {
        "model": "microsoft-decision-1",
        "answers": {
            "route": {"type": "choice", "choice": "medium", "probabilities": {"low": 0.2, "medium": 0.7, "high": 0.1}},
            "skill": {"type": "choice", "choice": "tools",
                      "probabilities": {"coding": 0.1, "reasoning": 0.1, "tools": 0.6, "writing": 0.1, "extraction": 0.1}},
        },
        "usage": {"input_tokens": 1000, "output_tokens": 2},
    }


def client():
    session = Mock()
    session.post.return_value = SimpleNamespace(status_code=200, json=payload, headers={})
    credential = Mock()
    credential.get_token.return_value = SimpleNamespace(token="test-token", expires_on=9999999999)
    router = DecisionOneRouter("https://r.services.ai.azure.com/openai/v1/", "decision-1",
                               credential=credential, session=session)
    return router, session, credential


def test_native_contract_reuses_the_same_questions_and_caches_credentials():
    router, session, credential = client()
    request = router.request_payload("Pick a hotel.", "Four guests.")
    session.post.assert_not_called()
    first = router.classify("Pick a hotel.", "Four guests.")
    router.classify("Another task.")
    call = session.post.call_args_list[0]
    assert call.args[0] == "https://r.services.ai.azure.com" + API_PATH
    assert call.kwargs["json"] == request
    assert request["state"] == "User request:\nPick a hotel.\nAvailable context:\nFour guests."
    assert request["model"] == "decision-1" and set(request["questions"]) == {"route", "skill"}
    assert "smallest tier" in request["questions"]["route"]["instructions"]
    assert call.kwargs["allow_redirects"] is False and "headers" not in request
    credential.get_token.assert_called_once_with(TOKEN_SCOPE)
    assert first.tier == "medium" and first.skill == "tools" and first.confidence == 0.7
    assert first.cost_usd == pytest.approx(1002 * 0.042 / 1e6)
    assert first.input_tokens == 1000 and first.output_tokens == 2


@pytest.mark.parametrize("url", [
    "http://r.services.ai.azure.com", "https://evil.example", "https://r.services.ai.azure.com.evil.example",
    "https://r.services.ai.azure.com?key=x", "https://r.services.ai.azure.com#fragment",
    "https://user:password@r.services.ai.azure.com", "https://r.services.ai.azure.com:444",
    "https://r.services.ai.azure.com/other",
])
def test_invalid_urls_are_rejected_before_credentials_or_network(url):
    with pytest.raises(ValueError, match="HTTPS Azure"):
        DecisionOneRouter(url)


@pytest.mark.parametrize("case", ["wrong_model", "missing_skill", "unknown_option", "bad_sum", "nan", "bool", "wrong_choice", "missing_usage"])
def test_malformed_decisions_are_not_converted_into_fake_probabilities(case):
    router, session, _ = client()
    data = deepcopy(payload())
    if case == "wrong_model":
        data["model"] = "different-model"
    elif case == "missing_skill":
        del data["answers"]["skill"]
    elif case == "unknown_option":
        data["answers"]["route"]["probabilities"]["extra"] = 0.0
    elif case == "bad_sum":
        data["answers"]["route"]["probabilities"]["low"] = 0.5
    elif case == "nan":
        data["answers"]["route"]["probabilities"]["low"] = float("nan")
    elif case == "bool":
        data["answers"]["route"]["probabilities"]["low"] = True
    elif case == "wrong_choice":
        data["answers"]["route"]["choice"] = "low"
    else:
        del data["usage"]["output_tokens"]
    session.post.return_value.json = lambda: data
    with pytest.raises(ValueError):
        router.classify("x")
    assert session.post.call_count == 1


def test_transient_failure_retries_identical_body_but_auth_failures_do_not(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr("routing_lab.decision_one.time.sleep", sleep)
    router, session, _ = client()
    session.post.side_effect = [
        SimpleNamespace(status_code=408, headers={}),
        SimpleNamespace(status_code=429, headers={"Retry-After": "2"}),
        SimpleNamespace(status_code=200, json=payload),
    ]
    assert router.classify("x").tier == "medium"
    assert session.post.call_count == 3
    assert all(c.kwargs["json"] == session.post.call_args.kwargs["json"] for c in session.post.call_args_list)
    assert sleep.call_args_list[-1].args == (2.0,)
    session.post.reset_mock()
    session.post.side_effect = [SimpleNamespace(status_code=401)]
    with pytest.raises(RuntimeError, match="401"):
        router.classify("x")
    assert session.post.call_count == 1


def test_connection_retries_are_bounded(monkeypatch):
    monkeypatch.setattr("routing_lab.decision_one.time.sleep", lambda _: None)
    router, session, _ = client()
    session.post.side_effect = requests.Timeout("timeout")
    with pytest.raises(requests.Timeout):
        router.classify("x")
    assert session.post.call_count == 3
