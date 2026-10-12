"""Offline tests for the local Azure Clef client; no real tokens or HTTP requests."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

pytest.importorskip("azure.identity", reason="Install the endpoint extra to test remote clients")
pytest.importorskip("requests", reason="Install the endpoint extra to test remote clients")

from routing_lab.cli import create_parser, execute
from routing_lab.data import ROUTES
from routing_lab import clef
from routing_lab.endpoint import TOKEN_SCOPE, ClefEndpointBackend
from routing_lab.data import Contract
from routing_lab.taxonomy import TAXONOMIES, taxonomy_routes

URL = "https://clef-routing-exp.eastus2.inference.ml.azure.com/score"


@pytest.fixture
def remote(monkeypatch):
    credential = Mock()
    credential.get_token.return_value = SimpleNamespace(token="test-token-not-real", expires_on=9999999999)
    monkeypatch.setattr("azure.identity.AzureCliCredential", Mock(return_value=credential))
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = 200
    payload = {
        "experiment_model": clef.model_info("flash"),
        "answers": {"route": {"choice": "repository_qa", "probabilities": {
            route: float(route == "repository_qa") for route in ROUTES
        }}},
        "usage": {"input_tokens": 32, "output_tokens": 0, "state_tokens_dropped": 3},
    }
    response.json.return_value = payload
    post = Mock(return_value=response)
    monkeypatch.setattr("requests.Session.post", post)
    return credential, post, response, payload


def test_native_endpoint_auth_contract_and_no_labels(remote, contracts, make_record):
    credential, post, _, _ = remote
    backend = ClefEndpointBackend(contracts, url=URL)
    record = make_record()
    result = backend.predict(record)
    assert result.route == "repository_qa"
    assert result.state_tokens_dropped == 3
    credential.get_token.assert_called_once_with(TOKEN_SCOPE)
    kwargs = post.call_args.kwargs
    assert kwargs["allow_redirects"] is False
    assert kwargs["timeout"] == (10, 190)
    assert kwargs["headers"]["Authorization"] == "Bearer test-token-not-real"
    assert set(kwargs["json"]) == {"state", "questions"}
    assert list(kwargs["json"]["questions"]["route"]["criteria"]) == list(ROUTES)
    assert "expected_route" not in kwargs["json"]
    assert "token" not in str(backend.metadata())


def test_predict_state_sends_the_given_text_verbatim(remote, contracts):
    _, post, _, _ = remote
    result = ClefEndpointBackend(contracts, url=URL).predict_state("Step: summarise the incident.")
    assert result.route == "repository_qa"
    assert post.call_args.kwargs["json"]["state"] == "Step: summarise the incident."


def test_max_length_omitted_keeps_body_unchanged(remote, contracts, make_record):
    _, post, _, _ = remote
    backend = ClefEndpointBackend(contracts, url=URL)
    backend.predict(make_record())
    assert set(post.call_args.kwargs["json"]) == {"state", "questions"}
    backend.close()


def test_max_length_override_is_included_in_request_body_when_set(remote, contracts, make_record):
    _, post, _, _ = remote
    backend = ClefEndpointBackend(contracts, url=URL, max_length=4096)
    backend.predict(make_record())
    assert post.call_args.kwargs["json"]["max_length"] == 4096
    assert backend.metadata()["max_length"] == 4096
    backend.close()


def test_connection_is_reused_across_requests_not_reopened_per_call(remote, contracts, make_record):
    """One persistent Session should back every predict() call, avoiding a fresh
    TCP/TLS handshake per request (previously measured as several seconds of
    client-only overhead versus the server's own sub-second processing time)."""
    import requests

    _, post, _, _ = remote
    backend = ClefEndpointBackend(contracts, url=URL)
    session_before = backend._session
    assert isinstance(session_before, requests.Session)
    backend.predict(make_record())
    backend.predict(make_record())
    assert backend._session is session_before
    assert post.call_count == 2
    backend.close()


def test_token_is_cached_across_calls_not_refetched_every_time(remote, contracts, make_record):
    """AzureCliCredential.get_token() shells out to a fresh `az` subprocess per call
    (measured at ~2-2.8s each); refetching on every predict() previously dominated
    per-request latency far more than network transit or model inference time."""
    credential, _, _, _ = remote
    backend = ClefEndpointBackend(contracts, url=URL)
    backend.predict(make_record())
    backend.predict(make_record())
    backend.predict(make_record())
    credential.get_token.assert_called_once_with(TOKEN_SCOPE)
    backend.close()


def test_token_is_refetched_once_actually_near_expiry(remote, contracts, make_record):
    credential, _, _, _ = remote
    backend = ClefEndpointBackend(contracts, url=URL)
    backend.predict(make_record())
    assert credential.get_token.call_count == 1
    backend._cached_token = SimpleNamespace(token="stale-token", expires_on=0)
    backend.predict(make_record())
    assert credential.get_token.call_count == 2
    backend.close()


@pytest.mark.parametrize("url", [
    "http://clef.eastus2.inference.ml.azure.com/score",
    "https://example.com/score",
    "https://clef.eastus2.inference.ml.azure.com.attacker.test/score",
    "https://user:secret@clef.eastus2.inference.ml.azure.com/score",
    URL + "?token=secret", URL + "#fragment", URL.replace("/score", "/other"),
    URL.replace("/score", ":8443/score"),
])
def test_reject_untrusted_urls(remote, contracts, url):
    with pytest.raises(ValueError, match="scoring_uri"):
        ClefEndpointBackend(contracts, url=url)
    remote[1].assert_not_called()


@pytest.mark.parametrize("status", [301, 302, 401, 403, 429, 500, 503])
def test_http_failure_not_hidden_or_retried(remote, contracts, make_record, status):
    _, post, response, _ = remote
    response.status_code = status
    response.text = "private source text must not leak into exception"
    with pytest.raises(RuntimeError, match=f"HTTP {status}") as error:
        ClefEndpointBackend(contracts, url=URL).predict(make_record())
    assert "private" not in str(error.value)
    post.assert_called_once()


def test_gateway_424_is_retried_and_succeeds(remote, contracts, make_record):
    """Observed live: the managed-endpoint gateway intermittently returns 424 while
    the scoring container's own access log shows 200 for the same request; an
    immediate retry of the identical request always succeeded in practice."""
    _, post, response, payload = remote
    failing = MagicMock()
    failing.__enter__.return_value = failing
    failing.status_code = 424
    failing.text = "gateway hiccup"
    post.side_effect = [failing, failing, response]
    backend = ClefEndpointBackend(contracts, url=URL)
    backend._sleep = Mock()
    result = backend.predict(make_record())
    assert result.route == "repository_qa"
    assert post.call_count == 3
    backend._sleep.assert_called()
    backend.close()


def test_gateway_424_gives_up_after_max_retries(remote, contracts, make_record):
    _, post, response, _ = remote
    failing = MagicMock()
    failing.__enter__.return_value = failing
    failing.status_code = 424
    failing.text = "gateway hiccup"
    post.side_effect = [failing, failing, failing, failing]
    backend = ClefEndpointBackend(contracts, url=URL, max_gateway_retries=3)
    backend._sleep = Mock()
    with pytest.raises(RuntimeError, match="HTTP 424"):
        backend.predict(make_record())
    assert post.call_count == 4
    backend.close()


def test_reject_different_revision(remote, contracts, make_record):
    remote[3]["experiment_model"]["revision"] = "unexpected-checkpoint"
    with pytest.raises(ValueError, match="provenance"):
        ClefEndpointBackend(contracts, url=URL).predict(make_record())


def test_reject_missing_usage(remote, contracts, make_record):
    remote[3].pop("usage")
    with pytest.raises(ValueError, match="contract"):
        ClefEndpointBackend(contracts, url=URL).predict(make_record())


def test_reject_missing_probabilities(remote, contracts, make_record):
    remote[3]["answers"]["route"]["probabilities"] = None
    with pytest.raises(ValueError, match="probabilities"):
        ClefEndpointBackend(contracts, url=URL).predict(make_record())


def test_auth_failure_propagates_without_http(remote, contracts, make_record):
    remote[0].get_token.side_effect = RuntimeError("Login required")
    with pytest.raises(RuntimeError, match="Login required"):
        ClefEndpointBackend(contracts, url=URL).predict(make_record())
    remote[1].assert_not_called()


def test_cli_remote_gate_precedes_client_creation(tmp_path, write_dataset, make_record):
    data = write_dataset([make_record()])
    args = create_parser().parse_args([
        "run", "--backend", "clef-endpoint", "--data", str(data),
        "--output", str(tmp_path / "run"),
    ])
    with pytest.raises(ValueError, match="allow-remote"):
        execute(args)


def test_variant_selects_the_expected_provenance(remote, contracts, make_record):
    with pytest.raises(ValueError, match="provenance"):
        ClefEndpointBackend(contracts, url=URL, variant="clef").predict(make_record())
    assert ClefEndpointBackend(contracts, url=URL, variant="flash").predict(make_record()).route


@pytest.mark.parametrize("taxonomy", TAXONOMIES)
def test_endpoint_selected_taxonomy_and_mismatch_before_network(
    remote, contracts, make_record, taxonomy,
):
    routes = taxonomy_routes(taxonomy)
    catalog = {route: Contract(**next(iter(contracts.values())).model_dump()) for route in routes}
    record = make_record(taxonomy_id=taxonomy, expected_route=routes[0])
    remote[3]["answers"]["route"] = {
        "choice": routes[0], "probabilities": {route: float(route == routes[0]) for route in routes},
    }
    backend = ClefEndpointBackend(catalog, url=URL, taxonomy_id=taxonomy)
    decision = backend.predict(record)
    assert decision.taxonomy_id == taxonomy
    assert decision.route == routes[0]
    assert set(remote[1].call_args.kwargs["json"]) == {"state", "questions"}
    assert list(remote[1].call_args.kwargs["json"]["questions"]["route"]["criteria"]) == list(routes)
    remote[0].get_token.reset_mock()
    remote[1].reset_mock()
    other = next(key for key in TAXONOMIES if key != taxonomy)
    with pytest.raises(ValueError, match="taxonomy"):
        backend.predict(make_record(taxonomy_id=other, expected_route=taxonomy_routes(other)[0]))
    remote[0].get_token.assert_not_called()
    remote[1].assert_not_called()
    remote[3]["answers"]["route"]["probabilities"] = {"unknown": 1.0}
    with pytest.raises(ValueError, match="Probabilities"):
        backend.predict(record)


ADAPTER = "a" * 64


def tuned_payload(payload, adapter=ADAPTER):
    payload["experiment_model"] = {**clef.model_info("flash"), "fine_tuned": {"adapter_sha256": adapter}}


def test_deployment_header_targets_a_named_deployment_only_when_asked(remote, contracts, make_record):
    _, post, _, _ = remote
    plain = ClefEndpointBackend(contracts, url=URL)
    plain.predict(make_record())
    assert "azureml-model-deployment" not in post.call_args.kwargs["headers"]
    targeted = ClefEndpointBackend(contracts, url=URL, deployment="flash-ft-gpu")
    targeted.predict(make_record())
    assert post.call_args.kwargs["headers"]["azureml-model-deployment"] == "flash-ft-gpu"
    assert targeted.metadata()["deployment"] == "flash-ft-gpu"


@pytest.mark.parametrize("deployment", ["", "AB", "has space", "x" * 40, "under_score"])
def test_deployment_names_are_validated(remote, contracts, deployment):
    with pytest.raises(ValueError, match="Deployment names"):
        ClefEndpointBackend(contracts, url=URL, deployment=deployment)


def test_pinned_adapter_accepts_only_the_matching_finetuned_response(remote, contracts, make_record):
    backend = ClefEndpointBackend(contracts, url=URL, adapter_sha256=ADAPTER)
    tuned_payload(remote[3])
    assert backend.predict(make_record()).route == "repository_qa"
    assert "adapter hash" in backend.metadata()["provenance_policy"]
    tuned_payload(remote[3], adapter="c" * 64)
    with pytest.raises(ValueError, match="provenance"):
        backend.predict(make_record())
    remote[3]["experiment_model"] = clef.model_info("flash")
    with pytest.raises(ValueError, match="provenance"):
        backend.predict(make_record())


def test_unpinned_client_refuses_a_finetuned_response(remote, contracts, make_record):
    tuned_payload(remote[3])
    with pytest.raises(ValueError, match="provenance"):
        ClefEndpointBackend(contracts, url=URL).predict(make_record())


def test_pinned_adapter_must_be_a_sha256_digest(remote, contracts):
    with pytest.raises(ValueError, match="SHA-256"):
        ClefEndpointBackend(contracts, url=URL, adapter_sha256="not-a-hash")


def test_cli_deployment_flags_are_endpoint_only(tmp_path, write_dataset, make_record):
    data = write_dataset([make_record()])
    args = create_parser().parse_args([
        "run", "--backend", "rules", "--deployment", "flash-ft-gpu", "--data", str(data),
        "--output", str(tmp_path / "run")])
    with pytest.raises(ValueError, match="clef-endpoint backend"):
        execute(args)

LOCAL = "http://clef-server:5001/score"


def test_a_local_container_url_is_refused_unless_local_mode_is_asked_for(remote, contracts):
    with pytest.raises(ValueError):
        ClefEndpointBackend(contracts, url=LOCAL)
    for bad in ("https://clef-server:5001/score", "http://clef-server/score", "http://clef-server:5001/other",
                "http://user:pw@clef-server:5001/score", "http://clef-server:5001/score?x=1", "http://CLEF-SERVER:5001/score#f",
                "http://169.254.169.254:80/score"):
        with pytest.raises(ValueError):
            ClefEndpointBackend(contracts, url=bad, local=True)
    assert ClefEndpointBackend(contracts, url=LOCAL, local=True).url == LOCAL
    with pytest.raises(ValueError):  # local mode does not widen the Azure URL rules either way
        ClefEndpointBackend(contracts, url=URL, local=True)


def test_local_mode_asks_for_no_identity_and_sends_no_token(remote, contracts, make_record):
    credential, post, _, _ = remote
    backend = ClefEndpointBackend(contracts, url=LOCAL, local=True)
    assert backend.credential is None
    assert backend.predict(make_record()).route == "repository_qa"
    credential.get_token.assert_not_called()
    headers = post.call_args.kwargs["headers"]
    assert "Authorization" not in headers and headers["Content-Type"] == "application/json"
    assert backend.metadata()["auth"] == "none (local container)"


def test_a_wrong_model_from_a_local_container_is_still_refused(remote, contracts, make_record):
    _, _, _, payload = remote
    payload["experiment_model"] = {**clef.model_info("flash"), "revision": "other"}
    with pytest.raises(ValueError):
        ClefEndpointBackend(contracts, url=LOCAL, local=True).predict(make_record())


def test_the_azure_identity_mode_is_chosen_by_environment(monkeypatch):
    import azure.identity as identity

    from routing_lab import azure_auth

    built = {}
    for name in ("AzureCliCredential", "DefaultAzureCredential", "ManagedIdentityCredential"):
        monkeypatch.setattr(identity, name, lambda *a, _n=name, **k: built.setdefault("used", (_n, k)))
    monkeypatch.delenv(azure_auth.AUTH_ENV, raising=False)
    azure_auth.make_credential()
    assert built.pop("used")[0] == "AzureCliCredential" and azure_auth.credential_name() == "AzureCliCredential"
    monkeypatch.setenv(azure_auth.AUTH_ENV, "default")
    azure_auth.make_credential()
    name, kwargs = built.pop("used")
    assert name == "DefaultAzureCredential" and kwargs["exclude_interactive_browser_credential"] is True
    monkeypatch.setenv(azure_auth.AUTH_ENV, "managed-identity")
    monkeypatch.setenv("AZURE_CLIENT_ID", "client-1")
    azure_auth.make_credential()
    assert built.pop("used") == ("ManagedIdentityCredential", {"client_id": "client-1"})
    monkeypatch.setenv(azure_auth.AUTH_ENV, "browser")
    with pytest.raises(ValueError):
        azure_auth.make_credential()
