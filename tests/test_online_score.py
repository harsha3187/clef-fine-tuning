"""Offline tests for the standalone Azure scorer; never import or execute Clef."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import subprocess
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import pytest

from routing_lab import clef
from routing_lab.backends import questions
from routing_lab.data import ROUTES, Contract
from routing_lab.taxonomy import SKILLS, TAXONOMIES, taxonomy_routes

SCORE_PATH = Path(__file__).resolve().parents[1] / "azure" / "online" / "code" / "score.py"


class FakeResponse:
    """Match the narrow AMLResponse constructor without installing Azure serving."""

    def __init__(self, message: dict[str, Any], status_code: int, *, json_str: bool) -> None:
        assert json_str is True
        self.body = message
        self.status_code = status_code


@pytest.fixture
def scorer(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    for name in (
        "azureml_inference_server_http",
        "azureml_inference_server_http.api",
        "azureml_inference_server_http.api.aml_response",
    ):
        module = ModuleType(name)
        module.AMLResponse = FakeResponse
        monkeypatch.setitem(sys.modules, name, module)
    spec = importlib.util.spec_from_file_location("online_score_test", SCORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    module.decide = Mock()
    monkeypatch.setattr(module.clef, "decide", module.decide)
    return module


@pytest.fixture
def payload(contracts: dict[str, Contract]) -> dict[str, Any]:
    return {"state": "Explain the synthetic scheduler.", "questions": questions(contracts)}


@pytest.fixture
def native() -> dict[str, Any]:
    return {
        "answers": {"route": {
            "choice": "repository_qa",
            "probabilities": {route: float(route == "repository_qa") for route in ROUTES},
            "confidence": 1.0,
        }},
        "usage": {
            "input_tokens": 32, "output_tokens": 0, "state_tokens_dropped": 4,
        },
    }


def test_init_is_pinned_gpu_and_idempotent(scorer: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    load = Mock(return_value=Mock())
    monkeypatch.setattr(scorer.clef, "load", load)
    assert scorer._runtime is None
    scorer.init()
    scorer.init()
    load.assert_called_once_with("flash", device="cuda")


def test_azure_loader_without_module_registration(payload: dict[str, Any]) -> None:
    spec = importlib.util.spec_from_file_location("azure_unregistered_entry", SCORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.name not in sys.modules
    spec.loader.exec_module(module)
    validated = module.Request.model_validate(payload)
    assert validated.state == payload["state"]
    assert validated.questions.route.type == "choice"


def test_native_response_and_metadata(
    scorer: ModuleType, payload: dict[str, Any], native: dict[str, Any],
) -> None:
    supplied = deepcopy(native)
    supplied["experiment_model"] = {"revision": "untrusted"}
    scorer._runtime = Mock()
    scorer.decide.return_value = supplied
    result = scorer.run(json.dumps(payload))
    scorer.decide.assert_called_once_with(
        scorer._runtime, payload["state"], payload["questions"], max_length=clef.DEFAULT_MAX_LENGTH,
    )
    assert result["answers"] == native["answers"]
    assert result["usage"] == native["usage"]
    assert result["experiment_model"] == clef.model_info("flash")
    assert result["experiment_model"]["revision"] == clef.RELEASES["flash"].revision
    result["experiment_model"]["revision"] = "client mutation"
    assert scorer.MODEL["revision"] == clef.RELEASES["flash"].revision
    with pytest.raises(TypeError):
        scorer.MODEL["revision"] = "mutation"


def test_criteria_order_is_preserved(
    scorer: ModuleType, payload: dict[str, Any], native: dict[str, Any],
) -> None:
    criteria = payload["questions"]["route"]["criteria"]
    payload["questions"]["route"]["criteria"] = dict(reversed(list(criteria.items())))
    scorer._runtime = Mock()
    scorer.decide.return_value = native
    scorer.run(json.dumps(payload))
    actual = scorer.decide.call_args.args[2]["route"]["criteria"]
    assert list(actual) == list(reversed(ROUTES))


def test_local_client_matches_real_scorer_contract(
    scorer, native, contracts, make_record, monkeypatch,
) -> None:
    pytest.importorskip("azure.identity")
    pytest.importorskip("requests")
    from routing_lab.endpoint import ClefEndpointBackend

    credential = Mock()
    credential.get_token.return_value = SimpleNamespace(token="fake-token")
    monkeypatch.setattr("azure.identity.AzureCliCredential", Mock(return_value=credential))
    scorer._runtime = Mock()
    scorer.decide.return_value = native

    def send(url, **kwargs):
        reply = scorer.run(json.dumps(kwargs["json"]))
        assert isinstance(reply, dict), reply
        response = MagicMock()
        response.__enter__.return_value = response
        response.status_code = 200
        response.json.return_value = reply
        return response

    # A plain function assigned to Session.post would be bound (self injected) via the
    # descriptor protocol; wrap in a Mock so it is called with exactly (url, **kwargs),
    # matching the persistent-session client's actual self._session.post(...) call.
    monkeypatch.setattr("requests.Session.post", Mock(side_effect=send))
    backend = ClefEndpointBackend(
        contracts, url="https://example-endpoint.eastus2.inference.ml.azure.com/score",
    )
    decision = backend.predict(make_record())
    assert decision.route == "repository_qa"
    assert decision.probabilities == native["answers"]["route"]["probabilities"]
    assert decision.state_tokens_dropped == 4


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(state=""),
    lambda p: p.update(state=" \n\t"),
    lambda p: p.update(state=42),
    lambda p: p.update(state=["batch"]),
    lambda p: p.update(revision="override"),
    lambda p: p.update(experiment_model={"variant": "clef"}),
    lambda p: p.pop("questions"),
    lambda p: p["questions"].update(other=p["questions"]["route"]),
    lambda p: p["questions"]["route"].update(type="score"),
    lambda p: p["questions"]["route"].update(extra="not allowed"),
    lambda p: p["questions"]["route"].update(instructions=" "),
    lambda p: p["questions"]["route"].update(instructions="x" * 4_097),
    lambda p: p["questions"]["route"].update(criteria=[]),
    lambda p: p["questions"]["route"]["criteria"].pop("fallback"),
    lambda p: p["questions"]["route"]["criteria"].update(unknown="other"),
    lambda p: p["questions"]["route"]["criteria"].update(fallback=""),
    lambda p: p["questions"]["route"]["criteria"].update(fallback=" "),
    lambda p: p["questions"]["route"]["criteria"].update(fallback=1),
    lambda p: p["questions"]["route"]["criteria"].update(fallback="x" * 2_049),
])
def test_invalid_requests_never_call_model(
    scorer: ModuleType, payload: dict[str, Any], mutation: Any, caplog: pytest.LogCaptureFixture,
) -> None:
    scorer._runtime = Mock()
    mutation(payload)
    result = scorer.run(json.dumps(payload))
    assert result.status_code == 400
    assert result.body["error"]["code"] == "invalid_request"
    scorer.decide.assert_not_called()
    assert caplog.text == ""


@pytest.mark.parametrize("raw", ["not json PRIVATE", "[]", "null", "42", "{}", None, {}, '"\\ud800"'])
def test_invalid_json_or_top_level(scorer: ModuleType, raw: Any) -> None:
    result = scorer.run(raw)
    assert result.status_code == 400
    assert "PRIVATE" not in json.dumps(result.body)


def test_duplicate_keys_rejected(scorer: ModuleType, payload: dict[str, Any]) -> None:
    raw = json.dumps(payload).replace('"state":', '"state": "other", "state":', 1)
    assert scorer.run(raw).status_code == 400


def test_wire_size_bound(scorer: ModuleType) -> None:
    assert scorer.run(" " * (scorer.MAX_REQUEST_BYTES + 1)).status_code == 413
    assert scorer.run("é" * (scorer.MAX_REQUEST_BYTES // 2 + 1)).status_code == 413


def test_state_over_max_chars_is_rejected(scorer: ModuleType, payload: dict[str, Any]) -> None:
    payload["state"] = "x" * (scorer.MAX_STATE_CHARS + 1)
    scorer._runtime = Mock()
    result = scorer.run(json.dumps(payload))
    assert result.status_code == 400
    scorer.decide.assert_not_called()


def test_valid_boundary_state(
    scorer: ModuleType, payload: dict[str, Any], native: dict[str, Any],
) -> None:
    payload["state"] = "x" * scorer.MAX_STATE_CHARS
    scorer._runtime = Mock()
    scorer.decide.return_value = native
    assert scorer.run(json.dumps(payload))["usage"] == native["usage"]


def test_max_length_override_is_passed_through_when_set(
    scorer: ModuleType, payload: dict[str, Any], native: dict[str, Any],
) -> None:
    payload["max_length"] = 4096
    scorer._runtime = Mock()
    scorer.decide.return_value = native
    scorer.run(json.dumps(payload))
    scorer.decide.assert_called_once_with(
        scorer._runtime, payload["state"], payload["questions"], max_length=4096,
    )


@pytest.mark.parametrize("mutation", [
    lambda p: p.update(max_length=511),
    lambda p: p.update(max_length=clef.MAX_LENGTH_CEILING + 1),
    lambda p: p.update(max_length="4096"),
    lambda p: p.update(max_state_tokens=100),
    lambda p: p.update(images=["x"]),
])
def test_max_length_out_of_bounds_and_media_are_rejected(
    scorer: ModuleType, payload: dict[str, Any], mutation: Any,
) -> None:
    scorer._runtime = Mock()
    mutation(payload)
    result = scorer.run(json.dumps(payload))
    assert result.status_code == 400
    scorer.decide.assert_not_called()


def test_uninitialized_does_not_lazy_load(scorer: ModuleType, payload: dict[str, Any]) -> None:
    assert scorer.run(json.dumps(payload)).status_code == 503
    assert scorer._runtime is None


def test_model_errors_are_sanitized(
    scorer: ModuleType, payload: dict[str, Any], caplog: pytest.LogCaptureFixture,
) -> None:
    scorer._runtime = Mock()
    scorer.decide.side_effect = RuntimeError("PRIVATE request content")
    result = scorer.run(json.dumps(payload))
    assert result.status_code == 500
    assert "PRIVATE" not in json.dumps(result.body)
    assert caplog.text == ""


def test_serialization_errors_are_sanitized(scorer: ModuleType, payload: dict[str, Any]) -> None:
    scorer._runtime = Mock()
    scorer.decide.return_value = {"answers": {}, "usage": {"bad": float("nan")}}
    assert scorer.run(json.dumps(payload)).status_code == 500


def test_taxonomy_and_upload_scope(scorer: ModuleType) -> None:
    assert scorer.ROUTES == ROUTES
    deployment = SCORE_PATH.parent.parent / "deployment.yml"
    text = deployment.read_text(encoding="utf-8")
    assert "  code: code\n" in text
    assert "  scoring_script: score.py\n" in text
    assert "instance_type: Standard_NC40ads_H100_v5\n" in text
    assert "instance_count: 1\n" in text
    assert "  max_concurrent_requests_per_instance: 1\n" in text
    assert "app_insights_enabled: false\n" in text


@pytest.mark.parametrize("taxonomy", TAXONOMIES)
def test_bounded_exact_catalogs_and_selected_model_distribution(scorer, taxonomy):
    routes = taxonomy_routes(taxonomy)
    payload = {"state": "Synthetic input", "questions": {"route": {
        "type": "choice", "instructions": "Pick one owner",
        "criteria": {route: f"Own {route}" for route in routes},
    }}}
    native = {
        "answers": {"route": {"choice": routes[0], "probabilities": {
            route: float(route == routes[0]) for route in routes
        }}},
        "usage": {"input_tokens": 1, "output_tokens": 1, "state_tokens_dropped": 0},
    }
    scorer._runtime = Mock()
    scorer.decide.return_value = native
    assert scorer.run(json.dumps(payload))["answers"] == native["answers"]
    scorer.decide.assert_called_once()
    scorer.decide.reset_mock()
    for mutation in ("missing", "unknown", "union"):
        bad = deepcopy(payload)
        criteria = bad["questions"]["route"]["criteria"]
        if mutation == "missing":
            criteria.pop(routes[0])
        elif mutation == "unknown":
            criteria["unknown"] = criteria.pop(routes[0])
        else:
            criteria.update({route: route for key in TAXONOMIES for route in taxonomy_routes(key)})
        assert scorer.run(json.dumps(bad)).status_code == 400
    scorer.decide.assert_not_called()
    native["answers"]["route"]["choice"] = "unknown"
    assert scorer.run(json.dumps(payload)).status_code == 500
    native["answers"]["route"]["choice"] = routes[0]
    native["answers"]["route"]["probabilities"] = list(routes)
    assert scorer.run(json.dumps(payload)).status_code == 500


def capability_payload() -> dict[str, Any]:
    tiers = taxonomy_routes("capability-v1")
    return {"state": "User request:\nReformat this date.", "questions": {
        "route": {"type": "choice", "instructions": "Pick the smallest tier", "criteria": {t: f"Tier {t}" for t in tiers}},
        "skill": {"type": "choice", "instructions": "Name the work", "criteria": {s: f"Skill {s}" for s in SKILLS}}}}


def capability_native() -> dict[str, Any]:
    tiers = taxonomy_routes("capability-v1")
    return {"answers": {
        "route": {"choice": "low", "probabilities": {"low": 0.7, "medium": 0.2, "high": 0.1}},
        "skill": {"choice": "extraction", "probabilities": {s: (0.6 if s == "extraction" else 0.1) for s in SKILLS}}},
        "usage": {"input_tokens": 40, "output_tokens": 0, "state_tokens_dropped": 0}} if tiers else {}


def test_capability_tier_and_skill_are_answered_together(scorer: ModuleType) -> None:
    payload, native = capability_payload(), capability_native()
    scorer._runtime = Mock()
    scorer.decide.return_value = native
    result = scorer.run(json.dumps(payload))
    assert result["answers"] == native["answers"] and result["usage"] == native["usage"]
    asked = scorer.decide.call_args.args[2]
    assert list(asked) == ["route", "skill"] and set(asked["skill"]["criteria"]) == set(SKILLS)
    # Without a skill question nothing extra is asked of the model.
    scorer.decide.reset_mock()
    route_only = {"state": payload["state"], "questions": {"route": payload["questions"]["route"]}}
    scorer.decide.return_value = {**native, "answers": {"route": native["answers"]["route"]}}
    assert "answers" in scorer.run(json.dumps(route_only))
    assert list(scorer.decide.call_args.args[2]) == ["route"]


@pytest.mark.parametrize("mutation", ["missing_skill", "unknown_skill", "blank", "skill_only", "extra_head", "wrong_type", "route_union"])
def test_the_skill_question_is_validated_before_the_model_runs(scorer: ModuleType, mutation: str) -> None:
    payload = capability_payload()
    skill = payload["questions"]["skill"]["criteria"]
    if mutation == "missing_skill":
        skill.pop("coding")
    elif mutation == "unknown_skill":
        skill["astrology"] = skill.pop("coding")
    elif mutation == "blank":
        skill["coding"] = "  "
    elif mutation == "skill_only":
        del payload["questions"]["route"]
    elif mutation == "extra_head":
        payload["questions"]["difficulty"] = payload["questions"]["skill"]
    elif mutation == "wrong_type":
        payload["questions"]["skill"]["type"] = "free_text"
    else:
        payload["questions"]["route"]["criteria"].update({s: s for s in SKILLS})
    scorer._runtime = Mock()
    assert scorer.run(json.dumps(payload)).status_code == 400
    scorer.decide.assert_not_called()


@pytest.mark.parametrize("breakage", ["no_skill_answer", "bad_skill_choice", "skill_not_summing", "extra_answer"])
def test_a_wrong_model_answer_for_either_head_is_an_error(scorer: ModuleType, breakage: str) -> None:
    native = capability_native()
    if breakage == "no_skill_answer":
        del native["answers"]["skill"]
    elif breakage == "bad_skill_choice":
        native["answers"]["skill"]["choice"] = "astrology"
    elif breakage == "skill_not_summing":
        native["answers"]["skill"]["probabilities"]["coding"] = 0.9
    else:
        native["answers"]["extra"] = native["answers"]["route"]
    scorer._runtime = Mock()
    scorer.decide.return_value = native
    assert scorer.run(json.dumps(capability_payload())).status_code == 500


def test_packaged_scorer_uses_exact_canonical_registry_offline(tmp_path):
    pytest.importorskip("azure.ai.ml")
    manage_path = SCORE_PATH.parent.parent / "manage.py"
    spec = importlib.util.spec_from_file_location("offline_packager", manage_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    package = module.package_scoring_code(tmp_path / "code")
    source = SCORE_PATH.parents[3] / "routing_lab" / "taxonomy.py"
    assert (package / "routing_lab" / "taxonomy.py").read_bytes() == source.read_bytes()
    assert sorted(str(path.relative_to(package)) for path in package.rglob("*") if path.is_file()) == [
        str(Path("routing_lab") / "__init__.py"), str(Path("routing_lab") / "clef.py"),
        str(Path("routing_lab") / "taxonomy.py"), "score.py",
    ]
    snippet = (
        "import sys; sys.path.insert(0, sys.argv[1]); import score; "
        "from routing_lab.taxonomy import TAXONOMIES, taxonomy_routes; "
        "assert score._runtime is None; "
        "[score.Request.model_validate({'state':'Synthetic', 'questions':{'route':"
        "{'type':'choice','instructions':'Pick one','criteria':"
        "{r:r for r in taxonomy_routes(t)}}}}) for t in TAXONOMIES]; "
        "assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-I", "-c", snippet, str(package)], check=True,
                   cwd=tmp_path, capture_output=True, text=True)


def make_export(tmp_path: Path, nested: bool = True) -> Path:
    export = tmp_path / "model" / ("clef-flash-router-ft" if nested else "") / "export"
    (export / clef.ADAPTER_DIR).mkdir(parents=True)
    (export / clef.ADAPTER_DIR / "adapter_model.safetensors").write_bytes(b"adapter weights")
    (export / clef.HEAD_FILE).write_bytes(b"head weights")
    (export / clef.FINETUNE_META).write_text(json.dumps({
        "format": 1, "base": clef.model_info("flash"), "temperature": 0.937, "best_epoch": 3,
        "bundle_sha256": "b" * 64}), encoding="utf-8")
    return tmp_path / "model"


def test_finetuned_init_loads_the_export_calibrated_and_reports_its_hashes(
    scorer: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any],
    native: dict[str, Any],
) -> None:
    root = make_export(tmp_path)
    monkeypatch.setattr(scorer, "FINETUNED", True)
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(root))
    load = Mock(return_value=Mock())
    monkeypatch.setattr(scorer.clef, "load", load)
    scorer.init()
    scorer.init()
    load.assert_called_once()
    assert load.call_args.kwargs["calibrated"] is True and load.call_args.args == ("flash",)
    assert load.call_args.kwargs["snapshot"].name == "export"
    scorer.decide.return_value = native
    reported = scorer.run(json.dumps(payload))["experiment_model"]
    assert {k: v for k, v in reported.items() if k != "fine_tuned"} == clef.model_info("flash")
    assert reported["fine_tuned"] == {
        "adapter_sha256": hashlib.sha256(b"adapter weights").hexdigest(),
        "head_sha256": hashlib.sha256(b"head weights").hexdigest(),
        "temperature": 0.937, "best_epoch": 3, "bundle_sha256": "b" * 64}


def test_finetuned_deployment_never_falls_back_to_the_base_model(
    scorer: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(scorer, "FINETUNED", True)
    load = Mock()
    monkeypatch.setattr(scorer.clef, "load", load)
    monkeypatch.delenv("AZUREML_MODEL_DIR", raising=False)
    with pytest.raises(RuntimeError, match="registered model"):
        scorer.init()
    monkeypatch.setenv("AZUREML_MODEL_DIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="exactly one"):
        scorer.init()
    load.assert_not_called()
    assert scorer._runtime is None


def test_finetuned_deployment_spec_pins_the_registered_model_and_flag() -> None:
    text = (SCORE_PATH.parent.parent / "deployment-finetuned.yml").read_text(encoding="utf-8")
    assert "name: flash-ft-gpu" in text and "model: azureml:clef-flash-router-ft:1" in text
    assert 'CLEF_FINETUNED: "true"' in text and "instance_type: Standard_NC40ads_H100_v5" in text
    base = (SCORE_PATH.parent.parent / "deployment.yml").read_text(encoding="utf-8")
    assert "CLEF_FINETUNED" not in base and "model:" not in base
    assert "peft==0.21.1" in (SCORE_PATH.parent.parent / "conda.yml").read_text(encoding="utf-8")
