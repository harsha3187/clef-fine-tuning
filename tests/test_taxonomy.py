"""Offline versioned routing tests; remote clients and model loading are mocked."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from routing_lab import clef, cli
from routing_lab.backends import (
    INSTRUCTIONS, AzureLLMBackend, ClefBackend, ConstantBackend, Decision, RulesBackend,
    TfidfBackend, questions, state,
)
from routing_lab.data import (
    ROUTES, Contract, Record, load_contracts, load_records, split_reviewed,
)
from routing_lab.evaluation import run_experiment, summarize
from routing_lab.prompt_experiment import expanded_contracts, paired_changes
from routing_lab.taxonomy import (
    TAXONOMIES, abstention_routes, taxonomy_for_routes, taxonomy_routes,
)


@pytest.fixture(params=TAXONOMIES)
def taxonomy(request):
    return request.param


@pytest.fixture
def catalog(taxonomy):
    return {route: Contract(description=f"Own {route}", tools=[], can_write=False,
                            success_criteria="Resolve the task")
            for route in taxonomy_routes(taxonomy)}


@pytest.fixture
def record(make_record, taxonomy):
    return make_record(taxonomy_id=taxonomy, expected_route=taxonomy_routes(taxonomy)[0])


def test_exact_versioned_registry():
    assert taxonomy_routes("coding-v1") == ROUTES
    assert taxonomy_routes("copilot-coding-v2") == (
        "debugging", "code_generation", "refactoring", "test_generation",
        "code_review", "security_review", "performance_optimization", "repository_qa",
        "research", "planning", "documentation", "build_release", "clarification", "fallback",
    )
    assert taxonomy_routes("tool-api-v1") == (
        "tool_lookup", "tool_action", "tool_workflow", "web_research",
        "memory_retrieval", "clarification", "no_matching_tool",
    )
    with pytest.raises(ValueError, match="Unknown taxonomy"):
        taxonomy_routes("unversioned")


@pytest.mark.parametrize("taxonomy", ["copilot-coding-v2", "tool-api-v1"])
def test_parent_catalogs_use_shared_initial_owner_policy(taxonomy):
    path = Path(__file__).resolve().parents[1] / "contracts" / f"{taxonomy}.json"
    contracts = load_contracts(path, taxonomy_id=taxonomy)
    question = questions(contracts)["route"]
    assert question == questions(contracts, taxonomy_id=taxonomy)["route"]
    assert question["criteria"] == {key: value.description for key, value in contracts.items()}
    instructions = question["instructions"]
    assert "Select one initial owning agent, not a sequence" in instructions
    assert "Pure explanation/search is repository_qa" not in instructions
    if taxonomy == "copilot-coding-v2":
        assert "not downstream solvability" in instructions
        assert "checkout alone must not force clarification or fallback" in instructions
        assert "issue intent and repository identity are clear" in instructions
        assert "intent changes which agent should own" in instructions
    else:
        assert "only through explicitly offered capabilities" in instructions
        assert "only unrelated business APIs is no_matching_tool, not code_generation" in instructions
        assert "cannot be obtained through offered tools" in instructions
        assert "Select one initial owning coding agent" not in instructions


def test_legacy_instructions_remain_unchanged(contracts):
    assert questions(contracts)["route"]["instructions"] == INSTRUCTIONS


def test_records_contracts_and_review_gates(taxonomy, catalog, record, tmp_path, write_dataset):
    routes = taxonomy_routes(taxonomy)
    assert record.taxonomy_id == taxonomy
    assert Record.model_validate_json(record.model_dump_json()) == record
    for changes in ({"expected_route": "unknown"}, {"acceptable_routes": ["unknown"]},
                    {"taxonomy_id": "unknown"}):
        with pytest.raises(ValidationError):
            Record.model_validate({**record.model_dump(), **changes})
    path = tmp_path / "contracts.json"
    payload = {key: value.model_dump() for key, value in catalog.items()}
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_contracts(path, taxonomy) == catalog
    if taxonomy != "coding-v1":
        with pytest.raises(ValueError, match="taxonomy"):
            load_contracts(path)
    payload["unknown"] = payload.pop(routes[0])
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly"):
        load_contracts(path, taxonomy)
    draft = Record.model_validate({**record.model_dump(), "review_status": "unreviewed", "reviewer": ""})
    path = write_dataset([draft])
    assert load_records(path, taxonomy_id=taxonomy) == [draft]
    with pytest.raises(ValueError, match="reviewed"):
        load_records(path, taxonomy_id=taxonomy, reviewed_only=True)
    with pytest.raises(ValueError, match="Review"):
        split_reviewed([draft])


def test_decisions_require_exact_probability_catalog(taxonomy):
    routes = taxonomy_routes(taxonomy)
    probabilities = {route: float(route == routes[0]) for route in routes}
    assert Decision(taxonomy_id=taxonomy, route=routes[0], probabilities=probabilities)
    for values in ({"unknown": 1.0}, {**probabilities, "unknown": 0.0},
                   {route: 0.0 for route in routes}):
        with pytest.raises(ValidationError):
            Decision(taxonomy_id=taxonomy, route=routes[0], probabilities=values)
    with pytest.raises(ValidationError):
        Decision(taxonomy_id=taxonomy, route="unknown")
    if taxonomy != "coding-v1":
        with pytest.raises(ValidationError):
            Decision(route=routes[0], probabilities=probabilities)


def test_questions_and_state_never_expose_labels_or_provenance(taxonomy, catalog, record):
    question = questions(catalog, reverse=True)
    assert list(question["route"]["criteria"]) == list(reversed(taxonomy_routes(taxonomy)))
    assert set(question["route"]) == {"type", "instructions", "criteria"}
    assert question == questions(catalog, reverse=True, taxonomy_id=taxonomy)
    assert record.taxonomy_id not in json.dumps(question)
    assert state(record, "query") == record.query
    assert state(record, "context") == (
        f"User request:\n{record.query}\nAvailable context:\n{record.context}"
    )
    for selected in TAXONOMIES:
        if selected != taxonomy:
            with pytest.raises(ValueError, match="match"):
                questions(catalog, taxonomy_id=selected)
    with pytest.raises(ValueError, match="exactly"):
        questions(dict(list(catalog.items())[:-1]))


def test_models_use_same_selected_instructions_and_criteria(
    taxonomy, catalog, record, monkeypatch,
):
    pytest.importorskip("openai")
    pytest.importorskip("azure.identity")
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "offline-test")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://offline-test.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_VERSION", "offline-test")
    monkeypatch.setattr("azure.identity.DefaultAzureCredential", Mock())
    monkeypatch.setattr("azure.identity.get_bearer_token_provider", Mock())
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps({"route": record.expected_route})))],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=2),
    )
    client = Mock()
    client.chat.completions.create.return_value = response
    monkeypatch.setattr("openai.AzureOpenAI", Mock(return_value=client))
    probabilities = {route: float(route == record.expected_route) for route in catalog}
    decide = Mock(return_value={
        "answers": {"route": {"choice": record.expected_route, "probabilities": probabilities}},
        "usage": {"input_tokens": 11, "output_tokens": 0, "state_tokens_dropped": 0},
    })
    loader = Mock(return_value=clef.ClefRuntime(None, None, None, "flash"))
    monkeypatch.setattr(clef, "load", loader)
    monkeypatch.setattr(clef, "decide", decide)
    llm = AzureLLMBackend(catalog, taxonomy_id=taxonomy, reverse=True)
    model = ClefBackend(catalog, taxonomy_id=taxonomy)
    for backend in (llm, model):
        result = backend.predict(record)
        assert result.taxonomy_id == taxonomy
        assert result.route == record.expected_route
        assert backend.metadata()["taxonomy_id"] == taxonomy
    question = questions(catalog, reverse=True, taxonomy_id=taxonomy)["route"]
    assert llm.system == (question["instructions"] + "\nOptions:\n" + json.dumps(question["criteria"])
                          + '\nReturn only JSON with one key "route".')
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[1] == {"role": "user", "content": state(record, "context")}
    assert decide.call_args.args[1:] == (state(record, "context"), model.question)
    assert llm.predict(record).probabilities is None
    response.choices[0].message.content = '{"route":"unknown"}'
    with pytest.raises(ValidationError):
        llm.predict(record)
    other = next(value for value in TAXONOMIES if value != taxonomy)
    wrong = Record.model_validate({
        **record.model_dump(), "taxonomy_id": other,
        "expected_route": taxonomy_routes(other)[0], "acceptable_routes": [taxonomy_routes(other)[0]],
    })
    client.chat.completions.create.reset_mock()
    decide.reset_mock()
    for backend in (llm, model):
        with pytest.raises(ValueError, match="taxonomy"):
            backend.predict(wrong)
    client.chat.completions.create.assert_not_called()
    decide.assert_not_called()
    loader.reset_mock()
    with pytest.raises(ValueError, match="match"):
        ClefBackend(catalog, taxonomy_id=other)
    loader.assert_not_called()


def test_baselines_use_real_selected_classes(taxonomy, make_record, record):
    constant = ConstantBackend(taxonomy_id=taxonomy)
    assert constant.predict(record).route in taxonomy_routes(taxonomy)
    assert constant.predict(record).probabilities is None
    if taxonomy != "coding-v1":
        with pytest.raises(ValueError, match="only taxonomy coding-v1"):
            RulesBackend(taxonomy_id=taxonomy)
        with pytest.raises(ValueError, match="taxonomy"):
            RulesBackend().predict(record)
    training = [make_record(taxonomy_id=taxonomy, expected_route=route, query=f"Request for {route}")
                for route in taxonomy_routes(taxonomy)]
    model = TfidfBackend(training, taxonomy_id=taxonomy)
    decision = model.predict(record)
    assert decision.taxonomy_id == taxonomy
    assert set(decision.probabilities) == set(taxonomy_routes(taxonomy))
    assert set(model.model.classes_) == set(taxonomy_routes(taxonomy))
    with pytest.raises(ValueError, match="every route"):
        TfidfBackend(training[:-1])


def test_metrics_cover_selected_catalog_and_abstention(taxonomy, make_record):
    routes = taxonomy_routes(taxonomy)
    records = [make_record(taxonomy_id=taxonomy, expected_route=route) for route in routes]
    decisions = [Decision(taxonomy_id=taxonomy, route=route,
                          probabilities={key: float(key == route) for key in routes}) for route in routes]
    summary = summarize(records, decisions, [1.0] * len(records))
    assert summary["taxonomy_id"] == taxonomy
    assert summary["route_order"] == list(routes)
    assert summary["strict_accuracy"] == summary["macro_f1"] == 1.0
    assert summary["confusion_matrix"] == [[int(a == b) for a in routes] for b in routes]
    assert summary["probability_metrics"]["multiclass_brier"] == 0.0
    assert summary["probability_metrics"]["ece_10_bins"] == 0.0
    assert summary["abstention_recall"] == (1.0 if abstention_routes(taxonomy) else None)
    for point in summary["probability_metrics"]["risk_coverage"]:
        assert point["coverage"] == pytest.approx(1 - len(abstention_routes(taxonomy)) / len(routes))
    if "fallback" not in routes:
        assert summary["fallback_rate"] is summary["fallback_recall"] is None
    else:
        assert summary["fallback_recall"] == 1.0


def test_mixed_records_and_mismatched_decisions_rejected(make_record, write_dataset, tmp_path):
    old = make_record(expected_route="debugging")
    new = make_record(taxonomy_id="copilot-coding-v2", expected_route="debugging")
    mixed = [old, new]
    with pytest.raises(ValueError, match="Mixed taxonomies"):
        load_records(write_dataset(mixed))
    with pytest.raises(ValueError, match="Mixed taxonomies"):
        split_reviewed(mixed)
    with pytest.raises(ValueError, match="Mixed taxonomies"):
        TfidfBackend(mixed)
    with pytest.raises(ValueError, match="Mixed taxonomies"):
        summarize(mixed, [Decision(route="debugging")] * 2, [1.0] * 2)
    with pytest.raises(ValueError, match="Decision taxonomy"):
        summarize([new], [Decision(route="debugging")], [1.0])
    output = tmp_path / "no-run"
    with pytest.raises(ValueError, match="Backend taxonomy"):
        run_experiment(ConstantBackend(), [new], output)
    assert not output.exists()


def test_cli_taxonomy_selection_and_persisted_artifacts(
    taxonomy, catalog, record, write_dataset, tmp_path, capsys,
):
    data = write_dataset([record])
    path = tmp_path / "contracts.json"
    path.write_text(json.dumps({key: value.model_dump() for key, value in catalog.items()}), encoding="utf-8")
    output = tmp_path / "run"
    arguments = ["run", "--data", str(data), "--contracts", str(path),
                 "--backend", "constant", "--output", str(output)]
    if taxonomy != "coding-v1":
        with pytest.raises(ValueError, match="taxonomy"):
            cli.execute(cli.create_parser().parse_args(arguments))
        assert not output.exists()
    cli.execute(cli.create_parser().parse_args(arguments + ["--taxonomy", taxonomy]))
    capsys.readouterr()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["taxonomy_id"] == taxonomy
    assert manifest["backend"]["taxonomy_id"] == taxonomy
    assert manifest["status"] == "complete"
    prediction = json.loads((output / "predictions.jsonl").read_text(encoding="utf-8"))
    assert prediction["taxonomy_id"] == prediction["decision"]["taxonomy_id"] == taxonomy


def test_description_ablation_retains_selected_taxonomy(taxonomy, catalog, tmp_path):
    path = tmp_path / "criteria.json"
    criteria = {key: f"Expanded {key}" for key in catalog}
    path.write_text(json.dumps(criteria), encoding="utf-8")
    expanded = expanded_contracts(catalog, path)
    assert taxonomy_for_routes(expanded) == taxonomy
    assert [value.description for value in expanded.values()] == list(criteria.values())


def test_cli_default_contract_path_tracks_taxonomy(
    taxonomy, catalog, record, write_dataset, tmp_path, monkeypatch, capsys,
):
    data = write_dataset([record])
    folder = tmp_path / "contracts"
    folder.mkdir()
    path = folder / ("agents.json" if taxonomy == "coding-v1" else f"{taxonomy}.json")
    path.write_text(json.dumps({key: value.model_dump() for key, value in catalog.items()}), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    args = cli.create_parser().parse_args([
        "run", "--data", str(data), "--taxonomy", taxonomy,
        "--backend", "constant", "--output", str(tmp_path / "result"),
    ])
    cli.execute(args)
    result = json.loads(capsys.readouterr().out)
    assert result["taxonomy_id"] == taxonomy


def test_paired_comparison_rejects_different_taxonomies():
    before = {"id": "one", "expected_route": "debugging", "acceptable_routes": ["debugging"],
              "decision": {"route": "debugging"}}
    after = {**before, "taxonomy_id": "copilot-coding-v2"}
    with pytest.raises(ValueError, match="taxonomies"):
        paired_changes([before], [after])
