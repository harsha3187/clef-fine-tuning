"""Offline validation of prompt ablation controls and paired reporting."""

import json
from pathlib import Path

import pytest

from routing_lab.backends import ConstantBackend, questions
from routing_lab.data import ROUTES
from routing_lab.prompt_experiment import expanded_contracts, paired_changes, run_prompt_comparison

ROOT = Path(__file__).resolve().parents[1]
CRITERIA = ROOT / "contracts" / "criteria-expanded-v1.json"


def test_expansion_changes_only_descriptions(contracts):
    expanded = expanded_contracts(contracts, CRITERIA)
    assert list(expanded) == list(contracts)
    assert questions(expanded)["route"]["instructions"] == questions(contracts)["route"]["instructions"]
    for route in ROUTES:
        assert expanded[route].description != contracts[route].description
        assert expanded[route].model_dump(exclude={"description"}) == contracts[route].model_dump(
            exclude={"description"}
        )
        assert contracts[route].description == f"Synthetic owner for {route}"


@pytest.mark.parametrize("change", ["missing", "blank", "long", "not-string"])
def test_bad_criteria_rejected(tmp_path, contracts, change):
    criteria = {route: "Valid criterion." for route in ROUTES}
    if change == "missing":
        criteria.pop("fallback")
    else:
        criteria["fallback"] = {"blank": " ", "long": "x" * 2049, "not-string": 5}[change]
    path = tmp_path / "criteria.json"
    path.write_text(json.dumps(criteria), encoding="utf-8")
    with pytest.raises(ValueError):
        expanded_contracts(contracts, path)


def row(record_id, route):
    return {
        "id": record_id, "expected_route": "debugging", "acceptable_routes": ["debugging"],
        "decision": {"route": route},
    }


def test_pair_by_id_not_position_and_count_both_directions():
    result = paired_changes(
        [row("a", "code_generation"), row("b", "debugging"), row("c", "fallback")],
        [row("c", "code_review"), row("b", "fallback"), row("a", "debugging")],
    )
    assert (result["corrected"], result["regressed"], result["changed"]) == (1, 1, 3)


@pytest.mark.parametrize("candidate", [
    [], [row("b", "debugging")], [row("a", "debugging"), row("a", "fallback")],
    [{**row("a", "debugging"), "expected_route": "fallback"}],
])
def test_pairing_rejects_missing_duplicate_or_changed_labels(candidate):
    with pytest.raises(ValueError):
        paired_changes([row("a", "debugging")], candidate)


def test_remote_gate_before_io(tmp_path):
    with pytest.raises(ValueError, match="allow_remote"):
        run_prompt_comparison(
            contracts_path=tmp_path / "absent", criteria_path=CRITERIA,
            data_path=tmp_path / "absent", url="not-used", output=tmp_path / "out",
        )
    assert not (tmp_path / "out").exists()


def test_comparison_is_persistent_and_preserves_baseline(
    tmp_path, monkeypatch, write_dataset, make_record,
):
    created = []

    def backend(contracts, **kwargs):
        created.append(questions(contracts))
        return ConstantBackend("debugging" if len(created) == 2 else "fallback")

    monkeypatch.setattr("routing_lab.prompt_experiment.ClefEndpointBackend", backend)
    data = write_dataset([make_record(expected_route="debugging", review_status="unreviewed")])
    output = tmp_path / "comparison"
    kwargs = dict(
        contracts_path=ROOT / "contracts" / "agents.json", criteria_path=CRITERIA,
        data_path=data, url="unused-offline", output=output, allow_remote=True,
    )
    result = run_prompt_comparison(**kwargs)
    assert result["status"] == "complete"
    assert result["delta_expanded_minus_baseline"]["strict_accuracy"] == 1
    assert result["paired"]["corrected"] == 1
    assert json.loads((output / "comparison.json").read_text()) == result
    assert created[0] != created[1]
    manifests = [json.loads((output / name / "manifest.json").read_text())
                 for name in ("baseline", "expanded_v1")]
    assert manifests[0]["dataset_digest"] == manifests[1]["dataset_digest"]
    with pytest.raises(FileExistsError):
        run_prompt_comparison(**kwargs)


def test_failed_second_arm_leaves_comparison_incomplete(
    tmp_path, monkeypatch, write_dataset, make_record,
):
    count = 0

    def backend(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError("Synthetic endpoint failure")
        return ConstantBackend()

    monkeypatch.setattr("routing_lab.prompt_experiment.ClefEndpointBackend", backend)
    output = tmp_path / "failed"
    with pytest.raises(RuntimeError, match="Synthetic endpoint failure"):
        run_prompt_comparison(
            contracts_path=ROOT / "contracts" / "agents.json", criteria_path=CRITERIA,
            data_path=write_dataset([make_record()]), url="unused-offline",
            output=output, allow_remote=True,
        )
    report = json.loads((output / "comparison.json").read_text())
    assert report["status"] == "incomplete"
    assert set(report["results"]) == {"baseline"}
