from __future__ import annotations

import json
import shutil

import pytest

from capability_lab.bundle import HELD_OUT_FAMILIES, assign_splits, build_bundle
from capability_lab.tasks import build_tasks
from routing_lab import finetune_capability as fc


def fake_labels(tasks):
    tiers = ("low", "low", "low", "medium", "high")
    return [{"task_id": t.id, "family": t.family, "skill": t.skill, "level": t.level, "tier": tiers[i % 5],
             "pass_fraction": 0.5, "unsolved": False, "cheapest_passing": "m", "cheapest_cost": 0.0, "endpoints": 10, "passing": ["m"]}
            for i, t in enumerate(tasks)]


@pytest.fixture(scope="module")
def tasks():
    return build_tasks(per_family=20)


def test_splits_hold_out_whole_families_and_cover_every_other_family(tasks):
    labels = fake_labels(tasks)
    splits = assign_splits(labels)
    for label in labels:
        name = splits[label["task_id"]]
        assert (name == "test_ood") == (label["family"] in HELD_OUT_FAMILIES)
    trained = {label["family"] for label in labels if splits[label["task_id"]] == "train"}
    assert trained == {label["family"] for label in labels} - set(HELD_OUT_FAMILIES)
    sizes = {name: sum(v == name for v in splits.values()) for name in ("train", "validation", "calibration", "test_id")}
    assert sizes["train"] > 3 * sizes["validation"] > 0 and sizes["test_id"] > 0 and sizes["calibration"] > 0
    assert splits == assign_splits(labels)


def test_the_bundle_verifies_and_contains_no_test_rows(tasks, tmp_path):
    labels = fake_labels(tasks)
    manifest = build_bundle(tasks, labels, tmp_path / "b", tasks_hash="x", thresholds={"low_at_least": 0.7, "high_below": 0.35}, endpoints=["m"])
    splits = assign_splits(labels)
    checked, examples = fc.validate_inputs(tmp_path / "b")
    assert checked == manifest and {p.name for p in (tmp_path / "b").iterdir()} == {"train.jsonl", "validation.jsonl", "calibration.jsonl", "manifest.json"}
    in_bundle = {row["id"] for rows in examples.values() for row in rows}
    assert in_bundle == {t for t, s in splits.items() if s in ("train", "validation", "calibration")}
    row = examples["train"][0]
    assert row["answers"]["route"] in ("low", "medium", "high") and set(row["questions"]) == {"route", "skill"}
    assert row["state"].startswith("User request:\n")


def test_tampering_and_leakage_are_refused(tasks, tmp_path):
    labels = fake_labels(tasks)
    build_bundle(tasks, labels, tmp_path / "b", tasks_hash="x", thresholds={}, endpoints=["m"])
    bundle = tmp_path / "b"
    copy = tmp_path / "c"
    shutil.copytree(bundle, copy)
    with (copy / "train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        fc.verify_bundle(copy)
    shutil.copytree(bundle, tmp_path / "d")
    (tmp_path / "d" / "test.jsonl").write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="unexpected"):
        fc.verify_bundle(tmp_path / "d")

    leaked = tmp_path / "e"
    shutil.copytree(bundle, leaked)
    first = json.loads((leaked / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    manifest = json.loads((leaked / "manifest.json").read_text(encoding="utf-8"))
    rows = (leaked / "validation.jsonl").read_text(encoding="utf-8").splitlines()
    rows[0] = json.dumps({**json.loads(rows[0]), "state": first["state"], "id": "other"})
    (leaked / "validation.jsonl").write_text("\n".join(rows) + "\n", encoding="utf-8")
    manifest["files"]["validation.jsonl"]["sha256"] = fc.sha256(leaked / "validation.jsonl")
    (leaked / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        fc.validate_inputs(leaked)


def test_head_metrics_on_a_known_case():
    class Encoded:
        pass

    raw = [{"route": [5.0, 0.0, 0.0]}, {"route": [0.0, 5.0, 0.0]}, {"route": [5.0, 0.0, 0.0]}, {"route": [0.0, 0.0, 5.0]}]
    rows = [(Encoded(), {"route": t}) for t in (0, 1, 1, 2)]
    metrics = fc.head_metrics(raw, rows, "route")
    assert metrics["accuracy"] == 0.75 and metrics["majority_accuracy"] == 0.5
    assert 0.7 < metrics["macro_f1"] < 0.9
    hot = fc.head_metrics(raw, rows, "route", temperature=100.0)
    assert hot["accuracy"] == 0.75 and hot["ece_10_bins"] < metrics["ece_10_bins"] + 0.5 and hot["brier"] < 0.8

def test_extra_training_sets_join_the_bundle_and_evaluation_prompts_are_refused(tasks, tmp_path):
    from capability_lab.bundle import TRAINING_ONLY

    labels = fake_labels(tasks)
    extra = [x for x in labels if x["family"] == "arithmetic_steps"]
    splits = {**assign_splits(labels), **assign_splits(extra, (), fractions=TRAINING_ONLY)}
    assert not any(splits[x["task_id"]] == "test_id" for x in extra)
    manifest = build_bundle(tasks, labels, tmp_path / "v2", tasks_hash="x", thresholds={}, endpoints=["m"], splits=splits,
                            extra_sets=[{"set": "tools2", "tasks_hash": "h"}], test_sets=[{"set": "tools", "tasks_hash": "g"}])
    assert manifest["extra_training_sets"][0]["set"] == "tools2" and manifest["evaluation_only_sets"][0]["set"] == "tools"
    fc.validate_inputs(tmp_path / "v2")
    leaked = next(t.prompt for t in tasks if splits.get(t.id) == "train")
    with pytest.raises(ValueError, match="held-out evaluation"):
        build_bundle(tasks, labels, tmp_path / "bad", tasks_hash="x", thresholds={}, endpoints=["m"], splits=splits, forbidden_prompts={leaked})
