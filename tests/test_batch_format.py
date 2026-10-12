"""Batch packing, exact-key JSON shape and per-row evidence isolation."""

import json
from pathlib import Path

import pytest

from routing_lab.annotation import AnnotationItem, digest
from routing_lab.annotation_judge import _validate_schema
from routing_lab.batch_format import (
    Batch, batch_characters, batch_payload, batch_prompt, batch_schema,
    build_batches, validate_batch,
)
from routing_lab.data import load_contracts

ROOT = Path(__file__).resolve().parents[1]


def item(index, taxonomy="copilot-coding-v2", query=None):
    fields = {"query": query or f"Function number {index} returns an incorrect value.", "context": "{}"}
    return AnnotationItem(
        id=f"private-source-id-{index}", taxonomy_id=taxonomy,
        source="swe-bench" if taxonomy == "copilot-coding-v2" else "bfcl",
        category="hidden-category", native_split="test", native_id=str(index),
        group_id=f"group-{index}", repository="example/repo", **fields,
        source_file_sha256="a" * 64, input_sha256=digest(fields), flags=[],
        eligibility="ready", eligibility_reason="fixture", development_exposed=False,
    )


@pytest.fixture
def catalogs():
    return {name: load_contracts(ROOT / "contracts" / f"{name}.json", taxonomy_id=name)
            for name in ("copilot-coding-v2", "tool-api-v1")}


def wire(row, route="debugging"):
    return {
        "verdict": "label", "route": route, "alternative_route": None,
        "rationale": "The requested functional repair belongs to the debugging owner.",
        "evidence": {"field": "query", "quote": row.query},
        "missing_information": None,
    }


def test_pack_twentythree_rows_fit_in_one_batch_at_the_default_size(catalogs):
    batches = build_batches([item(i) for i in range(23)], catalogs)
    assert [len(batch.items) for batch in batches] == [23]


def test_pack_twentythree_rows_as_ten_ten_three_with_explicit_batch_size(catalogs):
    batches = build_batches([item(i) for i in range(23)], catalogs, batch_size=10)
    assert [len(batch.items) for batch in batches] == [10, 10, 3]
    assert len({batch.id for batch in batches}) == 3


def test_packing_never_mixes_tracks(catalogs):
    rows = [item(0), item(1, "tool-api-v1"), item(2)]
    batches = build_batches(rows, catalogs)
    assert len(batches) == 2
    assert [len(batch.items) for batch in batches] == [2, 1]


def test_packing_keeps_native_splits_and_exposure_separate(catalogs):
    rows = [item(0), item(1).model_copy(update={"native_split": "dev"}),
            item(2).model_copy(update={"development_exposed": True})]
    batches = build_batches(rows, catalogs)
    assert len(batches) == 3
    with pytest.raises(ValueError, match="native splits"):
        Batch(tuple(rows))


def test_char_bound_splits_without_truncating(catalogs):
    rows = [item(0, query="A" * 15000), item(1, query="B" * 15000)]
    limit = batch_characters(Batch((rows[0],)), catalogs["copilot-coding-v2"]) + 10
    batches = build_batches(rows, catalogs, max_input_chars=limit)
    assert len(batches) == 2
    assert batches[0].items[0].query == "A" * 15000
    with pytest.raises(ValueError, match="Single row"):
        build_batches(rows, catalogs, max_input_chars=1024)


def test_exact_keyed_schema_uses_supported_fixed_slots():
    batch = Batch((item(0), item(1)))
    schema = batch_schema(batch)
    _validate_schema(schema)
    rows_schema = schema["properties"]["annotations"]
    assert rows_schema["required"] == ["item_000", "item_001"]
    assert rows_schema["additionalProperties"] is False
    evidence = schema["$defs"]["annotation"]["properties"]["evidence"]
    assert evidence["type"] == "object"
    assert "maxItems" not in json.dumps(schema)
    assert "maxLength" not in json.dumps(schema)


def test_payload_keys_are_opaque_stable_and_reverse_only_input_order(catalogs):
    batch = Batch((item(0), item(1)))
    first = batch_payload(batch, 0)
    second = batch_payload(batch, 1)
    assert list(first["items"]) == ["item_000", "item_001"]
    assert list(second["items"]) == ["item_001", "item_000"]
    assert first["items"] == second["items"]
    assert "private-source-id" not in json.dumps(first)
    assert "hidden-category" not in json.dumps(first)
    assert set(first["items"]["item_000"]) == {"query", "context"}
    assert batch_prompt(catalogs["copilot-coding-v2"], 0) != batch_prompt(catalogs["copilot-coding-v2"], 1)


def test_results_map_by_key_not_output_order():
    batch = Batch((item(0), item(1)))
    output = {"annotations": {"item_001": wire(batch.items[1]), "item_000": wire(batch.items[0])}}
    valid = validate_batch(output, batch)
    assert set(valid) == {row.id for row in batch.items}
    assert valid[batch.items[0].id].evidence[0].quote == batch.items[0].query
    assert len(valid[batch.items[0].id].evidence) == 1


@pytest.mark.parametrize("keys", [["item_000"], ["item_000", "item_001", "extra"], ["item_001", "extra"]])
def test_missing_or_unknown_keys_reject_whole_batch(keys):
    batch = Batch((item(0), item(1)))
    with pytest.raises(ValueError, match="opaque row key"):
        validate_batch({"annotations": {key: wire(batch.items[0]) for key in keys}}, batch)


def test_evidence_text_is_not_checked_against_source_or_other_rows():
    """Row attribution is guaranteed by the opaque key mapping, not quote content."""
    batch = Batch((item(0), item(1)))
    output = {"annotations": {key: wire(row) for key, row in batch.slots.items()}}
    output["annotations"]["item_001"]["evidence"]["quote"] = batch.items[0].query
    valid = validate_batch(output, batch)
    assert valid[batch.items[1].id].evidence[0].quote == batch.items[0].query
    with pytest.raises(ValueError, match="empty"):
        output["annotations"]["item_001"]["evidence"]["quote"] = "   "
        validate_batch(output, batch)


def test_six_evidence_quotes_cannot_fit_the_wire_shape():
    batch = Batch((item(0),))
    result = wire(batch.items[0])
    result["evidence"] = [result["evidence"]] * 6
    with pytest.raises(ValueError):
        validate_batch({"annotations": {"item_000": result}}, batch)


def test_distinct_alternative_normalizes_but_duplicates_are_rejected():
    batch = Batch((item(0),))
    result = wire(batch.items[0])
    result["alternative_route"] = "code_generation"
    valid = validate_batch({"annotations": {"item_000": result}}, batch)
    assert valid[batch.items[0].id].acceptable_routes == ["debugging", "code_generation"]
    result["alternative_route"] = "debugging"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_batch({"annotations": {"item_000": result}}, batch)


def test_insufficient_context_is_an_abstention_not_a_guessed_label():
    batch = Batch((item(0),))
    result = wire(batch.items[0])
    result.update(verdict="insufficient_context", route=None, missing_information="The intended result is absent.")
    valid = validate_batch({"annotations": {"item_000": result}}, batch)
    assert valid[batch.items[0].id].route is None
    assert valid[batch.items[0].id].acceptable_routes == []


def test_clarification_label_may_carry_missing_information_other_labels_may_not():
    """A clarification-routed label is a confident decision, not an abstention."""
    batch = Batch((item(0),))
    result = wire(batch.items[0], route="clarification")
    result["missing_information"] = "Specify which repository and file the request concerns."
    valid = validate_batch({"annotations": {"item_000": result}}, batch)
    assert valid[batch.items[0].id].route == "clarification"
    assert valid[batch.items[0].id].missing_information == [result["missing_information"]]
    other = wire(batch.items[0], route="debugging")
    other["missing_information"] = "Specify which repository and file the request concerns."
    with pytest.raises(ValueError, match="clarification"):
        validate_batch({"annotations": {"item_000": other}}, batch)


@pytest.mark.parametrize("batch_size", [0, 101])
def test_batch_size_is_bounded(catalogs, batch_size):
    with pytest.raises(ValueError):
        build_batches([item(0)], catalogs, batch_size=batch_size)


def test_batch_of_hundred_rows_is_accepted_and_packs_in_one_batch_when_small(catalogs):
    batches = build_batches([item(i) for i in range(100)], catalogs, batch_size=100)
    assert [len(batch.items) for batch in batches] == [100]
    assert len(batches[0].slots) == 100
    with pytest.raises(ValueError, match="between one and 100 rows"):
        Batch(tuple(item(i) for i in range(101)))


def test_ineligible_or_duplicate_rows_are_rejected(catalogs):
    row = item(0)
    with pytest.raises(ValueError, match="Duplicate"):
        build_batches([row, row], catalogs)
    blocked = row.model_copy(update={"eligibility": "blocked_context"})
    with pytest.raises(ValueError, match="annotation-ready"):
        build_batches([blocked], catalogs)
