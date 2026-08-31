from collections import Counter
import random

from pocketsql.data.semantic import SOURCE_WEIGHTS, build_focused_splits, mix_sources
from pocketsql.model.semantic_plan import semantic_plan_to_sql, serialize_semantic_plan


def test_focused_semantic_data_is_schema_disjoint_executable_and_contrastive():
    splits = build_focused_splits(5, 10101)
    schema_sets = {name: {record["schema_id"] for record in records} for name, records in splits.items()}
    intents = {record["semantic_intent"] for records in splits.values() for record in records}

    assert {name: len(records) for name, records in splits.items()} == {"train": 36, "validation": 12, "test": 12}
    assert schema_sets["train"].isdisjoint(schema_sets["validation"] | schema_sets["test"])
    assert schema_sets["validation"].isdisjoint(schema_sets["test"])
    assert {
        "contrast_parent_projection",
        "contrast_named_parent_rows",
        "relationship_join_name",
        "relationship_join_name_location",
        "contrast_maximum_aggregate",
    } <= intents
    for records in splits.values():
        for record in records:
            assert semantic_plan_to_sql(serialize_semantic_plan(record["query_plan"])) == record["sql"]
            assert record["evaluation_only"] is False


def test_natural_gate_withholds_close_paraphrases_from_training_templates():
    train = build_focused_splits(4, 10101)["train"]
    gate = build_focused_splits(2, 10102, heldout=True, schema_prefix="semantic_gate")["natural_gate"]
    train_relationships = [record for record in train if record["semantic_intent"] == "relationship_join_name"]
    gate_relationships = [record for record in gate if record["semantic_intent"] == "relationship_join_name"]

    assert train_relationships and gate_relationships
    assert {record["question"] for record in train_relationships}.isdisjoint(
        record["question"] for record in gate_relationships
    )
    assert all(" where " in record["question"] or " whose " in record["question"] for record in gate_relationships)
    assert all(" INNER JOIN " in record["sql"] and " WHERE " in record["sql"] for record in gate_relationships)
    assert all(record["evaluation_only"] is True for record in gate)


def test_joined_aggregate_curriculum_is_optional_and_executable():
    splits = build_focused_splits(5, 20260830, include_joined_aggregates=True)
    records = [record for split in splits.values() for record in split]
    joined_aggregate_intents = {
        record["semantic_intent"]
        for record in records
        if record["semantic_intent"].startswith("relationship_join_")
        and record["query_plan"].get("aggregate")
    }

    assert {name: len(split) for name, split in splits.items()} == {
        "train": 48,
        "validation": 16,
        "test": 16,
    }
    assert {
        "relationship_join_sum_name_location",
        "relationship_join_average_location_status",
        "relationship_join_maximum_name",
        "relationship_join_count_name_location",
    } <= joined_aggregate_intents
    assert all(" INNER JOIN " in record["sql"] for record in records if record["semantic_intent"] in joined_aggregate_intents)


def test_semantic_mix_uses_fixed_source_proportions_and_prefixes_ids():
    sources = {
        source: [
            {"id": f"id_{index}", "schema_id": f"schema_{index}", "query_plan": {"family": source}}
            for index in range(40)
        ]
        for source in SOURCE_WEIGHTS
    }

    mixed, quotas = mix_sources(sources, 40, 7)

    assert quotas == SOURCE_WEIGHTS
    assert Counter(record["semantic_source"] for record in mixed) == Counter(SOURCE_WEIGHTS)
    assert all(record["id"].startswith(record["semantic_source"] + ":") for record in mixed)
    assert all(record["schema_id"].startswith(record["semantic_source"] + ":") for record in mixed)
