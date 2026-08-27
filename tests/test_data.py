import json
import random
import sqlite3

from pocketsql.data.challenge import write_challenge_dataset
from pocketsql.data.casual_dev import write_casual_dev_dataset
from pocketsql.data.column_copy_dev import write_column_copy_dev_dataset
from pocketsql.data.generate import build_records, dataset_quality_report, plans_for
from pocketsql.data.grounding_dev import write_grounding_dev_dataset
from pocketsql.data.populate import populate
from pocketsql.data.render_sql import render_sql
from pocketsql.data.schemas import make_schema
from pocketsql.data.validate import validate_sql


def test_generation_is_deterministic_and_split_by_schema():
    first = build_records(10, 20, 7)
    second = build_records(10, 20, 7)
    assert first == second
    split_ids = [{record["schema_id"] for record in records} for records in first.values()]
    assert not split_ids[0] & split_ids[1]
    assert not split_ids[0] & split_ids[2]
    assert not split_ids[1] & split_ids[2]


def test_schema_vocabulary_is_diverse_and_not_shared_across_splits():
    splits = build_records(30, 20, 7)
    shapes = {name: {record["schema_sql"] for record in records} for name, records in splits.items()}
    assert len(shapes["train"]) > 5
    assert not shapes["train"] & shapes["test"]
    assert not shapes["train"] & shapes["validation"]


def test_generated_sql_executes_and_all_families_exist():
    splits = build_records(5, 13, 42)
    records = [record for split in splits.values() for record in split]
    expected = {"select", "distinct", "filter", "and_filter", "or_filter", "count", "sum", "avg", "min", "max", "group", "order_limit", "join"}
    assert expected <= {record["query_plan"]["family"] for record in records}
    for record in records:
        connection = sqlite3.connect(":memory:")
        connection.executescript(record["database_sql"])
        assert validate_sql(connection, record["sql"])[0]
        connection.close()


def test_weighted_generation_oversamples_join_and_compound_filters():
    weights = {"join": 4, "and_filter": 3, "or_filter": 3}
    splits = build_records(20, 50, 17, weights)
    records = [record for split in splits.values() for record in split]
    families = [record["query_plan"]["family"] for record in records]
    assert families.count("join") / len(families) >= 0.18
    assert (families.count("and_filter") + families.count("or_filter")) / len(families) >= 0.25


def test_harder_join_variants_execute_and_quality_report_has_no_schema_leakage():
    splits = build_records(8, 45, 11, {"join": 4, "and_filter": 3, "or_filter": 3})
    records = [record for split in splits.values() for record in split]
    joins = [record for record in records if record["query_plan"]["family"] == "join"]
    assert any(" WHERE " in record["sql"] for record in joins)
    assert any(" ORDER BY " in record["sql"] for record in joins)
    for record in joins:
        connection = sqlite3.connect(":memory:")
        connection.executescript(record["database_sql"])
        assert validate_sql(connection, record["sql"])[0]
        connection.close()
    report = dataset_quality_report(splits)
    assert report["schema_disjoint"]
    assert all(count == 0 for count in report["schema_split_overlap"].values())
    assert all(split["duplicate_schema_question_pairs"] == 0 for split in report["splits"].values())


def test_balanced_generation_retains_every_planned_example():
    generation_stats = {}
    splits = build_records(20, 75, 11, {"join": 4, "and_filter": 3, "or_filter": 3}, generation_stats=generation_stats)
    assert generation_stats == {
        "planned_examples": 1500,
        "retained_examples": 1500,
        "discarded_invalid_sql": 0,
        "discarded_duplicate_questions": 0,
    }
    assert sum(len(records) for records in splits.values()) == generation_stats["planned_examples"]


def test_challenge_data_uses_held_out_identifiers(tmp_path):
    reference_splits = build_records(10, 20, 5)
    reference = tmp_path / "train.jsonl"
    reference.write_text(
        "".join(json.dumps(record) + "\n" for record in reference_splits["train"]), encoding="utf-8"
    )
    output = tmp_path / "challenge"
    counts = write_challenge_dataset(output, schemas=8, examples_per_schema=30, seed=13, reference_data=reference)
    records = [json.loads(line) for line in (output / "challenge.jsonl").read_text(encoding="utf-8").splitlines()]
    report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))
    assert counts == {"challenge": len(records), "schemas": 8}
    assert all(record["schema_id"].startswith("challenge_") for record in records)
    assert report["profile"] == "identifier_held_out_challenge"
    assert report["generation"]["discarded_invalid_sql"] == 0
    assert report["generation"]["planned_examples"] == 8 * 30
    assert report["splits"]["challenge"]["unseen_identifier_rate_vs_reference"] > 0.5


def test_grounding_dev_uses_unique_opaque_identifiers(tmp_path):
    output = tmp_path / "grounding-dev"
    counts = write_grounding_dev_dataset(output, schemas=6, examples_per_schema=20, seed=19)
    records = [json.loads(line) for line in (output / "opaque.jsonl").read_text(encoding="utf-8").splitlines()]
    report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))

    assert counts == {"opaque": 120, "schemas": 6}
    assert all(record["schema_id"].startswith("opaque_") for record in records)
    assert all("CREATE TABLE x_" in record["schema_sql"] for record in records)
    assert report["profile"] == "opaque_identifier_grounding_dev"
    assert report["training_use_allowed"] is False
    assert report["generation"]["discarded_invalid_sql"] == 0


def test_population_respects_foreign_keys():
    schema = make_schema(0, __import__("random").Random(4))
    connection = sqlite3.connect(":memory:")
    populate(connection, schema, __import__("random").Random(4))
    child = schema.tables[1]
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute(f"SELECT COUNT(*) FROM {child.name}").fetchone()[0] > 0


def test_mixed_language_data_contains_casual_single_column_projections():
    splits = build_records(
        10,
        75,
        91,
        {"join": 4, "and_filter": 3, "or_filter": 3, "select": 8},
        question_style="mixed",
    )
    records = [record for split in splits.values() for record in split]
    single = [
        record
        for record in records
        if record["query_plan"]["family"] == "select" and len(record["query_plan"]["columns"]) == 1
    ]

    assert single
    assert any("show me" in record["question"] for record in single)
    assert any(record["question"].startswith("what are the ") for record in single)
    assert any(
        record["query_plan"]["family"] == "select" and len(record["query_plan"]["columns"]) > 1
        for record in records
    )
    assert all(record["sql"].upper().startswith("SELECT ") for record in single)


def test_casual_dev_uses_held_out_language_and_is_evaluation_only(tmp_path):
    output = tmp_path / "casual-dev"
    counts = write_casual_dev_dataset(output, schemas=6, examples_per_schema=30, seed=23)
    records = [json.loads(line) for line in (output / "casual.jsonl").read_text(encoding="utf-8").splitlines()]
    report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))

    assert counts == {"casual": 180, "schemas": 6}
    assert any("I'd like" in record["question"] or "pull up" in record["question"] for record in records)
    assert report["profile"] == "held_out_casual_language_dev"
    assert report["training_use_allowed"] is False


def test_column_copy_dev_requests_every_physical_column(tmp_path):
    output = tmp_path / "column-copy-dev"
    counts = write_column_copy_dev_dataset(output, schemas=5, seed=29)
    records = [json.loads(line) for line in (output / "column_copy.jsonl").read_text(encoding="utf-8").splitlines()]
    report = json.loads((output / "quality_report.json").read_text(encoding="utf-8"))

    by_schema: dict[str, set[tuple[str, str]]] = {}
    for record in records:
        by_schema.setdefault(record["schema_id"], set()).add(
            (record["query_plan"]["table"], record["query_plan"]["columns"][0])
        )
    assert counts == {"column_copy": len(records), "schemas": 5}
    assert all(len(columns) >= 9 for columns in by_schema.values())
    assert {"parent_id", "child_id", "parent_fk", "name", "location", "amount", "status"} <= set(report["roles"])
    assert report["training_use_allowed"] is False


def test_position_robust_schedule_selects_every_shuffled_column():
    rng = random.Random(101)
    role_positions = set()
    schema_sizes = set()
    weights = {"join": 4, "and_filter": 3, "or_filter": 3, "select": 8}

    for schema_index in range(20):
        schema = make_schema(schema_index, rng)
        plans = plans_for(schema, 75, weights)
        expected = {
            (table.name, column.name)
            for table in schema.tables
            for column in table.columns
        }
        selected = {
            (plan.table, column)
            for plan in plans
            if plan.family == "select"
            for column in plan.columns
        }
        parent_table, parent_id = schema.role("parent_id")
        role_positions.add(next(index for index, column in enumerate(parent_table.columns) if column == parent_id))
        schema_sizes.add(len(expected))

        assert expected <= selected

    assert len(role_positions) > 1
    assert len(schema_sizes) > 1
