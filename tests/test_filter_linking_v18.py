import json
from pathlib import Path
import sqlite3

from pocketsql.data.filter_linking_v18 import build_filter_linking_dataset
from pocketsql.model.schema_grounding import canonicalize_inputs


def _source(schema_id: str, table: str) -> dict:
    schema_sql = (
        f"CREATE TABLE {table} ("
        "item_id INTEGER PRIMARY KEY, status TEXT, category TEXT, note TEXT, "
        "amount REAL, score REAL);"
    )
    connection = sqlite3.connect(":memory:")
    connection.executescript(schema_sql)
    connection.executemany(
        f"INSERT INTO {table} VALUES (?, ?, ?, ?, ?, ?)",
        [
            (1, "open", "a", "first", 10.0, 1.0),
            (2, "closed", "b", "second", 20.0, 2.0),
            (3, "pending", "c", "third", 30.0, 3.0),
        ],
    )
    database_sql = "\n".join(connection.iterdump())
    connection.close()
    return {
        "id": schema_id,
        "schema_id": schema_id,
        "schema_sql": schema_sql,
        "database_sql": database_sql,
    }


def _write(path: Path, record: dict) -> None:
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_filter_linking_dataset_is_schema_disjoint_explicit_and_counterfactual(tmp_path: Path):
    train_source = tmp_path / "train_source.jsonl"
    validation_source = tmp_path / "validation_source.jsonl"
    _write(train_source, _source("train_schema", "work_items"))
    _write(validation_source, _source("validation_schema", "service_items"))
    output = tmp_path / "filter_linking"

    report = build_filter_linking_dataset(
        output, train_source, validation_source, seed=123
    )

    assert report["schema_overlap"] == 0
    assert report["exact_question_overlap"] == 0
    assert report["all_targets_explicit"] is True
    assert report["records"] == {"train": 15, "validation": 10}
    assert report["counterfactual_groups"]["validation"] == 4
    assert report["counterfactual_groups"]["validation_complete"] == 4

    records = [json.loads(line) for line in (output / "validation.jsonl").open()]
    assert all(record.get("database_sql") for record in records)
    assert all(record["query_plan"]["family"] == "v18_filter_column_contrast" for record in records)
    for record in records:
        _, grounded, mapping = canonicalize_inputs(
            record["schema_sql"], record["question"], "permuted", True, False
        )
        target = record["filter_link_target"]
        assert mapping.column_to_slot[target] in grounded
