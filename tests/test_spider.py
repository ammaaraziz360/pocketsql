from __future__ import annotations

from collections import Counter
import json
import sqlite3

import pytest

from pocketsql.data.spider import (
    SpiderRejected,
    contamination_matches,
    convert_example,
    schema_sql_from_metadata,
    training_contamination_index,
)
from pocketsql.data.spider_training import mix_human_curriculum
from pocketsql.data.schema_linking import linking_family, weighted_resample


def metadata() -> dict:
    return {
        "db_id": "shop",
        "table_names_original": ["customers", "orders"],
        "column_names_original": [
            [-1, "*"],
            [0, "customer_id"],
            [0, "name"],
            [0, "city"],
            [1, "order_id"],
            [1, "customer_id"],
            [1, "total"],
        ],
        "column_types": ["text", "number", "text", "text", "number", "number", "number"],
        "primary_keys": [1, 4],
        "foreign_keys": [[5, 1]],
    }


def database(tmp_path):
    path = tmp_path / "shop.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT);"
        "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL);"
        "INSERT INTO customers VALUES (1, 'Ada', 'Houston'), (2, 'Lin', 'Boston');"
        "INSERT INTO orders VALUES (10, 1, 75), (11, 2, 20);"
    )
    connection.close()
    return path


def source(**overrides) -> dict:
    value = {
        "db_id": "shop",
        "question": "Show order totals for customers in Houston",
        "query": (
            "SELECT orders.total FROM customers JOIN orders "
            "ON customers.customer_id = orders.customer_id WHERE customers.city = 'Houston'"
        ),
    }
    value.update(overrides)
    return value


def test_schema_sql_uses_clean_identifiers_primary_keys_and_references():
    schema = schema_sql_from_metadata(metadata())

    assert "customer_id REAL PRIMARY KEY" in schema
    assert "customer_id REAL REFERENCES customers(customer_id)" in schema
    assert schema.count("CREATE TABLE") == 2


def test_convert_example_keeps_human_source_and_normalizes_supported_sql(tmp_path):
    record = convert_example(source(), 7, metadata(), database(tmp_path), "test")

    assert record["id"] == "spider_test_0007"
    assert record["source"]["human_authored"] is True
    assert record["source"]["training_use_allowed"] is False
    assert record["query_plan"]["family"] == "spider_join"
    assert record["database_path"] == "databases/shop.sqlite"
    assert record["sql"].startswith("SELECT orders.total FROM customers INNER JOIN orders")


def test_convert_example_rejects_non_foreign_key_join(tmp_path):
    bad = source(
        query=(
            "SELECT orders.total FROM customers JOIN orders "
            "ON customers.customer_id = orders.order_id WHERE customers.city = 'Houston'"
        )
    )

    with pytest.raises(SpiderRejected, match="join_is_not_declared_foreign_key"):
        convert_example(bad, 1, metadata(), database(tmp_path), "test")


def test_convert_example_rejects_unmentioned_filter_literal(tmp_path):
    bad = source(question="Show order totals for customers in the requested city")

    with pytest.raises(SpiderRejected, match="literal_not_mentioned_in_question"):
        convert_example(bad, 1, metadata(), database(tmp_path), "test")


def test_contamination_requires_a_record_level_pair_not_only_common_sql(tmp_path):
    training_path = tmp_path / "train.jsonl"
    training_record = {
        "id": "train_1",
        "schema_sql": "CREATE TABLE customers (name TEXT);",
        "question": "Show customer names",
        "sql": "SELECT name FROM customers;",
    }
    training_path.write_text(json.dumps(training_record) + "\n")
    indexes, hashes = training_contamination_index([training_path])

    same_pair = dict(training_record)
    common_shape = {
        "schema_sql": "CREATE TABLE vendors (name TEXT);",
        "question": "Show vendor names",
        "sql": "SELECT name FROM vendors;",
    }

    assert contamination_matches(same_pair, indexes)
    assert contamination_matches(common_shape, indexes) == {}
    assert str(training_path) in hashes


def test_human_curriculum_hits_fraction_and_preserves_source_labels():
    base = [{"id": f"base_{index}", "schema_id": f"base_schema_{index}"} for index in range(10)]
    human = [{"id": f"human_{index}", "schema_id": f"human_schema_{index}"} for index in range(2)]

    mixed, counts = mix_human_curriculum(base, human, 10, 0.3, 42)

    assert counts == {"v11": 7, "spider_human": 3}
    assert Counter(record["semantic_source"] for record in mixed) == counts
    assert len(mixed) == 10


def test_schema_linking_resampling_increases_hard_family_weight_deterministically():
    records = []
    for index in range(8):
        records.append(
            {
                "id": f"replay_{index}",
                "schema_id": f"schema_{index}",
                "query_plan": {"filters": [], "group_by": [], "join_table": None, "aggregate": None},
            }
        )
    for index in range(2):
        records.append(
            {
                "id": f"join_{index}",
                "schema_id": f"join_schema_{index}",
                "query_plan": {"filters": [], "group_by": [], "join_table": "other", "aggregate": None},
            }
        )

    selected, quotas = weighted_resample(
        records,
        20,
        {"replay": 1.0, "join": 4.0},
        7,
        "link",
    )

    assert quotas == {"replay": 10, "join": 10}
    assert Counter(linking_family(record) for record in selected) == quotas
    assert selected == weighted_resample(records, 20, {"replay": 1.0, "join": 4.0}, 7, "link")[0]
    assert all(record["semantic_source"] == "link" for record in selected)
