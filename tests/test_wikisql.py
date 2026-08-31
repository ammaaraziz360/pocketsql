from __future__ import annotations

from collections import Counter
import json

import pytest

from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.data.render_sql import render_sql
from pocketsql.data.wikisql import (
    DATASET_REVISION,
    WikiSQLRejected,
    _number,
    balanced_schema_take,
    convert_example,
    write_pilot,
)


def source_table(**overrides) -> dict:
    table = {
        "id": "1-100",
        "name": "table_100",
        "header": ["Order ID", "Status/Type", "Total $"],
        "types": ["real", "text", "real"],
        "rows": [[1, "Pending", 15], [2, "Complete", 5]],
    }
    table.update(overrides)
    return table


def source_example(**overrides) -> dict:
    source = {
        "question": "Which order id has status/type Pending and total greater than 10?",
        "table_id": "1-100",
        "sql": {"sel": 0, "agg": 0, "conds": [[1, 0, "Pending"], [2, 1, "10"]]},
    }
    source.update(overrides)
    return source


def test_convert_example_executes_and_preserves_copyable_text_literal():
    record = convert_example(source_table(), source_example(), "train")

    assert record["query_plan"]["family"] == "wikisql_filter"
    assert record["sql"] == (
        "SELECT order_id FROM table_100 WHERE status_type = 'Pending' AND total > 10;"
    )
    assert record["source"]["revision"] == DATASET_REVISION
    assert "INSERT INTO \"table_100\"" in record["database_sql"]


def test_oversized_integer_is_kept_as_sqlite_real():
    value = _number("999999999999999999999999")

    assert isinstance(value, float)


def test_reserved_header_is_safely_prefixed():
    table = source_table(
        header=["Order", "Status"],
        types=["real", "text"],
        rows=[[1, "Pending"]],
    )
    source = source_example(
        question="Which field order has status Pending?",
        sql={"sel": 0, "agg": 0, "conds": [[1, 0, "Pending"]]},
    )

    record = convert_example(table, source, "train")

    assert "field_order" in record["schema_sql"]
    assert record["sql"].startswith("SELECT field_order")


def test_count_is_normalized_to_star_without_requiring_arbitrary_source_column():
    table = source_table(header=["Unused Value", "Status"], types=["real", "text"], rows=[[1, "Pending"]])
    source = source_example(
        question="How many rows have status equal to Pending?",
        sql={"sel": 0, "agg": 3, "conds": [[1, 0, "Pending"]]},
    )

    record = convert_example(table, source, "train")

    assert record["sql"] == "SELECT COUNT(*) FROM table_100 WHERE status = 'Pending';"
    assert record["query_plan"]["aggregate_column"] is None


def test_convert_example_rejects_unresolved_required_column():
    source = source_example(question="Which value is Pending and greater than 10?")

    with pytest.raises(WikiSQLRejected, match="unresolved_schema_link"):
        convert_example(source_table(), source, "train")


def fake_candidate(split: str, index: int, family: str) -> dict:
    plan = QueryPlan(
        family,
        f"table_{split}_{index}",
        ("name",) if family == "wikisql_filter" else (),
        aggregate="COUNT" if family == "wikisql_aggregate" else None,
        filters=(Filter("status", "=", "open"),),
    )
    return {
        "id": f"{split}_{index}",
        "schema_id": f"schema_{split}_{index}",
        "schema_sql": f"CREATE TABLE table_{split}_{index} (name TEXT, status TEXT);",
        "database_sql": (
            f"CREATE TABLE table_{split}_{index} (name TEXT, status TEXT); "
            f"INSERT INTO table_{split}_{index} VALUES ('x', 'open');"
        ),
        "question": f"show name {index} where status is open",
        "sql": render_sql(plan),
        "query_plan": plan.normalized(),
        "difficulty": 1,
        "seed": None,
        "source": {"split": split},
    }


def candidates(split: str, count: int) -> list[dict]:
    return [
        fake_candidate(split, index, "wikisql_aggregate" if index % 3 == 0 else "wikisql_filter")
        for index in range(count)
    ]


def test_balanced_schema_take_respects_family_mix_and_schema_exclusions():
    selected = balanced_schema_take(candidates("test", 60), 12, 42, {"schema_test_1"})

    assert len(selected) == 12
    assert len({record["schema_id"] for record in selected}) == 12
    assert Counter(record["query_plan"]["family"] for record in selected) == {
        "wikisql_filter": 8,
        "wikisql_aggregate": 4,
    }
    assert "schema_test_1" not in {record["schema_id"] for record in selected}


def test_write_pilot_uses_official_splits_and_freezes_gate(tmp_path):
    pools = {
        "train": candidates("train", 60),
        "validation": candidates("validation", 30),
        "test": candidates("test", 60),
    }
    counts = write_pilot(
        tmp_path,
        pools,
        {name: Counter() for name in pools},
        pilot_records=30,
        gate_records=10,
        seed=91,
    )
    report = json.loads((tmp_path / "quality_report.json").read_text())

    assert counts == {"train": 24, "validation": 3, "test": 3, "external_gate": 10}
    assert report["schema_disjoint"] is True
    assert report["source"]["official_split_used"] is True
    assert report["selection"]["training_use_allowed"] == {
        "train": True,
        "validation": False,
        "test": False,
        "external_gate": False,
    }
