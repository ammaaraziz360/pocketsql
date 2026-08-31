from __future__ import annotations

from collections import Counter
import json

import pytest

from pocketsql.data.gretel import (
    DATASET_REVISION,
    GretelRejected,
    balanced_take,
    convert_row,
    mix_records,
    query_plan_from_sql,
    write_pilot,
)
from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.data.render_sql import render_sql


def source_row(**overrides) -> dict:
    row = {
        "id": 17,
        "domain": "retail",
        "sql_complexity": "basic SQL",
        "sql_task_type": "analytics and reporting",
        "sql_prompt": "Show customer names in Houston",
        "sql_context": (
            "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT);"
            "INSERT INTO customers VALUES (1, 'Ada', 'Houston'), (2, 'Lin', 'Boston');"
        ),
        "sql": "SELECT c.name FROM customers c WHERE c.city = 'Houston';",
    }
    row.update(overrides)
    return row


def test_query_plan_normalizes_aliases_and_supported_operations():
    plan = query_plan_from_sql(
        "SELECT c.name, o.total FROM customers c JOIN orders o ON c.customer_id = o.customer_id "
        "WHERE c.city = 'Houston' AND o.total >= 50 ORDER BY o.total DESC LIMIT 5;"
    )

    assert plan.family == "gretel_join"
    assert plan.columns == ("customers.name", "orders.total")
    assert plan.join_on == ("customers.customer_id", "orders.customer_id")
    assert plan.filters == (
        Filter("customers.city", "=", "Houston"),
        Filter("orders.total", ">=", 50),
    )
    assert render_sql(plan) == (
        "SELECT customers.name, orders.total FROM customers INNER JOIN orders "
        "ON customers.customer_id = orders.customer_id WHERE customers.city = 'Houston' "
        "AND orders.total >= 50 ORDER BY orders.total DESC LIMIT 5;"
    )


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers;",
        "SELECT name FROM customers WHERE city IN ('Houston', 'Boston');",
        "SELECT name FROM customers WHERE city = 'Houston' OR name = 'Ada' AND customer_id > 1;",
        "SELECT status, SUM(total), COUNT(*) FROM orders GROUP BY status;",
        "SELECT name FROM customers UNION SELECT name FROM prospects;",
    ],
)
def test_query_plan_rejects_queries_outside_current_ir(sql):
    with pytest.raises(GretelRejected):
        query_plan_from_sql(sql)


def test_convert_row_executes_and_checks_grounding_equivalence():
    record = convert_row(source_row(), "test")

    assert record["sql"] == "SELECT customers.name FROM customers WHERE customers.city = 'Houston';"
    assert record["query_plan"]["family"] == "gretel_filter"
    assert record["source"]["revision"] == DATASET_REVISION
    assert "INSERT INTO \"customers\"" in record["database_sql"]


def test_query_plan_preserves_grouped_aggregate_output_position():
    plan = query_plan_from_sql("SELECT status, SUM(total) FROM orders GROUP BY status;")

    assert plan.aggregate_position == 1
    assert render_sql(plan) == "SELECT status, SUM(total) FROM orders GROUP BY status;"


def test_convert_row_rejects_literal_not_grounded_by_question():
    with pytest.raises(GretelRejected, match="literal_not_mentioned_in_question"):
        convert_row(source_row(sql_prompt="Show customer names in the requested city"), "train")


def test_convert_row_rejects_unsafe_context():
    with pytest.raises(GretelRejected, match="unsafe_or_unparseable_context"):
        convert_row(source_row(sql_context="PRAGMA user_version = 1; CREATE TABLE customers (name TEXT);"), "train")


def fake_candidate(index: int, family: str) -> dict:
    plan = QueryPlan(family, f"table_{index}", ("name",))
    return {
        "id": f"external_{index}",
        "schema_id": f"schema_{index}",
        "schema_sql": f"CREATE TABLE table_{index} (name TEXT);",
        "database_sql": f"CREATE TABLE table_{index} (name TEXT); INSERT INTO table_{index} VALUES ('x');",
        "question": f"show name {index}",
        "sql": render_sql(plan),
        "query_plan": plan.normalized(),
        "difficulty": 1,
        "seed": None,
        "source": {"domain": f"domain_{index % 5}"},
    }


def test_balanced_take_never_reuses_a_schema():
    families = tuple(
        (
            "gretel_select",
            "gretel_filter",
            "gretel_aggregate",
            "gretel_group",
            "gretel_join",
            "gretel_distinct",
            "gretel_order_limit",
        )
    )
    candidates = [fake_candidate(index, families[index % len(families)]) for index in range(70)]
    selected = balanced_take(candidates, 30, 42, {"schema_0", "schema_1"})

    assert len(selected) == 30
    assert len({record["schema_id"] for record in selected}) == 30
    assert not {"schema_0", "schema_1"} & {record["schema_id"] for record in selected}


def test_write_pilot_keeps_gate_evaluation_only_and_all_splits_disjoint(tmp_path):
    families = tuple(
        (
            "gretel_select",
            "gretel_filter",
            "gretel_aggregate",
            "gretel_group",
            "gretel_join",
            "gretel_distinct",
            "gretel_order_limit",
        )
    )
    candidates = [fake_candidate(index, families[index % len(families)]) for index in range(80)]
    counts = write_pilot(tmp_path, candidates, Counter(), 30, 10, 91)
    report = json.loads((tmp_path / "quality_report.json").read_text())

    assert counts == {"train": 24, "validation": 3, "test": 3, "external_gate": 10}
    assert report["schema_disjoint"] is True
    assert report["selection"]["training_use_allowed"]["external_gate"] is False
    assert report["selection"]["training_use_allowed"]["validation"] is False
    assert report["selection"]["training_use_allowed"]["test"] is False
    assert report["selection"]["training_use_allowed"]["train"] is True
    assert report["source"]["official_split_used"] is False
    assert report["source"]["resplit_key"] == "normalized_schema_sha256"


def test_mix_records_hits_requested_fraction_without_mutating_inputs():
    reference = [{"id": f"reference_{index}"} for index in range(16)]
    external = [{"id": f"external_{index}"} for index in range(4)]

    mixed = mix_records(reference, external, 0.2, 12)

    assert len(mixed) == 20
    assert sum(record["id"].startswith("external") for record in mixed) == 4
    assert len(reference) == 16


def test_mix_records_can_keep_total_size_while_reducing_external_fraction():
    reference = [{"id": f"reference_{index}"} for index in range(30)]
    external = [{"id": f"external_{index}"} for index in range(10)]

    mixed = mix_records(reference, external, 0.1, 12, total_records=20)

    assert len(mixed) == 20
    assert sum(record["id"].startswith("external") for record in mixed) == 2
