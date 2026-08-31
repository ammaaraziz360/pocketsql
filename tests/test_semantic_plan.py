from types import SimpleNamespace

import pytest

from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.data.render_sql import render_sql
from pocketsql.inference import _finish_target
from pocketsql.model.schema_grounding import canonicalize_inputs, canonicalize_record
from pocketsql.model.semantic_plan import (
    SemanticPlanError,
    parse_semantic_plan,
    semantic_plan_to_sql,
    serialize_semantic_plan,
)
from pocketsql.training.dataset import format_record


def complex_plan() -> QueryPlan:
    return QueryPlan(
        "join_multi_filter",
        "orders",
        ("orders.order_id",),
        aggregate="COUNT",
        distinct=True,
        filters=(
            Filter("customers.name", "=", "O'Brien & Sons"),
            Filter("customers.city", "=", "Dallas"),
        ),
        filter_connector="AND",
        group_by=("orders.order_id",),
        order_by="orders.order_id",
        descending=True,
        limit=5,
        join_table="customers",
        join_on=("orders.customer_id", "customers.customer_id"),
        aggregate_position=1,
    )


def test_semantic_plan_round_trip_preserves_rendered_sql():
    plan = complex_plan()
    encoded = serialize_semantic_plan(plan)
    decoded = parse_semantic_plan(encoded)

    assert encoded == (
        "T orders | S orders.order_id | A COUNT * 1 | D | "
        "J customers orders.customer_id customers.customer_id | "
        "F AND customers.name = 'O''Brien & Sons' & customers.city = 'Dallas' | "
        "G orders.order_id | O orders.order_id DESC | L 5"
    )
    assert render_sql(decoded) == render_sql(plan)


def test_semantic_plan_rejects_unknown_duplicate_and_unsafe_clauses():
    with pytest.raises(SemanticPlanError, match="unknown"):
        parse_semantic_plan("T customers | S name | DROP customers")
    with pytest.raises(SemanticPlanError, match="duplicate"):
        parse_semantic_plan("T customers | T orders | S name")
    with pytest.raises(SemanticPlanError, match="identifier"):
        parse_semantic_plan("T customers | S name;DELETE")
    with pytest.raises(SemanticPlanError, match="unterminated"):
        parse_semantic_plan("T customers | S name | F AND name = 'Max")


def test_training_target_and_inference_use_the_same_grounded_semantic_plan():
    schema = (
        "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT); "
        "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id));"
    )
    plan = QueryPlan(
        "join_multi_filter",
        "orders",
        ("orders.*",),
        filters=(Filter("customers.name", "=", "Max"), Filter("customers.city", "=", "Dallas")),
        join_table="customers",
        join_on=("orders.customer_id", "customers.customer_id"),
    )
    record = {
        "schema_sql": schema,
        "question": "show me customer orders where name is Max and city is Dallas",
        "sql": render_sql(plan),
        "query_plan": plan.normalized(),
    }
    canonical = canonicalize_record(record, "permuted", True)
    grounded_schema, grounded_question, mapping = canonicalize_inputs(schema, record["question"], "permuted", True)
    target = serialize_semantic_plan(canonical["query_plan"])

    assert canonical["schema_sql"] == grounded_schema
    assert "value0" in target and "value1" in target
    assert "Max" not in target and "Dallas" not in target
    assert _finish_target(
        target,
        SimpleNamespace(target_format="semantic_plan"),
        mapping,
        schema,
        grounded_question,
    ) == record["sql"]


def test_format_record_uses_plan_target_without_changing_the_legacy_envelope():
    plan = QueryPlan("select", "customers", ("name",))
    record = {
        "schema_sql": "CREATE TABLE customers (name TEXT);",
        "question": "show customer names",
        "sql": render_sql(plan),
        "query_plan": plan.normalized(),
    }

    formatted = format_record(record, target_format="semantic_plan")

    assert "<sql>T customers | S name</sql>" in formatted
    assert "SELECT name" not in formatted
    assert semantic_plan_to_sql("T customers | S name") == "SELECT name FROM customers;"
