from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.evaluation.schema_oracle import (
    apply_literal_oracle,
    apply_operation_oracle,
    apply_schema_oracle,
)


def test_all_schema_links_preserve_decoded_operations():
    predicted = QueryPlan(
        family="semantic_plan",
        table="table9",
        columns=("column9",),
        aggregate="SUM",
        aggregate_column="column8",
        distinct=True,
        filters=(Filter("column7", ">", "value0"),),
        filter_connector="OR",
        group_by=("column6",),
        order_by="column5",
        descending=True,
        limit=3,
        join_table="table8",
        join_on=("table9.column1", "table8.column2"),
        aggregate_position=1,
    )
    gold = QueryPlan(
        family="semantic_plan",
        table="table0",
        columns=("table0.column3",),
        aggregate="COUNT",
        aggregate_column=None,
        filters=(Filter("table1.column4", "=", "value0"),),
        group_by=("table0.column3",),
        order_by="table0.column3",
        join_table="table1",
        join_on=("table0.column0", "table1.column1"),
    )

    oracle = apply_schema_oracle(predicted, gold, "all_schema_links")

    assert oracle.table == gold.table
    assert oracle.join_table == gold.join_table
    assert oracle.join_on == gold.join_on
    assert oracle.columns == gold.columns
    assert oracle.aggregate_column == gold.columns[0]
    assert oracle.filters[0].column == gold.filters[0].column
    assert oracle.group_by == gold.group_by
    assert oracle.order_by == gold.order_by
    assert oracle.aggregate == "SUM"
    assert oracle.distinct is True
    assert oracle.filters[0].operator == ">"
    assert oracle.filters[0].value == "value0"
    assert oracle.filter_connector == "OR"
    assert oracle.descending is True
    assert oracle.limit == 3
    assert oracle.aggregate_position == 1


def test_projection_oracle_preserves_predicted_selection_arity():
    predicted = QueryPlan("semantic_plan", "table0", ("column8", "column9"))
    gold = QueryPlan("semantic_plan", "table0", ("column1",))

    oracle = apply_schema_oracle(predicted, gold, "projection")

    assert oracle.columns == ("column1", "column1")


def test_operation_oracle_uses_gold_shape_but_keeps_decoded_filter_values():
    predicted = QueryPlan(
        family="semantic_plan",
        table="table9",
        columns=("column9",),
        filters=(Filter("column8", ">", "value1"),),
        limit=7,
    )
    gold = QueryPlan(
        family="semantic_plan",
        table="table0",
        columns=(),
        aggregate="COUNT",
        filters=(
            Filter("table1.column2", "=", "value0"),
            Filter("table1.column3", "<", "value2"),
        ),
        filter_connector="OR",
        join_table="table1",
        join_on=("table0.column0", "table1.column1"),
    )

    oracle = apply_operation_oracle(predicted, gold)

    assert oracle.aggregate == "COUNT"
    assert oracle.columns == ()
    assert oracle.join_table == "table1"
    assert oracle.filter_connector == "OR"
    assert [item.operator for item in oracle.filters] == ["=", "<"]
    assert oracle.filters[0].value == "value1"
    assert str(oracle.filters[1].value).startswith("__oracle_missing_literal_")
    assert oracle.limit is None


def test_literal_oracle_changes_only_filter_values():
    predicted = QueryPlan(
        family="semantic_plan",
        table="table9",
        columns=("column9",),
        aggregate="SUM",
        aggregate_column="column9",
        filters=(Filter("column8", ">", "value1"),),
        filter_connector="OR",
        descending=True,
        limit=7,
    )
    gold = QueryPlan(
        family="semantic_plan",
        table="table0",
        columns=("column1",),
        filters=(Filter("column2", "=", "value0"),),
    )

    oracle = apply_literal_oracle(predicted, gold)

    assert oracle.filters == (Filter("column8", ">", "value0"),)
    assert oracle.table == predicted.table
    assert oracle.columns == predicted.columns
    assert oracle.aggregate == predicted.aggregate
    assert oracle.filter_connector == predicted.filter_connector
    assert oracle.descending == predicted.descending
    assert oracle.limit == predicted.limit
