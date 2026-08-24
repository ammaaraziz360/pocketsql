import sqlite3

from pocketsql.data.generate import build_records
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


def test_population_respects_foreign_keys():
    schema = make_schema(0, __import__("random").Random(4))
    connection = sqlite3.connect(":memory:")
    populate(connection, schema, __import__("random").Random(4))
    child = schema.tables[1]
    assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    assert connection.execute(f"SELECT COUNT(*) FROM {child.name}").fetchone()[0] > 0