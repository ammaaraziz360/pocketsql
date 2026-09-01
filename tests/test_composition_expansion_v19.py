import json
from pathlib import Path
import sqlite3

from pocketsql.data.composition_expansion_v19 import build_composition_expansion_dataset
from pocketsql.data.composition_replay_v19 import build_composition_replay_mixture
from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.data.render_sql import render_sql
from pocketsql.data.validate import validate_sql


def _source_records(schema_id: str, parent: str, child: str) -> list[dict]:
    schema_sql = (
        f"CREATE TABLE {parent} (company_id INTEGER PRIMARY KEY, company_name TEXT, city TEXT);\n"
        f"CREATE TABLE {child} (order_id INTEGER PRIMARY KEY, company_id INTEGER "
        f"REFERENCES {parent}(company_id), total REAL, status TEXT);"
    )
    connection = sqlite3.connect(":memory:")
    connection.executescript(schema_sql)
    connection.executemany(
        f"INSERT INTO {parent} VALUES (?, ?, ?)",
        [(1, "Acme", "Dallas"), (2, "Nova", "Dallas"), (3, "Atlas", "Austin")],
    )
    connection.executemany(
        f"INSERT INTO {child} VALUES (?, ?, ?, ?)",
        [
            (1, 1, 10.0, "open"),
            (2, 1, 20.0, "closed"),
            (3, 2, 30.0, "open"),
            (4, 2, 40.0, "closed"),
            (5, 3, 50.0, "open"),
            (6, 3, 60.0, "closed"),
        ],
    )
    database_sql = "\n".join(connection.iterdump())
    connection.close()
    join_on = (f"{child}.company_id", f"{parent}.company_id")
    plans = {
        "v17_project_name": QueryPlan("source", parent, ("company_name",)),
        "v17_project_id_location": QueryPlan(
            "source", parent, ("company_id", "city")
        ),
        "v17_child_two_filter": QueryPlan(
            "source",
            child,
            ("order_id", "total"),
            filters=(Filter("status", "=", "open"), Filter("total", ">", 10)),
        ),
        "v17_join_amount_name": QueryPlan(
            "source",
            child,
            (f"{child}.total",),
            filters=(Filter(f"{parent}.company_name", "=", "Acme"),),
            join_table=parent,
            join_on=join_on,
        ),
    }
    return [
        {
            "id": f"{schema_id}:{index}",
            "schema_id": schema_id,
            "schema_sql": schema_sql,
            "database_sql": database_sql,
            "question": intent,
            "sql": render_sql(plan),
            "query_plan": plan.normalized(),
            "intent": intent,
        }
        for index, (intent, plan) in enumerate(plans.items())
    ]


def _write(path: Path, records: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_v19_composition_dataset_is_disjoint_explicit_and_execution_checked(tmp_path: Path):
    train_source = tmp_path / "train.jsonl"
    validation_source = tmp_path / "validation.jsonl"
    fresh_source = tmp_path / "fresh.jsonl"
    _write(train_source, _source_records("train_schema", "companies", "orders"))
    _write(validation_source, _source_records("validation_schema", "vendors", "shipments"))
    _write(fresh_source, _source_records("fresh_schema", "studios", "releases"))
    output = tmp_path / "composition_v19"

    report = build_composition_expansion_dataset(
        output, train_source, validation_source, fresh_source, seed=19
    )

    assert report["splits"]["train"]["records"] == 48
    assert report["splits"]["validation"]["records"] == 48
    assert report["splits"]["fresh_gate"]["records"] == 48
    assert report["splits"]["train"]["execution_checked"] == 48
    assert report["splits"]["train"]["explicit_schema_roles"]["rate"] == 1.0
    assert report["splits"]["train"]["counterfactual_groups"] == 7
    assert (
        report["splits"]["train"]["counterfactual_groups_with_distinct_results"]
        == 7
    )
    assert not any(report["isolation"].values())

    validation = [
        json.loads(line) for line in (output / "validation.jsonl").open(encoding="utf-8")
    ]
    assert all(record.get("database_sql") for record in validation)
    assert all(record["source"]["training_use_allowed"] is False for record in validation)
    for record in validation:
        connection = sqlite3.connect(":memory:")
        connection.executescript(record["database_sql"])
        valid, rows = validate_sql(connection, record["sql"])
        connection.close()
        assert valid and rows

    operators = {
        record["query_plan"]["filters"][0]["operator"]
        for record in validation
        if record["intent"].startswith("child_amount_")
    }
    assert operators == {">", "<", ">=", "<="}


def test_v19_replay_keeps_all_composition_and_samples_ordinary_data(tmp_path: Path):
    composition = tmp_path / "composition.jsonl"
    replay = tmp_path / "replay.jsonl"
    composition_records = [
        {
            "id": f"composition_{index}",
            "schema_sql": f"CREATE TABLE c{index} (id INTEGER);",
            "question": f"show composition {index}",
            "sql": f"SELECT id FROM c{index};",
            "query_plan": {"family": "compose"},
        }
        for index in range(2)
    ]
    replay_records = [
        {
            "id": f"replay_{index}",
            "schema_sql": f"CREATE TABLE r{index} (id INTEGER);",
            "question": f"show replay {index}",
            "sql": f"SELECT id FROM r{index};",
            "query_plan": {"family": "ordinary"},
            "v17_source": "test",
        }
        for index in range(6)
    ]
    _write(composition, composition_records)
    _write(replay, replay_records)

    report = build_composition_replay_mixture(
        tmp_path / "mixture", composition, replay, replay_records=4, seed=20
    )

    assert report["records"] == {
        "composition": 2,
        "ordinary_replay": 4,
        "total": 6,
    }
    records = [
        json.loads(line)
        for line in (tmp_path / "mixture" / "train.jsonl").open(encoding="utf-8")
    ]
    assert sum(record["v19_replay_kind"] == "composition" for record in records) == 2
    assert sum(record["v19_replay_kind"] == "ordinary" for record in records) == 4
