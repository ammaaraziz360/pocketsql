from pocketsql.evaluation.evaluate import evaluate


def make_record(family: str, sql: str) -> dict:
    return {
        "schema_sql": "CREATE TABLE t (id INTEGER, name TEXT);",
        "database_sql": "CREATE TABLE t (id INTEGER, name TEXT); INSERT INTO t VALUES (1, 'a'), (2, 'b');",
        "sql": sql,
        "difficulty": 1,
        "query_plan": {"family": family},
    }


def test_by_family_rates_are_fractions_not_raw_counts():
    records = [make_record("select", "SELECT name FROM t;") for _ in range(5)]
    result = evaluate(records, [record["sql"] for record in records])
    rates = result["by_family"]["difficulty_1:select"]
    assert all(0.0 <= value <= 1.0 for value in rates.values())
    assert rates["exact_match"] == 1.0


def test_by_family_rates_reflect_partial_correctness():
    records = [make_record("select", "SELECT name FROM t;") for _ in range(4)]
    predictions = ["SELECT name FROM t;", "SELECT name FROM t;", "not sql", "not sql"]
    result = evaluate(records, predictions)
    rates = result["by_family"]["difficulty_1:select"]
    assert rates["exact_match"] == 0.5


def test_by_schema_includes_metrics_and_example_counts():
    records = [make_record("select", "SELECT name FROM t;") for _ in range(2)]
    records[0]["schema_id"] = "schema_a"
    records[1]["schema_id"] = "schema_b"
    result = evaluate(records, [records[0]["sql"], "not sql"])
    assert result["by_schema"]["schema_a"]["examples"] == 1
    assert result["by_schema"]["schema_a"]["execution_accuracy"] == 1.0
    assert result["by_schema"]["schema_b"]["execution_accuracy"] == 0.0


def test_report_aggregates_query_family_and_input_schema_complexity():
    records = [make_record("select", "SELECT name FROM t;") for _ in range(2)]
    records[1]["schema_sql"] += " CREATE TABLE u (id INTEGER);"

    result = evaluate(records, [record["sql"] for record in records])

    assert result["records"] == 2
    assert result["by_query_family"]["select"]["examples"] == 2
    assert result["by_schema_complexity"]["1_tables"]["examples"] == 1
    assert result["by_schema_complexity"]["2_tables"]["examples"] == 1
    assert result["execution_accuracy_given_valid"] == 1.0


def test_failure_counts_distinguish_format_execution_and_result_errors():
    records = [make_record("select", "SELECT name FROM t;") for _ in range(3)]
    result = evaluate(records, ["not sql", "SELECT missing FROM t;", "SELECT id FROM t;"])
    assert result["failure_counts"]["unsafe_or_incomplete_output"]["count"] == 1
    assert result["failure_counts"]["sqlite_execution_error"]["count"] == 1
    assert result["failure_counts"]["wrong_result"]["count"] == 1


def test_unordered_execution_comparison_handles_null_and_text_rows():
    record = make_record("select", "SELECT name FROM t;")
    record["database_sql"] = "CREATE TABLE t (id INTEGER, name TEXT); INSERT INTO t VALUES (1, NULL), (2, 'b');"

    result = evaluate([record], ["SELECT name FROM t;"])

    assert result["execution_accuracy"] == 1.0


def test_counterfactual_pair_accuracy_requires_both_variants_to_be_correct():
    records = [make_record("select", "SELECT name FROM t;") for _ in range(4)]
    for index, record in enumerate(records):
        record["counterfactual_group"] = f"pair_{index // 2}"
        record["counterfactual_change"] = "projection"
    predictions = [records[0]["sql"], records[1]["sql"], records[2]["sql"], "SELECT id FROM t;"]

    result = evaluate(records, predictions)

    assert result["counterfactual_pairs"] == 2
    assert result["counterfactual_pair_accuracy"] == 0.5
    assert result["counterfactual_pair_accuracy_by_change"] == {"projection": 0.5}


def test_evaluation_opens_relative_sqlite_database_read_only(tmp_path):
    import sqlite3

    database_dir = tmp_path / "databases"
    database_dir.mkdir()
    database_path = database_dir / "tiny.sqlite"
    connection = sqlite3.connect(database_path)
    connection.executescript("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1), (2);")
    connection.close()
    record = make_record("select", "SELECT id FROM t;")
    record.pop("database_sql")
    record["database_path"] = "databases/tiny.sqlite"

    result = evaluate([record], ["SELECT id FROM t;"], tmp_path)

    assert result["execution_accuracy"] == 1.0


def test_record_database_base_dir_supports_training_loaded_records(tmp_path):
    import sqlite3

    database_dir = tmp_path / "databases"
    database_dir.mkdir()
    database_path = database_dir / "tiny.sqlite"
    connection = sqlite3.connect(database_path)
    connection.executescript("CREATE TABLE t (id INTEGER); INSERT INTO t VALUES (1);")
    connection.close()
    record = make_record("select", "SELECT id FROM t;")
    record.pop("database_sql")
    record["database_path"] = "databases/tiny.sqlite"
    record["_database_base_dir"] = str(tmp_path)

    result = evaluate([record], [record["sql"]])

    assert result["execution_accuracy"] == 1.0


def test_semantic_component_metrics_explain_query_errors():
    record = make_record("filter", "SELECT name FROM t WHERE id = 1;")
    record["query_plan"] = {
        "family": "filter",
        "table": "t",
        "columns": ["name"],
        "aggregate": None,
        "aggregate_column": None,
        "distinct": False,
        "filters": [{"column": "id", "operator": "=", "value": 1}],
        "filter_connector": "AND",
        "group_by": [],
        "order_by": None,
        "descending": False,
        "limit": None,
        "join_table": None,
        "join_on": None,
        "aggregate_position": 0,
    }

    result = evaluate([record], ["SELECT name FROM t WHERE id = 2;"])

    assert result["semantic_components"]["tables"] == 1.0
    assert result["semantic_components"]["projection"] == 1.0
    assert result["semantic_components"]["filters"] == 0.0
    assert result["semantic_components"]["semantic_plan_match"] == 0.0
