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


def test_failure_counts_distinguish_format_execution_and_result_errors():
    records = [make_record("select", "SELECT name FROM t;") for _ in range(3)]
    result = evaluate(records, ["not sql", "SELECT missing FROM t;", "SELECT id FROM t;"])
    assert result["failure_counts"]["unsafe_or_incomplete_output"]["count"] == 1
    assert result["failure_counts"]["sqlite_execution_error"]["count"] == 1
    assert result["failure_counts"]["wrong_result"]["count"] == 1
