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
