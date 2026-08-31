from __future__ import annotations

from pathlib import Path
import sqlite3

from pocketsql.data.validate import is_read_only_select


def _bounded(connection: sqlite3.Connection, max_progress_calls: int = 500) -> None:
    progress_calls = 0

    def progress() -> int:
        nonlocal progress_calls
        progress_calls += 1
        return int(progress_calls > max_progress_calls)

    connection.set_progress_handler(progress, 10_000)


def execute_read_only(schema_sql: str, sql: str, row_limit: int = 1000) -> list[tuple] | None:
    if not is_read_only_select(sql):
        return None
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(schema_sql)
        connection.execute("PRAGMA query_only = ON")
        _bounded(connection)
        cursor = connection.execute(sql)
        return cursor.fetchmany(row_limit)
    except sqlite3.Error:
        return None
    finally:
        connection.close()


def execute_read_only_database(database_path: Path, sql: str, row_limit: int = 1000) -> list[tuple] | None:
    """Execute one checked query against an existing SQLite file without allowing writes."""
    if not is_read_only_select(sql):
        return None
    try:
        connection = sqlite3.connect(database_path.resolve().as_uri() + "?mode=ro", uri=True)
    except sqlite3.Error:
        return None
    try:
        connection.execute("PRAGMA query_only = ON")
        _bounded(connection)
        cursor = connection.execute(sql)
        return cursor.fetchmany(row_limit)
    except sqlite3.Error:
        return None
    finally:
        connection.close()
