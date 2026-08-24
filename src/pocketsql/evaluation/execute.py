from __future__ import annotations

import sqlite3

from pocketsql.data.validate import is_read_only_select


def execute_read_only(schema_sql: str, sql: str, row_limit: int = 1000) -> list[tuple] | None:
    if not is_read_only_select(sql):
        return None
    connection = sqlite3.connect(":memory:")
    try:
        connection.executescript(schema_sql)
        connection.execute("PRAGMA query_only = ON")
        cursor = connection.execute(sql)
        return cursor.fetchmany(row_limit)
    except sqlite3.Error:
        return None
    finally:
        connection.close()