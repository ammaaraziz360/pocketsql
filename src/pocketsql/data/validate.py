from __future__ import annotations

import re
import sqlite3

FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA|REPLACE|VACUUM)\b", re.I)


def is_read_only_select(sql: str) -> bool:
    statement = sql.strip()
    if not re.match(r"^SELECT\b", statement, re.I) or FORBIDDEN.search(statement):
        return False
    return len([part for part in statement.split(";") if part.strip()]) == 1


def validate_sql(connection: sqlite3.Connection, sql: str, require_rows: bool = True) -> tuple[bool, list[tuple]]:
    if not is_read_only_select(sql):
        return False, []
    try:
        rows = connection.execute(sql).fetchall()
    except sqlite3.Error:
        return False, []
    return (bool(rows) or not require_rows), rows