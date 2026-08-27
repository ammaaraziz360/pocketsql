from __future__ import annotations

import random
import sqlite3

from .schemas import STATUS_VALUES, Schema


def populate(connection: sqlite3.Connection, schema: Schema, rng: random.Random, rows: int = 12) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(schema.sql())
    parent, child = schema.tables
    parent_id = schema.role("parent_id")[1].name
    parent_name = schema.role("name")[1].name
    parent_location = schema.role("location")[1].name
    child_id = schema.role("child_id")[1].name
    child_parent_id = schema.role("parent_fk")[1].name
    child_amount = schema.role("amount")[1].name
    child_status = schema.role("status")[1].name
    cities = ("Austin", "Boston", "Chicago", "Denver")
    connection.executemany(
        f"INSERT INTO {parent.name} ({parent_id}, {parent_name}, {parent_location}) VALUES (?, ?, ?)",
        [(index, f"{schema.domain}_name_{index}", rng.choice(cities)) for index in range(1, rows + 1)],
    )
    # Each status receives examples in every amount band.  This guarantees the
    # high-threshold filter and join variants have non-empty gold results while
    # retaining per-schema variation in values and foreign-key assignments.
    amount_bands = (25, 80, 140, 210, 300, 420)
    child_rows = []
    for index in range(1, rows * 2 + 1):
        band = amount_bands[((index - 1) // len(STATUS_VALUES)) % len(amount_bands)]
        child_rows.append((index, rng.randint(1, rows), round(band + rng.uniform(0, 20), 2), STATUS_VALUES[index % len(STATUS_VALUES)]))
    connection.executemany(
        f"INSERT INTO {child.name} ({child_id}, {child_parent_id}, {child_amount}, {child_status}) VALUES (?, ?, ?, ?)",
        child_rows,
    )
    for column in parent.columns:
        if column.role and column.role.startswith("parent_extra_"):
            connection.executemany(
                f"UPDATE {parent.name} SET {column.name} = ? WHERE {parent_id} = ?",
                [(f"{column.role}_{index}", index) for index in range(1, rows + 1)],
            )
    for column in child.columns:
        if column.role and column.role.startswith("child_extra_"):
            connection.executemany(
                f"UPDATE {child.name} SET {column.name} = ? WHERE {child_id} = ?",
                [(f"{column.role}_{index}", index) for index in range(1, rows * 2 + 1)],
            )
    connection.commit()
