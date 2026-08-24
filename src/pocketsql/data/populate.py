from __future__ import annotations

import random
import sqlite3

from .schemas import Schema


def populate(connection: sqlite3.Connection, schema: Schema, rng: random.Random, rows: int = 12) -> None:
    connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(schema.sql())
    parent, child = schema.tables
    cities = ("Austin", "Boston", "Chicago", "Denver")
    connection.executemany(
        f"INSERT INTO {parent.name} VALUES (?, ?, ?)",
        [(index, f"{schema.domain}_name_{index}", rng.choice(cities)) for index in range(1, rows + 1)],
    )
    status_column = child.columns[-1].name
    states = ("shipped", "open", "complete", "cancelled")
    connection.executemany(
        f"INSERT INTO {child.name} VALUES (?, ?, ?, ?)",
        [(index, rng.randint(1, rows), round(rng.uniform(5, 500), 2), states[index % len(states)]) for index in range(1, rows * 2 + 1)],
    )
    connection.commit()