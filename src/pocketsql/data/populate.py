from __future__ import annotations

import random
import sqlite3

from .schemas import CITY_VALUES, STATUS_VALUES, Schema


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
    cities = list(CITY_VALUES)
    rng.shuffle(cities)
    parent_rows = [
        (index, f"{schema.domain}_name_{index}", cities[(index - 1) % len(cities)])
        for index in range(1, rows + 1)
    ]
    connection.executemany(
        f"INSERT INTO {parent.name} ({parent_id}, {parent_name}, {parent_location}) VALUES (?, ?, ?)",
        parent_rows,
    )
    # Each status receives examples in every amount band.  This guarantees the
    # high-threshold filter and join variants have non-empty gold results while
    # retaining per-schema variation in values and foreign-key assignments.
    amount_bands = (25, 80, 140, 210, 300, 420)
    parents_by_city: dict[str, list[int]] = {city: [] for city in CITY_VALUES}
    for identifier, _, city in parent_rows:
        parents_by_city[city].append(identifier)
    for identifiers in parents_by_city.values():
        rng.shuffle(identifiers)
    child_rows = []
    index = 1
    # Every location/status combination receives a child. This makes composed
    # join predicates discriminative instead of silently producing empty gold
    # results merely because two independently sampled filters never co-occurred.
    for city_index, city in enumerate(CITY_VALUES):
        identifiers = parents_by_city[city]
        for status_index, status_value in enumerate(STATUS_VALUES):
            band = amount_bands[city_index % len(amount_bands)]
            child_rows.append(
                (
                    index,
                    identifiers[status_index % len(identifiers)],
                    round(band + rng.uniform(0, 20), 2),
                    status_value,
                )
            )
            index += 1
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
