from __future__ import annotations

from dataclasses import dataclass
import random


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    primary_key: bool = False
    references: tuple[str, str] | None = None


@dataclass(frozen=True)
class Table:
    name: str
    columns: tuple[Column, ...]


@dataclass(frozen=True)
class Schema:
    schema_id: str
    domain: str
    tables: tuple[Table, ...]

    def sql(self) -> str:
        statements = []
        for table in self.tables:
            fields = []
            for column in table.columns:
                field = f"{column.name} {column.type}"
                if column.primary_key:
                    field += " PRIMARY KEY"
                if column.references:
                    field += f" REFERENCES {column.references[0]}({column.references[1]})"
                fields.append(field)
            statements.append(f"CREATE TABLE {table.name} ({', '.join(fields)});")
        return "\n".join(statements)

    def table(self, name: str) -> Table:
        return next(table for table in self.tables if table.name == name)


DOMAINS = {
    "retail": (("customers", ("customer_id", "name", "city")), ("orders", ("order_id", "customer_id", "total", "status"))),
    "library": (("members", ("member_id", "name", "city")), ("loans", ("loan_id", "member_id", "fee", "state"))),
    "school": (("students", ("student_id", "name", "city")), ("enrollments", ("enrollment_id", "student_id", "score", "status"))),
    "restaurant": (("diners", ("diner_id", "name", "city")), ("visits", ("visit_id", "diner_id", "amount", "status"))),
    "events": (("attendees", ("attendee_id", "name", "city")), ("tickets", ("ticket_id", "attendee_id", "price", "status"))),
}


def make_schema(index: int, rng: random.Random) -> Schema:
    domain = list(DOMAINS)[index % len(DOMAINS)]
    (parent_name, parent_columns), (child_name, child_columns) = DOMAINS[domain]
    parent_id, _, _ = parent_columns
    child_id, foreign_key, amount, status = child_columns
    if rng.random() < 0.5:
        status = rng.choice([status, "state", "category"])
    parent = Table(parent_name, (Column(parent_id, "INTEGER", True), Column("name", "TEXT"), Column("city", "TEXT")))
    child = Table(child_name, (Column(child_id, "INTEGER", True), Column(foreign_key, "INTEGER", references=(parent_name, parent_id)), Column(amount, "REAL"), Column(status, "TEXT")))
    return Schema(f"schema_{index:04d}", domain, (parent, child))