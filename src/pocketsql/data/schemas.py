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


STATUS_VALUES = ("shipped", "open", "complete", "cancelled")

# Per-domain synonym pools so different schema instances of the same domain get
# genuinely distinct table/column vocabulary, not just a repeated fixed template.
DOMAIN_VOCAB = {
    "retail": {
        "parent_table": ("customers", "shoppers", "clients"),
        "child_table": ("orders", "purchases", "invoices"),
        "id_column": ("customer_id", "client_id", "account_id"),
        "child_id_column": ("order_id", "purchase_id", "invoice_id"),
        "name_column": ("name", "full_name", "customer_name"),
        "location_column": ("city", "region", "home_city"),
        "amount_column": ("total", "amount", "order_total"),
        "status_column": ("status", "state", "category"),
    },
    "library": {
        "parent_table": ("members", "patrons", "borrowers"),
        "child_table": ("loans", "checkouts", "borrowings"),
        "id_column": ("member_id", "patron_id", "borrower_id"),
        "child_id_column": ("loan_id", "checkout_id", "borrowing_id"),
        "name_column": ("name", "full_name", "member_name"),
        "location_column": ("city", "region", "home_city"),
        "amount_column": ("fee", "fine", "charge"),
        "status_column": ("status", "state", "category"),
    },
    "school": {
        "parent_table": ("students", "learners", "pupils"),
        "child_table": ("enrollments", "registrations", "sections"),
        "id_column": ("student_id", "learner_id", "pupil_id"),
        "child_id_column": ("enrollment_id", "registration_id", "section_id"),
        "name_column": ("name", "full_name", "student_name"),
        "location_column": ("city", "region", "home_city"),
        "amount_column": ("score", "grade", "gpa"),
        "status_column": ("status", "state", "category"),
    },
    "restaurant": {
        "parent_table": ("diners", "guests", "patrons"),
        "child_table": ("visits", "reservations", "checks"),
        "id_column": ("diner_id", "guest_id", "patron_id"),
        "child_id_column": ("visit_id", "reservation_id", "check_id"),
        "name_column": ("name", "full_name", "guest_name"),
        "location_column": ("city", "region", "home_city"),
        "amount_column": ("amount", "bill_total", "check_total"),
        "status_column": ("status", "state", "category"),
    },
    "events": {
        "parent_table": ("attendees", "registrants", "guests"),
        "child_table": ("tickets", "registrations", "passes"),
        "id_column": ("attendee_id", "registrant_id", "guest_id"),
        "child_id_column": ("ticket_id", "registration_id", "pass_id"),
        "name_column": ("name", "full_name", "attendee_name"),
        "location_column": ("city", "region", "home_city"),
        "amount_column": ("price", "fee", "ticket_price"),
        "status_column": ("status", "state", "category"),
    },
}


def make_schema(index: int, rng: random.Random) -> Schema:
    domain = list(DOMAIN_VOCAB)[index % len(DOMAIN_VOCAB)]
    vocab = DOMAIN_VOCAB[domain]
    parent_table = rng.choice(vocab["parent_table"])
    child_table = rng.choice(vocab["child_table"])
    id_column = rng.choice(vocab["id_column"])
    child_id_column = rng.choice(vocab["child_id_column"])
    name_column = rng.choice(vocab["name_column"])
    location_column = rng.choice(vocab["location_column"])
    amount_column = rng.choice(vocab["amount_column"])
    status_column = rng.choice(vocab["status_column"])
    parent = Table(parent_table, (Column(id_column, "INTEGER", True), Column(name_column, "TEXT"), Column(location_column, "TEXT")))
    child = Table(child_table, (Column(child_id_column, "INTEGER", True), Column(id_column, "INTEGER", references=(parent_table, id_column)), Column(amount_column, "REAL"), Column(status_column, "TEXT")))
    return Schema(f"schema_{index:04d}", domain, (parent, child))