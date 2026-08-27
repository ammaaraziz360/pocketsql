from __future__ import annotations

from dataclasses import dataclass
import random
import string


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    primary_key: bool = False
    references: tuple[str, str] | None = None
    role: str | None = None


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

    def role(self, role: str) -> tuple[Table, Column]:
        for table in self.tables:
            for column in table.columns:
                if column.role == role:
                    return table, column
        raise KeyError(f"Schema {self.schema_id} has no column role {role!r}")


STATUS_VALUES = ("shipped", "open", "complete", "cancelled")

# Per-domain synonym pools so different schema instances of the same domain get
# genuinely distinct table/column vocabulary, not just a repeated fixed template.
DOMAIN_VOCAB = {
    "retail": {
        "parent_table": ("customers", "shoppers", "clients", "buyers", "accounts", "customer_records"),
        "child_table": ("orders", "purchases", "invoices", "sales", "transactions", "order_records"),
        "id_column": ("customer_id", "client_id", "account_id", "buyer_id", "customer_key", "account_key"),
        "child_id_column": ("order_id", "purchase_id", "invoice_id", "sale_id", "transaction_id", "order_key"),
        "name_column": ("name", "full_name", "customer_name", "display_name", "contact_name", "account_name"),
        "location_column": ("city", "region", "home_city", "market", "territory", "postal_area"),
        "amount_column": ("total", "amount", "order_total", "net_amount", "sale_value", "invoice_total"),
        "status_column": ("status", "state", "category", "order_state", "lifecycle", "sale_status"),
    },
    "library": {
        "parent_table": ("members", "patrons", "borrowers", "readers", "cardholders", "library_users"),
        "child_table": ("loans", "checkouts", "borrowings", "circulations", "lends", "loan_records"),
        "id_column": ("member_id", "patron_id", "borrower_id", "reader_id", "cardholder_id", "member_key"),
        "child_id_column": ("loan_id", "checkout_id", "borrowing_id", "circulation_id", "lend_id", "loan_key"),
        "name_column": ("name", "full_name", "member_name", "reader_name", "display_name", "contact_name"),
        "location_column": ("city", "region", "home_city", "branch_area", "district", "postal_area"),
        "amount_column": ("fee", "fine", "charge", "balance", "penalty", "amount_due"),
        "status_column": ("status", "state", "category", "loan_state", "circulation_status", "phase"),
    },
    "school": {
        "parent_table": ("students", "learners", "pupils", "scholars", "enrollees", "student_records"),
        "child_table": ("enrollments", "registrations", "sections", "courses", "class_records", "placements"),
        "id_column": ("student_id", "learner_id", "pupil_id", "scholar_id", "enrollee_id", "student_key"),
        "child_id_column": ("enrollment_id", "registration_id", "section_id", "course_id", "class_id", "placement_id"),
        "name_column": ("name", "full_name", "student_name", "learner_name", "display_name", "legal_name"),
        "location_column": ("city", "region", "home_city", "district", "campus", "postal_area"),
        "amount_column": ("score", "grade", "gpa", "mark", "credit_value", "result"),
        "status_column": ("status", "state", "category", "enrollment_state", "standing", "course_status"),
    },
    "restaurant": {
        "parent_table": ("diners", "guests", "patrons", "visitors", "tableside_guests", "guest_records"),
        "child_table": ("visits", "reservations", "checks", "tabs", "bookings", "dining_records"),
        "id_column": ("diner_id", "guest_id", "patron_id", "visitor_id", "guest_key", "diner_key"),
        "child_id_column": ("visit_id", "reservation_id", "check_id", "tab_id", "booking_id", "dining_id"),
        "name_column": ("name", "full_name", "guest_name", "diner_name", "display_name", "contact_name"),
        "location_column": ("city", "region", "home_city", "neighborhood", "district", "postal_area"),
        "amount_column": ("amount", "bill_total", "check_total", "tab_total", "spend", "meal_total"),
        "status_column": ("status", "state", "category", "booking_status", "visit_state", "service_phase"),
    },
    "events": {
        "parent_table": ("attendees", "registrants", "guests", "participants", "delegates", "event_people"),
        "child_table": ("tickets", "registrations", "passes", "admissions", "badges", "ticket_records"),
        "id_column": ("attendee_id", "registrant_id", "guest_id", "participant_id", "delegate_id", "attendee_key"),
        "child_id_column": ("ticket_id", "registration_id", "pass_id", "admission_id", "badge_id", "ticket_key"),
        "name_column": ("name", "full_name", "attendee_name", "participant_name", "display_name", "contact_name"),
        "location_column": ("city", "region", "home_city", "venue_area", "district", "postal_area"),
        "amount_column": ("price", "fee", "ticket_price", "admission_fee", "cost", "registration_fee"),
        "status_column": ("status", "state", "category", "registration_state", "attendance_status", "pass_state"),
    },
}

# This vocabulary is deliberately isolated from DOMAIN_VOCAB. It is used only
# for the realistic challenge set, so its identifiers are lexically unseen by
# models trained on the standard synthetic corpus.
CHALLENGE_VOCAB = {
    "commerce": {
        "parent_table": ("organizations", "merchants", "vendors", "subscribers"),
        "child_table": ("contracts", "statements", "settlements", "payments"),
        "id_column": ("organization_key", "merchant_key", "vendor_key", "subscriber_key"),
        "child_id_column": ("contract_key", "statement_key", "settlement_key", "payment_key"),
        "name_column": ("legal_title", "trading_name", "vendor_label", "subscriber_label"),
        "location_column": ("billing_zone", "service_area", "operating_region", "postal_sector"),
        "amount_column": ("contract_value", "statement_balance", "settlement_value", "payment_value"),
        "status_column": ("contract_phase", "statement_state", "settlement_phase", "payment_state"),
    },
    "logistics": {
        "parent_table": ("depots", "carriers", "warehouses", "suppliers"),
        "child_table": ("shipments", "manifests", "deliveries", "dispatches"),
        "id_column": ("depot_key", "carrier_key", "warehouse_key", "supplier_key"),
        "child_id_column": ("shipment_key", "manifest_key", "delivery_key", "dispatch_key"),
        "name_column": ("depot_label", "carrier_name", "warehouse_label", "supplier_name"),
        "location_column": ("service_zone", "route_region", "storage_region", "supply_region"),
        "amount_column": ("freight_cost", "manifest_value", "delivery_cost", "dispatch_cost"),
        "status_column": ("shipment_phase", "manifest_state", "delivery_phase", "dispatch_state"),
    },
    "software": {
        "parent_table": ("workspaces", "tenants", "projects", "repositories"),
        "child_table": ("deployments", "builds", "releases", "jobs"),
        "id_column": ("workspace_key", "tenant_key", "project_key", "repository_key"),
        "child_id_column": ("deployment_key", "build_key", "release_key", "job_key"),
        "name_column": ("workspace_label", "tenant_label", "project_title", "repository_title"),
        "location_column": ("hosting_region", "service_region", "deployment_zone", "runtime_zone"),
        "amount_column": ("compute_cost", "build_cost", "release_cost", "job_cost"),
        "status_column": ("deployment_state", "build_state", "release_phase", "job_state"),
    },
}


def _style_identifier(name: str, style: str, is_table: bool) -> str:
    if style == "plain":
        return name
    if style == "camel":
        parts = name.split("_")
        return parts[0] + "".join(part.capitalize() for part in parts[1:])
    if style == "prefixed":
        return ("tbl_" if is_table else "col_") + name
    return name + ("_data" if is_table else "_value")


def make_schema(index: int, rng: random.Random, vocabularies: dict | None = None, schema_prefix: str = "schema") -> Schema:
    vocabularies = vocabularies or DOMAIN_VOCAB
    domain = list(vocabularies)[index % len(vocabularies)]
    vocab = vocabularies[domain]
    style = rng.choice(("plain", "camel", "prefixed", "suffixed"))
    parent_table = _style_identifier(rng.choice(vocab["parent_table"]), style, True)
    child_table = _style_identifier(rng.choice(vocab["child_table"]), style, True)
    id_column = _style_identifier(rng.choice(vocab["id_column"]), style, False)
    child_id_column = _style_identifier(rng.choice(vocab["child_id_column"]), style, False)
    name_column = _style_identifier(rng.choice(vocab["name_column"]), style, False)
    location_column = _style_identifier(rng.choice(vocab["location_column"]), style, False)
    amount_column = _style_identifier(rng.choice(vocab["amount_column"]), style, False)
    status_column = _style_identifier(rng.choice(vocab["status_column"]), style, False)
    parent_extra_names = rng.sample(("created_at", "contact_note", "source_code", "active_flag", "segment_note", "external_ref"), rng.randint(1, 3))
    child_extra_names = rng.sample(("reference_code", "processed_at", "source_channel", "remarks", "batch_note", "external_ref"), rng.randint(1, 3))
    parent_columns = [
        Column(id_column, "INTEGER", True, role="parent_id"),
        Column(name_column, "TEXT", role="name"),
        Column(location_column, "TEXT", role="location"),
        *[
            Column(_style_identifier(name, style, False), "TEXT", role=f"parent_extra_{extra_index}")
            for extra_index, name in enumerate(parent_extra_names)
        ],
    ]
    child_columns = [
        Column(child_id_column, "INTEGER", True, role="child_id"),
        Column(id_column, "INTEGER", references=(parent_table, id_column), role="parent_fk"),
        Column(amount_column, "REAL", role="amount"),
        Column(status_column, "TEXT", role="status"),
        *[
            Column(_style_identifier(name, style, False), "TEXT", role=f"child_extra_{extra_index}")
            for extra_index, name in enumerate(child_extra_names)
        ],
    ]
    rng.shuffle(parent_columns)
    rng.shuffle(child_columns)
    parent = Table(parent_table, tuple(parent_columns))
    child = Table(child_table, tuple(child_columns))
    return Schema(f"{schema_prefix}_{index:04d}", domain, (parent, child))


def make_opaque_schema(index: int, rng: random.Random, schema_prefix: str = "opaque") -> Schema:
    """Build a structurally normal schema whose identifiers carry no semantic hints."""
    used: set[str] = set()

    def identifier() -> str:
        while True:
            candidate = "x_" + "".join(rng.choice(string.ascii_lowercase) for _ in range(10))
            if candidate not in used:
                used.add(candidate)
                return candidate

    parent_table, child_table = identifier(), identifier()
    parent_id, child_id = identifier(), identifier()
    name, location, amount, status = identifier(), identifier(), identifier(), identifier()
    extra_parent, extra_child = identifier(), identifier()
    parent = Table(
        parent_table,
        (
            Column(parent_id, "INTEGER", True, role="parent_id"),
            Column(name, "TEXT", role="name"),
            Column(location, "TEXT", role="location"),
            Column(extra_parent, "TEXT", role="parent_extra_0"),
        ),
    )
    child = Table(
        child_table,
        (
            Column(child_id, "INTEGER", True, role="child_id"),
            Column(parent_id, "INTEGER", references=(parent_table, parent_id), role="parent_fk"),
            Column(amount, "REAL", role="amount"),
            Column(status, "TEXT", role="status"),
            Column(extra_child, "TEXT", role="child_extra_0"),
        ),
    )
    return Schema(f"{schema_prefix}_{index:04d}", "opaque", (parent, child))
