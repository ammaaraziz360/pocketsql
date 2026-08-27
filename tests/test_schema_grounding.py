from pocketsql.inference import _finish_sql
from pocketsql.model.schema_grounding import canonicalize_inputs, canonicalize_record
from pocketsql.model.tokenizer import ByteTokenizer
from pocketsql.training.dataset import encode_record, format_record


SCHEMA = """CREATE TABLE organizations (organization_key INTEGER PRIMARY KEY, legal_title TEXT, billing_zone TEXT);
CREATE TABLE contracts (contract_key INTEGER PRIMARY KEY, organization_key INTEGER REFERENCES organizations(organization_key), contract_value REAL, contract_phase TEXT);"""
QUESTION = "Show legal_title and contract_value for every organizations entry and its contracts."
SQL = "SELECT organizations.legal_title, contracts.contract_value FROM organizations INNER JOIN contracts ON organizations.organization_key = contracts.organization_key;"


def test_identifier_canonicalization_round_trips_schema_question_and_sql():
    canonical_schema, canonical_question, mapping = canonicalize_inputs(SCHEMA, QUESTION)
    canonical_sql = mapping.canonicalize(SQL)

    assert "organizations" not in canonical_schema
    assert "legal_title" not in canonical_question
    assert canonical_sql == "SELECT table0.column1, table1.column4 FROM table0 INNER JOIN table1 ON table0.column0 = table1.column0;"
    assert mapping.restore(canonical_schema) == SCHEMA
    assert mapping.restore(canonical_question) == QUESTION
    assert mapping.restore(canonical_sql) == SQL


def test_grounding_filter_rejects_identifiers_not_present_in_schema():
    _, _, mapping = canonicalize_inputs(SCHEMA, QUESTION)
    valid = "SELECT table0.column1 FROM table0;"

    assert mapping.accepts_sql(valid)
    assert _finish_sql(valid, mapping) == "SELECT organizations.legal_title FROM organizations;"
    assert not mapping.accepts_sql("SELECT checks.charge FROM checks;")
    assert not mapping.accepts_sql("SELECT table0.column99 FROM table0;")
    assert _finish_sql("SELECT checks.charge FROM checks;", mapping) == ""


def test_grounding_filter_rejects_columns_from_the_wrong_table():
    _, _, mapping = canonicalize_inputs(SCHEMA, QUESTION)

    assert _finish_sql("SELECT table0.column1 FROM table0;", mapping, SCHEMA) == "SELECT organizations.legal_title FROM organizations;"
    assert _finish_sql("SELECT table0.column1 FROM table1;", mapping, SCHEMA) == ""


def test_training_format_and_mask_use_canonical_targets_when_enabled():
    record = {"schema_sql": SCHEMA, "question": QUESTION, "sql": SQL}
    canonical = canonicalize_record(record)
    formatted = format_record(record, canonicalize_identifiers=True)

    assert canonical["sql"] in formatted
    assert "organizations" not in formatted
    ids, mask = encode_record(record, ByteTokenizer(), 1024, canonicalize_identifiers=True)
    assert len(ids) == len(mask)
    assert any(mask)


def test_casual_singular_and_plural_mentions_link_to_schema_slots():
    schema = "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT);"
    _, question, _ = canonicalize_inputs(schema, "show me customer names")

    assert question == "show me table0 column1"


def test_prefixed_and_camel_case_identifiers_have_human_aliases():
    schema = "CREATE TABLE tbl_customerRecords (col_customerId INTEGER, col_fullName TEXT);"
    _, question, _ = canonicalize_inputs(schema, "list the full names for each customer record")

    assert question == "list the column1 for each table0"


def test_permuted_slots_round_trip_and_vary_identifier_positions():
    slots = set()
    for index in range(20):
        schema = f"CREATE TABLE customers_{index} (customer_id_{index} INTEGER, name_{index} TEXT, city_{index} TEXT);"
        canonical_schema, _, mapping = canonicalize_inputs(
            schema,
            f"show name_{index} from customers_{index}",
            slot_strategy="permuted",
        )
        slots.add(mapping.column_to_slot[f"name_{index}"])
        assert mapping.restore(canonical_schema) == schema

    assert len(slots) > 1


def test_column_alias_preserves_an_embedded_table_hint():
    schema = """CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id), total REAL);"""

    _, customer_question, mapping = canonicalize_inputs(schema, "what are the customer_id?", "permuted")
    _, order_question, _ = canonicalize_inputs(schema, "show me order ids", "permuted")

    assert customer_question == f"what are the {mapping.table_to_slot['customers']} {mapping.column_to_slot['customer_id']}?"
    assert order_question == f"show me {mapping.table_to_slot['orders']} {mapping.column_to_slot['order_id']}"


def test_pluralized_underscore_identifier_links_to_the_schema_column():
    schema = """CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, city TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id));"""

    _, question, mapping = canonicalize_inputs(schema, "what are the customers_id and city?", "permuted")

    assert question == (
        f"what are the {mapping.table_to_slot['customers']} "
        f"{mapping.column_to_slot['customer_id']} and {mapping.column_to_slot['city']}?"
    )
