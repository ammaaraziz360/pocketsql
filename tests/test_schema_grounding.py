from types import SimpleNamespace

from pocketsql.inference import _explicit_filter_column_hints, _finish_sql, _finish_target
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


def test_grounding_accepts_and_restores_table_wildcard_join():
    schema = """CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, city TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id));"""
    _, question, mapping = canonicalize_inputs(
        schema,
        "show me orders from customer from houston",
        "permuted",
        True,
    )
    orders = mapping.table_to_slot["orders"]
    customers = mapping.table_to_slot["customers"]
    customer_id = mapping.column_to_slot["customer_id"]
    city = mapping.column_to_slot["city"]
    canonical_sql = (
        f"SELECT {orders}.* FROM {orders} INNER JOIN {customers} "
        f"ON {orders}.{customer_id} = {customers}.{customer_id} "
        f"WHERE {customers}.{city} = value0;"
    )

    assert "value0" in question
    assert mapping.accepts_sql(canonical_sql)
    assert mapping.restore(canonical_sql) == (
        "SELECT orders.* FROM orders INNER JOIN customers "
        "ON orders.customer_id = customers.customer_id "
        "WHERE customers.city = 'houston';"
    )


def test_literal_grounding_handles_count_for_location_without_binding_group_words():
    schema = """CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, city TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, status TEXT);"""
    _, count_question, count_mapping = canonicalize_inputs(
        schema,
        "what is the customer count for Boston",
        "permuted",
        True,
    )
    _, group_question, group_mapping = canonicalize_inputs(
        schema,
        "how many orders are there in each status category",
        "permuted",
        True,
    )

    assert count_question.endswith(f"count for value0 by {count_mapping.column_to_slot['city']}")
    assert [binding.value for binding in count_mapping.literals] == ["Boston"]
    assert "value0" not in group_question
    assert group_mapping.literals == ()


def test_literal_grounding_does_not_treat_aggregate_total_as_a_location():
    schema = "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, amount REAL);"
    _, question, mapping = canonicalize_inputs(
        schema,
        "how much amount is there in total for orders as long as amount above 30",
        "permuted",
        True,
    )

    assert "in total" in question
    assert "above value0" in question
    assert [binding.value for binding in mapping.literals] == [30]


def test_literal_slot_order_ignores_digits_inside_column_slots():
    schema = "CREATE TABLE orders (a TEXT, b TEXT, c TEXT, d TEXT, status TEXT);"
    _, question, mapping = canonicalize_inputs(
        schema,
        "show rows where status is cancelled, only return 4",
        "ordered",
        True,
    )

    assert mapping.column_to_slot["status"] == "column4"
    assert [binding.value for binding in mapping.literals] == ["cancelled", 4]
    assert "column4 is value0" in question
    assert question.endswith("value1")


def test_ambiguous_identifier_alias_is_not_extracted_as_a_location_value():
    schema = """CREATE TABLE tbl_customers (col_account_id INTEGER PRIMARY KEY, col_postal_area TEXT);
CREATE TABLE tbl_sales (col_transaction_id INTEGER PRIMARY KEY, col_sale_value REAL);"""
    _, question, mapping = canonicalize_inputs(
        schema,
        "what is the largest sale in sales",
        "permuted",
        True,
    )

    assert "value0" not in question
    assert mapping.literals == ()


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


def test_punctuated_identifier_alias_links_to_schema_slot():
    schema = "CREATE TABLE regions (state_territory TEXT, current_slogan TEXT);"
    _, question, mapping = canonicalize_inputs(schema, "show state/territory and current-slogan")

    assert question == (
        f"show {mapping.column_to_slot['state_territory']} and "
        f"{mapping.column_to_slot['current_slogan']}"
    )


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


def test_permuted_slots_are_invariant_to_schema_whitespace():
    canonical = """CREATE TABLE vendors (vendor_id INTEGER PRIMARY KEY, company_name TEXT, region TEXT, notes TEXT);
CREATE TABLE shipments (shipment_id INTEGER PRIMARY KEY, vendor_id INTEGER REFERENCES vendors(vendor_id), freight_cost REAL, delivery_status TEXT, tracking_note TEXT);"""
    pretty = """CREATE TABLE vendors (
      vendor_id INTEGER PRIMARY KEY,
      company_name TEXT,
      region TEXT,
      notes TEXT
    );

    CREATE TABLE shipments (
      shipment_id INTEGER PRIMARY KEY,
      vendor_id INTEGER REFERENCES vendors(vendor_id),
      freight_cost REAL,
      delivery_status TEXT,
      tracking_note TEXT
    );"""
    single_line = canonical.replace(";\n", "; ")

    grounded = [canonicalize_inputs(schema, "show vendor shipments", "permuted") for schema in (canonical, pretty, single_line)]
    mappings = [item[2] for item in grounded]

    assert mappings[0].table_to_slot == mappings[1].table_to_slot == mappings[2].table_to_slot
    assert mappings[0].column_to_slot == mappings[1].column_to_slot == mappings[2].column_to_slot
    assert grounded[0][0] == grounded[1][0] == grounded[2][0]


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


def test_literal_grounding_restores_location_shorthand():
    schema = "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT);"
    _, question, mapping = canonicalize_inputs(
        schema,
        "what are the customer city from houston?",
        "permuted",
        canonicalize_literals=True,
    )
    city = mapping.column_to_slot["city"]
    customers = mapping.table_to_slot["customers"]

    assert "houston" not in question.casefold()
    assert "value0" in question
    assert _finish_sql(
        f"SELECT {city} FROM {customers} WHERE {city} = value0;",
        mapping,
        schema,
    ) == "SELECT city FROM customers WHERE city = 'houston';"


def test_inference_repairs_one_schema_invalid_implicit_location_predicate():
    schema = """CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id));"""
    _, question, mapping = canonicalize_inputs(
        schema,
        "what are the customer city from houston?",
        "permuted",
        True,
    )
    customers = mapping.table_to_slot["customers"]
    city = mapping.column_to_slot["city"]
    order_id = mapping.column_to_slot["order_id"]

    assert _finish_sql(
        f"SELECT {city} FROM {customers} WHERE {order_id} = value0;",
        mapping,
        schema,
        question,
    ) == "SELECT city FROM customers WHERE city = 'houston';"


def test_inference_does_not_repair_ambiguous_or_non_location_predicates():
    ambiguous_schema = "CREATE TABLE customers (name TEXT, city TEXT, region TEXT);"
    _, ambiguous_question, ambiguous_mapping = canonicalize_inputs(
        ambiguous_schema,
        "show customer names from houston",
        "permuted",
        True,
    )
    customers = ambiguous_mapping.table_to_slot["customers"]
    name = ambiguous_mapping.column_to_slot["name"]
    assert _finish_sql(
        f"SELECT {name} FROM {customers} WHERE column99 = value0;",
        ambiguous_mapping,
        ambiguous_schema,
        ambiguous_question,
    ) == ""

    schema = """CREATE TABLE customers (name TEXT, city TEXT);
CREATE TABLE orders (order_id INTEGER, status TEXT);"""
    _, status_question, mapping = canonicalize_inputs(
        schema,
        "show order ids where status is complete",
        "permuted",
        True,
    )
    orders = mapping.table_to_slot["orders"]
    order_id = mapping.column_to_slot["order_id"]
    name = mapping.column_to_slot["name"]
    assert _finish_sql(
        f"SELECT {order_id} FROM {orders} WHERE {name} = value0;",
        mapping,
        schema,
        status_question,
    ) == ""


def test_implicit_location_schema_linking_exposes_the_shuffled_city_slot():
    schema = """CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT, note TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL, status TEXT);"""
    _, question, mapping = canonicalize_inputs(
        schema,
        "show me the number of customers from Houston",
        "permuted",
        True,
    )

    assert "value0" in question
    assert question.endswith(f"by {mapping.column_to_slot['city']}")
    assert mapping.column_to_slot["name"] not in question


def test_training_literal_grounding_canonicalizes_filters_and_limits():
    record = {
        "schema_sql": "CREATE TABLE customers (name TEXT, city TEXT);",
        "question": "show name where city is Houston, only return 3",
        "sql": "SELECT name FROM customers WHERE city = 'Houston' LIMIT 3;",
        "query_plan": {
            "filters": [{"column": "city", "operator": "=", "value": "Houston"}],
            "limit": 3,
        },
    }

    canonical = canonicalize_record(record, canonicalize_literals=True)

    assert "Houston" not in canonical["question"]
    assert "Houston" not in canonical["sql"]
    assert "WHERE column1 = value0 LIMIT value1;" in canonical["sql"]


def test_training_leaves_unrecognized_quoted_literal_available_for_copying():
    record = {
        "schema_sql": "CREATE TABLE orders (order_id INTEGER, status TEXT);",
        "question": "which order id belongs to the pending status?",
        "sql": "SELECT order_id FROM orders WHERE status = 'pending';",
        "query_plan": {
            "filters": [{"column": "status", "operator": "=", "value": "pending"}],
        },
    }

    canonical = canonicalize_record(record, "permuted", canonicalize_literals=True)
    schema, question, mapping = canonicalize_inputs(
        record["schema_sql"], record["question"], "permuted", canonicalize_literals=True
    )

    assert mapping.literals == ()
    assert canonical["schema_sql"] == schema
    assert canonical["question"] == question
    assert "'pending'" in canonical["sql"]
    assert canonical["sql"] == mapping.canonicalize_sql(record["sql"])
    assert mapping.accepts_sql(canonical["sql"])


def test_inference_literal_grounding_copies_multiple_values_and_limit():
    schema = "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, status TEXT, total REAL);"
    _, question, mapping = canonicalize_inputs(
        schema,
        "show order ids where status is open or status is complete, only return 3",
        "permuted",
        canonicalize_literals=True,
    )
    orders = mapping.table_to_slot["orders"]
    order_id = mapping.column_to_slot["order_id"]
    status = mapping.column_to_slot["status"]

    assert all(slot in question for slot in ("value0", "value1", "value2"))
    assert _finish_sql(
        f"SELECT {order_id} FROM {orders} WHERE {status} = value0 OR {status} = value1 LIMIT value2;",
        mapping,
        schema,
    ) == "SELECT order_id FROM orders WHERE status = 'open' OR status = 'complete' LIMIT 3;"


def test_numeric_comparison_words_do_not_become_literal_slots():
    schema = "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, status TEXT, total REAL);"
    _, question, mapping = canonicalize_inputs(
        schema,
        "show order ids where total is greater than 50 and status is shipped",
        "permuted",
        canonicalize_literals=True,
    )

    assert [binding.value for binding in mapping.literals] == [50, "shipped"]
    assert "greater than value0" in question
    assert "value1" in question


def test_literal_grounding_handles_implicit_name_value():
    schema = "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT);"
    _, question, mapping = canonicalize_inputs(
        schema,
        "show me customer with the name john",
        "permuted",
        canonicalize_literals=True,
    )

    assert [binding.value for binding in mapping.literals] == ["john"]
    assert question == (
        f"show me {mapping.table_to_slot['customers']} with the "
        f"{mapping.column_to_slot['name']} value0"
    )


def test_semantic_target_drops_only_unmentioned_hallucinated_filters():
    schema = "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT);"
    _, question, mapping = canonicalize_inputs(
        schema,
        "show me customer names",
        "permuted",
        canonicalize_literals=True,
    )
    customers = mapping.table_to_slot["customers"]
    name = mapping.column_to_slot["name"]
    target = f"T {customers} | S {name} | F AND {name} = 'Maya'"

    assert _finish_target(
        target,
        SimpleNamespace(target_format="semantic_plan"),
        mapping,
        schema,
        question,
    ) == "SELECT name FROM customers;"


def test_semantic_target_removes_unrequested_and_duplicate_projection_columns():
    schema = "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT);"
    _, question, mapping = canonicalize_inputs(
        schema,
        "what are the customer_id?",
        "permuted",
        canonicalize_literals=True,
    )
    customers = mapping.table_to_slot["customers"]
    customer_id = mapping.column_to_slot["customer_id"]
    name = mapping.column_to_slot["name"]

    assert _finish_target(
        f"T {customers} | S {customer_id},{name},{name}",
        SimpleNamespace(target_format="semantic_plan"),
        mapping,
        schema,
        question,
    ) == "SELECT customer_id FROM customers;"


def test_semantic_target_grounds_compound_relationship_head_and_filter_owner():
    schema = """CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT);
CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id), total REAL);"""
    _, question, mapping = canonicalize_inputs(
        schema,
        "show me customer orders where name is max and city is dallas",
        "permuted",
        canonicalize_literals=True,
    )
    customers = mapping.table_to_slot["customers"]
    orders = mapping.table_to_slot["orders"]
    customer_id = mapping.column_to_slot["customer_id"]
    name = mapping.column_to_slot["name"]
    city = mapping.column_to_slot["city"]
    reversed_target = (
        f"T {customers} | S {customers}.* | J {orders} {customers}.{customer_id} {orders}.{customer_id} | "
        f"F AND {orders}.{name} = value0 & {orders}.{customer_id} = value1"
    )

    assert _finish_target(
        reversed_target,
        SimpleNamespace(target_format="semantic_plan"),
        mapping,
        schema,
        question,
    ) == (
        "SELECT orders.* FROM orders INNER JOIN customers "
        "ON customers.customer_id = orders.customer_id "
        "WHERE customers.name = 'max' AND customers.city = 'dallas';"
    )


def test_schema_linking_hints_keep_readable_labels_beside_stable_slots():
    schema, question, mapping = canonicalize_inputs(
        SCHEMA,
        QUESTION,
        "permuted",
        schema_linking_hints=True,
    )

    assert "SCHEMA LINKS" in schema
    assert f"{mapping.table_to_slot['organizations']}=organizations" in schema
    assert f"{mapping.column_to_slot['legal_title']}=legal title" in schema
    assert "organizations" not in schema.split("SCHEMA LINKS", 1)[0]
    assert mapping.table_to_slot["organizations"] in question


def test_training_and_inference_schema_linking_hints_match():
    record = {
        "schema_sql": SCHEMA,
        "question": QUESTION,
        "sql": SQL,
    }
    canonical = canonicalize_record(record, "permuted", schema_linking_hints=True)
    schema, question, _ = canonicalize_inputs(
        SCHEMA,
        QUESTION,
        "permuted",
        schema_linking_hints=True,
    )

    assert canonical["schema_sql"] == schema
    assert canonical["question"] == question


def test_explicit_filter_hints_bind_direct_reversed_and_inherited_values():
    hints = _explicit_filter_column_hints(
        "show column2 where column4 is value0 and column7 value1 or value2 plus value3 by column9"
    )

    assert hints == {
        "value0": "column4",
        "value1": "column7",
        "value2": "column7",
        "value3": "column9",
    }


def test_explicit_filter_hints_do_not_guess_from_loose_projection_proximity():
    assert _explicit_filter_column_hints("show the average column5 before value0") == {}
