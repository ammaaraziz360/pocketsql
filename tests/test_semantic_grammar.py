from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.model.schema_grounding import canonicalize_inputs
from pocketsql.model.semantic_grammar import SemanticPlanGrammar
from pocketsql.model.semantic_plan import serialize_semantic_plan
from pocketsql.model.tokenizer import ByteTokenizer
from pocketsql.inference import _constrained_next_token, _token_pieces
import mlx.core as mx


def grammar():
    schema = (
        "CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT, city TEXT); "
        "CREATE TABLE orders (order_id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(customer_id), total REAL);"
    )
    _, _, mapping = canonicalize_inputs(
        schema,
        "show order totals for customers where name is Max and city is Dallas",
        "permuted",
        True,
    )
    return SemanticPlanGrammar.from_mapping(mapping), mapping


def test_every_character_prefix_of_serialized_plan_is_allowed():
    constraint, mapping = grammar()
    plan = QueryPlan(
        "join",
        mapping.table_to_slot["orders"],
        (f"{mapping.table_to_slot['orders']}.{mapping.column_to_slot['total']}",),
        filters=(
            Filter(f"{mapping.table_to_slot['customers']}.{mapping.column_to_slot['name']}", "=", "value0"),
            Filter(f"{mapping.table_to_slot['customers']}.{mapping.column_to_slot['city']}", "=", "value1"),
        ),
        join_table=mapping.table_to_slot["customers"],
        join_on=(
            f"{mapping.table_to_slot['orders']}.{mapping.column_to_slot['customer_id']}",
            f"{mapping.table_to_slot['customers']}.{mapping.column_to_slot['customer_id']}",
        ),
    )
    target = serialize_semantic_plan(plan)

    assert all(constraint.is_prefix(target[:index]) for index in range(len(target) + 1))
    assert constraint.is_complete(target)


def test_grammar_rejects_unknown_slots_bad_order_and_unclosed_literals():
    constraint, _ = grammar()

    assert not constraint.is_prefix("T table99")
    assert not constraint.is_prefix("T table0 | J")
    assert not constraint.is_complete("T table0")
    assert not constraint.is_complete("T table0 | S column0 | F AND column0 = 'Max")


def test_grammar_allows_canonical_separator_in_token_sized_steps():
    constraint, mapping = grammar()
    table = mapping.table_to_slot["customers"]
    column = mapping.column_to_slot["name"]

    for target in (f"T {table}", f"T {table} ", f"T {table} |", f"T {table} | ", f"T {table} | S {column}"):
        assert constraint.is_prefix(target)
    assert constraint.is_complete(f"T {table} | S {column}")


def test_logit_constraint_blocks_higher_scoring_invalid_token_and_allows_stop_only_when_complete():
    constraint, mapping = grammar()
    tokenizer = ByteTokenizer()
    scores = [0.0] * tokenizer.vocab_size
    scores[ord("X")] = 10.0
    scores[ord("T")] = 5.0
    scores[tokenizer.sql_end_id] = 20.0

    assert _constrained_next_token(
        mx.array(scores), "", constraint, tokenizer, _token_pieces(tokenizer)
    ) == ord("T")

    complete = f"T {mapping.table_to_slot['customers']} | S {mapping.column_to_slot['name']}"
    assert _constrained_next_token(
        mx.array(scores), complete, constraint, tokenizer, _token_pieces(tokenizer)
    ) == tokenizer.sql_end_id
