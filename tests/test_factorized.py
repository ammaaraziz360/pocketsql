from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
from mlx.utils import tree_flatten

from pocketsql.data.query_ast import Filter, QueryPlan
from pocketsql.inference import (
    _apply_factorized_schema_links,
    _finish_generated_target,
    load_model_from_checkpoint,
)
from pocketsql.model.factorized import (
    FactorizedPocketSQLTransformer,
    FactorizedSchemaConfig,
    decode_schema_link_logits,
)
from pocketsql.model.schema_grounding import canonicalize_inputs
from pocketsql.model.tokenizer import ByteTokenizer
from pocketsql.model.transformer import ModelConfig, PocketSQLTransformer
from pocketsql.training.checkpoint import initialize_model, save_checkpoint
from pocketsql.training.schema_links import schema_link_batch
from pocketsql.training.train import (
    _schema_supervision,
    build_model,
    configure_trainable_parameters,
    train_step,
)
from pocketsql.training.dataset import make_batch


SCHEMA = (
    "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT); "
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER REFERENCES customers(id), total REAL);"
)


def _factorized_config() -> dict:
    return {
        "model_architecture": "factorized_schema",
        "layers": 2,
        "hidden_dim": 128,
        "heads": 4,
        "ffn_dim": 256,
        "context_length": 256,
        "canonicalize_identifiers": True,
        "identifier_slot_strategy": "ordered",
        "canonicalize_literals": True,
        "target_format": "semantic_plan",
        "max_table_slots": 8,
        "max_column_slots": 32,
        "max_projection_slots": 4,
        "max_filter_slots": 4,
        "max_group_slots": 2,
    }


def _record() -> dict:
    return {
        "schema_id": "customers_orders",
        "difficulty": 2,
        "schema_sql": SCHEMA,
        "question": "show customer names for orders above 10",
        "sql": (
            "SELECT customers.name FROM customers INNER JOIN orders "
            "ON customers.id = orders.customer_id WHERE orders.total > 10;"
        ),
        "query_plan": {
            "family": "join",
            "table": "customers",
            "columns": ["customers.name"],
            "join_table": "orders",
            "join_on": ["customers.id", "orders.customer_id"],
            "filters": [{"column": "orders.total", "operator": ">", "value": 10}],
        },
    }


def test_schema_link_batch_assigns_role_specific_slots():
    labels, masks = schema_link_batch(
        [_record()],
        identifier_slot_strategy="ordered",
        canonicalize_literals=True,
        schema_linking_hints=False,
        max_table_slots=8,
        max_column_slots=32,
        max_projection_slots=4,
        max_filter_slots=4,
        max_group_slots=2,
    )

    assert labels["table"] == [0]
    assert labels["join_table"] == [1]
    assert labels["projection_column"][0][0] == 1
    assert labels["projection_owner"][0][0] == 0
    assert labels["join_column"] == [[0, 2]]
    assert labels["filter_column"][0][0] == 3
    assert labels["filter_owner"][0][0] == 1
    assert masks["projection_column"] == [[True, False, False, False]]


def test_factorized_model_trains_language_and_schema_heads_together():
    import mlx.optimizers as optim

    tokenizer = ByteTokenizer()
    config = _factorized_config()
    model = build_model(config, tokenizer.vocab_size)
    tokens, mask = make_batch(
        [_record(), _record()],
        tokenizer,
        config["context_length"],
        True,
        "ordered",
        True,
        "semantic_plan",
        False,
    )
    supervision = _schema_supervision(model, [_record(), _record()], tokens, tokenizer, config)

    loss = train_step(
        model,
        optim.AdamW(learning_rate=1e-3),
        mx.array(tokens),
        mx.array(mask),
        supervision,
        1.0,
    )

    assert loss > 0
    logits = model.schema_link_logits(mx.array([tokens[0][: tokens[0].index(tokenizer.sql_start_id) + 1]]))
    assert logits["table"].shape == (1, 8)
    assert logits["projection_column"].shape == (1, 4, 33)


def test_factorized_dual_prompt_adds_labels_only_for_schema_heads():
    tokenizer = ByteTokenizer()
    config = _factorized_config()
    config.update(
        {
            "factorized_schema_linking_hints": True,
            "schema_linking_max_tables": 8,
            "schema_linking_max_columns": 32,
            "context_length": 512,
        }
    )
    model = build_model(config, tokenizer.vocab_size)
    tokens, _ = make_batch(
        [_record()],
        tokenizer,
        config["context_length"],
        True,
        "ordered",
        True,
        "semantic_plan",
        False,
    )

    schema_tokens, prompt_positions, _, _ = _schema_supervision(
        model, [_record()], tokens, tokenizer, config
    )

    assert schema_tokens is not None
    assert "SCHEMA LINKS" not in tokenizer.decode(tokens[0])
    assert "SCHEMA LINKS" in tokenizer.decode(schema_tokens[0].tolist())
    assert prompt_positions.tolist()[0] == schema_tokens[0].tolist().index(tokenizer.sql_start_id)


def test_schema_head_phase_freezes_decoder_and_can_train_extended_positions():
    tokenizer = ByteTokenizer()
    config = _factorized_config()
    config.update({"train_schema_heads_only": True, "train_schema_prompt_positions": True})
    model = build_model(config, tokenizer.vocab_size)

    configure_trainable_parameters(model, config)

    names = {name for name, _ in tree_flatten(model.trainable_parameters())}
    assert names
    assert all(name.startswith("schema_") or name.startswith("position.") for name in names)
    assert any(name.startswith("position.") for name in names)


def test_factorized_links_replace_each_schema_role_and_join_path():
    _, _, mapping = canonicalize_inputs(SCHEMA, "show customer names for orders above 10")
    predicted = QueryPlan(
        "semantic_plan",
        "table1",
        ("column3",),
        filters=(Filter("column0", ">", 10),),
    )
    links = {
        "star_column": 32,
        "table": 0,
        "join_table": 1,
        "projection_column": (1, 0, 0, 0),
        "projection_owner": (0, 0, 0, 0),
        "aggregate_column": 0,
        "aggregate_owner": 0,
        "join_column": (0, 2),
        "filter_column": (3, 0, 0, 0),
        "filter_owner": (1, 0, 0, 0),
        "group_column": (0, 0),
        "group_owner": (0, 0),
        "order_column": 0,
        "order_owner": 0,
    }

    linked = _apply_factorized_schema_links(predicted, links, mapping)

    assert linked.table == "table0"
    assert linked.join_table == "table1"
    assert linked.join_on == ("table1.column2", "table0.column0") or linked.join_on == (
        "table0.column0",
        "table1.column2",
    )
    assert linked.columns == ("table0.column1",)
    assert linked.filters[0].column == "table1.column3"
    assert linked.filters[0].operator == ">"


def test_factorized_decoder_only_emits_physical_owner_column_pairs():
    tokenizer = ByteTokenizer()
    config = FactorizedSchemaConfig(8, 32, 4, 4, 2)
    model = FactorizedPocketSQLTransformer(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            layers=2,
            hidden_dim=128,
            heads=4,
            ffn_dim=256,
            context_length=256,
        ),
        config,
    )
    _, _, mapping = canonicalize_inputs(SCHEMA, "show customer names for orders above 10")
    logits = model.schema_link_logits(mx.zeros((1, 12), dtype=mx.int32))
    prediction = decode_schema_link_logits(logits, [mapping], config)[0]

    assert 0.0 < prediction["confidence"]["table_join"] <= 1.0
    assert len(prediction["confidence"]["projection"]) == config.max_projection_slots

    if prediction["join_table"] is not None:
        assert mapping.declared_joins(
            f"table{prediction['table']}",
            f"table{prediction['join_table']}",
        )
    for columns, owners in (
        (prediction["projection_column"], prediction["projection_owner"]),
        (prediction["filter_column"], prediction["filter_owner"]),
        (prediction["group_column"], prediction["group_owner"]),
        ((prediction["aggregate_column"],), (prediction["aggregate_owner"],)),
        ((prediction["order_column"],), (prediction["order_owner"],)),
    ):
        for column, owner in zip(columns, owners):
            if column == prediction["star_column"]:
                continue
            raw_column = mapping.slot_to_raw[f"column{column}"]
            raw_owner = mapping.slot_to_raw[f"table{owner}"]
            assert raw_owner in mapping.column_to_tables[raw_column]


def test_factorized_fallback_never_replaces_a_valid_baseline_plan():
    class Model:
        target_format = "semantic_plan"
        factorized_schema_mode = "fallback"

    _, question, mapping = canonicalize_inputs(SCHEMA, "show customer names")
    wrong_links = {
        "star_column": 32,
        "table": 1,
        "join_table": None,
        "projection_column": (3, 0, 0, 0),
        "projection_owner": (1, 0, 0, 0),
        "aggregate_column": 0,
        "aggregate_owner": 1,
        "join_column": (0, 2),
        "filter_column": (3, 0, 0, 0),
        "filter_owner": (1, 0, 0, 0),
        "group_column": (0, 0),
        "group_owner": (1, 1),
        "order_column": 0,
        "order_owner": 1,
    }

    sql = _finish_generated_target(
        "T table0 | S column1",
        Model(),
        mapping,
        SCHEMA,
        question,
        wrong_links,
    )

    assert sql == "SELECT name FROM customers;"


def test_confidence_mode_never_replaces_a_valid_baseline_plan():
    class Model:
        target_format = "semantic_plan"
        factorized_schema_mode = "confidence"
        factorized_schema_confidence_threshold = 0.8

    _, question, mapping = canonicalize_inputs(SCHEMA, "show customer names")
    wrong_links = {
        "star_column": 32,
        "table": 1,
        "join_table": None,
        "projection_column": (3, 0, 0, 0),
        "projection_owner": (1, 0, 0, 0),
        "aggregate_column": 0,
        "aggregate_owner": 1,
        "join_column": (0, 2),
        "filter_column": (3, 0, 0, 0),
        "filter_owner": (1, 0, 0, 0),
        "group_column": (0, 0),
        "group_owner": (1, 1),
        "order_column": 0,
        "order_owner": 1,
        "confidence": {
            "table_join": 1.0,
            "projection": (1.0, 1.0, 1.0, 1.0),
        },
    }

    sql = _finish_generated_target(
        "T table0 | S column1",
        Model(),
        mapping,
        SCHEMA,
        question,
        wrong_links,
    )

    assert sql == "SELECT name FROM customers;"


def test_confidence_gating_applies_only_high_confidence_residual_roles():
    _, _, mapping = canonicalize_inputs(SCHEMA, "show customer names")
    plan = QueryPlan("semantic_plan", "table0", ("column0",))
    links = {
        "star_column": 32,
        "table": 0,
        "join_table": None,
        "projection_column": (1, 0, 0, 0),
        "projection_owner": (0, 0, 0, 0),
        "aggregate_column": 0,
        "aggregate_owner": 0,
        "join_column": (0, 2),
        "filter_column": (0, 0, 0, 0),
        "filter_owner": (0, 0, 0, 0),
        "group_column": (0, 0),
        "group_owner": (0, 0),
        "order_column": 0,
        "order_owner": 0,
        "confidence": {
            "table_join": 0.4,
            "projection": (0.95, 0.0, 0.0, 0.0),
            "aggregate": 0.0,
            "join_column": 0.0,
            "filter": (0.0, 0.0, 0.0, 0.0),
            "group": (0.0, 0.0),
            "order": 0.0,
        },
    }

    linked = _apply_factorized_schema_links(plan, links, mapping, confidence_threshold=0.9)

    assert linked.table == "table0"
    assert linked.join_table is None
    assert linked.columns == ("column1",)


def test_legacy_checkpoint_initializes_new_heads_and_factorized_checkpoint_loads(tmp_path: Path):
    tokenizer = ByteTokenizer()
    config = _factorized_config()
    model_config = ModelConfig(
        vocab_size=tokenizer.vocab_size,
        layers=2,
        hidden_dim=128,
        heads=4,
        ffn_dim=256,
        context_length=256,
    )
    legacy = PocketSQLTransformer(model_config)
    factorized = FactorizedPocketSQLTransformer(
        model_config,
        FactorizedSchemaConfig(8, 32, 4, 4, 2),
    )
    legacy_path = tmp_path / "legacy"
    save_checkpoint(legacy_path, legacy, {"config": config}, tokenizer=tokenizer)

    initialize_model(legacy_path, factorized, allow_new_schema_heads=True)

    factorized_path = tmp_path / "factorized"
    save_checkpoint(factorized_path, factorized, {"config": config}, tokenizer=tokenizer)
    restored = load_model_from_checkpoint(str(factorized_path), tokenizer)
    assert isinstance(restored, FactorizedPocketSQLTransformer)
