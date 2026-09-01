from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from pocketsql.inference import load_model_from_checkpoint
from pocketsql.model.factorized import FactorizedSchemaConfig
from pocketsql.model.schema_grounding import canonicalize_inputs
from pocketsql.model.structured import (
    StructuredPocketSQLTransformer,
    StructuredQueryConfig,
    literal_span_labels,
    prompt_layout,
    structured_query_plan,
)
from pocketsql.model.tokenizer import ByteTokenizer, load_tokenizer
from pocketsql.training.checkpoint import initialize_model, save_checkpoint
from pocketsql.training.dataset import make_batch
from pocketsql.training.schema_links import structured_prompt_batch, structured_query_batch
from pocketsql.training.train import (
    _classification_loss,
    _schema_supervision,
    build_model,
    configure_trainable_parameters,
    train_step,
)


SCHEMA = (
    "CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT); "
    "CREATE TABLE orders (id INTEGER PRIMARY KEY, "
    "customer_id INTEGER REFERENCES customers(id), total REAL);"
)


def _config() -> dict:
    return {
        "model_architecture": "structured_v18",
        "layers": 2,
        "hidden_dim": 128,
        "heads": 4,
        "ffn_dim": 256,
        "context_length": 320,
        "canonicalize_identifiers": True,
        "identifier_slot_strategy": "ordered",
        "canonicalize_literals": True,
        "target_format": "semantic_plan",
        "max_table_slots": 8,
        "max_column_slots": 32,
        "max_projection_slots": 4,
        "max_filter_slots": 4,
        "max_group_slots": 2,
        "max_literal_span_tokens": 12,
        "max_limit_value": 20,
        "structured_plan_mode": "replace",
    }


def _record() -> dict:
    return {
        "schema_id": "customers_orders",
        "difficulty": 2,
        "schema_sql": SCHEMA,
        "question": "show customer names for orders where total is 10",
        "sql": (
            "SELECT customers.name FROM customers INNER JOIN orders "
            "ON customers.id = orders.customer_id WHERE orders.total = 10;"
        ),
        "query_plan": {
            "family": "join",
            "table": "customers",
            "columns": ["customers.name"],
            "join_table": "orders",
            "join_on": ["customers.id", "orders.customer_id"],
            "filters": [{"column": "orders.total", "operator": "=", "value": 10}],
        },
    }


def test_prompt_layout_locates_schema_question_and_literal_candidates():
    tokenizer = ByteTokenizer()
    model = build_model(_config(), tokenizer.vocab_size)
    schema, question, mapping = canonicalize_inputs(
        SCHEMA,
        _record()["question"],
        "ordered",
        True,
    )
    prompt = f"<bos><schema>{schema}</schema><question>{question}</question><sql>"

    layout = prompt_layout(
        prompt,
        tokenizer,
        mapping,
        model.schema_config,
        model.structured_config,
    )

    assert sum(layout["table_mask"]) == 2
    assert sum(layout["column_mask"]) == 4
    assert any(layout["question_mask"])
    assert layout["tokens"][layout["prompt_position"]] == tokenizer.sql_start_id
    assert tokenizer.decode([layout["tokens"][layout["table_positions"][0]]]) == "0"

    labels, masks = literal_span_labels(prompt, tokenizer, ["value0"], 4)
    assert masks["filter_start"][0] is True
    start = labels["filter_start"][0]
    end = labels["filter_end"][0]
    assert tokenizer.decode(layout["tokens"][start : end + 1]) == "value0"


def test_structured_labels_cover_operations_and_literal_copy():
    operation_labels, operation_masks = structured_query_batch(
        [_record()],
        identifier_slot_strategy="ordered",
        canonicalize_literals=True,
        schema_linking_hints=False,
        max_projection_slots=4,
        max_filter_slots=4,
        max_group_slots=2,
        max_limit_value=20,
    )

    assert operation_labels["selection_arity"] == [1]
    assert operation_labels["join_presence"] == [1]
    assert operation_labels["filter_count"] == [1]
    assert operation_labels["filter_operator"][0][0] == 0
    assert operation_masks["filter_operator"][0][0] is True

    tokenizer = ByteTokenizer()
    model = build_model(_config(), tokenizer.vocab_size)
    _, _, literal_labels, literal_masks = structured_prompt_batch(
        [_record()], tokenizer, model, _config()
    )
    assert literal_masks["filter_start"][0][0] is True
    assert literal_labels["filter_start"][0][0] <= literal_labels["filter_end"][0][0]


def test_structured_model_trains_all_three_head_families():
    tokenizer = ByteTokenizer()
    config = _config()
    model = build_model(config, tokenizer.vocab_size)
    assert isinstance(model, StructuredPocketSQLTransformer)
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

    schema_logits, operation_logits, literal_logits = model.structured_logits(
        supervision["prompt_tokens"], supervision["layout"]
    )
    assert schema_logits["table"].shape == (2, 8)
    assert schema_logits["projection_column"].shape == (2, 4, 33)
    assert operation_logits["filter_count"].shape == (2, 5)
    assert literal_logits["filter_start"].shape[:2] == (2, 4)
    assert literal_logits["filter_end"].shape == literal_logits["filter_start"].shape

    loss = train_step(
        model,
        optim.AdamW(learning_rate=1e-3),
        mx.array(tokens),
        mx.array(mask),
        supervision,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    assert loss > 0


def test_structured_predictions_compile_through_declared_join():
    _, _, mapping = canonicalize_inputs(
        SCHEMA,
        _record()["question"],
        "ordered",
        True,
    )
    operation = {
        "selection_arity": 1,
        "aggregate": 0,
        "aggregate_target": 0,
        "aggregate_position": 0,
        "distinct": 0,
        "join_presence": 1,
        "filter_count": 1,
        "filter_operator": [0, 0, 0, 0],
        "filter_connector": 0,
        "group_count": 0,
        "order_presence": 0,
        "descending": 0,
        "limit": 0,
    }
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

    plan = structured_query_plan(
        operation,
        links,
        ("value0", None, None, None),
        mapping,
        build_model(_config(), ByteTokenizer().vocab_size).structured_config,
    )

    assert plan is not None
    assert plan.table == "table0"
    assert plan.join_table == "table1"
    assert plan.columns == ("table0.column1",)
    assert plan.filters[0].column == "table1.column3"
    assert plan.filters[0].value == "value0"


def test_v16_weights_initialize_v18_and_v18_checkpoint_reloads(tmp_path: Path):
    tokenizer = ByteTokenizer()
    v16_config = {**_config(), "model_architecture": "factorized_schema"}
    v16 = build_model(v16_config, tokenizer.vocab_size)
    v16_path = tmp_path / "v16"
    save_checkpoint(v16_path, v16, {"config": v16_config}, tokenizer=tokenizer)
    v18 = build_model(_config(), tokenizer.vocab_size)

    initialize_model(v16_path, v18, allow_new_schema_heads=True)

    v18_path = tmp_path / "v18"
    save_checkpoint(v18_path, v18, {"config": _config()}, tokenizer=tokenizer)
    restored = load_model_from_checkpoint(str(v18_path), tokenizer)
    assert isinstance(restored, StructuredPocketSQLTransformer)


def test_real_bpe_offsets_support_schema_and_multiword_literal_spans():
    tokenizer_path = Path("artifacts/tokenizers/position-robust-v8-bpe.json")
    if not tokenizer_path.exists():
        pytest.skip("workspace BPE tokenizer is unavailable")
    tokenizer = load_tokenizer(tokenizer_path)
    schema, question, mapping = canonicalize_inputs(
        SCHEMA,
        'show customer names where name is "Tillman Ernser"',
        "permuted",
        True,
    )
    prompt = f"<bos><schema>{schema}</schema><question>{question}</question><sql>"
    layout = prompt_layout(
        prompt,
        tokenizer,
        mapping,
        FactorizedSchemaConfig(8, 32, 4, 4, 2),
        StructuredQueryConfig(max_literal_span_tokens=12, max_limit_value=20),
    )
    labels, masks = literal_span_labels(prompt, tokenizer, ["Tillman Ernser"], 4)

    assert tokenizer.decode(layout["tokens"]) == prompt
    assert layout["tokens"][layout["prompt_position"]] == tokenizer.sql_start_id
    assert masks["filter_end"][0] is True
    start, end = labels["filter_start"][0], labels["filter_end"][0]
    assert "Tillman Ernser" in tokenizer.decode(layout["tokens"][start : end + 1])


def test_classification_loss_can_emphasize_one_schema_role():
    logits = {
        "easy": mx.array([[8.0, -8.0]]),
        "filter_column": mx.array([[8.0, -8.0]]),
    }
    labels = {"easy": mx.array([0]), "filter_column": mx.array([1])}
    masks = {"easy": mx.array([True]), "filter_column": mx.array([True])}

    ordinary = _classification_loss(logits, labels, masks)
    emphasized = _classification_loss(
        logits, labels, masks, {"filter_column": 8.0}
    )

    assert float(emphasized.item()) > float(ordinary.item())


def test_structured_calibration_can_freeze_schema_heads():
    tokenizer = ByteTokenizer()
    config = {
        **_config(),
        "train_schema_heads_only": True,
        "train_head_prefixes": ["operation_", "literal_"],
    }
    model = build_model(config, tokenizer.vocab_size)

    configure_trainable_parameters(model, config)

    names = {name for name, _ in tree_flatten(model.trainable_parameters())}
    assert names
    assert all(name.startswith(("operation_", "literal_")) for name in names)
    assert any(name.startswith("operation_") for name in names)
    assert any(name.startswith("literal_") for name in names)
