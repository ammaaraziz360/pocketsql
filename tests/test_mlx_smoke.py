import pytest

mx = pytest.importorskip("mlx.core")

from pocketsql.model.tokenizer import ByteTokenizer
from pocketsql.model.transformer import ModelConfig, PocketSQLTransformer
from pocketsql.training.checkpoint import initialize_model, load_checkpoint, save_checkpoint
from pocketsql.training.interpolate import interpolate_checkpoints
from pocketsql.training.train import train_step


def tiny_model():
    tokenizer = ByteTokenizer()
    return PocketSQLTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, layers=2, hidden_dim=128, heads=4, ffn_dim=512, context_length=256))


def test_model_forward_dimensions():
    model = tiny_model()
    logits = model(mx.zeros((2, 16), dtype=mx.int32))
    assert logits.shape == (2, 16, ByteTokenizer().vocab_size)


def test_cached_forward_matches_full_forward_for_next_token():
    mx.random.seed(0)
    model = tiny_model()
    tokens = mx.array([[1, 2, 3, 4, 5]])
    full_logits = model(tokens)
    prompt_logits, cache = model.forward_with_cache(tokens[:, :-1])
    cached_logits, _ = model.forward_with_cache(tokens[:, -1:], cache)
    mx.eval(full_logits, prompt_logits, cached_logits)
    assert mx.max(mx.abs(full_logits[:, -1] - cached_logits[:, -1])).item() < 0.02
    assert mx.array_equal(mx.argmax(full_logits[:, -1], axis=-1), mx.argmax(cached_logits[:, -1], axis=-1)).item()


def test_checkpoint_round_trip(tmp_path):
    model = tiny_model()
    tokenizer = ByteTokenizer()
    save_checkpoint(tmp_path / "checkpoint", model, {"step": 1}, tokenizer=tokenizer)
    restored = tiny_model()
    assert load_checkpoint(tmp_path / "checkpoint", restored) == {"step": 1}
    assert (tmp_path / "checkpoint" / "tokenizer.json").exists()


def test_weight_initialization_can_extend_only_the_position_table(tmp_path):
    tokenizer = ByteTokenizer()
    source = tiny_model()
    save_checkpoint(tmp_path / "source", source, {"config": {"context_length": 256}}, tokenizer=tokenizer)
    expanded = PocketSQLTransformer(
        ModelConfig(
            vocab_size=tokenizer.vocab_size,
            layers=2,
            hidden_dim=128,
            heads=4,
            ffn_dim=512,
            context_length=320,
        )
    )

    metadata = initialize_model(tmp_path / "source", expanded)
    mx.eval(source.position.weight, expanded.position.weight)

    assert metadata["config"]["context_length"] == 256
    assert expanded.position.weight.shape == (320, 128)
    assert mx.array_equal(source.position.weight, expanded.position.weight[:256]).item()


def test_checkpoint_interpolation_writes_inference_only_checkpoint(tmp_path):
    import json

    tokenizer = ByteTokenizer()
    config = {
        "layers": 2,
        "hidden_dim": 128,
        "heads": 4,
        "ffn_dim": 512,
        "context_length": 256,
        "canonicalize_identifiers": False,
        "identifier_slot_strategy": "ordered",
        "canonicalize_literals": False,
    }
    base = tmp_path / "base"
    fine_tuned = tmp_path / "fine-tuned"
    output = tmp_path / "interpolated"
    save_checkpoint(base, tiny_model(), {"config": config, "epoch": 1}, tokenizer=tokenizer)
    save_checkpoint(fine_tuned, tiny_model(), {"config": config, "epoch": 2}, tokenizer=tokenizer)

    interpolate_checkpoints(base, fine_tuned, output, 0.5)

    metadata = json.loads((output / "metadata.json").read_text())
    assert (output / "weights.safetensors").exists()
    assert metadata["resumable"] is False
    assert metadata["weight_interpolation"]["fine_tuned_weight"] == 0.5
    with pytest.raises(FileExistsError):
        interpolate_checkpoints(base, fine_tuned, output, 0.5)


def test_tiny_model_completes_one_training_step():
    import mlx.optimizers as optim

    model = tiny_model()
    tokens = mx.array([[1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
    mask = mx.array([[False, False, True, True, True], [False, False, True, True, True]])
    loss = train_step(model, optim.AdamW(learning_rate=1e-3), tokens, mask)
    assert loss >= 0


def test_resume_from_checkpoint_continues_training(tmp_path):
    from pocketsql.training.train import train

    records = [
        {"schema_sql": "CREATE TABLE t (id INTEGER, name TEXT);", "question": f"show name {i}", "sql": f"SELECT name FROM t WHERE id = {i};"}
        for i in range(8)
    ]
    config = {"seed": 1, "layers": 2, "hidden_dim": 128, "heads": 4, "ffn_dim": 512, "context_length": 256, "batch_size": 4, "grad_accum_steps": 1, "learning_rate": 5e-4, "warmup_steps": 2, "epochs": 1}
    checkpoint_dir = tmp_path / "checkpoint"
    _, first_losses = train(records, config, checkpoint_dir)
    assert len(first_losses) == 2
    config["epochs"] = 2
    _, second_losses = train(records, config, checkpoint_dir, resume_from=checkpoint_dir)
    assert len(second_losses) == 2


def test_training_saves_best_validation_checkpoint_and_stops_early(tmp_path, monkeypatch):
    import json
    import pocketsql.training.train as training

    records = [
        {"schema_sql": "CREATE TABLE t (id INTEGER, name TEXT);", "question": "show names", "sql": "SELECT name FROM t;"}
        for _ in range(4)
    ]
    validation_losses = iter((1.0, 0.8, 0.9, 1.0))
    monkeypatch.setattr(training, "evaluate_loss", lambda *_: next(validation_losses))
    config = {
        "seed": 1,
        "layers": 2,
        "hidden_dim": 128,
        "heads": 4,
        "ffn_dim": 512,
        "context_length": 256,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "learning_rate": 5e-4,
        "warmup_steps": 0,
        "epochs": 4,
        "early_stopping_patience": 1,
    }
    checkpoint_dir = tmp_path / "checkpoint"

    _, losses = training.train(records, config, checkpoint_dir, val_records=records)

    assert len(losses) == 3
    best_checkpoint_dir = checkpoint_dir.with_name("checkpoint-best")
    best_metadata = json.loads((best_checkpoint_dir / "metadata.json").read_text())
    final_metadata = json.loads((checkpoint_dir / "metadata.json").read_text())
    assert best_metadata["epoch"] == 2
    assert best_metadata["val_loss"] == 0.8
    assert final_metadata["best_epoch"] == 2
    assert final_metadata["stopped_early"] is True


def test_training_saves_best_execution_checkpoint(tmp_path, monkeypatch):
    import json
    import pocketsql.training.train as training

    records = [
        {
            "schema_id": "schema_a",
            "schema_sql": "CREATE TABLE t (id INTEGER, name TEXT);",
            "question": "show names",
            "sql": "SELECT name FROM t;",
            "difficulty": 1,
            "query_plan": {"family": "select"},
        }
        for _ in range(4)
    ]
    monkeypatch.setattr(training, "evaluate_loss", lambda *_: 1.0)
    execution_scores = iter((0.25, 0.75, 0.5))
    monkeypatch.setattr(
        training,
        "evaluate_model",
        lambda *_, **__: {"execution_accuracy": next(execution_scores), "exact_match": 0.0, "executable": 0.0, "syntactically_valid": 1.0},
    )
    config = {
        "seed": 1,
        "layers": 2,
        "hidden_dim": 128,
        "heads": 4,
        "ffn_dim": 512,
        "context_length": 256,
        "batch_size": 4,
        "grad_accum_steps": 1,
        "learning_rate": 5e-4,
        "warmup_steps": 0,
        "epochs": 3,
        "validation_execution_every": 1,
        "validation_execution_max_examples": 2,
    }
    checkpoint_dir = tmp_path / "checkpoint"

    training.train(records, config, checkpoint_dir, val_records=records)

    metadata = json.loads((checkpoint_dir.with_name("checkpoint-best-execution") / "metadata.json").read_text())
    assert metadata["epoch"] == 2
    assert metadata["validation_metric"] == "execution_accuracy"
    assert metadata["validation_score"] == 0.75


def test_validation_subset_spreads_samples_across_schemas_and_families():
    from pocketsql.training.train import validation_subset

    records = [
        {"schema_id": f"schema_{schema_index}", "query_plan": {"family": family}}
        for schema_index in range(4)
        for family in ("select", "join", "filter")
    ]
    subset = validation_subset(records, 4)

    assert {record["schema_id"] for record in subset} == {f"schema_{index}" for index in range(4)}
    assert {record["query_plan"]["family"] for record in subset} == {"select", "join", "filter"}


def test_batched_generation_matches_single_prompt_generation():
    from pocketsql.inference import generate_sql_batch, generate_sql_batch_with_targets

    model = tiny_model()
    tokenizer = ByteTokenizer()
    schema = "CREATE TABLE t (id INTEGER, name TEXT);"
    question = "show names"

    single = generate_sql_batch(model, [schema], [question], tokenizer, max_tokens=4)
    batched = generate_sql_batch(model, [schema, schema], [question, question], tokenizer, max_tokens=4)
    checked, targets = generate_sql_batch_with_targets(
        model,
        [schema, schema],
        [question, question],
        tokenizer,
        max_tokens=4,
    )

    assert batched == single * 2
    assert checked == batched
    assert len(targets) == 2
