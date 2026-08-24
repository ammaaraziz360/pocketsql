import pytest

mx = pytest.importorskip("mlx.core")

from pocketsql.model.tokenizer import ByteTokenizer
from pocketsql.model.transformer import ModelConfig, PocketSQLTransformer
from pocketsql.training.checkpoint import load_checkpoint, save_checkpoint
from pocketsql.training.train import train_step


def tiny_model():
    tokenizer = ByteTokenizer()
    return PocketSQLTransformer(ModelConfig(vocab_size=tokenizer.vocab_size, layers=2, hidden_dim=128, heads=4, ffn_dim=512, context_length=256))


def test_model_forward_dimensions():
    model = tiny_model()
    logits = model(mx.zeros((2, 16), dtype=mx.int32))
    assert logits.shape == (2, 16, ByteTokenizer().vocab_size)


def test_checkpoint_round_trip(tmp_path):
    model = tiny_model()
    save_checkpoint(tmp_path / "checkpoint", model, {"step": 1})
    restored = tiny_model()
    assert load_checkpoint(tmp_path / "checkpoint", restored) == {"step": 1}


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