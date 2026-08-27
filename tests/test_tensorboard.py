from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from pocketsql.training.tensorboard import TensorBoardLogger


def test_tensorboard_logger_writes_readable_scalar_events(tmp_path):
    logger = TensorBoardLogger(tmp_path / "run")
    logger.add_scalar("loss/train_epoch", 0.25, 3)
    logger.add_scalar("validation_execution/execution_accuracy", 0.75, 3)
    logger.add_text("validation/examples", "| Gold | Predicted |\n|---|---|\n| SELECT 1 | SELECT 2 |", 3)
    logger.close()

    events = EventAccumulator(str(tmp_path / "run"))
    events.Reload()
    assert events.Tags()["scalars"] == ["loss/train_epoch", "validation_execution/execution_accuracy"]
    assert events.Scalars("loss/train_epoch")[0].step == 3
    assert events.Scalars("loss/train_epoch")[0].value == 0.25
    assert "validation/examples" in events.Tags()["tensors"]
