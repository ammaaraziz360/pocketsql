"""Minimal TensorBoard scalar logging without a TensorFlow dependency."""
from __future__ import annotations

from pathlib import Path
import time

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.compat.proto.tensor_pb2 import TensorProto
from tensorboard.compat.proto.tensor_shape_pb2 import TensorShapeProto
from tensorboard.compat.proto.types_pb2 import DT_STRING
from tensorboard.plugins.text.metadata import create_summary_metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter


class TensorBoardLogger:
    """Write scalar events consumable by the ``tensorboard`` command-line app."""

    def __init__(self, log_dir: Path):
        log_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir = log_dir
        self._writer = EventFileWriter(str(log_dir))

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        event = Event(
            wall_time=time.time(),
            step=step,
            summary=Summary(value=[Summary.Value(tag=tag, simple_value=float(value))]),
        )
        self._writer.add_event(event)

    def add_text(self, tag: str, value: str, step: int) -> None:
        tensor = TensorProto(
            dtype=DT_STRING,
            string_val=[value.encode("utf-8")],
            tensor_shape=TensorShapeProto(dim=[TensorShapeProto.Dim(size=1)]),
        )
        event = Event(
            wall_time=time.time(),
            step=step,
            summary=Summary(
                value=[
                    Summary.Value(
                        tag=tag,
                        tensor=tensor,
                        metadata=create_summary_metadata(display_name=tag, description=""),
                    )
                ]
            ),
        )
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
