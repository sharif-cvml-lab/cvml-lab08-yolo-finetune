"""Progress bridge between ultralytics and the GUI (fixed infrastructure).

`TrainingReporter.attach(model)` registers ultralytics callbacks. During training they
call `emit(event)` with plain dictionaries, which the training thread forwards to the
Qt user interface:

    {"type": "start",  "epochs": 40}
    {"type": "batch",  "epoch": 3, "epochs": 40, "batch": 2, "batches": 5}
    {"type": "epoch",  "epoch": 3, "epochs": 40, "train_loss": 2.1,
                       "val_loss": 2.4 or None, "map50": 0.31 or None}
"""
from __future__ import annotations

from typing import Callable


def _to_float(value) -> float | None:
    try:
        if hasattr(value, "sum"):
            value = value.sum()
        if hasattr(value, "item"):
            value = value.item()
        number = float(value)
    except Exception:
        return None
    return number if number == number else None  # drop NaN


class TrainingReporter:
    def __init__(self, emit: Callable[[dict], None] | None = None):
        self._emit = emit or (lambda event: None)
        self.stop_requested = False
        self._batch = 0

    def request_stop(self) -> None:
        self.stop_requested = True

    def attach(self, model) -> None:
        model.add_callback("on_train_start", self._on_train_start)
        model.add_callback("on_train_epoch_start", self._on_epoch_start)
        model.add_callback("on_train_batch_end", self._on_batch_end)
        model.add_callback("on_fit_epoch_end", self._on_fit_epoch_end)

    # ---- ultralytics callbacks (each receives the trainer) --------------- #
    def _on_train_start(self, trainer) -> None:
        self._emit({"type": "start", "epochs": int(trainer.epochs)})

    def _on_epoch_start(self, trainer) -> None:
        self._batch = 0

    def _on_batch_end(self, trainer) -> None:
        self._batch += 1
        try:
            batches = len(trainer.train_loader)
        except Exception:
            batches = max(self._batch, 1)
        self._emit({
            "type": "batch",
            "epoch": int(trainer.epoch),
            "epochs": int(trainer.epochs),
            "batch": self._batch,
            "batches": batches,
        })
        if self.stop_requested:
            trainer.stop = True

    def _on_fit_epoch_end(self, trainer) -> None:
        metrics = getattr(trainer, "metrics", {}) or {}
        val_parts = [_to_float(metrics.get(k)) for k in
                     ("val/box_loss", "val/cls_loss", "val/dfl_loss")]
        val_loss = sum(v for v in val_parts if v is not None) if any(
            v is not None for v in val_parts) else None
        self._emit({
            "type": "epoch",
            "epoch": int(trainer.epoch),
            "epochs": int(trainer.epochs),
            "train_loss": _to_float(getattr(trainer, "tloss", None)),
            "val_loss": val_loss,
            "map50": _to_float(metrics.get("metrics/mAP50(B)")),
        })
        if self.stop_requested:
            trainer.stop = True
