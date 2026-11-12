"""Background training thread (fixed infrastructure).

Runs the student's `finetune.train_custom_detector` so the GUI never freezes.
`finetune.py` is re-imported on every run: edit it, press Train again, no restart needed.
"""
from __future__ import annotations

import importlib
import traceback
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

import config
from config import TrainingParameters
from model_utils import resolve_weights
from training_reporter import TrainingReporter


class TrainThread(QThread):
    progress = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    finished_ok = pyqtSignal(str)
    failed = pyqtSignal(str, str)  # kind ("todo" | "import" | "stopped" | "error"), message

    def __init__(self, params: TrainingParameters, data_yaml: str, output_dir: str,
                 run_name: str, parent=None):
        super().__init__(parent)
        self._params = params
        self._data_yaml = data_yaml
        self._output_dir = output_dir
        self._run_name = run_name  # every training gets its own folder: output_dir / run_name
        self._reporter = TrainingReporter(self.progress.emit)

    def request_stop(self) -> None:
        self._reporter.request_stop()

    def run(self) -> None:
        try:
            self.log_message.emit("Loading finetune.py and ultralytics (the first run can take a while)...")
            import finetune

            # finetune.py reads RUN_NAME from config.py when it is (re)loaded: setting it here
            # gives every training its own folder without touching the student's file.
            config.RUN_NAME = self._run_name
            finetune = importlib.reload(finetune)
            best = finetune.train_custom_detector(
                dataset_config=self._data_yaml,
                pretrained_weights=resolve_weights(self._params.weights),
                epochs=self._params.epochs,
                batch_size=self._params.batch_size,
                output_dir=self._output_dir,
                reporter=self._reporter,
            )
            if (not best or not Path(best).exists()) and self._reporter.stop_requested:
                self.failed.emit("stopped", "Training was stopped before the first epoch finished.")
                return
            if not best or not Path(best).exists():
                self.failed.emit(
                    "error",
                    "train_custom_detector must return the path of an existing weights file "
                    f"(got: {best!r}).",
                )
                return
            self.finished_ok.emit(str(best))
        except NotImplementedError as error:
            self.failed.emit("todo", str(error))
        except ModuleNotFoundError as error:
            self.failed.emit(
                "import",
                f"{error}. Install the requirements first:  pip install -r requirements.txt",
            )
        except Exception as error:  # noqa: BLE001 - show any student error in the GUI
            self.log_message.emit(traceback.format_exc())
            self.failed.emit("error", f"{type(error).__name__}: {error}")
