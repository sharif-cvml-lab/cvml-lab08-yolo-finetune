"""Step 2 - Train page (fixed infrastructure)."""
from __future__ import annotations

import shutil
import time

from PyQt6.QtCore import QPointF, QRectF, QSettings, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config import (
    CUSTOM_MODEL_PATH,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_WEIGHTS,
    IMG_SIZE,
    LEARNING_RATE,
    MODELS_DIR,
    OPTIMIZER,
    RUNS_DIR,
    WEIGHT_CHOICES,
    TrainingParameters,
)
from dataset import DatasetError, DatasetStore
from gui_common import ValueSlider, create_sidebar
from train_thread import TrainThread


class LossPlot(QWidget):
    """Tiny loss-curve widget (training = green, validation = orange)."""

    def __init__(self):
        super().__init__()
        self.train: list[float] = []
        self.val: list[float | None] = []
        self._dark = True
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_dark(self, dark: bool) -> None:
        self._dark = dark
        self.update()

    def clear(self) -> None:
        self.train, self.val = [], []
        self.update()

    def add_point(self, train: float | None, val: float | None) -> None:
        if train is None:
            return
        self.train.append(train)
        self.val.append(val)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0a0d11" if self._dark else "#e9edf2"))
        muted = QColor("#8e98a6")
        plot = QRectF(46, 12, self.width() - 62, self.height() - 34)
        if not self.train:
            painter.setPen(muted)
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), "Loss curves appear here during training")
            return
        values = self.train + [v for v in self.val if v is not None]
        low, high = min(values), max(values)
        if high - low < 1e-6:
            high = low + 1.0
        count = max(len(self.train) - 1, 1)

        painter.setPen(QPen(muted, 1, Qt.PenStyle.DotLine))
        for step in range(5):
            y = plot.top() + plot.height() * step / 4
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.drawText(QRectF(2, y - 8, 40, 16), int(Qt.AlignmentFlag.AlignRight), f"{high - (high - low) * step / 4:.2f}")
        painter.drawText(QRectF(plot.left(), plot.bottom() + 4, plot.width(), 16),
                         int(Qt.AlignmentFlag.AlignCenter), f"epoch 1 ... {len(self.train)}")

        def point(index: int, value: float) -> QPointF:
            x = plot.left() + plot.width() * index / count
            y = plot.bottom() - plot.height() * (value - low) / (high - low)
            return QPointF(x, y)

        for series, color in ((self.train, "#4cc9a0"), (self.val, "#ffb703")):
            painter.setPen(QPen(QColor(color), 2))
            previous = None
            for index, value in enumerate(series):
                if value is None:
                    continue
                current = point(index, value)
                if previous is not None:
                    painter.drawLine(previous, current)
                painter.drawEllipse(current, 2.5, 2.5)
                previous = current


class TrainPage(QWidget):
    status_changed = pyqtSignal(str)
    training_finished = pyqtSignal(str)
    go_to_compare = pyqtSignal()

    def __init__(self, store: DatasetStore, settings: QSettings, parent=None):
        super().__init__(parent)
        self.store = store
        self.settings = settings
        self.thread: TrainThread | None = None
        self._start_time = 0.0
        self._fraction = 0.0
        self._build_ui()
        self._restore_settings()
        self.refresh_summary()
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._update_time)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        card = QFrame()
        card.setObjectName("VideoCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        title = QLabel("Fine-Tuning Progress")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        grid = QGridLayout()
        self.metric_labels: dict[str, QLabel] = {}
        for column, key in enumerate(("Epoch", "Train loss", "Val loss", "mAP50", "Elapsed", "ETA")):
            caption = QLabel(key)
            caption.setObjectName("Metric")
            value = QLabel("--")
            value.setObjectName("MetricValue")
            grid.addWidget(caption, 0, column)
            grid.addWidget(value, 1, column)
            self.metric_labels[key] = value
        layout.addLayout(grid)

        self.plot = LossPlot()
        layout.addWidget(self.plot, 1)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(170)
        self.log.setPlainText("Ready. Review the dataset summary, then press 'Train Model'.")
        layout.addWidget(self.log)
        splitter.addWidget(card)

        scroll, side = create_sidebar()
        model_box = QGroupBox("Pretrained Model")
        model_layout = QVBoxLayout(model_box)
        self.weights_combo = QComboBox()
        self.weights_combo.setEditable(True)
        self.weights_combo.addItems(list(WEIGHT_CHOICES))
        model_layout.addWidget(self.weights_combo)
        note = QLabel("Small 'nano' models are used so training stays fast on a CPU. "
                      "If the file is not in weights/ it is downloaded automatically (internet needed once).")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        model_layout.addWidget(note)
        side.addWidget(model_box)

        params_box = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout(params_box)
        self.epochs = ValueSlider("Epochs", 1, 400, DEFAULT_EPOCHS)
        self.batch = ValueSlider("Batch size", 1, 32, DEFAULT_BATCH_SIZE)
        params_layout.addWidget(self.epochs)
        params_layout.addWidget(self.batch)
        fixed = QLabel(f"Fixed in config.py:  image size {IMG_SIZE}  |  {OPTIMIZER}, lr {LEARNING_RATE}  |  "
                       "only the new class outputs are trained (COCO classes stay frozen)")
        fixed.setObjectName("Muted")
        fixed.setWordWrap(True)
        params_layout.addWidget(fixed)
        side.addWidget(params_box)

        dataset_box = QGroupBox("Dataset")
        dataset_layout = QVBoxLayout(dataset_box)
        self.summary_label = QLabel("")
        self.summary_label.setWordWrap(True)
        dataset_layout.addWidget(self.summary_label)
        side.addWidget(dataset_box)

        actions_box = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_box)
        self.train_button = QPushButton("Train Model")
        self.train_button.setObjectName("accent")
        self.train_button.clicked.connect(self._start_training)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("danger")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_training)
        self.compare_button = QPushButton("Open Before / After")
        self.compare_button.clicked.connect(self.go_to_compare)
        actions_layout.addWidget(self.train_button)
        actions_layout.addWidget(self.stop_button)
        actions_layout.addWidget(self.compare_button)
        side.addWidget(actions_box)
        side.addStretch(1)
        splitter.addWidget(scroll)
        splitter.setSizes([1000, 420])
        root.addWidget(splitter)

    # ------------------------------------------------------------ lifecycle
    def activate(self) -> None:
        self.refresh_summary()

    def deactivate(self) -> None:
        pass

    def apply_theme(self, dark: bool) -> None:
        self.plot.set_dark(dark)

    def is_training(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def shutdown(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.thread.request_stop()
            self.thread.wait(20000)
        self._save_settings()

    def reset_defaults(self) -> None:
        if self.is_training():
            return
        self.weights_combo.setCurrentText(DEFAULT_WEIGHTS)
        self.epochs.set_value(DEFAULT_EPOCHS)
        self.batch.set_value(DEFAULT_BATCH_SIZE)
        self._save_settings()

    def _restore_settings(self) -> None:
        self.weights_combo.setCurrentText(self.settings.value("train/weights", DEFAULT_WEIGHTS))
        self.epochs.set_value(int(self.settings.value("train/epochs", DEFAULT_EPOCHS)))
        self.batch.set_value(int(self.settings.value("train/batch", DEFAULT_BATCH_SIZE)))

    def _save_settings(self) -> None:
        self.settings.setValue("train/weights", self.weights_combo.currentText().strip() or DEFAULT_WEIGHTS)
        self.settings.setValue("train/epochs", self.epochs.value())
        self.settings.setValue("train/batch", self.batch.value())
        self.settings.sync()

    def refresh_summary(self) -> None:
        stats = self.store.stats()
        names = self.store.classes()
        lines = [f"{len(names)} custom class(es), ids 80-{79 + len(names)}" if names else "No custom classes yet.",
                 f"Train images: {stats.train_images}   Val images: {stats.val_images}"]
        if names:
            lines.append("Boxes: " + ", ".join(f"{n} {c}" for n, c in zip(names, stats.instances)))
        errors, warnings = self.store.check_ready()
        lines += [f"✗ {message}" for message in errors] + [f"! {message}" for message in warnings]
        self.summary_label.setText("\n".join(lines))

    # ------------------------------------------------------------- training
    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _start_training(self) -> None:
        if self.is_training():
            return
        try:
            yaml_path = self.store.write_data_yaml()
        except DatasetError as error:
            QMessageBox.warning(self, "Dataset not ready", str(error))
            return
        self._save_settings()
        params = TrainingParameters(
            weights=self.weights_combo.currentText().strip() or DEFAULT_WEIGHTS,
            epochs=self.epochs.value(),
            batch_size=self.batch.value(),
        )
        self.plot.clear()
        self.log.clear()
        self.progress_bar.setValue(0)
        self._fraction = 0.0
        for label in self.metric_labels.values():
            label.setText("--")
        self._append_log(f"Starting: {params.weights}, {params.epochs} epochs, batch {params.batch_size}.")
        _, warnings = self.store.check_ready()
        for message in warnings:
            self._append_log(f"Warning: {message}")

        self.thread = TrainThread(params, str(yaml_path), str(RUNS_DIR), parent=self)
        self.thread.progress.connect(self._on_event)
        self.thread.log_message.connect(self._append_log)
        self.thread.finished_ok.connect(self._on_finished)
        self.thread.failed.connect(self._on_failed)
        self.thread.finished.connect(self._on_thread_done)
        self._start_time = time.monotonic()
        self._timer.start()
        self._set_running(True)
        self.thread.start()
        self.status_changed.emit("Training started (running in a background thread).")

    def _stop_training(self) -> None:
        if self.thread is not None:
            self.thread.request_stop()
            self.stop_button.setEnabled(False)
            self._append_log("Stop requested: finishing the current step...")
            self.status_changed.emit("Stopping training...")

    def _set_running(self, running: bool) -> None:
        self.train_button.setEnabled(not running)
        self.stop_button.setEnabled(running)
        self.weights_combo.setEnabled(not running)
        self.epochs.setEnabled(not running)
        self.batch.setEnabled(not running)

    def _on_event(self, event: dict) -> None:
        kind = event.get("type")
        epochs = max(int(event.get("epochs", 1)), 1)
        if kind == "start":
            self._append_log(f"Training on {epochs} epoch(s)...")
            self.metric_labels["Epoch"].setText(f"0 / {epochs}")
        elif kind == "batch":
            batches = max(int(event["batches"]), 1)
            self._fraction = (int(event["epoch"]) + int(event["batch"]) / batches) / epochs
            self.progress_bar.setValue(int(min(self._fraction, 1.0) * 1000))
            self.metric_labels["Epoch"].setText(f"{int(event['epoch']) + 1} / {epochs}")
        elif kind == "epoch":
            epoch = int(event["epoch"]) + 1
            self._fraction = epoch / epochs
            self.progress_bar.setValue(int(min(self._fraction, 1.0) * 1000))
            train, val, map50 = event.get("train_loss"), event.get("val_loss"), event.get("map50")
            self.metric_labels["Epoch"].setText(f"{epoch} / {epochs}")
            self.metric_labels["Train loss"].setText(f"{train:.3f}" if train is not None else "--")
            self.metric_labels["Val loss"].setText(f"{val:.3f}" if val is not None else "--")
            self.metric_labels["mAP50"].setText(f"{map50:.3f}" if map50 is not None else "--")
            self.plot.add_point(train, val)
            self._append_log(
                f"Epoch {epoch}/{epochs}  train {train if train is None else round(train, 3)}  "
                f"val {val if val is None else round(val, 3)}  mAP50 {map50 if map50 is None else round(map50, 3)}"
            )
            self.status_changed.emit(f"Training: epoch {epoch}/{epochs}")

    def _update_time(self) -> None:
        elapsed = time.monotonic() - self._start_time
        self.metric_labels["Elapsed"].setText(self._format_time(elapsed))
        if self._fraction > 0.02:
            self.metric_labels["ETA"].setText(self._format_time(elapsed * (1 - self._fraction) / self._fraction))

    @staticmethod
    def _format_time(seconds: float) -> str:
        seconds = int(seconds)
        return f"{seconds // 60:02d}:{seconds % 60:02d}"

    def _on_finished(self, best_path: str) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best_path, CUSTOM_MODEL_PATH)
        self.progress_bar.setValue(1000)
        self._append_log(f"Done. Best weights copied to {CUSTOM_MODEL_PATH}")
        self.status_changed.emit("Training finished. Open Step 3 to compare Base and Custom YOLO.")
        self.training_finished.emit(str(CUSTOM_MODEL_PATH))

    def _on_failed(self, kind: str, message: str) -> None:
        self._append_log(f"Training did not run: {message}")
        self.status_changed.emit("Training stopped: see the log.")
        if kind == "todo":
            QMessageBox.information(
                self, "finetune.py is not finished",
                f"{message}\n\nOpen finetune.py, complete the TODOs, save the file and press Train again.\n"
                "(Steps 1 and 3 keep working in the meantime.)",
            )
        elif kind == "import":
            QMessageBox.warning(self, "Missing package", message)
        else:
            QMessageBox.warning(self, "Training failed", message)

    def _on_thread_done(self) -> None:
        self._timer.stop()
        self._update_time()
        self._set_running(False)
        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None
