"""Step 2 - Train page (fixed infrastructure).

Two tabs on the left:
  * Training  - live progress, loss curves and log of the training that is running
  * History   - every earlier training: compare their curves, open them in Step 3, delete them
"""
from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import QItemSelectionModel, QSettings, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from config import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_WEIGHTS,
    IMG_SIZE,
    LEARNING_RATE,
    OPTIMIZER,
    WEIGHT_CHOICES,
    TrainingParameters,
)
from dataset import DatasetError, DatasetStore
from gui_common import ValueSlider, create_sidebar
from gui_plot import RUN_COLORS, TRAIN_COLOR, VAL_COLOR, CurvePlot, Series
from history import (
    STATUS_FINISHED,
    STATUS_INTERRUPTED,
    STATUS_STOPPED,
    HistoryError,
    RunHistory,
    RunRecord,
)
from train_thread import TrainThread

LOSS_Y_LABEL = "Loss (box + class + dfl)"
LOSS_EXPLANATION = (
    "<b style='color:%s'>Green</b> = training loss (on the images the model learns from). "
    "<b style='color:%s'>Orange</b> = validation loss (on images it never trains on). "
    "Loss measures how wrong the predictions are: <b>lower is better</b>. "
    "If green keeps falling but orange goes up, the model is memorising the photos (overfitting)."
) % (TRAIN_COLOR, VAL_COLOR)

HISTORY_COLUMNS = ("Date", "Name", "Model", "Epochs", "Batch", "Classes", "Images train / val",
                   "Best mAP50", "Final val loss", "Status")
HISTORY_METRICS = (
    ("Validation loss", "Validation loss (lower is better)", lambda p: p.val_loss),
    ("Training loss", "Training loss (lower is better)", lambda p: p.train_loss),
    ("mAP50 (accuracy)", "mAP50 (higher is better)", lambda p: p.map50),
)


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def short_date(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%d %b %H:%M")
    except ValueError:
        return iso


def fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


class TrainPage(QWidget):
    status_changed = pyqtSignal(str)
    training_finished = pyqtSignal(str)       # run_id of the new training
    training_state_changed = pyqtSignal(bool)  # True while a training is running
    history_changed = pyqtSignal()
    go_to_compare = pyqtSignal(str)           # run_id to select in Step 3 ("" = keep the current one)

    def __init__(self, store: DatasetStore, history: RunHistory, settings: QSettings, parent=None):
        super().__init__(parent)
        self.store = store
        self.history = history
        self.settings = settings
        self.thread: TrainThread | None = None
        self._record: RunRecord | None = None
        self._stop_requested = False
        self._log_lines: list[str] = []
        self._train_curve: list[float | None] = []
        self._val_curve: list[float | None] = []
        self._start_time = 0.0
        self._fraction = 0.0
        self._dark = True
        self._build_ui()
        self._restore_settings()
        self.refresh_summary()
        self.refresh_history()
        self._timer = QTimer(self)
        self._timer.setInterval(500)
        self._timer.timeout.connect(self._update_time)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_training_tab(), "Training")
        self.tabs.addTab(self._build_history_tab(), "History")
        self.tabs.currentChanged.connect(lambda _i: self.refresh_history())
        splitter.addWidget(self.tabs)
        splitter.addWidget(self._build_sidebar())
        splitter.setSizes([1000, 420])
        root.addWidget(splitter)

    def _build_training_tab(self) -> QWidget:
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

        tips = {
            "Epoch": "One epoch = the model has seen every training image once.",
            "Train loss": "How wrong the model is on the training images (lower is better).",
            "Val loss": "How wrong the model is on validation images it never trains on (lower is better).",
            "mAP50": "Detection accuracy on the validation images, from 0 to 1 (higher is better).",
            "Elapsed": "Time since the training started.",
            "ETA": "Estimated time left.",
        }
        grid = QGridLayout()
        self.metric_labels: dict[str, QLabel] = {}
        for column, key in enumerate(tips):
            caption = QLabel(key)
            caption.setObjectName("Metric")
            caption.setToolTip(tips[key])
            value = QLabel("--")
            value.setObjectName("MetricValue")
            value.setToolTip(tips[key])
            grid.addWidget(caption, 0, column)
            grid.addWidget(value, 1, column)
            self.metric_labels[key] = value
        layout.addLayout(grid)

        self.plot = CurvePlot("Epoch", LOSS_Y_LABEL, "Loss curves appear here during training")
        layout.addWidget(self.plot, 1)
        explanation = QLabel(LOSS_EXPLANATION)
        explanation.setObjectName("Muted")
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self._set_log("Ready. Review the dataset summary, then press 'Train Model'.")
        layout.addWidget(self.log)
        return card

    def _build_history_tab(self) -> QWidget:
        card = QFrame()
        card.setObjectName("VideoCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)
        head = QHBoxLayout()
        title = QLabel("Training History")
        title.setObjectName("PanelTitle")
        hint = QLabel("Select one or more rows (Ctrl / Shift) to compare their curves.")
        hint.setObjectName("Muted")
        head.addWidget(title)
        head.addStretch(1)
        head.addWidget(hint)
        layout.addLayout(head)

        self.table = QTableWidget(0, len(HISTORY_COLUMNS))
        self.table.setHorizontalHeaderLabels(HISTORY_COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(140)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self._on_history_selection)
        layout.addWidget(self.table, 2)

        metric_row = QHBoxLayout()
        metric_row.addWidget(QLabel("Compare:"))
        self.metric_combo = QComboBox()
        for label, _axis, _getter in HISTORY_METRICS:
            self.metric_combo.addItem(label)
        self.metric_combo.currentIndexChanged.connect(self._on_history_selection)
        metric_row.addWidget(self.metric_combo)
        metric_row.addStretch(1)
        layout.addLayout(metric_row)

        self.history_plot = CurvePlot("Epoch", HISTORY_METRICS[0][1], "Select a training above to see its curve")
        layout.addWidget(self.history_plot, 2)

        self.details_label = QLabel("")
        self.details_label.setObjectName("Muted")
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)
        self.history_log = QPlainTextEdit()
        self.history_log.setReadOnly(True)
        self.history_log.setMaximumHeight(90)
        layout.addWidget(self.history_log)

        buttons = QHBoxLayout()
        self.open_compare_button = QPushButton("Open in Before / After")
        self.open_compare_button.setObjectName("accent")
        self.open_compare_button.clicked.connect(self._open_selected_in_compare)
        self.open_folder_button = QPushButton("Open Folder")
        self.open_folder_button.clicked.connect(self._open_selected_folder)
        self.delete_button = QPushButton("Delete Selected")
        self.delete_button.setObjectName("danger")
        self.delete_button.clicked.connect(self._delete_selected)
        for button in (self.open_compare_button, self.open_folder_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        buttons.addWidget(self.delete_button)
        layout.addLayout(buttons)
        self._update_history_buttons()
        return card

    def _build_sidebar(self):
        scroll, side = create_sidebar()
        model_box = QGroupBox("Pretrained Model")
        model_layout = QVBoxLayout(model_box)
        self.weights_combo = QComboBox()
        self.weights_combo.setEditable(True)
        self.weights_combo.addItems(list(WEIGHT_CHOICES))
        model_layout.addWidget(self.weights_combo)
        note = QLabel("Small 'nano' models are used so training stays fast on a CPU. "
                      "If the file is not in assets/weights/ it is downloaded there automatically "
                      "(internet needed once).")
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
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Run name"))
        self.run_name_edit = QLineEdit()
        self.run_name_edit.setPlaceholderText("optional, e.g. more_epochs")
        self.run_name_edit.setToolTip("A label that helps you recognise this training in the History tab.")
        name_row.addWidget(self.run_name_edit, 1)
        params_layout.addLayout(name_row)
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
        self.compare_button.clicked.connect(lambda: self.go_to_compare.emit(""))
        actions_layout.addWidget(self.train_button)
        actions_layout.addWidget(self.stop_button)
        actions_layout.addWidget(self.compare_button)
        side.addWidget(actions_box)
        side.addStretch(1)
        return scroll

    # ------------------------------------------------------------ lifecycle
    def activate(self) -> None:
        self.refresh_summary()
        self.refresh_history()

    def deactivate(self) -> None:
        pass

    def apply_theme(self, dark: bool) -> None:
        self._dark = dark
        self.plot.set_dark(dark)
        self.history_plot.set_dark(dark)

    def is_training(self) -> bool:
        return self.thread is not None and self.thread.isRunning()

    def shutdown(self) -> None:
        if self.thread is not None and self.thread.isRunning():
            self.thread.request_stop()
            if not self.thread.wait(20000):
                self.thread.terminate()
                self.thread.wait(2000)
            self._stop_requested = True
            self._finalize_record(STATUS_STOPPED)
        self._save_settings()

    def reset_defaults(self) -> None:
        if self.is_training():
            return
        self.weights_combo.setCurrentText(DEFAULT_WEIGHTS)
        self.epochs.set_value(DEFAULT_EPOCHS)
        self.batch.set_value(DEFAULT_BATCH_SIZE)
        self.run_name_edit.clear()
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

    # ---------------------------------------------------------------- log
    def _set_log(self, text: str) -> None:
        self._log_lines = [text] if text else []
        self.log.setPlainText(text)

    def _append_log(self, text: str) -> None:
        self._log_lines.append(text)
        self.log.appendPlainText(text)

    # ------------------------------------------------------------- training
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
        stats = self.store.stats()
        self._record = self.history.create(
            model=params.weights, epochs=params.epochs, batch=params.batch_size,
            classes=self.store.classes(), train_images=stats.train_images, val_images=stats.val_images,
            name=self.run_name_edit.text(),
        )
        self.tabs.setCurrentIndex(0)
        self.plot.clear()
        self._train_curve, self._val_curve = [], []
        self._set_log("")
        self.progress_bar.setValue(0)
        self._fraction = 0.0
        self._stop_requested = False
        for label in self.metric_labels.values():
            label.setText("--")
        self._append_log(f"Starting: {params.weights}, {params.epochs} epochs, batch {params.batch_size}.")
        self._append_log(f"Results will be saved in {self._record.path}")
        _, warnings = self.store.check_ready()
        for message in warnings:
            self._append_log(f"Warning: {message}")

        self.thread = TrainThread(params, str(yaml_path), str(self.history.root), self._record.run_id, parent=self)
        self.thread.progress.connect(self._on_event)
        self.thread.log_message.connect(self._append_log)
        self.thread.finished_ok.connect(self._on_finished)
        self.thread.failed.connect(self._on_failed)
        self.thread.finished.connect(self._on_thread_done)
        self._start_time = time.monotonic()
        self._timer.start()
        self._set_running(True)
        self.thread.start()
        self.refresh_history()
        self.status_changed.emit("Training started (running in a background thread).")

    def _stop_training(self) -> None:
        if self.thread is not None:
            self._stop_requested = True
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
        self.run_name_edit.setEnabled(not running)
        self.training_state_changed.emit(running)

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
            self.metric_labels["Train loss"].setText(fmt(train))
            self.metric_labels["Val loss"].setText(fmt(val))
            self.metric_labels["mAP50"].setText(fmt(map50))
            if train is not None:
                self._train_curve.append(train)
                self._val_curve.append(val)
                self.plot.set_series([
                    Series("Train loss", self._train_curve, TRAIN_COLOR),
                    Series("Validation loss", self._val_curve, VAL_COLOR),
                ])
            self._append_log(f"Epoch {epoch}/{epochs}  train {fmt(train)}  val {fmt(val)}  mAP50 {fmt(map50)}")
            self.status_changed.emit(f"Training: epoch {epoch}/{epochs}")

    def _update_time(self) -> None:
        elapsed = time.monotonic() - self._start_time
        self.metric_labels["Elapsed"].setText(format_seconds(elapsed))
        if self._fraction > 0.02:
            self.metric_labels["ETA"].setText(format_seconds(elapsed * (1 - self._fraction) / self._fraction))

    def _finalize_record(self, status: str) -> None:
        """Write the final state of the running training into its history folder."""
        record, self._record = self._record, None
        if record is None:
            return
        if not record.has_weights:
            self.history.discard(record)  # nothing useful was produced
            return
        self.history.finish(record, status, time.monotonic() - self._start_time, "\n".join(self._log_lines))

    def _on_finished(self, best_path: str) -> None:
        status = STATUS_STOPPED if self._stop_requested else STATUS_FINISHED
        run_id = self._record.run_id if self._record is not None else ""
        if status == STATUS_FINISHED:
            self.progress_bar.setValue(1000)
        self._append_log("Stopped by you. The best weights so far were kept." if self._stop_requested
                         else "Done.")
        self._append_log(f"Model saved in {best_path}")
        self._finalize_record(status)
        self.refresh_history()
        self.history_changed.emit()
        self.status_changed.emit("Training finished. Open Step 3 to compare Base and Custom YOLO.")
        self.training_finished.emit(run_id)

    def _on_failed(self, kind: str, message: str) -> None:
        self._append_log(f"Training did not run: {message}")
        self.status_changed.emit("Training stopped: see the log.")
        self._finalize_record(STATUS_INTERRUPTED)
        self.refresh_history()
        self.history_changed.emit()
        if kind == "todo":
            QMessageBox.information(
                self, "finetune.py is not finished",
                f"{message}\n\nOpen finetune.py, complete the TODOs, save the file and press Train again.\n"
                "(Steps 1 and 3 keep working in the meantime.)",
            )
        elif kind == "import":
            QMessageBox.warning(self, "Missing package", message)
        elif kind == "stopped":
            pass  # the user asked for it, the log already says so
        else:
            QMessageBox.warning(self, "Training failed", message)

    def _on_thread_done(self) -> None:
        self._timer.stop()
        self._update_time()
        self._set_running(False)
        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None

    # -------------------------------------------------------------- history
    def _running_id(self) -> str:
        return self._record.run_id if self._record is not None and self.is_training() else ""

    def refresh_history(self) -> None:
        selected = set(self._selected_ids())
        runs = self.history.list_runs()
        self.table.blockSignals(True)
        self.table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            info = run.summary()
            running = run.run_id == self._running_id()
            cells = (
                short_date(run.created),
                run.name or "-",
                Path(run.model).name,
                f"{info['epochs_done']} / {run.epochs}",
                str(run.batch),
                ", ".join(run.classes) if run.classes else "-",
                f"{run.train_images} / {run.val_images}",
                fmt(info["best_map50"]),
                fmt(info["final_val_loss"]),
                "running..." if running else run.status,
            )
            for column, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, run.run_id)
                self.table.setItem(row, column, item)
            if run.run_id in selected:
                self.table.selectionModel().select(
                    self.table.model().index(row, 0),
                    QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
                )
        self.table.blockSignals(False)
        self._on_history_selection()

    def _selected_ids(self) -> list[str]:
        rows = sorted({index.row() for index in self.table.selectionModel().selectedRows()})
        ids = []
        for row in rows:
            item = self.table.item(row, 0)
            if item is not None:
                ids.append(item.data(Qt.ItemDataRole.UserRole))
        return ids

    def _update_history_buttons(self) -> None:
        count = len(self._selected_ids()) if hasattr(self, "table") else 0
        self.open_compare_button.setEnabled(count == 1)
        self.open_folder_button.setEnabled(count == 1)
        self.delete_button.setEnabled(count >= 1)

    def _on_history_selection(self, *_args) -> None:
        self._update_history_buttons()
        ids = self._selected_ids()
        _label, axis_title, getter = HISTORY_METRICS[self.metric_combo.currentIndex()]
        self.history_plot.set_labels("Epoch", axis_title)
        series = []
        records = []
        for position, run_id in enumerate(ids):
            record = self.history.get(run_id)
            if record is None:
                continue
            records.append(record)
            name = record.name or f"{short_date(record.created)} {Path(record.model).stem} e{record.epochs}"
            series.append(Series(name, [getter(p) for p in record.curve()], RUN_COLORS[position % len(RUN_COLORS)]))
        self.history_plot.set_series(series)

        if len(records) == 1:
            record = records[0]
            info = record.summary()
            best = (f"best mAP50 {info['best_map50']:.3f} at epoch {info['best_epoch']}"
                    if info["best_map50"] is not None else "no mAP50 yet")
            self.details_label.setText(
                f"{record.title}\nModel {record.model}  |  batch {record.batch}  |  classes: "
                f"{', '.join(record.classes) or '-'}  |  {record.train_images} train / {record.val_images} val images  |  "
                f"{best}  |  time {format_seconds(record.duration_sec)}  |  folder: {record.path}"
            )
            self.history_log.setPlainText(record.log_path.read_text(encoding="utf-8")
                                          if record.log_path.exists() else "")
        elif len(records) > 1:
            self.details_label.setText(f"{len(records)} trainings selected: curves are drawn together.")
            self.history_log.clear()
        else:
            self.details_label.setText("Every training is saved in assets/outputs/trainings/ - one folder each.")
            self.history_log.clear()

    def _open_selected_in_compare(self) -> None:
        ids = self._selected_ids()
        if len(ids) != 1:
            return
        record = self.history.get(ids[0])
        if record is None or not record.has_weights:
            QMessageBox.information(self, "No model", "This training has no saved model to compare.")
            return
        self.go_to_compare.emit(record.run_id)

    def _open_selected_folder(self) -> None:
        ids = self._selected_ids()
        if len(ids) == 1:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.history.root / ids[0])))

    def _delete_selected(self) -> None:
        ids = self._selected_ids()
        if not ids:
            return
        running = self._running_id()
        if running in ids:
            QMessageBox.information(self, "Training is running",
                                    "Stop the running training before deleting it.")
            ids = [i for i in ids if i != running]
            if not ids:
                return
        answer = QMessageBox.question(
            self, "Delete trainings",
            f"Delete {len(ids)} training(s)?\nTheir results, logs and saved models are removed permanently.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        errors = []
        for run_id in ids:
            try:
                self.history.delete(run_id)
            except HistoryError as error:
                errors.append(str(error))
        self.refresh_history()
        self.history_changed.emit()
        if errors:
            QMessageBox.warning(self, "Could not delete", "\n".join(errors))
        else:
            self.status_changed.emit(f"Deleted {len(ids)} training(s).")
