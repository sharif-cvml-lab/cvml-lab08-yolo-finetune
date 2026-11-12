"""Step 3 - Before / After page (fixed infrastructure)."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from config import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CONFIDENCE,
    DEFAULT_WEIGHTS,
    FILE_FILTER,
)
from gui_common import ValueSlider, create_sidebar, create_video_card
from history import RunHistory
from inference_thread import InferenceThread
from thread_utils import call_when_idle, retire_thread

BASE_TITLE = "Base YOLO  (pretrained, 80 COCO classes)"
CUSTOM_TITLE = "Custom YOLO  (after your fine-tuning)"


class ComparePage(QWidget):
    status_changed = pyqtSignal(str)
    fps_changed = pyqtSignal(float)

    def __init__(self, history: RunHistory, settings: QSettings, parent=None):
        super().__init__(parent)
        self.history = history
        self.settings = settings
        self.thread: InferenceThread | None = None
        self.source_type = "camera"
        self.selected_file = ""
        self.base_weights = DEFAULT_WEIGHTS
        self.custom_weights: str | None = None
        self.run_id = ""
        self._active = False
        self._start_token = 0
        self._build_ui()
        self._restore_settings()
        self.refresh_runs()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        video_side = QWidget()
        grid = QGridLayout(video_side)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(10)
        self.base_panel = create_video_card(BASE_TITLE)
        self.custom_panel = create_video_card(CUSTOM_TITLE)
        grid.addWidget(self.base_panel, 0, 0)
        grid.addWidget(self.custom_panel, 0, 1)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        splitter.addWidget(video_side)

        scroll, layout = create_sidebar()

        source_box = QGroupBox("Input Source")
        source_layout = QVBoxLayout(source_box)
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera index"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 9)
        self.camera_spin.valueChanged.connect(self._camera_index_changed)
        camera_row.addWidget(self.camera_spin)
        source_layout.addLayout(camera_row)
        buttons = QHBoxLayout()
        self.webcam_button = QPushButton("Webcam")
        self.webcam_button.clicked.connect(self._use_webcam)
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self._browse_file)
        self.upload_button = QPushButton("Upload")
        self.upload_button.setObjectName("accent")
        self.upload_button.setEnabled(False)
        self.upload_button.clicked.connect(self._upload_selected_file)
        buttons.addWidget(self.webcam_button)
        buttons.addWidget(self.browse_button)
        buttons.addWidget(self.upload_button)
        source_layout.addLayout(buttons)
        self.file_label = QLabel("No offline image/video selected")
        self.file_label.setObjectName("Muted")
        self.file_label.setWordWrap(True)
        source_layout.addWidget(self.file_label)
        actions = QHBoxLayout()
        self.pause_button = QPushButton("Pause")
        self.pause_button.clicked.connect(self.toggle_pause)
        self.restart_button = QPushButton("Restart")
        self.restart_button.clicked.connect(self._restart_source)
        actions.addWidget(self.pause_button)
        actions.addWidget(self.restart_button)
        source_layout.addLayout(actions)
        layout.addWidget(source_box)

        detect_box = QGroupBox("Detection")
        detect_layout = QVBoxLayout(detect_box)
        self.confidence = ValueSlider("Confidence %", 5, 95, DEFAULT_CONFIDENCE)
        self.confidence.slider.valueChanged.connect(self._confidence_changed)
        detect_layout.addWidget(self.confidence)
        layout.addWidget(detect_box)

        model_box = QGroupBox("Models")
        model_layout = QVBoxLayout(model_box)
        model_layout.addWidget(QLabel("Custom model (choose one of your trainings):"))
        self.run_combo = QComboBox()
        self.run_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.run_combo.setMinimumContentsLength(24)
        self.run_combo.currentIndexChanged.connect(self._on_run_selected)
        model_layout.addWidget(self.run_combo)
        self.base_label = QLabel("")
        self.custom_label = QLabel("")
        self.base_label.setWordWrap(True)
        self.custom_label.setWordWrap(True)
        self.custom_label.setObjectName("Muted")
        model_layout.addWidget(self.base_label)
        model_layout.addWidget(self.custom_label)
        refresh_button = QPushButton("Refresh List")
        refresh_button.clicked.connect(lambda: self.refresh_runs())
        model_layout.addWidget(refresh_button)
        layout.addWidget(model_box)

        note = QLabel("Try your custom objects in front of the webcam: the Base model cannot name them, "
                      "the Custom model can. Ids 0-79 are COCO, ids 80+ are your classes. "
                      "Every training you run in Step 2 appears in the list: pick another one to compare.")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        splitter.addWidget(scroll)
        splitter.setSizes([1100, 380])
        root.addWidget(splitter)
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_pause)

    # ------------------------------------------------------------ lifecycle
    def activate(self) -> None:
        self._active = True
        self.refresh_runs()
        self._restart_thread()

    def deactivate(self) -> None:
        self._active = False
        self._stop_thread()

    def apply_theme(self, dark: bool) -> None:
        pass

    def shutdown(self) -> None:
        self._stop_thread()
        self._save_settings()

    def reset_defaults(self) -> None:
        self.camera_spin.setValue(DEFAULT_CAMERA_INDEX)
        self.confidence.set_value(DEFAULT_CONFIDENCE)
        self._save_settings()

    # ------------------------------------------------------------- settings
    def _restore_settings(self) -> None:
        self.camera_spin.setValue(self.settings.value("compare/camera_index", DEFAULT_CAMERA_INDEX, int))
        self.confidence.set_value(int(self.settings.value("compare/confidence", DEFAULT_CONFIDENCE)))
        self.run_id = str(self.settings.value("compare/run_id", "") or "")

    def _save_settings(self) -> None:
        self.settings.setValue("compare/camera_index", self.camera_spin.value())
        self.settings.setValue("compare/confidence", self.confidence.value())
        self.settings.setValue("compare/run_id", self.run_id)
        self.settings.sync()

    # ---------------------------------------------------- trained models list
    def refresh_runs(self, select: str | None = None) -> None:
        """Fill the list with every training that has a model; keep (or choose) the selection."""
        wanted = select or self.run_id
        runs = self.history.usable_runs()
        self.run_combo.blockSignals(True)
        self.run_combo.clear()
        for run in runs:
            map50 = run.summary()["best_map50"]
            accuracy = f"   mAP50 {map50:.2f}" if map50 is not None else ""
            self.run_combo.addItem(f"{run.title}{accuracy}", run.run_id)
        index = self.run_combo.findData(wanted) if wanted else -1
        if index < 0 and runs:
            index = 0  # newest training
        if not runs:
            self.run_combo.addItem("No trained model yet")
        self.run_combo.setCurrentIndex(index)
        self.run_combo.setEnabled(bool(runs))
        self.run_combo.blockSignals(False)
        self._apply_selected_run(restart=False)

    def select_run(self, run_id: str) -> None:
        self.refresh_runs(select=run_id)
        if self._active:
            self._restart_thread()

    def _on_run_selected(self, _index: int) -> None:
        self._apply_selected_run(restart=False)
        if self.thread is not None:
            self.thread.set_models(self.base_weights, self.custom_weights)
        self._save_settings()

    def _apply_selected_run(self, restart: bool) -> None:
        run_id = self.run_combo.currentData() if self.run_combo.isEnabled() else None
        record = self.history.get(run_id) if run_id else None
        if record is not None and record.has_weights:
            self.run_id = record.run_id
            self.custom_weights = str(record.weights_path)
            # compare against the pretrained model THIS training started from
            self.base_weights = record.model or DEFAULT_WEIGHTS
            self.custom_panel.title_label.setText(f"Custom YOLO  -  {record.title}")
            self.custom_label.setText(
                f"Trained on: {', '.join(record.classes) or '-'}  ({record.train_images} train / "
                f"{record.val_images} val images, batch {record.batch})")
        else:
            self.run_id = ""
            self.custom_weights = None
            self.base_weights = self.settings.value("train/weights", DEFAULT_WEIGHTS) or DEFAULT_WEIGHTS
            self.custom_panel.title_label.setText(CUSTOM_TITLE)
            self.custom_label.setText("Train a model in Step 2 and it will appear here.")
        self.base_label.setText(f"Base: {self.base_weights}  (the model your training started from)")

    # --------------------------------------------------------------- thread
    def _restart_thread(self) -> None:
        """Stop the running thread (if any) and start a new one as soon as the old one is gone."""
        self._start_token += 1
        token = self._start_token
        self._detach_thread()

        def start() -> None:
            if token == self._start_token and self._active and self.thread is None:
                self._start_thread_now()

        call_when_idle(start)

    def _stop_thread(self) -> None:
        self._start_token += 1  # cancels a start that is still waiting
        self._detach_thread()

    def _detach_thread(self) -> None:
        thread, self.thread = self.thread, None
        if thread is None:
            return
        for signal in (thread.frames_ready, thread.status_changed):
            try:
                signal.disconnect()
            except TypeError:
                pass
        retire_thread(thread)  # never blocks and never deletes a running thread

    def _start_thread_now(self) -> None:
        self.thread = InferenceThread(
            camera_index=self.camera_spin.value(),
            source_type=self.source_type,
            file_path=self.selected_file,
            base_weights=self.base_weights,
            custom_weights=self.custom_weights,
            confidence=self.confidence.value() / 100.0,
        )
        self.thread.frames_ready.connect(self._update_frames)
        self.thread.status_changed.connect(self.status_changed)
        self.thread.start()
        self.pause_button.setText("Pause")
        self.status_changed.emit("Starting source...")

    def _update_frames(self, base_image, custom_image, fps: float) -> None:
        self.base_panel.image.set_frame(base_image)
        self.custom_panel.image.set_frame(custom_image)
        self.fps_changed.emit(fps)

    # ------------------------------------------------------------- controls
    def _confidence_changed(self, value: int) -> None:
        self._save_settings()
        if self.thread is not None:
            self.thread.set_confidence(value / 100.0)

    def _camera_index_changed(self, _value: int) -> None:
        self._save_settings()
        if self._active and self.source_type == "camera":
            self._restart_thread()

    def _use_webcam(self) -> None:
        self.source_type = "camera"
        if self._active:
            self._restart_thread()

    def _browse_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image or Video", "", FILE_FILTER)
        if not file_path:
            return
        self.selected_file = file_path
        self.file_label.setText(f"Selected: {Path(file_path).name}")
        self.upload_button.setEnabled(True)
        self.status_changed.emit("File selected. Click Upload to process it.")

    def _upload_selected_file(self) -> None:
        if not self.selected_file:
            self.status_changed.emit("No file selected. Click Browse first.")
            return
        self.source_type = "file"
        if self._active:
            self._restart_thread()

    def toggle_pause(self) -> None:
        if self.thread is None:
            return
        paused = self.pause_button.text() == "Pause"
        self.thread.set_paused(paused)
        self.pause_button.setText("Resume" if paused else "Pause")

    def _restart_source(self) -> None:
        if self.thread is not None:
            self.thread.restart_source()
            self.pause_button.setText("Pause")
            self.thread.set_paused(False)
            self.status_changed.emit("Restarting source...")
        elif self._active:
            self._restart_thread()
