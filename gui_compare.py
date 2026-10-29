"""Step 3 - Before / After page (fixed infrastructure)."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
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
    CUSTOM_MODEL_PATH,
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CONFIDENCE,
    DEFAULT_WEIGHTS,
    FILE_FILTER,
)
from gui_common import ValueSlider, create_sidebar, create_video_card
from inference_thread import InferenceThread


class ComparePage(QWidget):
    status_changed = pyqtSignal(str)
    fps_changed = pyqtSignal(float)

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.thread: InferenceThread | None = None
        self.source_type = "camera"
        self.selected_file = ""
        self.base_weights = DEFAULT_WEIGHTS
        self.custom_weights: str | None = str(CUSTOM_MODEL_PATH) if CUSTOM_MODEL_PATH.exists() else None
        self._active = False
        self._build_ui()
        self._restore_settings()

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
        self.base_panel = create_video_card("Base YOLO  (pretrained, 80 COCO classes)")
        self.custom_panel = create_video_card("Custom YOLO  (after your fine-tuning)")
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
        self.base_label = QLabel("")
        self.custom_label = QLabel("")
        self.base_label.setWordWrap(True)
        self.custom_label.setWordWrap(True)
        model_layout.addWidget(self.base_label)
        model_layout.addWidget(self.custom_label)
        reload_button = QPushButton("Reload Custom Model")
        reload_button.clicked.connect(self._reload_custom)
        model_layout.addWidget(reload_button)
        layout.addWidget(model_box)

        note = QLabel("Try your custom objects in front of the webcam: the Base model cannot name them, "
                      "the Custom model can. Ids 0-79 are COCO, ids 80+ are your classes.")
        note.setObjectName("Muted")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        splitter.addWidget(scroll)
        splitter.setSizes([1100, 380])
        root.addWidget(splitter)
        QShortcut(QKeySequence("Space"), self, activated=self.toggle_pause)
        self._update_model_labels()

    # ------------------------------------------------------------ lifecycle
    def activate(self) -> None:
        self._active = True
        self._update_model_labels()
        self._start_thread()

    def deactivate(self) -> None:
        self._active = False
        self._stop_thread()

    def apply_theme(self, dark: bool) -> None:
        pass

    def shutdown(self) -> None:
        self._stop_thread()
        self._save_settings()

    def set_custom_weights(self, path: str | None) -> None:
        self.custom_weights = path
        self._update_model_labels()
        if self.thread is not None:
            self.thread.set_custom_weights(path)

    def reset_defaults(self) -> None:
        self.camera_spin.setValue(DEFAULT_CAMERA_INDEX)
        self.confidence.set_value(DEFAULT_CONFIDENCE)
        self._save_settings()

    # ------------------------------------------------------------- settings
    def _restore_settings(self) -> None:
        self.camera_spin.setValue(self.settings.value("compare/camera_index", DEFAULT_CAMERA_INDEX, int))
        self.confidence.set_value(int(self.settings.value("compare/confidence", DEFAULT_CONFIDENCE)))
        self.base_weights = self.settings.value("train/weights", DEFAULT_WEIGHTS)
        self._update_model_labels()

    def _save_settings(self) -> None:
        self.settings.setValue("compare/camera_index", self.camera_spin.value())
        self.settings.setValue("compare/confidence", self.confidence.value())
        self.settings.sync()

    def _update_model_labels(self) -> None:
        self.base_weights = self.settings.value("train/weights", DEFAULT_WEIGHTS) or DEFAULT_WEIGHTS
        self.base_label.setText(f"Base: {self.base_weights}")
        if self.custom_weights and Path(self.custom_weights).exists():
            self.custom_label.setText(f"Custom: {Path(self.custom_weights).name}")
        else:
            self.custom_label.setText("Custom: not trained yet")

    # --------------------------------------------------------------- thread
    def _start_thread(self) -> None:
        self._stop_thread()
        self.thread = InferenceThread(
            camera_index=self.camera_spin.value(),
            source_type=self.source_type,
            file_path=self.selected_file,
            base_weights=self.base_weights,
            custom_weights=self.custom_weights,
            confidence=self.confidence.value() / 100.0,
            parent=self,
        )
        self.thread.frames_ready.connect(self._update_frames)
        self.thread.status_changed.connect(self.status_changed)
        self.thread.start()
        self.pause_button.setText("Pause")
        self.status_changed.emit("Starting source...")

    def _stop_thread(self) -> None:
        if self.thread is not None:
            self.thread.stop()
            self.thread.deleteLater()
            self.thread = None

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
            self._start_thread()

    def _use_webcam(self) -> None:
        self.source_type = "camera"
        if self._active:
            self._start_thread()

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
            self._start_thread()

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
            self._start_thread()

    def _reload_custom(self) -> None:
        path = str(CUSTOM_MODEL_PATH) if CUSTOM_MODEL_PATH.exists() else None
        self.set_custom_weights(path)
        self.status_changed.emit("Reloading custom model..." if path else "No custom model found. Train first.")
