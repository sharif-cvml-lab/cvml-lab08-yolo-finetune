"""Step 1 - Dataset Builder page (fixed infrastructure)."""
from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings, Qt, QTimer, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from annotation_canvas import AnnotationCanvas, class_color
from config import DEFAULT_CAMERA_INDEX, IMAGE_FILTER
from dataset import DatasetError, DatasetStore, ImageItem
from gui_common import VideoLabel, bgr_to_qimage, create_sidebar
from thread_utils import call_when_idle, retire_thread
from video_thread import CaptureThread


class DatasetPage(QWidget):
    status_changed = pyqtSignal(str)
    dataset_changed = pyqtSignal()

    def __init__(self, store: DatasetStore, settings: QSettings, parent=None):
        super().__init__(parent)
        self.store = store
        self.settings = settings
        self.items: list[ImageItem] = []
        self.current: ImageItem | None = None
        self._dirty = False
        self._loading = False
        self._capture_thread: CaptureThread | None = None
        self._last_frame = None
        self._captured = 0
        self._checked: set[Path] = set()  # images ticked for bulk deletion
        self._locked = False

        self._build_ui()
        self._build_shortcuts()
        self._restore_settings()
        self._reload_classes()
        self._reload_items()
        self._open_row(0)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        card = QFrame()
        card.setObjectName("VideoCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(10, 10, 10, 10)
        card_layout.setSpacing(8)
        head = QHBoxLayout()
        self.canvas_title = QLabel("Annotation Canvas")
        self.canvas_title.setObjectName("PanelTitle")
        hint = QLabel("Drag = new box   |   Click = select   |   Right-click = delete")
        hint.setObjectName("Muted")
        head.addWidget(self.canvas_title)
        head.addStretch(1)
        head.addWidget(hint)
        card_layout.addLayout(head)

        self.canvas = AnnotationCanvas()
        self.canvas.boxes_changed.connect(self._on_boxes_changed)
        self.canvas.blocked.connect(self.status_changed)
        self.live_label = VideoLabel()
        self.live_label.setText("Starting webcam...")
        self.stack = QStackedWidget()
        self.stack.addWidget(self.canvas)
        self.stack.addWidget(self.live_label)
        frame = QFrame()
        frame.setObjectName("VideoCanvas")
        frame_layout = QVBoxLayout(frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.addWidget(self.stack)
        card_layout.addWidget(frame, 1)

        toolbar = QHBoxLayout()
        self.prev_button = QPushButton("◀ Prev (A)")
        self.next_button = QPushButton("Next (D) ▶")
        self.undo_button = QPushButton("Undo (Ctrl+Z)")
        self.clear_button = QPushButton("Clear Boxes")
        self.save_button = QPushButton("Save (Ctrl+S)")
        self.save_button.setObjectName("accent")
        self.delete_button = QPushButton("Delete Image")
        self.delete_button.setObjectName("danger")
        self.prev_button.clicked.connect(lambda: self._navigate(-1))
        self.next_button.clicked.connect(lambda: self._navigate(1))
        self.undo_button.clicked.connect(self.canvas.undo)
        self.clear_button.clicked.connect(self.canvas.clear_boxes)
        self.save_button.clicked.connect(self._save_current)
        self.delete_button.clicked.connect(self._delete_current)
        for button in (self.prev_button, self.next_button, self.undo_button,
                       self.clear_button, self.save_button, self.delete_button):
            toolbar.addWidget(button)
        card_layout.addLayout(toolbar)
        splitter.addWidget(card)
        splitter.addWidget(self._build_sidebar())
        splitter.setSizes([1000, 420])
        root.addWidget(splitter)

    def _build_sidebar(self):
        scroll, layout = create_sidebar()

        images_box = QGroupBox("Images")
        images_layout = QVBoxLayout(images_box)
        row = QHBoxLayout()
        self.add_button = QPushButton("Add Images...")
        self.add_button.clicked.connect(self._add_images)
        self.webcam_button = QPushButton("Webcam")
        self.webcam_button.clicked.connect(self._toggle_webcam)
        self.capture_button = QPushButton("Capture (Space)")
        self.capture_button.setObjectName("accent")
        self.capture_button.setEnabled(False)
        self.capture_button.clicked.connect(self._capture)
        row.addWidget(self.add_button)
        row.addWidget(self.webcam_button)
        images_layout.addLayout(row)
        images_layout.addWidget(self.capture_button)
        camera_row = QHBoxLayout()
        camera_row.addWidget(QLabel("Camera index"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 9)
        self.camera_spin.valueChanged.connect(self._camera_index_changed)
        camera_row.addWidget(self.camera_spin)
        images_layout.addLayout(camera_row)
        self.image_list = QListWidget()
        self.image_list.setMinimumHeight(190)
        self.image_list.currentRowChanged.connect(self._on_row_changed)
        self.image_list.itemChanged.connect(self._on_item_checked)
        images_layout.addWidget(self.image_list)
        bulk_row = QHBoxLayout()
        self.check_all_button = QPushButton("Tick All")
        self.check_all_button.clicked.connect(lambda: self._set_all_checked(True))
        self.uncheck_all_button = QPushButton("Untick All")
        self.uncheck_all_button.clicked.connect(lambda: self._set_all_checked(False))
        self.delete_checked_button = QPushButton("Delete Ticked")
        self.delete_checked_button.setObjectName("danger")
        self.delete_checked_button.clicked.connect(self._delete_checked)
        for button in (self.check_all_button, self.uncheck_all_button, self.delete_checked_button):
            bulk_row.addWidget(button)
        images_layout.addLayout(bulk_row)
        self.list_info = QLabel("")
        self.list_info.setObjectName("Muted")
        self.list_info.setWordWrap(True)
        images_layout.addWidget(self.list_info)
        layout.addWidget(images_box)

        classes_box = QGroupBox("Classes (ids start at 80)")
        classes_layout = QVBoxLayout(classes_box)
        self.class_list = QListWidget()
        self.class_list.setMaximumHeight(130)
        self.class_list.currentRowChanged.connect(self._on_class_clicked)
        classes_layout.addWidget(self.class_list)
        add_row = QHBoxLayout()
        self.class_edit = QLineEdit()
        self.class_edit.setPlaceholderText("new class name, e.g. my_toy")
        self.class_edit.returnPressed.connect(self._add_class)
        add_class_button = QPushButton("Add")
        add_class_button.clicked.connect(self._add_class)
        add_row.addWidget(self.class_edit, 1)
        add_row.addWidget(add_class_button)
        classes_layout.addLayout(add_row)
        self.delete_class_button = QPushButton("Delete Selected Class")
        self.delete_class_button.setObjectName("danger")
        self.delete_class_button.setToolTip("Removes the class and every box that uses it.")
        self.delete_class_button.clicked.connect(self._delete_class)
        classes_layout.addWidget(self.delete_class_button)
        tip = QLabel("Press 1-9 to choose the class. Select a box, then pick a class to relabel it.\n"
                     "Label ONLY your own object: do not draw boxes around other things in the photo "
                     "(person, cup, ...), even if you can see them.")
        tip.setObjectName("Muted")
        tip.setWordWrap(True)
        classes_layout.addWidget(tip)
        layout.addWidget(classes_box)

        stats_box = QGroupBox("Dataset")
        stats_layout = QVBoxLayout(stats_box)
        self.stats_label = QLabel("")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        open_button = QPushButton("Open Dataset Folder")
        open_button.clicked.connect(
            lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.store.root)))
        )
        stats_layout.addWidget(open_button)
        layout.addWidget(stats_box)
        layout.addStretch(1)
        return scroll

    def _build_shortcuts(self) -> None:
        def add(key: str, slot) -> None:
            QShortcut(QKeySequence(key), self, activated=slot)

        add("A", lambda: self._navigate(-1))
        add("Left", lambda: self._navigate(-1))
        add("D", lambda: self._navigate(1))
        add("Right", lambda: self._navigate(1))
        add("Ctrl+S", self._save_current)
        add("Ctrl+Z", self.canvas.undo)
        add("Delete", self.canvas.delete_selected)
        add("Backspace", self.canvas.delete_selected)
        add("Space", self._capture)
        for number in range(1, 10):
            add(str(number), lambda n=number: self._select_class(n - 1))

    # ------------------------------------------------------------ lifecycle
    def activate(self) -> None:
        self._reload_classes()
        self._reload_items(select=self.current.path if self.current else None)
        self._open_row(self.image_list.currentRow())

    def deactivate(self) -> None:
        self._stop_webcam()
        self._commit_current()
        self._update_stats()

    def apply_theme(self, dark: bool) -> None:
        self.canvas.set_dark(dark)

    def set_locked(self, locked: bool) -> None:
        """Read-only while a training runs (the trainer reads the dataset files)."""
        self._locked = locked
        self.setEnabled(not locked)

    def reset_defaults(self) -> None:
        self.camera_spin.setValue(DEFAULT_CAMERA_INDEX)

    def shutdown(self) -> None:
        self._stop_webcam()
        self._commit_current()
        self._save_settings()

    # ------------------------------------------------------------- settings
    def _restore_settings(self) -> None:
        self.camera_spin.setValue(self.settings.value("dataset/camera_index", DEFAULT_CAMERA_INDEX, int))
        self._active_class = self.settings.value("dataset/active_class", 0, int)

    def _save_settings(self) -> None:
        self.settings.setValue("dataset/camera_index", self.camera_spin.value())
        self.settings.setValue("dataset/active_class", self.canvas.active_class())
        self.settings.sync()

    # --------------------------------------------------------------- classes
    def _reload_classes(self) -> None:
        names = self.store.classes()
        self.canvas.set_classes(names)
        active = min(getattr(self, "_active_class", 0), max(0, len(names) - 1))
        self.class_list.blockSignals(True)
        self.class_list.clear()
        for index, name in enumerate(names):
            item = QListWidgetItem(f"{index + 1}  ·  {name}   (id {80 + index})")
            item.setForeground(class_color(index))
            self.class_list.addItem(item)
        if names:
            self.class_list.setCurrentRow(active)
            self.canvas.set_active_class(active)
        self.class_list.blockSignals(False)

    def _on_class_clicked(self, row: int) -> None:
        if row >= 0:
            self._select_class(row)

    def _select_class(self, index: int) -> None:
        if not 0 <= index < len(self.store.classes()):
            return
        self._active_class = index
        self.class_list.blockSignals(True)
        self.class_list.setCurrentRow(index)
        self.class_list.blockSignals(False)
        self.canvas.set_active_class(index, retag_selected=True)
        self.status_changed.emit(f"Active class: {self.store.classes()[index]}")

    def _add_class(self) -> None:
        try:
            names = self.store.add_class(self.class_edit.text())
        except DatasetError as error:
            self.status_changed.emit(str(error))
            return
        self.class_edit.clear()
        self._active_class = len(names) - 1
        self._reload_classes()
        self._update_stats()
        self.dataset_changed.emit()
        self.status_changed.emit(f"Added class '{names[-1]}' (id {80 + len(names) - 1}).")

    def _delete_class(self) -> None:
        names = self.store.classes()
        row = self.class_list.currentRow()
        if not 0 <= row < len(names):
            self.status_changed.emit("Select the class you want to delete in the Classes list first.")
            return
        self._commit_current()
        boxes, images = self.store.class_usage(row)
        text = f"Delete the class '{names[row]}'?"
        if boxes:
            text += (f"\n\nIts {boxes} box(es) in {images} image(s) are deleted too. Images left without any box "
                     "go back to the 'not annotated' list. The ids of the classes after it move up by one.")
        if QMessageBox.question(self, "Delete class", text) != QMessageBox.StandardButton.Yes:
            return
        try:
            removed, unannotated = self.store.remove_class(row)
        except DatasetError as error:
            self.status_changed.emit(str(error))
            return
        remaining = len(names) - 1
        if row < self._active_class:
            self._active_class -= 1
        self._active_class = max(0, min(self._active_class, remaining - 1))
        self.current = None
        self._dirty = False
        self._reload_classes()
        self._goto(None)
        self.dataset_changed.emit()
        message = f"Deleted class '{names[row]}'."
        if removed:
            message += f" {removed} box(es) removed"
            message += f", {unannotated} image(s) are not annotated any more." if unannotated else "."
        self.status_changed.emit(message)

    # ---------------------------------------------------------------- items
    def _reload_items(self, select: Path | None = None) -> None:
        previous_row = self.image_list.currentRow()
        self.items = self.store.list_items()
        self._checked &= {item.path for item in self.items}  # forget images that no longer exist
        self._loading = True
        self.image_list.clear()
        target = -1
        for row, item in enumerate(self.items):
            count = self.store.count_boxes(item) if item.annotated else 0
            mark = "✓" if item.annotated else "○"
            suffix = f"   [{count} box{'es' if count != 1 else ''}]" if count else "   [not annotated]"
            entry = QListWidgetItem(f"{mark}  {item.name}{suffix}")
            entry.setFlags(entry.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            entry.setCheckState(Qt.CheckState.Checked if item.path in self._checked else Qt.CheckState.Unchecked)
            self.image_list.addItem(entry)
            if select is not None and item.path == select:
                target = row
        if target < 0 and self.items:
            target = min(max(previous_row, 0), len(self.items) - 1)
        if target >= 0:
            self.image_list.setCurrentRow(target)
        self._loading = False
        self._update_stats()

    def _open_row(self, row: int) -> None:
        if not self.items or row < 0 or row >= len(self.items):
            self.current = None
            self._dirty = False
            self.canvas.clear("No images yet.\nUse 'Add Images...' or the Webcam to create some.")
            self.canvas_title.setText("Annotation Canvas")
            return
        item = self.items[row]
        self.current = item
        if self.canvas.set_image(item.path):
            self.canvas.set_boxes(self.store.load_boxes(item))
        self._dirty = False
        state = "annotated" if item.annotated else "not annotated"
        self.canvas_title.setText(f"Annotation Canvas  -  {row + 1} / {len(self.items)}  -  {item.name}  ({state})")

    def _commit_current(self) -> bool:
        """Save the boxes of the current image when they changed. Returns True if saved."""
        if self.current is None or not self._dirty:
            return False
        boxes = self.canvas.boxes()
        before = self.current.location
        old_path = self.current.path
        self.current = self.store.save_boxes(self.current, boxes)
        if old_path in self._checked and old_path != self.current.path:
            self._checked.discard(old_path)  # the image moved (inbox <-> train / val): keep it ticked
            self._checked.add(self.current.path)
        self._dirty = False
        if boxes:
            self.status_changed.emit(f"Saved {len(boxes)} box(es) to {self.current.location}/{self.current.name}.")
        elif before != "inbox":
            self.status_changed.emit("All boxes removed: the image moved back to the unannotated list.")
        self.dataset_changed.emit()
        return True

    def _goto(self, path: Path | None) -> None:
        self._commit_current()
        self._reload_items(select=path)
        self._open_row(self.image_list.currentRow())

    def _on_row_changed(self, row: int) -> None:
        if self._loading or row < 0 or row >= len(self.items):
            return
        target = self.items[row].path
        if self.current is not None and target == self.current.path:
            return
        QTimer.singleShot(0, lambda path=target: self._goto(path))

    def _navigate(self, delta: int) -> None:
        if not self.items or self._capture_thread is not None:
            return
        paths = [item.path for item in self.items]
        index = paths.index(self.current.path) if self.current and self.current.path in paths else 0
        index = min(max(index + delta, 0), len(paths) - 1)
        self._goto(paths[index])

    def _on_boxes_changed(self) -> None:
        self._dirty = True
        self.status_changed.emit(f"{len(self.canvas.boxes())} box(es) - unsaved (Ctrl+S or move to another image).")

    def _save_current(self) -> None:
        if self.current is None:
            return
        self._commit_current()
        self._goto(self.current.path)

    def _delete_current(self) -> None:
        if self.current is None:
            return
        answer = QMessageBox.question(
            self, "Delete image", f"Delete {self.current.name} and its annotations?"
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        paths = [item.path for item in self.items]
        index = paths.index(self.current.path)
        neighbour = paths[index + 1] if index + 1 < len(paths) else (paths[index - 1] if index > 0 else None)
        self.store.delete_item(self.current)
        self.current = None
        self._dirty = False
        self._goto(neighbour)
        self.dataset_changed.emit()
        self.status_changed.emit("Image deleted.")

    def _update_stats(self) -> None:
        stats = self.store.stats()
        names = self.store.classes()
        lines = [f"Train: {stats.train_images}   Val: {stats.val_images}   Not annotated: {stats.inbox_images}"]
        if names:
            lines.append("   ".join(f"{n}: {c}" for n, c in zip(names, stats.instances)))
        else:
            lines.append("No custom classes yet.")
        self.stats_label.setText("\n".join(lines))
        self.list_info.setText(f"{len(self.items)} image(s). ✓ = saved in the dataset, ○ = waiting for boxes. "
                               "Tick images to delete many at once.")
        self.delete_checked_button.setText(f"Delete Ticked ({len(self._checked)})" if self._checked else "Delete Ticked")

    def _on_item_checked(self, entry: QListWidgetItem) -> None:
        if self._loading:
            return
        row = self.image_list.row(entry)
        if not 0 <= row < len(self.items):
            return
        path = self.items[row].path
        if entry.checkState() == Qt.CheckState.Checked:
            self._checked.add(path)
        else:
            self._checked.discard(path)
        self._update_stats()

    def _set_all_checked(self, checked: bool) -> None:
        self._checked = {item.path for item in self.items} if checked else set()
        self._loading = True
        for row in range(self.image_list.count()):
            self.image_list.item(row).setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
        self._loading = False
        self._update_stats()

    def _delete_checked(self) -> None:
        if not self._checked:
            self.status_changed.emit("Tick the images you want to delete first (checkbox in the list).")
            return
        count = len(self._checked)
        answer = QMessageBox.question(
            self, "Delete images", f"Delete {count} ticked image(s) and their annotations?")
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._stop_webcam()
        if self.current is not None and self.current.path in self._checked:
            self._dirty = False  # it is going to be deleted anyway: do not save it first
        else:
            self._commit_current()
        targets = [item for item in self.store.list_items() if item.path in self._checked]
        deleted = self.store.delete_items(targets)
        self._checked.clear()
        keep = self.current.path if self.current is not None and self.current.path.exists() else None
        self.current = None
        self._dirty = False
        self._goto(keep)
        self.dataset_changed.emit()
        self.status_changed.emit(f"Deleted {deleted} image(s).")

    # --------------------------------------------------------------- import
    def _add_images(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Select Images", "", IMAGE_FILTER)
        if not paths:
            return
        self._stop_webcam()
        self._commit_current()
        added, failed = self.store.import_images(paths)
        self._goto(added[0].path if added else None)
        message = f"Imported {len(added)} image(s)."
        if failed:
            message += f" {failed} file(s) could not be read."
        self.status_changed.emit(message + " Draw boxes, then move to the next image.")

    # --------------------------------------------------------------- webcam
    def _toggle_webcam(self) -> None:
        if self._capture_thread is None:
            self._start_webcam()
        else:
            self._stop_webcam()

    def _start_webcam(self) -> None:
        self._commit_current()
        self._captured = 0
        self._last_frame = None
        thread = CaptureThread(self.camera_spin.value())
        self._capture_thread = thread
        thread.frame_ready.connect(self._on_live_frame)
        thread.status_changed.connect(self.status_changed)
        # start only when no old camera thread is still shutting down (it may hold the device)
        call_when_idle(lambda: thread.start() if self._capture_thread is thread else None)
        self.live_label.clear_frame()
        self.live_label.setText("Starting webcam...")
        self.stack.setCurrentWidget(self.live_label)
        self.webcam_button.setText("Stop Webcam")
        self.capture_button.setEnabled(True)
        self.canvas_title.setText("Live Webcam  -  move the object around and capture many angles")

    def _stop_webcam(self) -> None:
        thread, self._capture_thread = self._capture_thread, None
        if thread is None:
            return
        for signal in (thread.frame_ready, thread.status_changed):
            try:
                signal.disconnect()
            except TypeError:
                pass
        retire_thread(thread)  # never blocks and never deletes a running thread
        self.stack.setCurrentWidget(self.canvas)
        self.webcam_button.setText("Webcam")
        self.capture_button.setEnabled(False)
        if self._captured:
            self._goto(self._first_unannotated_path())
            self.status_changed.emit(f"Captured {self._captured} image(s). Now draw boxes on them.")
        else:
            self._open_row(self.image_list.currentRow())

    def _first_unannotated_path(self) -> Path | None:
        for item in self.store.list_items():
            if not item.annotated:
                return item.path
        return self.current.path if self.current else None

    def _on_live_frame(self, frame) -> None:
        self._last_frame = frame
        self.live_label.set_frame(bgr_to_qimage(frame))

    def _capture(self) -> None:
        if self._capture_thread is None or self._last_frame is None:
            return
        self.store.add_frame(self._last_frame)
        self._captured += 1
        self.status_changed.emit(f"Captured {self._captured} image(s) this session.")

    def _camera_index_changed(self, _value: int) -> None:
        self._save_settings()
        if self._capture_thread is not None:
            self._stop_webcam()
            self._start_webcam()
