"""Before / After inference thread (fixed infrastructure).

Runs the original (Base) YOLO and the fine-tuned (Custom) YOLO on the same frames.
The app keeps working when ultralytics, the weights, or the custom model are missing:
the panels then show the raw frame together with a short explanation.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from config import INFER_IMG_SIZE, VIDEO_EXTENSIONS
from gui_common import bgr_to_qimage
from model_utils import get_device, resolve_weights
from video_thread import open_camera

_MODEL_CACHE: dict[tuple[str, float], object] = {}


def load_model(path: str):
    from ultralytics import YOLO

    file = Path(path)
    key = (str(path), file.stat().st_mtime if file.exists() else 0.0)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = YOLO(str(path))
    return _MODEL_CACHE[key]


def draw_message(frame: np.ndarray, message: str) -> np.ndarray:
    output = frame.copy()
    height = output.shape[0]
    cv2.rectangle(output, (0, height - 34), (output.shape[1], height), (20, 20, 20), -1)
    cv2.putText(output, message, (10, height - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (240, 240, 240), 1, cv2.LINE_AA)
    return output


class InferenceThread(QThread):
    frames_ready = pyqtSignal(object, object, float)  # base QImage, custom QImage, fps
    status_changed = pyqtSignal(str)

    def __init__(self, camera_index: int, source_type: str, file_path: str,
                 base_weights: str, custom_weights: str | None, confidence: float, parent=None):
        super().__init__(parent)
        self._camera_index = camera_index
        self._source_type = source_type
        self._file_path = file_path
        self._base_weights = base_weights
        self._custom_weights = custom_weights
        self._confidence = confidence
        self._running = True
        self._paused = False
        self._dirty = True
        self._restart = False
        self._reload_models = False
        self._base = None
        self._custom = None
        self._base_error = ""
        self._device = "cpu"

    # ---- controls (called from the GUI thread) --------------------------- #
    def set_confidence(self, value: float) -> None:
        self._confidence = value
        self._dirty = True

    def set_paused(self, paused: bool) -> None:
        self._paused = paused

    def restart_source(self) -> None:
        self._restart = True

    def set_custom_weights(self, path: str | None) -> None:
        self._custom_weights = path
        self._reload_models = True

    def set_models(self, base_weights: str, custom_weights: str | None) -> None:
        self._base_weights = base_weights
        self._custom_weights = custom_weights
        self._reload_models = True

    def request_stop(self) -> None:
        """Non-blocking: the thread ends by itself (see thread_utils.retire_thread)."""
        self._running = False

    # ---- models ---------------------------------------------------------- #
    def _load_models(self) -> None:
        self._base, self._custom, self._base_error = None, None, ""
        self.status_changed.emit("Loading models...")
        try:
            self._device = get_device()
            self._base = load_model(resolve_weights(self._base_weights))
        except ModuleNotFoundError:
            self._base_error = "ultralytics is not installed (pip install -r requirements.txt)"
        except Exception as error:  # noqa: BLE001
            self._base_error = f"cannot load {self._base_weights}: {error}"
        if self._custom_weights and Path(self._custom_weights).exists() and not self._base_error:
            try:
                self._custom = load_model(self._custom_weights)
            except Exception as error:  # noqa: BLE001
                self.status_changed.emit(f"Could not load the custom model: {error}")
        if self._base_error:
            self.status_changed.emit(f"Base model unavailable: {self._base_error}")
        else:
            custom = "ready" if self._custom is not None else "not trained yet"
            self.status_changed.emit(f"Base model ready. Custom model: {custom}.")

    def _detect(self, model, frame: np.ndarray) -> np.ndarray:
        result = model.predict(
            frame, imgsz=INFER_IMG_SIZE, conf=self._confidence,
            device=self._device, verbose=False,
        )[0]
        return result.plot()

    def _render(self, frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._base_error:
            message = "Base model unavailable - see status bar"
            return draw_message(frame, message), draw_message(frame, message)
        base = self._detect(self._base, frame)
        if self._custom is None:
            custom = draw_message(frame, "No custom model yet - train one in Step 2")
        else:
            custom = self._detect(self._custom, frame)
        return base, custom

    # ---- sources --------------------------------------------------------- #
    def _open_source(self):
        """Return (capture or None, static image or None)."""
        if self._source_type == "camera":
            capture = open_camera(self._camera_index)
            if capture is None:
                self.status_changed.emit(f"Cannot open camera {self._camera_index}.")
            return capture, None
        path = Path(self._file_path)
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            capture = cv2.VideoCapture(str(path))
            if not capture.isOpened():
                self.status_changed.emit(f"Cannot open video {path.name}.")
                return None, None
            return capture, None
        try:
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:  # noqa: BLE001
            image = None
        if image is None:
            self.status_changed.emit(f"Cannot read image {path.name}.")
        return None, image

    # ---- main loop ------------------------------------------------------- #
    def run(self) -> None:
        self._load_models()
        capture, static = self._open_source()
        last_frame = static
        fps, previous = 0.0, time.perf_counter()

        while self._running:
            if self._reload_models:
                self._reload_models = False
                self._load_models()
                self._dirty = True
            if self._restart:
                self._restart = False
                if capture is not None:
                    capture.release()
                capture, static = self._open_source()
                last_frame = static
                self._dirty = True

            live = capture is not None and not self._paused
            if live:
                ok, frame = capture.read()
                if not ok:
                    if self._source_type == "file":
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop videos
                    else:
                        self.status_changed.emit("Camera read failed, retrying...")
                        self.msleep(200)
                    continue
                last_frame = frame
            elif not self._dirty or last_frame is None:
                self.msleep(30)
                continue
            self._dirty = False

            try:
                base, custom = self._render(last_frame)
            except Exception as error:  # noqa: BLE001
                self.status_changed.emit(f"Inference error: {error}")
                self.msleep(300)
                continue

            now = time.perf_counter()
            instant = 1.0 / max(now - previous, 1e-6)
            fps = instant if fps == 0.0 else 0.85 * fps + 0.15 * instant
            previous = now
            self.frames_ready.emit(bgr_to_qimage(base), bgr_to_qimage(custom), fps)

        if capture is not None:
            capture.release()
