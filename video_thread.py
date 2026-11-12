"""Camera helpers (fixed infrastructure)."""
from __future__ import annotations

import sys

import cv2
from PyQt6.QtCore import QThread, pyqtSignal

from config import CAPTURE_HEIGHT, CAPTURE_WIDTH, TARGET_FPS


def open_camera(index: int):
    """Open a webcam, or return None when it is not available."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    capture = cv2.VideoCapture(index, backend)
    if not capture.isOpened():
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    capture.set(cv2.CAP_PROP_FPS, TARGET_FPS)
    return capture


class CaptureThread(QThread):
    """Streams raw BGR webcam frames (used by the Dataset Builder to capture photos)."""

    frame_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(self, camera_index: int, parent=None):
        super().__init__(parent)
        self._camera_index = camera_index
        self._running = True

    def run(self) -> None:
        capture = open_camera(self._camera_index)
        if capture is None:
            self.status_changed.emit(f"Cannot open camera {self._camera_index}.")
            return
        self.status_changed.emit("Webcam ready. Press Space (or Capture) to take a photo.")
        while self._running:
            ok, frame = capture.read()
            if not ok:
                self.status_changed.emit("Camera read failed, retrying...")
                self.msleep(200)
                continue
            self.frame_ready.emit(frame)
            self.msleep(int(1000 / TARGET_FPS))
        capture.release()

    def request_stop(self) -> None:
        """Non-blocking: the thread ends by itself (see thread_utils.retire_thread)."""
        self._running = False
