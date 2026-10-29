"""Small shared GUI building blocks (fixed infrastructure)."""
from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def bgr_to_qimage(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, _ = rgb.shape
    return QImage(rgb.data, width, height, 3 * width, QImage.Format.Format_RGB888).copy()


class VideoLabel(QLabel):
    """Scalable image view that preserves the source aspect ratio."""

    def __init__(self):
        super().__init__()
        self._pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(260, 190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, image: QImage) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self._refresh_pixmap()

    def clear_frame(self) -> None:
        self._pixmap = QPixmap()
        self.clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self) -> None:
        if self._pixmap.isNull():
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class ValueSlider(QWidget):
    def __init__(self, title: str, minimum: int, maximum: int, value: int, parent=None):
        super().__init__(parent)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.title_label = QLabel(title)
        self.title_label.setMinimumWidth(120)
        self.value_label = QLabel(str(value))
        self.value_label.setObjectName("ValueChip")
        self.value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.value_label)
        self.slider.valueChanged.connect(self._on_value_changed)

    def _on_value_changed(self, value: int) -> None:
        self.value_label.setText(str(value))

    def value(self) -> int:
        return self.slider.value()

    def set_value(self, value: int) -> None:
        self.slider.setValue(value)


def create_video_card(title: str) -> QFrame:
    card = QFrame()
    card.setObjectName("VideoCard")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(10, 10, 10, 10)
    layout.setSpacing(8)
    title_label = QLabel(title)
    title_label.setObjectName("PanelTitle")
    layout.addWidget(title_label)
    canvas = QFrame()
    canvas.setObjectName("VideoCanvas")
    canvas_layout = QVBoxLayout(canvas)
    canvas_layout.setContentsMargins(0, 0, 0, 0)
    image = VideoLabel()
    image.setText("Waiting for source...")
    canvas_layout.addWidget(image)
    layout.addWidget(canvas, 1)
    card.image = image
    card.title_label = title_label
    return card


def create_sidebar() -> tuple[QScrollArea, QVBoxLayout]:
    """Scrollable sidebar container; returns (scroll area, layout to add group boxes to)."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.setContentsMargins(4, 0, 4, 0)
    layout.setSpacing(10)
    scroll.setWidget(content)
    return scroll, layout
