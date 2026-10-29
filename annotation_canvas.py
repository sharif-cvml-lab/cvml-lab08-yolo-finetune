"""Image canvas for drawing bounding boxes (fixed infrastructure).

Mouse:  left-drag = new box | left-click = select box | right-click = delete box
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import QSizePolicy, QWidget

from dataset import Box

PALETTE = [
    "#4cc9f0", "#f72585", "#b5e48c", "#ffb703", "#9d4edd",
    "#06d6a0", "#ef476f", "#f78c6b", "#118ab2", "#e9ff70",
]


def class_color(index: int) -> QColor:
    return QColor(PALETTE[index % len(PALETTE)])


class AnnotationCanvas(QWidget):
    boxes_changed = pyqtSignal()
    blocked = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self._pixmap = QPixmap()
        self._boxes: list[Box] = []
        self._classes: list[str] = []
        self._active = 0
        self._selected = -1
        self._press: QPointF | None = None
        self._cursor: QPointF | None = None
        self._dark = True
        self._placeholder = "Add images (or capture from the webcam) to start annotating."
        self.setMouseTracking(True)
        self.setMinimumSize(360, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ---- public API ------------------------------------------------------ #
    def set_dark(self, dark: bool) -> None:
        self._dark = dark
        self.update()

    def set_image(self, path: Path | str) -> bool:
        pixmap = QPixmap(str(path))
        self._selected = -1
        self._press = None
        if pixmap.isNull():
            self.clear("Could not load this image.")
            return False
        self._pixmap = pixmap
        self.update()
        return True

    def clear(self, placeholder: str | None = None) -> None:
        self._pixmap = QPixmap()
        self._boxes = []
        self._selected = -1
        if placeholder is not None:
            self._placeholder = placeholder
        self.update()

    def set_classes(self, names: list[str]) -> None:
        self._classes = list(names)
        self._active = min(self._active, max(0, len(names) - 1))
        self.update()

    def set_active_class(self, index: int, retag_selected: bool = False) -> None:
        if not 0 <= index < len(self._classes):
            return
        self._active = index
        if retag_selected and 0 <= self._selected < len(self._boxes):
            self._boxes[self._selected] = replace(self._boxes[self._selected], cls=index)
            self.boxes_changed.emit()
        self.update()

    def active_class(self) -> int:
        return self._active

    def boxes(self) -> list[Box]:
        return list(self._boxes)

    def set_boxes(self, boxes: list[Box]) -> None:
        self._boxes = list(boxes)
        self._selected = -1
        self.update()

    def undo(self) -> None:
        if self._boxes:
            self._boxes.pop()
            self._selected = -1
            self.boxes_changed.emit()
            self.update()

    def delete_selected(self) -> None:
        if 0 <= self._selected < len(self._boxes):
            del self._boxes[self._selected]
            self._selected = -1
            self.boxes_changed.emit()
            self.update()

    def clear_boxes(self) -> None:
        if self._boxes:
            self._boxes = []
            self._selected = -1
            self.boxes_changed.emit()
            self.update()

    # ---- geometry -------------------------------------------------------- #
    def _image_rect(self) -> QRectF:
        if self._pixmap.isNull():
            return QRectF()
        width, height = self._pixmap.width(), self._pixmap.height()
        scale = min(self.width() / width, self.height() / height)
        draw_w, draw_h = width * scale, height * scale
        return QRectF((self.width() - draw_w) / 2, (self.height() - draw_h) / 2, draw_w, draw_h)

    @staticmethod
    def _box_rect(box: Box, image_rect: QRectF) -> QRectF:
        return QRectF(
            image_rect.left() + box.x1 * image_rect.width(),
            image_rect.top() + box.y1 * image_rect.height(),
            box.width * image_rect.width(),
            box.height * image_rect.height(),
        )

    @staticmethod
    def _clamp(point: QPointF, rect: QRectF) -> QPointF:
        return QPointF(
            min(max(point.x(), rect.left()), rect.right()),
            min(max(point.y(), rect.top()), rect.bottom()),
        )

    @staticmethod
    def _to_norm(point: QPointF, rect: QRectF) -> tuple[float, float]:
        return (
            (point.x() - rect.left()) / rect.width(),
            (point.y() - rect.top()) / rect.height(),
        )

    def _box_at(self, point: QPointF) -> int:
        rect = self._image_rect()
        for index in range(len(self._boxes) - 1, -1, -1):
            if self._box_rect(self._boxes[index], rect).contains(point):
                return index
        return -1

    # ---- mouse ----------------------------------------------------------- #
    def mousePressEvent(self, event) -> None:
        rect = self._image_rect()
        position = event.position()
        if rect.isEmpty() or not rect.contains(position):
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = position
            self._cursor = position
        elif event.button() == Qt.MouseButton.RightButton:
            index = self._box_at(position)
            if index >= 0:
                del self._boxes[index]
                self._selected = -1
                self.boxes_changed.emit()
                self.update()

    def mouseMoveEvent(self, event) -> None:
        self._cursor = event.position()
        self.update()

    def leaveEvent(self, event) -> None:
        self._cursor = None
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton or self._press is None:
            return
        rect = self._image_rect()
        start, self._press = self._press, None
        if rect.isEmpty():
            return
        end = self._clamp(event.position(), rect)
        moved = abs(end.x() - start.x()) + abs(end.y() - start.y())
        if moved < 8:  # a click, not a drag: select the box under the cursor
            index = self._box_at(end)
            self._selected = -1 if index == self._selected else index
            self.update()
            return
        if not self._classes:
            self.blocked.emit("Add a class first (Classes panel on the right).")
            self.update()
            return
        x1, y1 = self._to_norm(start, rect)
        x2, y2 = self._to_norm(end, rect)
        box = Box(self._active, x1, y1, x2, y2).normalized()
        if box.is_valid(0.01):
            self._boxes.append(box)
            self._selected = -1
            self.boxes_changed.emit()
        self.update()

    # ---- painting -------------------------------------------------------- #
    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0a0d11" if self._dark else "#e9edf2"))
        rect = self._image_rect()
        if rect.isEmpty():
            painter.setPen(QColor("#8e98a6"))
            painter.drawText(
                self.rect().adjusted(20, 20, -20, -20),
                int(Qt.AlignmentFlag.AlignCenter) | int(Qt.TextFlag.TextWordWrap),
                self._placeholder,
            )
            return

        painter.drawPixmap(rect, self._pixmap, QRectF(self._pixmap.rect()))

        for index, box in enumerate(self._boxes):
            color = class_color(box.cls)
            box_rect = self._box_rect(box, rect)
            selected = index == self._selected
            painter.setPen(QPen(color, 3 if selected else 2))
            fill = QColor(color)
            fill.setAlpha(55 if selected else 22)
            painter.setBrush(fill)
            painter.drawRect(box_rect)

            name = self._classes[box.cls] if 0 <= box.cls < len(self._classes) else "?"
            text_w = painter.fontMetrics().horizontalAdvance(name) + 10
            label_y = box_rect.top() - 18 if box_rect.top() - 18 >= rect.top() else box_rect.top()
            label_rect = QRectF(box_rect.left(), label_y, text_w, 18)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawRect(label_rect)
            painter.setPen(QColor("#10151b"))
            painter.drawText(label_rect, int(Qt.AlignmentFlag.AlignCenter), name)

        if self._press is not None and self._cursor is not None:
            color = class_color(self._active)
            pen = QPen(color, 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(QRectF(self._press, self._clamp(self._cursor, rect)).normalized())
        elif self._cursor is not None and rect.contains(self._cursor):
            guide = QColor("#ffffff" if self._dark else "#000000")
            guide.setAlpha(110)
            painter.setPen(QPen(guide, 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(rect.left(), self._cursor.y()), QPointF(rect.right(), self._cursor.y()))
            painter.drawLine(QPointF(self._cursor.x(), rect.top()), QPointF(self._cursor.x(), rect.bottom()))
