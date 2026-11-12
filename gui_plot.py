"""Curve plot with axis titles and a legend (fixed infrastructure).

Used twice: live training curves (Train = green, Validation = orange) and the history
comparison (one line per selected training).
"""
from __future__ import annotations

from dataclasses import dataclass

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

TRAIN_COLOR = "#4cc9a0"   # green
VAL_COLOR = "#ffb703"     # orange
RUN_COLORS = ("#4cc9a0", "#ffb703", "#4ea8ff", "#ff6b81", "#b388ff", "#ffd166", "#06d6a0", "#ef8354")


@dataclass
class Series:
    name: str
    values: list[float | None]
    color: str


class CurvePlot(QWidget):
    def __init__(self, x_label: str = "Epoch", y_label: str = "", empty_text: str = "Nothing to show yet"):
        super().__init__()
        self.series: list[Series] = []
        self.x_label = x_label
        self.y_label = y_label
        self.empty_text = empty_text
        self._dark = True
        self.setMinimumHeight(190)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_dark(self, dark: bool) -> None:
        self._dark = dark
        self.update()

    def set_series(self, series: list[Series]) -> None:
        self.series = series
        self.update()

    def set_labels(self, x_label: str, y_label: str) -> None:
        self.x_label, self.y_label = x_label, y_label
        self.update()

    def clear(self) -> None:
        self.series = []
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0a0d11" if self._dark else "#e9edf2"))
        muted = QColor("#8e98a6")
        text_color = QColor("#d7dde6" if self._dark else "#27303b")
        small = QFont(self.font())
        small.setPointSizeF(max(self.font().pointSizeF() - 1, 7))
        painter.setFont(small)

        left, top, right, bottom = 78, 30, 18, 38
        plot = QRectF(left, top, max(self.width() - left - right, 20), max(self.height() - top - bottom, 20))
        values = [v for s in self.series for v in s.values if v is not None]
        if not values:
            painter.setPen(muted)
            painter.drawText(self.rect(), int(Qt.AlignmentFlag.AlignCenter), self.empty_text)
            return

        low, high = min(values), max(values)
        if high - low < 1e-6:
            low, high = low - 0.5, high + 0.5
        count = max(max(len(s.values) for s in self.series) - 1, 1)

        # grid + y tick labels
        painter.setPen(QPen(muted, 1, Qt.PenStyle.DotLine))
        for step in range(5):
            y = plot.top() + plot.height() * step / 4
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
            painter.drawText(QRectF(plot.left() - 50, y - 8, 46, 16),
                             int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
                             f"{high - (high - low) * step / 4:.2f}")

        # x tick labels (first and last epoch) and axis titles
        painter.setPen(muted)
        painter.drawText(QRectF(plot.left() - 10, plot.bottom() + 3, 40, 14), int(Qt.AlignmentFlag.AlignLeft), "1")
        painter.drawText(QRectF(plot.right() - 30, plot.bottom() + 3, 40, 14), int(Qt.AlignmentFlag.AlignRight),
                         str(count + 1))
        painter.setPen(text_color)
        painter.drawText(QRectF(plot.left(), plot.bottom() + 18, plot.width(), 16),
                         int(Qt.AlignmentFlag.AlignCenter), self.x_label)
        painter.save()
        painter.translate(14, plot.top() + plot.height() / 2)
        painter.rotate(-90)
        painter.drawText(QRectF(-plot.height() / 2, -10, plot.height(), 20),
                         int(Qt.AlignmentFlag.AlignCenter), self.y_label)
        painter.restore()

        def point(index: int, value: float) -> QPointF:
            x = plot.left() + plot.width() * index / count
            y = plot.bottom() - plot.height() * (value - low) / (high - low)
            return QPointF(x, y)

        for series in self.series:
            painter.setPen(QPen(QColor(series.color), 2))
            previous = None
            for index, value in enumerate(series.values):
                if value is None:
                    continue
                current = point(index, value)
                if previous is not None:
                    painter.drawLine(previous, current)
                if len(series.values) <= 60:
                    painter.drawEllipse(current, 2.5, 2.5)
                previous = current

        # legend (top row, colour swatch + name)
        x = plot.left()
        for series in self.series:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(series.color))
            painter.drawRoundedRect(QRectF(x, 9, 14, 8), 3, 3)
            painter.setPen(text_color)
            label = series.name if len(series.name) <= 34 else series.name[:33] + "…"
            width = painter.fontMetrics().horizontalAdvance(label) + 8
            painter.drawText(QRectF(x + 18, 4, width, 18), int(Qt.AlignmentFlag.AlignVCenter), label)
            x += 18 + width + 14
            if x > plot.right():
                break
