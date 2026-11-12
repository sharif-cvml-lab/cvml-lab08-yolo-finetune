"""Dark / light Qt style sheets (fixed infrastructure)."""
from __future__ import annotations

from string import Template

_TEMPLATE = Template("""
QMainWindow, QWidget {
    background: $bg;
    color: $text;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QFrame#Header, QFrame#SidebarCard, QFrame#VideoCard, QFrame#StatusBar {
    background: $card;
    border: 1px solid $card_border;
    border-radius: 12px;
}
QFrame#VideoCanvas {
    background: $canvas;
    border: 1px solid $canvas_border;
    border-radius: 10px;
}
QLabel { background: transparent; }
QLabel#Title { font-size: 22px; font-weight: 700; }
QLabel#Subtitle, QLabel#Muted { color: $muted; }
QLabel#PanelTitle { font-size: 14px; font-weight: 700; }
QLabel#Metric { color: $metric; font-size: 12px; }
QLabel#MetricValue { font-size: 15px; font-weight: 700; }
QLabel#ValueChip {
    background: $chip_bg;
    border: 1px solid $chip_border;
    border-radius: 6px;
    padding: 2px 8px;
    min-width: 42px;
}
QPushButton {
    background: $btn_bg;
    color: $btn_text;
    border: 1px solid $btn_border;
    border-radius: 8px;
    padding: 8px 12px;
}
QPushButton:hover { background: $btn_hover; border-color: $btn_hover_border; }
QPushButton:pressed { background: $btn_pressed; }
QPushButton:disabled { color: $muted; background: $btn_pressed; }
QPushButton#accent { background: $accent; border-color: $accent_border; color: #ffffff; font-weight: 700; }
QPushButton#accent:hover { background: $accent_hover; }
QPushButton#accent:disabled { background: $btn_pressed; color: $muted; border-color: $btn_border; }
QPushButton#danger { background: $danger_bg; border-color: $danger_border; }
QPushButton#step { padding: 8px 18px; font-weight: 600; }
QPushButton#step:checked { background: $accent; border-color: $accent_border; color: #ffffff; }
QSpinBox, QLineEdit, QComboBox {
    background: $input_bg;
    border: 1px solid $btn_border;
    border-radius: 7px;
    padding: 6px;
}
QComboBox QAbstractItemView { background: $input_bg; selection-background-color: $list_sel; }
QSlider::groove:horizontal { height: 5px; background: $groove; border-radius: 3px; }
QSlider::sub-page:horizontal { background: $sub; border-radius: 3px; }
QSlider::handle:horizontal {
    background: $handle;
    border: 1px solid $handle_border;
    width: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QProgressBar {
    background: $input_bg;
    border: 1px solid $btn_border;
    border-radius: 8px;
    text-align: center;
    min-height: 18px;
}
QProgressBar::chunk { background: $sub; border-radius: 7px; }
QListWidget, QPlainTextEdit {
    background: $input_bg;
    border: 1px solid $btn_border;
    border-radius: 8px;
    padding: 4px;
}
QListWidget::item { padding: 3px 4px; border-radius: 4px; }
QListWidget::item:selected { background: $list_sel; color: $text; }
QGroupBox {
    border: 1px solid $card_border;
    border-radius: 10px;
    margin-top: 12px;
    padding-top: 12px;
    font-weight: 700;
}
QGroupBox::title { left: 12px; padding: 0 6px; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { width: 10px; background: transparent; }
QScrollBar::handle:vertical { background: $scroll; border-radius: 5px; min-height: 30px; }
QTabWidget::pane { border: none; background: transparent; top: -1px; }
QTabBar::tab {
    background: $btn_bg;
    color: $btn_text;
    border: 1px solid $btn_border;
    border-bottom: none;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    padding: 7px 20px;
    margin-right: 4px;
}
QTabBar::tab:selected { background: $accent; border-color: $accent_border; color: #ffffff; font-weight: 700; }
QTableWidget {
    background: $input_bg;
    border: 1px solid $btn_border;
    border-radius: 8px;
    gridline-color: $card_border;
}
QTableWidget::item { padding: 3px 6px; }
QTableWidget::item:selected { background: $list_sel; color: $text; }
QHeaderView::section {
    background: $btn_bg;
    color: $metric;
    border: none;
    border-bottom: 1px solid $btn_border;
    padding: 5px 8px;
    font-weight: 600;
}
QToolTip { background: $card; color: $text; border: 1px solid $card_border; }
""")

_DARK = dict(
    bg="#11151b", text="#e7ebf0", card="#171c23", card_border="#28303a",
    canvas="#0a0d11", canvas_border="#303945", muted="#8e98a6", metric="#9aa6b4",
    chip_bg="#202833", chip_border="#364250", btn_bg="#212833", btn_text="#eef2f6",
    btn_border="#35404d", btn_hover="#2a3441", btn_hover_border="#4e6175",
    btn_pressed="#18202a", accent="#3a7d67", accent_border="#4c9b80",
    accent_hover="#479276", danger_bg="#322126", danger_border="#5d333d",
    input_bg="#1a2028", groove="#2b333d", sub="#4d9d80", handle="#80c6ab",
    handle_border="#9ad9c0", scroll="#313b47", list_sel="#2b4a40",
)

_LIGHT = dict(
    bg="#f3f5f8", text="#1d242c", card="#ffffff", card_border="#d5dce4",
    canvas="#e9edf2", canvas_border="#c4ccd6", muted="#667282", metric="#556171",
    chip_bg="#eaeff4", chip_border="#cdd5de", btn_bg="#ffffff", btn_text="#1d242c",
    btn_border="#c4ccd6", btn_hover="#eef2f6", btn_hover_border="#8fa3b5",
    btn_pressed="#dde4eb", accent="#2f8a6c", accent_border="#26745a",
    accent_hover="#38a07d", danger_bg="#fbe9ec", danger_border="#e3b3bc",
    input_bg="#ffffff", groove="#cfd6de", sub="#3f9b7c", handle="#2f8a6c",
    handle_border="#1f6a52", scroll="#bcc6d1", list_sel="#cfe9df",
)


def build_style(dark: bool) -> str:
    return _TEMPLATE.substitute(_DARK if dark else _LIGHT)
