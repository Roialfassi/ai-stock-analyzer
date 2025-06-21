from typing import Optional

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from src.constants.constants import Constants


class StockInfoCard(QWidget):
    """Reusable card for key/val pairs"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {Constants.BG_CARD};
                border: 1px solid {Constants.BORDER};
                border-radius: 8px;
                padding: 10px;
                margin: 2px;
            }}
        """)
        vbox = QVBoxLayout(self)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-weight:bold; color:{Constants.PRIMARY};")
        vbox.addWidget(title_lbl)
        self.content_layout = QVBoxLayout()
        vbox.addLayout(self.content_layout)

    # ------------------------------------------------------------------
    def add_item(self, label: str, value: str, color: Optional[str] = None):
        row = QHBoxLayout()
        key_lbl = QLabel(label)
        key_lbl.setStyleSheet(f"color:{Constants.TEXT_SECONDARY};")
        key_lbl.setMinimumWidth(120)
        row.addWidget(key_lbl)
        row.addStretch()
        val_lbl = QLabel(str(value))
        style = "font-weight:bold;"
        if color:
            style += f" color:{color};"
        val_lbl.setStyleSheet(style)
        row.addWidget(val_lbl)
        self.content_layout.addLayout(row)

    def clear_items(self):
        while self.content_layout.count():
            child = self.content_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

