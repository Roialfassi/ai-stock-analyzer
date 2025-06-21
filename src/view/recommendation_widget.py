
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from src.constants.constants import Constants


class RecommendationWidget(QWidget):
    """Headline Buy/Hold/Sell + confidence bar"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(10, 10, 10, 10)

        self.rec_label = QLabel("HOLD")
        self.rec_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.rec_label.setStyleSheet(f"""
            QLabel {{
                font-size:32px; font-weight:bold;
                padding:18px; border-radius:10px;
                background:{Constants.PRIMARY}; color:white;
            }}
        """)
        vbox.addWidget(self.rec_label)

        self.conf_bar = QProgressBar()
        self.conf_bar.setTextVisible(True)
        self.conf_bar.setFormat("Confidence: %p%")
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{
                border:1px solid {Constants.BORDER}; border-radius:5px; height:22px;
            }}
            QProgressBar::chunk {{ background-color:{Constants.PRIMARY}; }}
        """)
        vbox.addWidget(self.conf_bar)

        self.risk_lbl = QLabel("Risk: MEDIUM")
        self.risk_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.risk_lbl.setStyleSheet("font-size:15px; padding:6px;")
        vbox.addWidget(self.risk_lbl)

    # ------------------------------------------------------------------
    def set_recommendation(self, rec: str, conf: int, risk: str):
        self.rec_label.setText(rec)
        self.conf_bar.setValue(conf)
        self.risk_lbl.setText(f"Risk: {risk}")

        colors = {"BUY": "#28a745", "SELL": "#e63946", "HOLD": Constants.PRIMARY}
        self.rec_label.setStyleSheet(f"""
            QLabel {{
                font-size:32px; font-weight:bold; padding:18px;
                border-radius:10px; background:{colors.get(rec, Constants.PRIMARY)};
                color:white;
            }}
        """)

        risk_css = {"LOW": "color:#28a745;",
                    "MEDIUM": "color:#f0a202;",
                    "HIGH": "color:#e63946;"}.get(risk, "")
        self.risk_lbl.setStyleSheet(f"font-size:15px; padding:6px; {risk_css}")
