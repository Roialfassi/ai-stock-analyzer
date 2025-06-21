from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

from src.constants.constants import Constants
from src.view.recommendation_widget import RecommendationWidget
from src.view.stock_info_card import StockInfoCard
from typing import Dict, List, Optional, Any


class AnalysisDisplay(QWidget):
    """Trip-column overview"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    # ------------------------------------------------------------------
    def init_ui(self):
        outer = QHBoxLayout(self)
        outer.setSpacing(10)

        # Left cards
        left = QVBoxLayout()
        self.info_card = StockInfoCard("Stock Information")
        self.metrics_card = StockInfoCard("Key Metrics")
        self.technical_card = StockInfoCard("Technical Indicators")
        for c in (self.info_card, self.metrics_card, self.technical_card):
            left.addWidget(c)
        left.addStretch()

        # Center
        center = QVBoxLayout()
        self.rec_widget = RecommendationWidget()
        self.valuation_card = StockInfoCard("Valuation")
        self.summary_txt = QTextEdit()
        self.summary_txt.setReadOnly(True)
        self.summary_txt.setMaximumHeight(110)
        self.summary_txt.setStyleSheet(f"""
            QTextEdit {{
                background:{Constants.BG_CARD}; border:1px solid {Constants.BORDER};
                padding:8px; border-radius:6px; font-size:12px;
            }}
        """)
        center.addWidget(self.rec_widget)
        center.addWidget(self.valuation_card)
        center.addWidget(self.summary_txt)
        center.addStretch()

        # Right strengths / risks
        right = QVBoxLayout()
        st_lbl = QLabel("Strengths")
        st_lbl.setStyleSheet("font-weight:bold; color:#28a745;")
        self.strengths = QListWidget()
        rk_lbl = QLabel("Risks")
        rk_lbl.setStyleSheet("font-weight:bold; color:#e63946;")
        self.risks = QListWidget()
        for widget in (self.strengths, self.risks):
            widget.setStyleSheet(f"""
                QListWidget {{
                    background:{Constants.BG_CARD}; border:1px solid {Constants.BORDER};
                    border-radius:5px; font-size:12px;
                }}
                QListWidget::item {{ padding:4px; }}
            """)
        right.addWidget(st_lbl)
        right.addWidget(self.strengths)
        right.addWidget(rk_lbl)
        right.addWidget(self.risks)

        # Assemble
        outer.addLayout(left, 1)
        outer.addLayout(center, 1)
        outer.addLayout(right, 1)

    # ------------------------------------------------------------------
    def display_stock_data(self, s: Dict[str, Any]):
        self.info_card.clear_items()
        self.metrics_card.clear_items()
        price_col = "#28a745" if s['change']['percent'] >= 0 else "#e63946"

        # Info
        for k, v in (("Symbol", s['symbol']), ("Company", s['name']),
                     ("Sector", s['sector']), ("Industry", s['industry']),
                     ("Price", f"${s['price']:.2f}"),
                     ("Change", f"{s['change']['percent']:+.2f}%")):
            col = price_col if k in ("Price", "Change") else None
            self.info_card.add_item(k, v, col)

        # Key metrics
        self.metrics_card.add_item("Market Cap", f"${s['market_cap']:,.0f}")
        self.metrics_card.add_item("P/E Ratio", f"{s['pe_ratio']:.2f}")
        self.metrics_card.add_item("Volume", f"{s['volume']:,}")
        self.metrics_card.add_item("Dividend Yield", f"{s['dividend_yield']:.2f}%")
        self.metrics_card.add_item("52W High", f"${s['52w_high']:.2f}")
        self.metrics_card.add_item("52W Low", f"${s['52w_low']:.2f}")

    def display_analysis(self, a: Dict[str, Any]):
        self.rec_widget.set_recommendation(a['recommendation'],
                                           a['confidence'],
                                           a['risk_level'])

        # Technical
        self.technical_card.clear_items()
        t = a['technical_indicators']
        trend_col = {"BULLISH": "#28a745",
                     "BEARISH": "#e63946",
                     "NEUTRAL": "#f0a202"}.get(t['trend'], Constants.TEXT_PRIMARY)
        self.technical_card.add_item("Trend", t['trend'], trend_col)
        self.technical_card.add_item("Support", f"${t['support_level']:.2f}")
        self.technical_card.add_item("Resistance", f"${t['resistance_level']:.2f}")

        # Valuation
        self.valuation_card.clear_items()
        v = a['valuation']
        val_col = {"UNDERVALUED": "#28a745",
                   "OVERVALUED": "#e63946",
                   "FAIR": "#f0a202"}.get(v['assessment'], Constants.TEXT_PRIMARY)
        self.valuation_card.add_item("Assessment", v['assessment'], val_col)
        self.valuation_card.add_item("Target Price", f"${a['price_target']:.2f}")
        self.valuation_card.add_item("Reasoning", v['reasoning'])

        # Lists / summary
        self.summary_txt.setText(a['summary'])
        self.strengths.clear()
        self.risks.clear()
        self.strengths.addItems([f"✓ {s}" for s in a['strengths']])
        self.risks.addItems([f"⚠ {r}" for r in a['risks']])

    # ------------------------------------------------------------------
    def clear_all(self):
        self.info_card.clear_items()
        self.metrics_card.clear_items()
        self.technical_card.clear_items()
        self.valuation_card.clear_items()
        self.summary_txt.clear()
        self.strengths.clear()
        self.risks.clear()
        self.rec_widget.set_recommendation("HOLD", 0, "MEDIUM")
