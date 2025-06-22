from typing import Dict, Any, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QTextOption
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QTextEdit,
    QListWidget,
    QLabel,
    QScrollArea,
)

from src.constants.constants import Constants
from src.view.recommendation_widget import RecommendationWidget
from src.view.stock_info_card import StockInfoCard


class AnalysisDisplay(QWidget):
    """Responsive three‑column stock analysis view with graceful clearing and improved spacing.

    The class avoids Python ≥3.10‑only syntax (e.g., ``A | B`` union types) so it runs cleanly
    on Python 3.8, which is still common in many production environments. All scrolling,
    sizing, and styling logic remains unchanged from the previous revision.
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _wrap_in_scroll(widget: QWidget) -> QScrollArea:
        """Return *widget* wrapped in a border‑less ``QScrollArea`` so every column scrolls
        independently and the layout stays usable on small screens."""
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(widget)
        return area

    def _auto_resize_text_edit(self, text_edit: QTextEdit, max_height: int = 0) -> None:
        """Shrink/grow ``text_edit`` to fit its document up to *max_height* (0 ⇒ unlimited)."""
        new_height = int(text_edit.document().size().height() + 10)
        if max_height:
            new_height = min(new_height, max_height)
        text_edit.setFixedHeight(new_height + 2)  # +2 for borders

    # ------------------------------------------------------------------
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # -------------------- Left column --------------------
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        self.info_card = StockInfoCard("Stock Information")
        self.metrics_card = StockInfoCard("Key Metrics")
        self.technical_card = StockInfoCard("Technical Indicators")

        for card in (self.info_card, self.metrics_card, self.technical_card):
            left_layout.addWidget(card)
        left_layout.addStretch()

        splitter.addWidget(self._wrap_in_scroll(left_container))

        # -------------------- Centre column --------------------
        centre_container = QWidget()
        centre_layout = QVBoxLayout(centre_container)
        centre_layout.setContentsMargins(0, 0, 0, 0)
        centre_layout.setSpacing(10)

        self.rec_widget = RecommendationWidget()
        self.valuation_card = StockInfoCard("Valuation")

        self.summary_txt = QTextEdit()
        self.summary_txt.setReadOnly(True)
        self.summary_txt.setWordWrapMode(QTextOption.WrapMode.WordWrap)
        self.summary_txt.setStyleSheet(
            f"""
            QTextEdit {{
                background: {Constants.BG_CARD};
                border: 1px solid {Constants.BORDER};
                padding: 8px;
                border-radius: 6px;
                font-size: 12px;
            }}
            """
        )

        centre_layout.addWidget(self.rec_widget)
        centre_layout.addWidget(self.valuation_card)
        centre_layout.addWidget(self.summary_txt)
        centre_layout.addStretch()

        splitter.addWidget(self._wrap_in_scroll(centre_container))

        # -------------------- Right column --------------------
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        st_lbl = QLabel("Strengths")
        st_lbl.setStyleSheet("font-weight:bold; color:#28a745;")
        self.strengths = QListWidget()

        rk_lbl = QLabel("Risks")
        rk_lbl.setStyleSheet("font-weight:bold; color:#e63946;")
        self.risks = QListWidget()

        for list_widget in (self.strengths, self.risks):
            list_widget.setStyleSheet(
                f"""
                QListWidget {{
                    background: {Constants.BG_CARD};
                    border: 1px solid {Constants.BORDER};
                    border-radius: 5px;
                    font-size: 12px;
                }}
                QListWidget::item {{ padding: 4px; }}
                """
            )

        right_layout.addWidget(st_lbl)
        right_layout.addWidget(self.strengths)
        right_layout.addWidget(rk_lbl)
        right_layout.addWidget(self.risks)
        right_layout.addStretch()

        splitter.addWidget(self._wrap_in_scroll(right_container))

        # Equal stretch for each column
        for i in range(3):
            splitter.setStretchFactor(i, 1)

        # Outer layout
        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(splitter)

        # Keep a reference for later use
        self._splitter = splitter

    # ------------------------------------------------------------------
    # Public API – populate widgets
    # ------------------------------------------------------------------
    def display_stock_data(self, s: Dict[str, Any]) -> None:
        self.info_card.clear_items()
        self.metrics_card.clear_items()

        price_col = "#28a745" if s["change"]["percent"] >= 0 else "#e63946"
        for label, value in (
            ("Symbol", s["symbol"]),
            ("Company", s["name"]),
            ("Sector", s["sector"]),
            ("Industry", s["industry"]),
            ("Price", f"${s['price']:.2f}"),
            ("Change", f"{s['change']['percent']:+.2f}%"),
        ):
            color = price_col if label in ("Price", "Change") else None
            self.info_card.add_item(label, value, color)

        self.metrics_card.add_item("Market Cap", f"${s['market_cap']:,.0f}")
        self.metrics_card.add_item("P/E Ratio", f"{s['pe_ratio']:.2f}")
        self.metrics_card.add_item("Volume", f"{s['volume']:,}")
        self.metrics_card.add_item("Dividend Yield", f"{s['dividend_yield']:.2f}%")
        self.metrics_card.add_item("52W High", f"${s['52w_high']:.2f}")
        self.metrics_card.add_item("52W Low", f"${s['52w_low']:.2f}")

    def display_analysis(self, a: Dict[str, Any]) -> None:
        # Recommendation
        self.rec_widget.set_recommendation(
            a["recommendation"], a["confidence"], a["risk_level"]
        )

        # Technicals
        self.technical_card.clear_items()
        t = a["technical_indicators"]
        trend_col = {
            "BULLISH": "#28a745",
            "BEARISH": "#e63946",
            "NEUTRAL": "#f0a202",
        }.get(t["trend"], Constants.TEXT_PRIMARY)
        self.technical_card.add_item("Trend", t["trend"], trend_col)
        self.technical_card.add_item("Support", f"${t['support_level']:.2f}")
        self.technical_card.add_item("Resistance", f"${t['resistance_level']:.2f}")

        # Valuation
        self.valuation_card.clear_items()
        v = a["valuation"]
        val_col = {
            "UNDERVALUED": "#28a745",
            "OVERVALUED": "#e63946",
            "FAIR": "#f0a202",
        }.get(v["assessment"], Constants.TEXT_PRIMARY)
        self.valuation_card.add_item("Assessment", v["assessment"], val_col)
        self.valuation_card.add_item("Target Price", f"${a['price_target']:.2f}")
        self.valuation_card.add_item("Reasoning", v["reasoning"])

        # Summary
        self.summary_txt.setText(a["summary"])
        self._auto_resize_text_edit(self.summary_txt, max_height=200)

        # Lists
        self.strengths.clear()
        self.risks.clear()
        self.strengths.addItems([f"✓ {item}" for item in a["strengths"]])
        self.risks.addItems([f"⚠ {item}" for item in a["risks"]])

    # ------------------------------------------------------------------
    # Reset state before next analysis
    # ------------------------------------------------------------------
    def clear_all(self) -> None:
        for card in (
            self.info_card,
            self.metrics_card,
            self.technical_card,
            self.valuation_card,
        ):
            card.clear_items()

        self.summary_txt.clear()
        self._auto_resize_text_edit(self.summary_txt)

        self.strengths.clear()
        self.risks.clear()

        self.rec_widget.set_recommendation("HOLD", 0, "MEDIUM")
