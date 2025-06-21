# !/usr/bin/env python3
"""
Stock Analyzer Pro – refined UI (Yahoo Finance-style) + Clear button
"""

import sys
import json
import logging
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any

# Third-party imports
import requests

try:
    import yfinance as yf
    import pandas as pd
    from PyQt6.QtWidgets import *
    from PyQt6.QtCore import *
    from PyQt6.QtGui import *
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install: pip install yfinance pandas PyQt6")
    sys.exit(1)

# --- logging -----------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# --- palette / style constants ----------------------------------------
PRIMARY = "#6200d5"  # Yahoo-style purple accent
BG_MAIN = "#ffffff"
BG_CARD = "#f5f5f5"
BORDER = "#d0d0d0"
TEXT_PRIMARY = "#111111"
TEXT_SECONDARY = "#555555"


# ----------------------------------------------------------------------
#                       Data & Analysis
# ----------------------------------------------------------------------
# --- Data Models ---
class StockData:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.ticker = yf.Ticker(symbol)
        self.info = {}
        self.history = pd.DataFrame()
        self.current_price = 0.0

    def fetch_data(self):
        """Fetch stock data from yfinance"""
        try:
            self.info = self.ticker.info
            self.history = self.ticker.history(period="3mo")
            if not self.history.empty:
                self.current_price = self.history['Close'].iloc[-1]
            return True
        except Exception as e:
            logger.error(f"Error fetching data for {self.symbol}: {e}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of stock data"""
        return {
            'symbol': self.symbol,
            'name': self.info.get('longName', self.symbol),
            'price': round(self.current_price, 2),
            'change': self._calculate_change(),
            'market_cap': self.info.get('marketCap', 0),
            'pe_ratio': round(self.info.get('trailingPE', 0), 2),
            'dividend_yield': round(self.info.get('dividendYield', 0) * 100, 2) if self.info.get(
                'dividendYield') else 0,
            'volume': self.info.get('volume', 0),
            '52w_high': round(self.info.get('fiftyTwoWeekHigh', 0), 2),
            '52w_low': round(self.info.get('fiftyTwoWeekLow', 0), 2),
            'sector': self.info.get('sector', 'Unknown'),
            'industry': self.info.get('industry', 'Unknown'),
            'employees': self.info.get('fullTimeEmployees', 0),
            'description': self.info.get('longBusinessSummary', '')[:200] + '...' if self.info.get(
                'longBusinessSummary') else '',
        }

    def _calculate_change(self) -> Dict[str, float]:
        """Calculate price change"""
        if self.history.empty:
            return {'amount': 0, 'percent': 0}

        prev_close = self.info.get('previousClose',
                                   self.history['Close'].iloc[-2] if len(self.history) > 1 else self.current_price)
        change = self.current_price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0

        return {
            'amount': round(change, 2),
            'percent': round(change_pct, 2)
        }


# --- Enhanced AI Analyzer ---
class SimpleAnalyzer:
    def __init__(self, api_key: Optional[str] = None, lm_studio_url: Optional[str] = None):
        self.api_key = api_key
        self.lm_studio_url = lm_studio_url
        self.has_ai = bool(api_key) or bool(lm_studio_url)
        if self.api_key:
            try:
                import openai
                openai.api_key = api_key
                self.openai = openai
            except ImportError:
                self.has_ai = False
                logger.warning("OpenAI package not installed, using LM Studio or basic analysis")

    def analyze_stock(self, stock_data: StockData) -> Dict[str, Any]:
        """Analyze stock with or without AI, return structured JSON"""
        summary = stock_data.get_summary()

        if not self.has_ai:
            return self._basic_analysis_json(summary)

        try:
            if self.lm_studio_url:
                return self._lm_studio_analysis(summary)
            else:
                return self._openai_analysis(summary)
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            return self._basic_analysis_json(summary)

    def _openai_analysis(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Use OpenAI for analysis"""
        prompt = self._create_analysis_prompt(summary)

        response = self.openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a stock analyst. Always respond with valid JSON only."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.7
        )

        try:
            return json.loads(response.choices[0].message.content)
        except json.JSONDecodeError:
            logger.error("Failed to parse OpenAI response as JSON")
            return self._basic_analysis_json(summary)

    def _lm_studio_analysis(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Use LM Studio for analysis"""
        prompt = self._create_analysis_prompt(summary)

        try:
            response = requests.post(
                f"{self.lm_studio_url}/v1/chat/completions",
                json={
                    "messages": [
                        {"role": "system", "content": "You are a stock analyst. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 500
                },
                timeout=30
            )

            if response.status_code == 200:
                content = response.json()['choices'][0]['message']['content']
                # Clean and parse JSON
                try:
                    # Remove potential markdown code blocks
                    cleaned_content = re.sub(r'```json|```', '', content).strip()
                    return json.loads(cleaned_content)
                except json.JSONDecodeError:
                    logger.error("Failed to parse LM Studio response as JSON")
                    return self._basic_analysis_json(summary)
            else:
                logger.error(f"LM Studio error: {response.status_code} - {response.text}")
                return self._basic_analysis_json(summary)

        except Exception as e:
            logger.error(f"LM Studio connection error: {e}")
            return self._basic_analysis_json(summary)

    def _create_analysis_prompt(self, summary: Dict[str, Any]) -> str:
        """Create analysis prompt for LLM"""
        return f"""
        Analyze this stock and provide a structured JSON response:
        
        Stock: {summary['symbol']} - {summary['name']}
        Price: ${summary['price']} ({summary['change']['percent']:+.2f}%)
        Market Cap: ${summary['market_cap']:,}
        P/E Ratio: {summary['pe_ratio']}
        52-Week Range: ${summary['52w_low']} - ${summary['52w_high']}
        Sector: {summary['sector']}
        
        Respond with ONLY a JSON object in this exact format:
        {{
            "recommendation": "BUY/HOLD/SELL",
            "confidence": 0-100,
            "price_target": number,
            "risk_level": "LOW/MEDIUM/HIGH",
            "summary": "Brief analysis summary",
            "strengths": ["strength1", "strength2", "strength3"],
            "risks": ["risk1", "risk2", "risk3"],
            "technical_indicators": {{
                "trend": "BULLISH/BEARISH/NEUTRAL",
                "support_level": number,
                "resistance_level": number
            }},
            "valuation": {{
                "assessment": "UNDERVALUED/FAIR/OVERVALUED",
                "reasoning": "Brief explanation"
            }}
        }}
        """

    def _basic_analysis_json(self, summary: Dict[str, Any]) -> Dict[str, Any]:
        """Basic rule-based analysis in JSON format"""
        # Price position calculation
        price_position = 0.5
        if summary['52w_high'] > 0 and summary['52w_low'] > 0:
            price_position = (summary['price'] - summary['52w_low']) / (summary['52w_high'] - summary['52w_low'])

        # P/E based valuation
        pe = summary['pe_ratio']
        if pe > 0:
            if pe < 15:
                valuation = "UNDERVALUED"
                valuation_reason = "Low P/E ratio suggests undervaluation"
            elif pe > 30:
                valuation = "OVERVALUED"
                valuation_reason = "High P/E ratio suggests overvaluation"
            else:
                valuation = "FAIR"
                valuation_reason = "P/E ratio within normal range"
        else:
            valuation = "UNKNOWN"
            valuation_reason = "No P/E ratio available"

        # Trend determination
        trend = "NEUTRAL"
        if summary['change']['percent'] > 2:
            trend = "BULLISH"
        elif summary['change']['percent'] < -2:
            trend = "BEARISH"

        # Recommendation logic
        if valuation == "UNDERVALUED" and trend != "BEARISH":
            recommendation = "BUY"
            confidence = 75
        elif valuation == "OVERVALUED" and trend != "BULLISH":
            recommendation = "SELL"
            confidence = 70
        else:
            recommendation = "HOLD"
            confidence = 60

        # Risk assessment
        if price_position > 0.8:
            risk_level = "HIGH"
        elif price_position < 0.2:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "recommendation": recommendation,
            "confidence": confidence,
            "price_target": round(summary['price'] * 1.1, 2),  # Simple 10% upside
            "risk_level": risk_level,
            "summary": f"{summary['name']} is currently {valuation.lower()} with a {trend.lower()} trend.",
            "strengths": [
                f"Trading at ${summary['price']}" + (" near 52-week low" if price_position < 0.2 else ""),
                f"Market cap of ${summary['market_cap']:,.0f}",
                f"In the {summary['sector']} sector" if summary['sector'] != 'Unknown' else "Established company"
            ],
            "risks": [
                "Near 52-week high" if price_position > 0.8 else "Market volatility",
                f"P/E ratio of {pe:.1f}" if pe > 25 else "Sector competition",
                "Limited dividend yield" if summary['dividend_yield'] < 1 else "Economic uncertainty"
            ],
            "technical_indicators": {
                "trend": trend,
                "support_level": round(summary['52w_low'], 2),
                "resistance_level": round(summary['52w_high'], 2)
            },
            "valuation": {
                "assessment": valuation,
                "reasoning": valuation_reason
            }
        }


# ----------------------------------------------------------------------
#                             UI widgets
# ----------------------------------------------------------------------
class StockInfoCard(QWidget):
    """Reusable card for key/val pairs"""

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {BG_CARD};
                border: 1px solid {BORDER};
                border-radius: 8px;
                padding: 10px;
                margin: 2px;
            }}
        """)
        vbox = QVBoxLayout(self)
        title_lbl = QLabel(title)
        title_lbl.setStyleSheet(f"font-weight:bold; color:{PRIMARY};")
        vbox.addWidget(title_lbl)
        self.content_layout = QVBoxLayout()
        vbox.addLayout(self.content_layout)

    # ------------------------------------------------------------------
    def add_item(self, label: str, value: str, color: Optional[str] = None):
        row = QHBoxLayout()
        key_lbl = QLabel(label)
        key_lbl.setStyleSheet(f"color:{TEXT_SECONDARY};")
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


# --- Stock Screener ---
class SimpleScreener:
    def __init__(self):
        # Popular stocks for screening
        self.stock_universe = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA',
            'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'HD',
            'DIS', 'BAC', 'ADBE', 'NFLX', 'PFE', 'KO', 'NKE', 'MCD',
            'INTC', 'VZ', 'T', 'XOM', 'CVX', 'ABBV', 'CRM'
        ]

    def screen_stocks(self, criteria: str) -> List[Dict[str, Any]]:
        """Simple screening based on criteria"""
        results = []
        criteria_lower = criteria.lower()

        # Determine what to look for
        filters = {
            'tech': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'ADBE', 'NFLX', 'CRM', 'INTC'],
            'dividend': ['JNJ', 'PG', 'KO', 'PFE', 'VZ', 'T', 'XOM', 'CVX', 'ABBV'],
            'retail': ['AMZN', 'WMT', 'HD', 'NKE', 'MCD'],
            'finance': ['JPM', 'V', 'MA', 'BAC']
        }

        # Select stocks based on criteria
        selected_stocks = self.stock_universe
        for keyword, stocks in filters.items():
            if keyword in criteria_lower:
                selected_stocks = stocks
                break

        # Fetch data for selected stocks
        for symbol in selected_stocks[:10]:  # Limit to 10 for speed
            try:
                stock = StockData(symbol)
                if stock.fetch_data():
                    summary = stock.get_summary()

                    # Apply additional filters
                    include = True

                    if 'undervalued' in criteria_lower and summary['pe_ratio'] > 20:
                        include = False
                    elif 'growth' in criteria_lower and summary['change']['percent'] < 0:
                        include = False
                    elif 'dividend' in criteria_lower and summary['dividend_yield'] == 0:
                        include = False

                    if include:
                        results.append(summary)

            except Exception as e:
                logger.error(f"Error screening {symbol}: {e}")

        # Sort by market cap
        results.sort(key=lambda x: x['market_cap'], reverse=True)
        return results


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
                background:{PRIMARY}; color:white;
            }}
        """)
        vbox.addWidget(self.rec_label)

        self.conf_bar = QProgressBar()
        self.conf_bar.setTextVisible(True)
        self.conf_bar.setFormat("Confidence: %p%")
        self.conf_bar.setStyleSheet(f"""
            QProgressBar {{
                border:1px solid {BORDER}; border-radius:5px; height:22px;
            }}
            QProgressBar::chunk {{ background-color:{PRIMARY}; }}
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

        colors = {"BUY": "#28a745", "SELL": "#e63946", "HOLD": PRIMARY}
        self.rec_label.setStyleSheet(f"""
            QLabel {{
                font-size:32px; font-weight:bold; padding:18px;
                border-radius:10px; background:{colors.get(rec, PRIMARY)};
                color:white;
            }}
        """)

        risk_css = {"LOW": "color:#28a745;",
                    "MEDIUM": "color:#f0a202;",
                    "HIGH": "color:#e63946;"}.get(risk, "")
        self.risk_lbl.setStyleSheet(f"font-size:15px; padding:6px; {risk_css}")


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
                background:{BG_CARD}; border:1px solid {BORDER};
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
                    background:{BG_CARD}; border:1px solid {BORDER};
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
                     "NEUTRAL": "#f0a202"}.get(t['trend'], TEXT_PRIMARY)
        self.technical_card.add_item("Trend", t['trend'], trend_col)
        self.technical_card.add_item("Support", f"${t['support_level']:.2f}")
        self.technical_card.add_item("Resistance", f"${t['resistance_level']:.2f}")

        # Valuation
        self.valuation_card.clear_items()
        v = a['valuation']
        val_col = {"UNDERVALUED": "#28a745",
                   "OVERVALUED": "#e63946",
                   "FAIR": "#f0a202"}.get(v['assessment'], TEXT_PRIMARY)
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


# ----------------------------------------------------------------------
#                           Main Window
# ----------------------------------------------------------------------
class StockAnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = SimpleAnalyzer()
        self.screener = SimpleScreener()
        self.current_stock = None
        self.init_ui()

    # ------------------------------------------------------------------
    def init_ui(self):
        self.setWindowTitle("Stock Analyzer Pro")
        self.resize(1200, 800)

        central = QWidget()
        self.setCentralWidget(central)
        vbox = QVBoxLayout(central)

        header = QLabel("📈 Stock Analyzer Pro")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        header.setStyleSheet(f"font-size:26px; font-weight:bold; color:{PRIMARY};")
        vbox.addWidget(header)

        # Search row
        row = QHBoxLayout()
        row.addWidget(QLabel("Symbol:"))
        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("e.g. AAPL")
        self.symbol_input.returnPressed.connect(self.analyze_stock)
        row.addWidget(self.symbol_input)

        self.analyze_btn = QPushButton("Analyze")
        self.analyze_btn.clicked.connect(self.analyze_stock)
        row.addWidget(self.analyze_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_all_views)
        self.clear_btn.setStyleSheet("background:#bbbbbb;")
        row.addWidget(self.clear_btn)
        row.addStretch()
        vbox.addLayout(row)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(f"""
            QTabBar::tab {{
                padding:10px; min-width:120px;
                background:{BG_CARD}; border:1px solid {BORDER};
            }}
            QTabBar::tab:selected {{
                background:{BG_MAIN}; border-bottom:3px solid {PRIMARY};
            }}
        """)
        self.analysis_tab = self.create_analysis_tab()
        self.screener_tab = self.create_screener_tab()
        self.settings_tab = self.create_settings_tab()
        self.tabs.addTab(self.analysis_tab, "Analysis")
        self.tabs.addTab(self.screener_tab, "Screener")
        self.tabs.addTab(self.settings_tab, "Settings")
        vbox.addWidget(self.tabs)

        self.statusBar().showMessage("Ready")
        self.set_app_style()

    # ------------------------------------------------------------------
    def set_app_style(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background:{BG_MAIN}; color:{TEXT_PRIMARY}; }}
            QLineEdit, QTextEdit {{
                background:{BG_MAIN}; border:1px solid {BORDER}; padding:6px;
            }}
            QPushButton {{
                background:{PRIMARY}; color:white; border:none;
                padding:8px 16px; border-radius:4px;
            }}
            QPushButton:hover {{ background:#4f00ba; }}
            QTableWidget {{ background:{BG_MAIN}; font-size:12px; }}
            QHeaderView::section {{
                background:{BG_CARD}; padding:4px; border:1px solid {BORDER};
            }}
            QGroupBox {{
                border:1px solid {BORDER}; border-radius:5px; margin-top:18px;
                padding-top:10px; font-weight:bold;
            }}
            QGroupBox::title {{
                subcontrol-origin:margin; subcontrol-position:top center;
                padding:0 5px;
            }}
        """)

    # ------------------------------------------------------------------
    def create_analysis_tab(self):
        w = QWidget()
        l = QVBoxLayout(w)
        self.analysis_disp = AnalysisDisplay()
        l.addWidget(self.analysis_disp)
        return w

    def create_screener_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)

        # controls
        ctl = QHBoxLayout()
        ctl.addWidget(QLabel("Screen:"))
        self.screen_input = QLineEdit()
        self.screen_input.setPlaceholderText("e.g. tech, dividend, undervalued")
        self.screen_input.returnPressed.connect(self.screen_stocks)
        ctl.addWidget(self.screen_input)
        btn = QPushButton("Run")
        btn.clicked.connect(self.screen_stocks)
        ctl.addWidget(btn)
        ctl.addStretch()
        v.addLayout(ctl)

        # table
        self.results_tbl = QTableWidget()
        self.results_tbl.setColumnCount(6)
        self.results_tbl.setHorizontalHeaderLabels(
            ["Symbol", "Name", "Price", "Change %", "Market Cap", "P/E"]
        )
        self.results_tbl.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.results_tbl.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.results_tbl.doubleClicked.connect(self.analyze_from_screener)
        self.results_tbl.setSortingEnabled(True)
        v.addWidget(self.results_tbl)
        return w

    def create_settings_tab(self):
        w = QWidget()
        v = QVBoxLayout(w)
        api_grp = QGroupBox("AI Settings")
        api_v = QVBoxLayout(api_grp)
        # OpenAI
        open_row = QHBoxLayout()
        open_row.addWidget(QLabel("OpenAI key:"))
        self.api_key_edit = QLineEdit()
        self.api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        open_row.addWidget(self.api_key_edit)
        api_v.addLayout(open_row)
        # LM studio
        lm_row = QHBoxLayout()
        lm_row.addWidget(QLabel("LM Studio URL:"))
        self.lm_url_edit = QLineEdit()
        lm_row.addWidget(self.lm_url_edit)
        api_v.addLayout(lm_row)

        save_btn = QPushButton("Save")
        save_btn.clicked.connect(self.save_api_key)
        api_v.addWidget(save_btn)
        v.addWidget(api_grp)
        v.addStretch()
        return w

    # ------------------------------------------------------------------
    def analyze_stock(self):
        symbol = self.symbol_input.text().strip().upper()
        if not symbol:
            QMessageBox.warning(self, "Missing", "Enter ticker symbol.")
            return
        self.statusBar().showMessage(f"Fetching {symbol}…")
        QApplication.processEvents()
        st = StockData(symbol)
        if not st.fetch_data():
            QMessageBox.critical(self, "Error", "Could not fetch data.")
            self.statusBar().showMessage("Error")
            return
        self.analysis_disp.display_stock_data(st.get_summary())
        analysis = self.analyzer.analyze_stock(st)
        self.analysis_disp.display_analysis(analysis)
        self.statusBar().showMessage(f"Analysis complete for {symbol}")

    # ------------------------------------------------------------------
    def screen_stocks(self):
        crit = self.screen_input.text().strip()
        if not crit:
            QMessageBox.warning(self, "Missing", "Enter screening criteria.")
            return
        self.statusBar().showMessage("Screening…")
        QApplication.processEvents()
        res = self.screener.screen_stocks(crit)
        self.results_tbl.setRowCount(0)
        for s in res:
            r = self.results_tbl.rowCount()
            self.results_tbl.insertRow(r)
            self.results_tbl.setItem(r, 0, QTableWidgetItem(s["symbol"]))
            self.results_tbl.setItem(r, 1, QTableWidgetItem(s["name"]))
            self.results_tbl.setItem(r, 2, QTableWidgetItem(f"${s['price']:.2f}"))
            chg = QTableWidgetItem(f"{s['change']['percent']:+.2f}%")
            chg.setForeground(QColor("#28a745" if s['change']['percent'] >= 0 else "#e63946"))
            self.results_tbl.setItem(r, 3, chg)
            self.results_tbl.setItem(r, 4, QTableWidgetItem(f"${s['market_cap']:,.0f}"))
            self.results_tbl.setItem(r, 5, QTableWidgetItem(f"{s['pe_ratio']:.2f}"))
        self.statusBar().showMessage(f"{len(res)} results")

    # ------------------------------------------------------------------
    def analyze_from_screener(self, index):
        symbol = self.results_tbl.item(index.row(), 0).text()
        self.tabs.setCurrentIndex(0)
        self.symbol_input.setText(symbol)
        self.analyze_stock()

    # ------------------------------------------------------------------
    def save_api_key(self):
        self.analyzer.api_key = self.api_key_edit.text().strip()
        self.analyzer.lm_studio_url = self.lm_url_edit.text().strip()
        self.analyzer.has_ai = bool(self.analyzer.api_key) or bool(self.analyzer.lm_studio_url)
        QMessageBox.information(self, "Saved", "Settings updated.")

    # ------------------------------------------------------------------
    def clear_all_views(self):
        self.symbol_input.clear()
        self.analysis_disp.clear_all()
        self.results_tbl.setRowCount(0)
        self.statusBar().showMessage("Cleared")


# ----------------------------------------------------------------------
def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Stock Analyzer Pro")
    app.setStyle("Fusion")
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    win = StockAnalyzerWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
