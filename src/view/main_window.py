from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *


# ----------------------------------------------------------------------
#                           Main Window
# ----------------------------------------------------------------------
from src.constants.constants import Constants
from src.model.stock_analyzer import StockAnalyzer
from src.model.stock_data import StockData
from src.model.stock_screener import StockScreener
from src.view.analysis_display import AnalysisDisplay


class StockAnalyzerWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.analyzer = StockAnalyzer()
        self.screener = StockScreener()
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
        header.setStyleSheet(f"font-size:26px; font-weight:bold; color:{Constants.PRIMARY};")
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
                background:{Constants.BG_CARD}; border:1px solid {Constants.BORDER};
            }}
            QTabBar::tab:selected {{
                background:{Constants.BG_MAIN}; border-bottom:3px solid {Constants.PRIMARY};
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
            QMainWindow {{ background:{Constants.BG_MAIN}; color:{Constants.TEXT_PRIMARY}; }}
            QLineEdit, QTextEdit {{
                background:{Constants.BG_MAIN}; border:1px solid {Constants.BORDER}; padding:6px;
            }}
            QPushButton {{
                background:{Constants.PRIMARY}; color:white; border:none;
                padding:8px 16px; border-radius:4px;
            }}
            QPushButton:hover {{ background:#4f00ba; }}
            QTableWidget {{ background:{Constants.BG_MAIN}; font-size:12px; }}
            QHeaderView::section {{
                background:{Constants.BG_CARD}; padding:4px; border:1px solid {Constants.BORDER};
            }}
            QGroupBox {{
                border:1px solid {Constants.BORDER}; border-radius:5px; margin-top:18px;
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

