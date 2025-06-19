# main_window.py - Application Shell and Orchestration

import sys
import asyncio
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QSplitter, QDockWidget, QMenuBar, QMenu, QToolBar,
    QStatusBar, QMessageBox, QFileDialog, QInputDialog,
    QListWidget, QListWidgetItem, QTabWidget, QTextEdit,
    QPushButton, QLineEdit, QComboBox, QLabel, QDialog,
    QDialogButtonBox, QFormLayout, QSpinBox, QDoubleSpinBox,
    QProgressDialog
)
from PyQt6.QtCore import (
    Qt, QTimer, QThread, pyqtSignal, QSettings,
    QSize, QPoint, QByteArray, QFileInfo
)
from PyQt6.QtGui import (
    QIcon, QKeySequence, QCloseEvent, QFont,
    QPalette, QColor
)

# Import all modules
from models import *
from market_data import MarketDataProvider
from llm_analyzer import (
    LLMProvider, OpenAIProvider, AnthropicProvider,
    GeminiProvider, HuggingFaceProvider, LMStudioProvider,
    StockAnalyzer, NaturalLanguageScreener
)
from ui_components import *
from screener import StockScreener
from portfolio import PortfolioManager

logger = logging.getLogger(__name__)


class AsyncWorker(QThread):
    """Worker thread for async operations"""

    result_ready = pyqtSignal(object)
    error_occurred = pyqtSignal(str)
    progress_update = pyqtSignal(int, str)

    def __init__(self, coro, parent=None):
        super().__init__(parent)
        self.coro = coro

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.coro)
            self.result_ready.emit(result)
        except Exception as e:
            logger.error(f"Async worker error: {e}")
            self.error_occurred.emit(str(e))
        finally:
            loop.close()


class SettingsDialog(QDialog):
    """Settings dialog for API keys and preferences"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setModal(True)
        self.setMinimumWidth(500)

        self.settings = QSettings("StockAnalyzer", "Settings")
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # Create tabs
        tabs = QTabWidget()

        # API Keys tab
        api_widget = QWidget()
        api_layout = QFormLayout(api_widget)

        self.openai_key = QLineEdit()
        self.openai_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.openai_key.setText(self.settings.value("openai_key", ""))
        api_layout.addRow("OpenAI API Key:", self.openai_key)

        self.anthropic_key = QLineEdit()
        self.anthropic_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.anthropic_key.setText(self.settings.value("anthropic_key", ""))
        api_layout.addRow("Anthropic API Key:", self.anthropic_key)

        self.gemini_key = QLineEdit()
        self.gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key.setText(self.settings.value("gemini_key", ""))
        api_layout.addRow("Google Gemini API Key:", self.gemini_key)

        self.huggingface_key = QLineEdit()
        self.huggingface_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.huggingface_key.setText(self.settings.value("huggingface_key", ""))
        api_layout.addRow("HuggingFace API Key:", self.huggingface_key)

        self.lmstudio_url = QLineEdit()
        self.lmstudio_url.setText(self.settings.value("lmstudio_url", "http://localhost:1234"))
        api_layout.addRow("LM Studio URL:", self.lmstudio_url)

        # Note about data source
        data_note = QLabel("📊 All market data is fetched from Yahoo Finance (yfinance)")
        data_note.setStyleSheet("color: #8b92a1; font-style: italic; padding: 10px;")
        api_layout.addRow(data_note)

        tabs.addTab(api_widget, "API Keys")

        # Preferences tab
        pref_widget = QWidget()
        pref_layout = QFormLayout(pref_widget)

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light"])
        self.theme_combo.setCurrentText(self.settings.value("theme", "Dark"))
        pref_layout.addRow("Theme:", self.theme_combo)

        self.llm_provider = QComboBox()
        self.llm_provider.addItems(["OpenAI", "Anthropic", "Google Gemini", "HuggingFace", "LM Studio"])
        self.llm_provider.setCurrentText(self.settings.value("llm_provider", "OpenAI"))
        pref_layout.addRow("LLM Provider:", self.llm_provider)

        self.update_interval = QSpinBox()
        self.update_interval.setRange(1, 60)
        self.update_interval.setValue(int(self.settings.value("update_interval", 5)))
        self.update_interval.setSuffix(" minutes")
        pref_layout.addRow("Update Interval:", self.update_interval)

        tabs.addTab(pref_widget, "Preferences")

        layout.addWidget(tabs)

        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok |
            QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.save_settings)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def save_settings(self):
        """Save settings"""
        self.settings.setValue("openai_key", self.openai_key.text())
        self.settings.setValue("anthropic_key", self.anthropic_key.text())
        self.settings.setValue("gemini_key", self.gemini_key.text())
        self.settings.setValue("huggingface_key", self.huggingface_key.text())
        self.settings.setValue("lmstudio_url", self.lmstudio_url.text())
        self.settings.setValue("theme", self.theme_combo.currentText())
        self.settings.setValue("llm_provider", self.llm_provider.currentText())
        self.settings.setValue("update_interval", self.update_interval.value())

        self.accept()


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self, startup_config: Optional[Dict[str, Any]] = None):
        super().__init__()

        # Store startup config
        self.startup_config = startup_config or {}

        # Initialize settings
        self.settings = QSettings("StockAnalyzer", "Settings")

        # Apply startup config to settings
        if startup_config:
            self.apply_startup_config(startup_config)

        # Initialize services
        self.init_services()

        # UI setup
        self.setWindowTitle("AI-Powered Stock Analysis")
        self.setGeometry(100, 100, 1400, 900)

        # Apply theme
        self.apply_theme()

        # Initialize UI
        self.init_ui()

        # Restore window state
        self.restore_state()

        # Start timers
        self.start_timers()

        # Show welcome message
        self.statusBar().showMessage("Welcome to AI Stock Analyzer", 3000)

        # Load initial data if not in demo mode
        if not self.startup_config.get('demo_mode', False):
            if not self.startup_config.get('skip_market_data', False):
                QTimer.singleShot(100, self.load_initial_data)

    def apply_startup_config(self, config: Dict[str, Any]):
        """Apply startup configuration to settings"""
        provider = config.get('provider', 'openai')

        # Set LLM provider
        self.settings.setValue('llm_provider', provider.title())

        # Set API keys and settings based on provider
        if provider == 'openai':
            self.settings.setValue('openai_key', config.get('api_key', ''))
            self.settings.setValue('openai_model', config.get('model', 'gpt-4'))
        elif provider == 'anthropic':
            self.settings.setValue('anthropic_key', config.get('api_key', ''))
            self.settings.setValue('anthropic_model', config.get('model', ''))
        elif provider == 'gemini':
            self.settings.setValue('gemini_key', config.get('api_key', ''))
        elif provider == 'huggingface':
            self.settings.setValue('huggingface_key', config.get('api_key', ''))
            self.settings.setValue('huggingface_model', config.get('model', ''))
        elif provider == 'lmstudio':
            settings = config.get('settings', {})
            self.settings.setValue('lmstudio_url', settings.get('url', 'http://localhost:1234'))
            self.settings.setValue('lmstudio_model', config.get('model', ''))

    def load_initial_data(self):
        """Load initial market data"""
        self.statusBar().showMessage("Loading market data...", 0)

        # Update market overview
        self.update_market_data()

        # Update watchlist
        self.update_watchlist_display()

        self.statusBar().showMessage("Ready", 3000)

    def init_services(self):
        """Initialize backend services"""
        # Market data provider
        self.data_provider = MarketDataProvider()

        # LLM provider
        self.llm_provider = self.create_llm_provider()

        # Analyzers
        self.stock_analyzer = StockAnalyzer(self.llm_provider)
        self.nl_screener = NaturalLanguageScreener(self.llm_provider)
        self.stock_screener = StockScreener(self.data_provider, self.nl_screener)
        # Portfolio manager
        self.portfolio_manager = PortfolioManager(self.data_provider)
        # Current state
        self.current_symbol = None
        self.current_analysis = None
        self.watchlist = []

    def create_llm_provider(self) -> LLMProvider:
        """Create LLM provider based on settings"""
        provider_name = self.settings.value("llm_provider", "OpenAI")

        if provider_name == "OpenAI":
            api_key = self.settings.value("openai_key", "")
            if api_key:
                return OpenAIProvider(api_key)
        elif provider_name == "Anthropic":
            api_key = self.settings.value("anthropic_key", "")
            if api_key:
                return AnthropicProvider(api_key)
        elif provider_name == "Google Gemini":
            api_key = self.settings.value("gemini_key", "")
            if api_key:
                return GeminiProvider(api_key)
        elif provider_name == "HuggingFace":
            api_key = self.settings.value("huggingface_key", "")
            if api_key:
                return HuggingFaceProvider(api_key)
        elif provider_name == "LM Studio":
            url = self.settings.value("lmstudio_url", "http://localhost:1234")
            return LMStudioProvider(url)

        # Fallback
        QMessageBox.warning(self, "Warning",
                            "No LLM provider configured. Please set API keys in settings.")
        return None

    def init_ui(self):
        """Initialize UI components"""
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Create menu bar
        self.create_menu_bar()

        # Create toolbar
        self.create_toolbar()

        # Create main content area
        content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel - Watchlist and Portfolio
        left_panel = self.create_left_panel()
        content_splitter.addWidget(left_panel)

        # Center panel - Main content
        self.content_tabs = QTabWidget()
        self.content_tabs.setTabsClosable(True)
        self.content_tabs.tabCloseRequested.connect(self.close_tab)

        # Add default tabs
        self.screener_tab = self.create_screener_tab()
        self.content_tabs.addTab(self.screener_tab, "Screener")

        content_splitter.addWidget(self.content_tabs)

        # Right panel - Market Overview
        right_panel = self.create_right_panel()
        content_splitter.addWidget(right_panel)

        # Set splitter proportions
        content_splitter.setSizes([250, 900, 250])

        main_layout.addWidget(content_splitter)

        # Create status bar
        self.create_status_bar()

    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        new_portfolio_action = file_menu.addAction("New Portfolio...")
        new_portfolio_action.triggered.connect(self.new_portfolio)

        import_action = file_menu.addAction("Import Portfolio...")
        import_action.triggered.connect(self.import_portfolio)

        export_action = file_menu.addAction("Export Portfolio...")
        export_action.triggered.connect(self.export_portfolio)

        file_menu.addSeparator()

        settings_action = file_menu.addAction("Settings...")
        settings_action.setShortcut(QKeySequence.StandardKey.Preferences)
        settings_action.triggered.connect(self.show_settings)

        file_menu.addSeparator()

        exit_action = file_menu.addAction("Exit")
        exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        exit_action.triggered.connect(self.close)

        # Analysis menu
        analysis_menu = menubar.addMenu("&Analysis")

        analyze_stock_action = analysis_menu.addAction("Analyze Stock...")
        analyze_stock_action.setShortcut(QKeySequence("Ctrl+A"))
        analyze_stock_action.triggered.connect(self.analyze_stock_dialog)

        screen_stocks_action = analysis_menu.addAction("Screen Stocks...")
        screen_stocks_action.setShortcut(QKeySequence("Ctrl+S"))
        screen_stocks_action.triggered.connect(self.show_screener)

        # View menu
        view_menu = menubar.addMenu("&View")

        # Theme submenu
        theme_menu = view_menu.addMenu("Theme")

        dark_theme_action = theme_menu.addAction("Dark")
        dark_theme_action.setCheckable(True)
        dark_theme_action.setChecked(self.settings.value("theme", "Dark") == "Dark")
        dark_theme_action.triggered.connect(lambda: self.change_theme("Dark"))

        light_theme_action = theme_menu.addAction("Light")
        light_theme_action.setCheckable(True)
        light_theme_action.setChecked(self.settings.value("theme", "Dark") == "Light")
        light_theme_action.triggered.connect(lambda: self.change_theme("Light"))

        # Help menu
        help_menu = menubar.addMenu("&Help")

        about_action = help_menu.addAction("About...")
        about_action.triggered.connect(self.show_about)

    def create_toolbar(self):
        """Create main toolbar"""
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Search box
        search_label = QLabel("Symbol: ")
        toolbar.addWidget(search_label)

        self.symbol_search = QLineEdit()
        self.symbol_search.setPlaceholderText("Enter stock symbol...")
        self.symbol_search.setMaximumWidth(150)
        self.symbol_search.returnPressed.connect(self.quick_analyze)
        toolbar.addWidget(self.symbol_search)

        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self.quick_analyze)
        toolbar.addWidget(analyze_btn)

        toolbar.addSeparator()

        # Screener search
        self.screener_search = QLineEdit()
        self.screener_search.setPlaceholderText("Natural language screening...")
        self.screener_search.setMinimumWidth(300)
        self.screener_search.returnPressed.connect(self.quick_screen)
        toolbar.addWidget(self.screener_search)

        screen_btn = QPushButton("Screen")
        screen_btn.clicked.connect(self.quick_screen)
        toolbar.addWidget(screen_btn)

    def create_left_panel(self) -> QWidget:
        """Create left panel with watchlist and portfolio"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Create tabs
        tabs = QTabWidget()

        # Watchlist tab
        watchlist_widget = QWidget()
        watchlist_layout = QVBoxLayout(watchlist_widget)

        # Watchlist controls
        watchlist_controls = QHBoxLayout()

        add_btn = QPushButton("Add")
        add_btn.clicked.connect(self.add_to_watchlist)
        watchlist_controls.addWidget(add_btn)

        remove_btn = QPushButton("Remove")
        remove_btn.clicked.connect(self.remove_from_watchlist)
        watchlist_controls.addWidget(remove_btn)

        watchlist_controls.addStretch()
        watchlist_layout.addLayout(watchlist_controls)

        # Watchlist
        self.watchlist_widget = QListWidget()
        self.watchlist_widget.itemDoubleClicked.connect(self.watchlist_item_clicked)
        watchlist_layout.addWidget(self.watchlist_widget)

        tabs.addTab(watchlist_widget, "Watchlist")

        # Portfolio tab
        portfolio_widget = QWidget()
        portfolio_layout = QVBoxLayout(portfolio_widget)

        # Portfolio selector
        self.portfolio_selector = QComboBox()
        self.portfolio_selector.currentTextChanged.connect(self.portfolio_changed)
        portfolio_layout.addWidget(self.portfolio_selector)

        # Portfolio summary
        self.portfolio_summary = PortfolioWidget()
        portfolio_layout.addWidget(self.portfolio_summary)

        tabs.addTab(portfolio_widget, "Portfolio")

        layout.addWidget(tabs)

        return panel

    def create_right_panel(self) -> QWidget:
        """Create right panel with market overview"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Market overview
        market_label = QLabel("Market Overview")
        market_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        layout.addWidget(market_label)

        # Market indices
        self.market_indices = QWidget()
        indices_layout = QVBoxLayout(self.market_indices)

        # Add sample indices
        for index, name in [("SPY", "S&P 500"), ("QQQ", "Nasdaq"), ("DIA", "Dow Jones")]:
            index_widget = self.create_index_widget(index, name)
            indices_layout.addWidget(index_widget)

        layout.addWidget(self.market_indices)

        # Top movers
        movers_label = QLabel("Top Movers")
        movers_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(movers_label)

        self.top_movers_list = QListWidget()
        layout.addWidget(self.top_movers_list)

        layout.addStretch()

        return panel

    def create_index_widget(self, symbol: str, name: str) -> QWidget:
        """Create market index display widget"""
        widget = QFrame()
        widget.setFrameStyle(QFrame.Shape.Box)
        layout = QVBoxLayout(widget)

        name_label = QLabel(name)
        name_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(name_label)

        price_label = QLabel("Loading...")
        price_label.setObjectName(f"{symbol}_price")
        layout.addWidget(price_label)

        change_label = QLabel("--")
        change_label.setObjectName(f"{symbol}_change")
        layout.addWidget(change_label)

        return widget

    def create_screener_tab(self) -> QWidget:
        """Create screener tab"""
        widget = QWidget()
        layout = QVBoxLayout(widget)

        # Screening presets
        preset_layout = QHBoxLayout()

        preset_label = QLabel("Presets:")
        preset_layout.addWidget(preset_label)

        preset_buttons = [
            ("Dividend Stocks", "dividend stocks with yield over 3%"),
            ("Growth Stocks", "high growth stocks with revenue growth over 15%"),
            ("Value Stocks", "undervalued stocks with PE under 15"),
            ("Tech Leaders", "large cap technology stocks")
        ]

        for name, query in preset_buttons:
            btn = QPushButton(name)
            btn.clicked.connect(lambda checked, q=query: self.run_screen(q))
            preset_layout.addWidget(btn)

        preset_layout.addStretch()
        layout.addLayout(preset_layout)

        # Results table
        self.screener_results = ScreenerResultsTable()
        self.screener_results.analyze_clicked.connect(self.analyze_from_screener)
        layout.addWidget(self.screener_results)

        return widget

    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Connection status
        self.connection_label = QLabel("Connected")
        self.connection_label.setStyleSheet("color: green;")
        self.status_bar.addPermanentWidget(self.connection_label)

        # Update time
        self.update_label = QLabel("Last update: --")
        self.status_bar.addPermanentWidget(self.update_label)

    def apply_theme(self):
        """Apply selected theme"""
        theme = self.settings.value("theme", "Dark")

        if theme == "Dark":
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #14151A;
                }
                QWidget {
                    background-color: #1B1D23;
                    color: #E2E4E9;
                }
                QTabWidget::pane {
                    border: 1px solid #303239;
                    background-color: #1B1D23;
                }
                QTabBar::tab {
                    background-color: #14151A;
                    padding: 8px 16px;
                    margin-right: 2px;
                }
                QTabBar::tab:selected {
                    background-color: #3B82F6;
                }
                QListWidget {
                    background-color: #14151A;
                    border: 1px solid #303239;
                }
                QListWidget::item:hover {
                    background-color: #2A2B32;
                }
                QListWidget::item:selected {
                    background-color: #3B82F6;
                }
                QPushButton {
                    background-color: #2A2B32;
                    border: 1px solid #303239;
                    padding: 5px 15px;
                    border-radius: 4px;
                }
                QPushButton:hover {
                    background-color: #373842;
                }
                QPushButton:pressed {
                    background-color: #1B1D23;
                }
                QLineEdit {
                    background-color: #14151A;
                    border: 1px solid #303239;
                    padding: 5px;
                    border-radius: 4px;
                }
                QComboBox {
                    background-color: #14151A;
                    border: 1px solid #303239;
                    padding: 5px;
                    border-radius: 4px;
                }
                QGroupBox {
                    border: 1px solid #303239;
                    border-radius: 4px;
                    margin-top: 8px;
                    padding-top: 8px;
                }
                QGroupBox::title {
                    subcontrol-origin: margin;
                    left: 10px;
                    padding: 0 5px 0 5px;
                }
            """)
        else:
            # Light theme
            self.setStyleSheet("""
                QMainWindow {
                    background-color: #FFFFFF;
                }
                QWidget {
                    background-color: #F9FAFB;
                    color: #111827;
                }
            """)

    def start_timers(self):
        """Start update timers"""
        # Market data update timer
        self.market_timer = QTimer()
        self.market_timer.timeout.connect(self.update_market_data)
        interval = self.settings.value("update_interval", 5) * 60 * 1000  # Convert to ms
        self.market_timer.start(interval)

        # Update immediately
        self.update_market_data()

    def restore_state(self):
        """Restore window state from settings"""
        geometry = self.settings.value("window_geometry")
        if geometry:
            self.restoreGeometry(QByteArray(geometry))

        state = self.settings.value("window_state")
        if state:
            self.restoreState(QByteArray(state))

        # Restore watchlist
        watchlist = self.settings.value("watchlist", [])
        if isinstance(watchlist, list):
            self.watchlist = watchlist
            self.update_watchlist_display()

    def save_state(self):
        """Save window state to settings"""
        self.settings.setValue("window_geometry", self.saveGeometry())
        self.settings.setValue("window_state", self.saveState())
        self.settings.setValue("watchlist", self.watchlist)

    # Slot implementations
    def show_settings(self):
        """Show settings dialog"""
        dialog = SettingsDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # Reload services with new settings
            self.llm_provider = self.create_llm_provider()
            self.stock_analyzer = StockAnalyzer(self.llm_provider)
            self.nl_screener = NaturalLanguageScreener(self.llm_provider)

            # Apply theme
            self.apply_theme()

            # Update timer interval
            interval = self.settings.value("update_interval", 5) * 60 * 1000
            self.market_timer.setInterval(interval)

    def change_theme(self, theme: str):
        """Change application theme"""
        self.settings.setValue("theme", theme)
        self.apply_theme()

    def toggle_theme(self):
        """Toggle between dark and light theme"""
        current_theme = self.settings.value("theme", "Dark")
        new_theme = "Light" if current_theme == "Dark" else "Dark"
        self.change_theme(new_theme)

        # Update button icon
        self.theme_btn.setText("☀️" if new_theme == "Light" else "🌙")

    def show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About AI Stock Analyzer",
            "AI-Powered Stock Analysis Desktop Application\n\n"
            "Version 1.0\n\n"
            "A sophisticated desktop application for stock market analysis "
            "that leverages LLM capabilities to filter stocks, perform "
            "fundamental and technical analysis, and provide investment insights."
        )

    def quick_analyze(self):
        """Quick analyze from toolbar"""
        symbol = self.symbol_search.text().strip().upper()
        if symbol:
            self.analyze_stock(symbol)

    def analyze_stock_dialog(self):
        """Show analyze stock dialog"""
        symbol, ok = QInputDialog.getText(
            self, "Analyze Stock", "Enter stock symbol:"
        )
        if ok and symbol:
            self.analyze_stock(symbol.upper())

    def analyze_stock(self, symbol: str):
        """Analyze a stock"""
        # Check if analysis tab already exists
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == f"Analysis: {symbol}":
                self.content_tabs.setCurrentIndex(i)
                return

        # Create new analysis tab
        analysis_widget = AnalysisDashboard()
        tab_index = self.content_tabs.addTab(
            analysis_widget, f"Analysis: {symbol}"
        )
        self.content_tabs.setCurrentIndex(tab_index)

        # Show loading
        loading = LoadingOverlay(analysis_widget)
        loading.resize(analysis_widget.size())
        loading.show_loading()

        # Create async task
        async def analyze():
            try:
                # Get stock data
                stock_data = await self.data_provider.get_stock_data(symbol)
                if not stock_data:
                    raise ValueError(f"Could not fetch data for {symbol}")

                # Get additional data
                financial_metrics = await self.data_provider.get_financial_statements(symbol)
                technical_indicators = await self.data_provider.get_technical_indicators(symbol)
                news_items = await self.data_provider.get_news(symbol)

                # Run analysis
                analysis = await self.stock_analyzer.analyze_stock(
                    stock_data,
                    financial_metrics,
                    technical_indicators,
                    news_items,
                    AnalysisType.COMPREHENSIVE
                )

                return analysis

            except Exception as e:
                logger.error(f"Analysis error: {e}")
                raise

        # Run async task
        worker = AsyncWorker(analyze())

        def on_analyze_complete(analysis_result: Optional[AnalysisResult]): # Added type hint
            try:
                loading.hide_loading()
                if analysis_result and isinstance(analysis_result, AnalysisResult):
                    if hasattr(analysis_widget, 'display_analysis'):
                        analysis_widget.display_analysis(analysis_result)
                    else: # pragma: no cover
                        logger.error("AnalysisDashboard is missing display_analysis method.")
                        QMessageBox.critical(self, "UI Error", "Cannot display analysis results: component is invalid.")
                        if self.content_tabs.indexOf(analysis_widget) != -1: self.content_tabs.removeTab(tab_index)
                        return

                    self.current_symbol = symbol
                    self.current_analysis = analysis_result # Store the successful analysis
                    self.statusBar().showMessage(f"Analysis complete for {symbol}", 3000)
                elif analysis_result is None:
                    logger.warning(f"Analysis for {symbol} returned no data (None).")
                    QMessageBox.warning(self, "Analysis Incomplete", f"Analysis for {symbol} did not return any data.")
                    if self.content_tabs.indexOf(analysis_widget) != -1: self.content_tabs.removeTab(tab_index)
            except Exception as e: # Catch errors during UI update itself # pragma: no cover
                logger.exception(f"Error displaying analysis results for {symbol}: {e}")
                QMessageBox.critical(self, "UI Error", f"Failed to display analysis results for {symbol}: {e}")
                if self.content_tabs.indexOf(analysis_widget) != -1 : self.content_tabs.removeTab(tab_index)

        def on_analyze_error(error_message: str):
            loading.hide_loading()
            logger.error(f"Async analysis error for {symbol}: {error_message}", exc_info=False)
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze {symbol}:\n{error_message.splitlines()[0]}")
            if self.content_tabs.indexOf(analysis_widget) != -1 : self.content_tabs.removeTab(tab_index)

        worker.result_ready.connect(on_analyze_complete)
        worker.error_occurred.connect(on_analyze_error)
        worker.start()

    def analyze_from_screener(self, symbol: str):
        """Analyze stock from screener results"""
        if not symbol: # pragma: no cover
            QMessageBox.warning(self, "Analyze Stock", "No symbol provided.")
            return
        self.analyze_stock(symbol)

    def show_screener(self):
        """Show screener tab"""
        # Find screener tab
        for i in range(self.content_tabs.count()):
            if self.content_tabs.tabText(i) == "Screener":
                self.content_tabs.setCurrentIndex(i)
                return

    def quick_screen(self):
        """Quick screen from toolbar"""
        query = self.screener_search.text().strip()
        if query:
            self.run_screen(query)

    def run_screen(self, query: str):
        """Run stock screening"""
        self.show_screener()

        # Show progress
        progress = QProgressDialog("Screening stocks...", "Cancel", 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setAutoClose(True)
        progress.show()

        # Create async task
        async def screen():
            return await self.stock_screener.screen_stocks(query)

        # Run async task
        worker = AsyncWorker(screen_coro()) # Renamed coro in previous patch

        def on_screen_complete(screen_result: Optional[StockScreenResult]):
            try:
                progress.close()
                if not hasattr(self, 'screener_results') or not self.screener_results: # pragma: no cover
                    logger.error("Screener results table widget is not available.")
                    return

                self.screener_results.clear_stocks()

                if screen_result and isinstance(screen_result, StockScreenResult):
                    if screen_result.matches:
                        for stock_data_item in screen_result.matches:
                            if isinstance(stock_data_item, StockData):
                                try:
                                    self.screener_results.add_stock(stock_data_item)
                                except Exception as e_add: # pragma: no cover
                                    logger.exception(f"Error adding stock {getattr(stock_data_item, 'symbol', 'Unknown')} to screener UI: {e_add}")
                                    QMessageBox.warning(self, "UI Update Error", f"Could not display stock {getattr(stock_data_item, 'symbol', 'Unknown')} in results.")
                            else:
                                logger.warning(f"Invalid item type '{type(stock_data_item)}' in screener matches, expected StockData.")

                        status_msg = f"Screening found {screen_result.total_count} stocks in {screen_result.execution_time:.1f}s"
                        self.statusBar().showMessage(status_msg, 5000)
                        if not screen_result.matches:
                             QMessageBox.information(self, "Screening Results", "No stocks matched your criteria to display (though some might have been found).")
                    else:
                        self.statusBar().showMessage("No stocks matched your screening criteria.", 3000)
                        QMessageBox.information(self, "Screening Results", "No stocks matched your criteria.")
                elif screen_result is None:
                     logger.info("Stock screening returned None (no data).")
                     QMessageBox.information(self, "Screening Results", "Screening completed but returned no data.")
            except AttributeError as ae: # pragma: no cover
                 logger.exception(f"Screener UI component missing: {ae}")
                 QMessageBox.critical(self, "UI Error", "Screener display component is missing. Cannot show results.")
            except Exception as e: # Catch any other errors during UI update # pragma: no cover
                logger.exception(f"Error updating UI with screening results: {e}")
                QMessageBox.critical(self, "UI Error", f"Failed to display screening results: {e}")

        def on_screen_error(error_message: str):
            progress.close()
            logger.error(f"Screening operation failed: {error_message}", exc_info=False)
            QMessageBox.critical(self, "Screening Error", f"Failed to screen stocks:\n{error_message.splitlines()[0]}")


        worker.result_ready.connect(on_screen_complete)
        worker.error_occurred.connect(on_screen_error)
        worker.start()


    def close_tab(self, index: int):
        """Close a content tab"""
        if self.content_tabs.tabText(index) != "Screener":  # Don't close screener
            self.content_tabs.removeTab(index)

    def add_to_watchlist(self):
        """Add symbol to watchlist"""
        symbol, ok = QInputDialog.getText(
            self, "Add to Watchlist", "Enter stock symbol:"
        )
        if ok and symbol:
            symbol = symbol.upper()
            if symbol not in self.watchlist:
                self.watchlist.append(symbol)
                self.update_watchlist_display()
                self.save_state()

    def remove_from_watchlist(self):
        """Remove selected symbol from watchlist"""
        current_item = self.watchlist_widget.currentItem()
        if current_item:
            symbol = current_item.text().split()[0]  # Get symbol part
            self.watchlist.remove(symbol)
            self.update_watchlist_display()
            self.save_state()

    def watchlist_item_clicked(self, item: QListWidgetItem):
        """Handle watchlist item double-click"""
        symbol = item.text().split()[0]  # Get symbol part
        self.analyze_stock(symbol)

    def update_watchlist_display(self):
        """Update watchlist display"""
        self.watchlist_widget.clear()

        # Create async task to get prices (renamed coro)
        async def get_watchlist_data_coro():
            # Gathers (symbol, price_or_error_string) tuples
            results = []
            if not hasattr(self, 'data_provider') or not self.data_provider: # Service check
                logger.error("Watchlist: MarketDataProvider not available for fetching prices.")
                # Return current watchlist symbols with an error message for each
                return [(s, "Data Service Error") for s in self.watchlist]

            for symbol_wl in self.watchlist:
                try:
                    stock_data_wl = await self.data_provider.get_stock_data(symbol_wl)
                    if stock_data_wl and stock_data_wl.current_price is not None:
                        results.append((symbol_wl, stock_data_wl.current_price))
                    elif stock_data_wl:
                        results.append((symbol_wl, "N/P"))
                    else:
                        results.append((symbol_wl, "No Data")) # Changed from "Error" to be more specific
                except Exception as e: # pragma: no cover
                    logger.error(f"Error fetching watchlist data for {symbol_wl}: {e}", exc_info=True)
                    results.append((symbol_wl, "Fetch Error"))
            return results

        worker = AsyncWorker(get_watchlist_data_coro())

        def on_watchlist_data_ready(data_results: list):
            try:
                if not hasattr(self, 'watchlist_widget') or not self.watchlist_widget: # pragma: no cover
                    logger.error("Watchlist widget not available for UI update.")
                    return
                self.watchlist_widget.clear()

                if not data_results and self.watchlist:
                    self.watchlist_widget.addItem("Could not fetch data for watchlist items.")
                elif not self.watchlist:
                     self.watchlist_widget.addItem("Watchlist is empty. Add symbols.")

                for symbol_res, price_or_error in data_results:
                    item = QListWidgetItem()
                    item_text = f"{symbol_res}"
                    if isinstance(price_or_error, (float, int)) and pd.notna(price_or_error):
                        item_text += f" - ${price_or_error:.2f}"
                    elif isinstance(price_or_error, str):
                        item_text += f" - ({price_or_error})"
                    else:
                        item_text += " - (Data N/A)"
                    item.setText(item_text)
                    self.watchlist_widget.addItem(item)
            except AttributeError as ae: # pragma: no cover
                 logger.exception(f"Watchlist UI component (e.g. watchlist_widget) missing: {ae}")
            except Exception as e: # pragma: no cover
                logger.exception(f"Error updating watchlist UI: {e}")
                if hasattr(self, 'watchlist_widget') and self.watchlist_widget:
                     try: self.watchlist_widget.addItem("Error updating display.")
                     except Exception: pass

        def on_watchlist_error(error_msg: str): # pragma: no cover
            logger.error(f"Async error updating watchlist: {error_msg}", exc_info=False)
            if hasattr(self, 'watchlist_widget') and self.watchlist_widget:
                try:
                    self.watchlist_widget.clear()
                    self.watchlist_widget.addItem(f"Error updating watchlist: {error_msg.split(':')[0]}.")
                except Exception: pass

        worker.result_ready.connect(on_watchlist_data_ready)
        worker.error_occurred.connect(on_watchlist_error)
        worker.start()


    def new_portfolio(self):
        """Create new portfolio"""
        name, ok = QInputDialog.getText(
            self, "New Portfolio", "Enter portfolio name:"
        )
        if ok and name:
            initial_cash, ok = QInputDialog.getDouble(
                self, "Initial Cash", "Enter initial cash balance:",
                value=10000, min=0, decimals=2
            )
            if ok:
                portfolio = self.portfolio_manager.create_portfolio(name, initial_cash)
                self.update_portfolio_selector()
                self.statusBar().showMessage(f"Created portfolio: {name}", 3000)

    def import_portfolio(self):
        """Import portfolio from file"""
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Import Portfolio", "", "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if file_name:
            try:
                with open(file_name, 'r') as f:
                    data = f.read()

                format = 'csv' if file_name.endswith('.csv') else 'json'
                name = QFileInfo(file_name).baseName()

                portfolio = self.portfolio_manager.import_portfolio(name, data, format)
                self.update_portfolio_selector()
                self.statusBar().showMessage(f"Imported portfolio: {name}", 3000)

            except Exception as e:
                QMessageBox.critical(self, "Import Error", f"Failed to import portfolio:\n{e}")

    def export_portfolio(self):
        """Export current portfolio"""
        current_portfolio = self.portfolio_selector.currentText()
        if not current_portfolio:
            QMessageBox.warning(self, "No Portfolio", "Please select a portfolio to export.")
            return

        file_name, _ = QFileDialog.getSaveFileName(
            self, "Export Portfolio", f"{current_portfolio}.csv",
            "CSV Files (*.csv);;JSON Files (*.json)"
        )
        if file_name:
            try:
                format = 'csv' if file_name.endswith('.csv') else 'json'
                data = self.portfolio_manager.export_portfolio(current_portfolio, format)

                with open(file_name, 'w') as f:
                    f.write(data)

                self.statusBar().showMessage(f"Exported portfolio to {file_name}", 3000)

            except Exception as e:
                QMessageBox.critical(self, "Export Error", f"Failed to export portfolio:\n{e}")

    def update_portfolio_selector(self):
        """Update portfolio dropdown"""
        current = self.portfolio_selector.currentText()
        self.portfolio_selector.clear()

        portfolios = self.portfolio_manager.list_portfolios()
        self.portfolio_selector.addItems(portfolios)

        if current in portfolios:
            self.portfolio_selector.setCurrentText(current)

    def portfolio_changed(self, name: str):
        """Handle portfolio selection change"""
        if name:
            async def update_portfolio_data_coro():
                if not self.portfolio_manager: return None

                portfolio = self.portfolio_manager.get_portfolio(name)
                if portfolio:
                    await self.portfolio_manager.update_portfolio(name)
                    return self.portfolio_manager.get_portfolio(name)
                logger.warning(f"Portfolio '{name}' not found before update attempt.")
                return None

            worker = AsyncWorker(update_portfolio_data_coro())

            def on_portfolio_update_complete(updated_portfolio: Optional[Portfolio]):
                try:
                    if not hasattr(self, 'portfolio_summary') or not self.portfolio_summary: # pragma: no cover
                        logger.error("Portfolio summary widget not available.")
                        return

                    if updated_portfolio and isinstance(updated_portfolio, Portfolio):
                        self.portfolio_summary.set_portfolio(updated_portfolio)
                    elif name:
                        logger.warning(f"Portfolio '{name}' data could not be retrieved or is None after update attempt.")
                        QMessageBox.warning(self, "Portfolio Update", f"Could not retrieve updated data for portfolio '{name}'. It might have been deleted or an error occurred.")
                        self.portfolio_summary.clear_portfolio()
                except AttributeError as ae: # pragma: no cover
                    logger.exception(f"Portfolio summary UI component missing: {ae}")
                    QMessageBox.critical(self, "UI Error", "Portfolio display component is missing.")
                except Exception as e: # pragma: no cover
                    logger.exception(f"Error updating portfolio UI for '{name}': {e}")
                    QMessageBox.critical(self, "UI Error", f"Failed to display portfolio '{name}': {e}")

            def on_portfolio_update_error(error_message: str):
                logger.error(f"Async error updating portfolio '{name}': {error_message}", exc_info=False)
                QMessageBox.critical(self, "Portfolio Update Error",
                                     f"Failed to update portfolio '{name}':\n{error_message.splitlines()[0]}")
                if hasattr(self, 'portfolio_manager') and self.portfolio_manager and \
                   hasattr(self, 'portfolio_summary') and self.portfolio_summary:
                    stale_portfolio = self.portfolio_manager.get_portfolio(name)
                    if stale_portfolio: self.portfolio_summary.set_portfolio(stale_portfolio, is_stale=True)
                    else: self.portfolio_summary.clear_portfolio()


            worker.result_ready.connect(on_portfolio_update_complete)
            worker.error_occurred.connect(on_portfolio_update_error)
            worker.start()
        except Exception as e: # pragma: no cover
            logger.exception(f"Error initiating portfolio change handling for '{name}': {e}")
            QMessageBox.critical(self, "Portfolio Error", f"Could not process portfolio change: {e}")


    def update_market_data(self):
        """Update market overview data"""
        if not self.data_provider:
            logger.warning("Market Overview: MarketDataProvider not available.")
            if hasattr(self, 'update_label') and self.update_label:
                self.update_label.setText("Last update: Service unavailable")
            return

        async def get_market_data_coro():
            if not self.data_provider: return None
            return await self.data_provider.get_market_overview()

        worker = AsyncWorker(get_market_data_coro())

        def on_market_data_ready(market_data_result: Optional[MarketOverview]):
            try:
                if not (market_data_result and isinstance(market_data_result, MarketOverview)):
                    logger.warning("Market overview update returned None or invalid data type.")
                    if hasattr(self, 'statusBar') and self.statusBar():
                        self.statusBar().showMessage("Failed to update market overview data.", 3000)
                    return

                index_name_to_symbol_map = {
                    "S&P 500": "SPY", "NASDAQ": "QQQ", "Dow Jones": "DIA",
                    "VIX": "VIX"
                }

                for display_name, data_val in market_data_result.indices.items():
                    obj_base_name = index_name_to_symbol_map.get(display_name, display_name.replace(" ", "_"))

                    price_label = self.findChild(QLabel, f"{obj_base_name}_price")
                    change_label = self.findChild(QLabel, f"{obj_base_name}_change")

                    if price_label and isinstance(data_val, dict) and 'price' in data_val:
                        price_val = data_val['price']
                        price_label.setText(f"${price_val:.2f}" if pd.notna(price_val) else "N/A")

                    if change_label and isinstance(data_val, dict) and 'change' in data_val and 'change_pct' in data_val:
                        change, change_pct = data_val['change'], data_val['change_pct']
                        if pd.notna(change) and pd.notna(change_pct):
                            color = "green" if change >= 0 else "red"
                            change_label.setText(f"{change:+.2f} ({change_pct:+.2f}%)")
                            change_label.setStyleSheet(f"color: {color};")
                        else:
                            change_label.setText("--")
                            change_label.setStyleSheet("")

                vix_widget_name = index_name_to_symbol_map.get("VIX", "VIX") + "_price"
                vix_price_label = self.findChild(QLabel, vix_widget_name)
                if vix_price_label and market_data_result.vix is not None and pd.notna(market_data_result.vix):
                    vix_price_label.setText(f"VIX: {market_data_result.vix:.2f}")
                elif vix_price_label:
                     vix_price_label.setText("VIX: N/A")


                if hasattr(self, 'update_label') and self.update_label:
                    self.update_label.setText(f"Last update: {datetime.now().strftime('%H:%M:%S')}")

            except AttributeError as ae: # pragma: no cover
                logger.exception(f"Market Data UI component missing: {ae}")
            except Exception as e: # pragma: no cover
                logger.exception(f"Error updating market data UI: {e}")
                if hasattr(self, 'statusBar') and self.statusBar():
                    self.statusBar().showMessage(f"Error updating market UI: {e}", 3000)

        def on_market_data_error(error_msg: str): # pragma: no cover
            logger.error(f"Async error updating market data: {error_msg}", exc_info=False)
            if hasattr(self, 'statusBar') and self.statusBar():
                self.statusBar().showMessage(f"Market data update failed: {error_msg.splitlines()[0]}.")


        worker.result_ready.connect(on_market_data_ready)
        worker.error_occurred.connect(on_market_data_error)
        worker.start()


    def closeEvent(self, event: QCloseEvent):
        """Handle window close event"""
        # Save state
        self.save_state()

        # Cleanup
        self.data_provider.cleanup()

        event.accept()


import asyncio


async def main():
    """Main application entry point"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("AI Stock Analyzer")
    app.setOrganizationName("StockAnalyzer")

    # Set application style
    app.setStyle("Fusion")

    # Import startup config
    from startup_config import StartupConfigDialog

    # Show startup configuration dialog
    config_dialog = StartupConfigDialog()

    config = None

    def on_config_complete(cfg):
        nonlocal config
        config = cfg

    config_dialog.config_complete.connect(on_config_complete)

    if config_dialog.exec() != QDialog.DialogCode.Accepted:
        # User cancelled
        sys.exit(0)

    if not config:
        QMessageBox.critical(None, "Error", "No configuration provided")
        sys.exit(1)

    # Create and show main window with config
    window = MainWindow(startup_config=config)
    window.show()

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    asyncio.run(main())
