from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from typing import Dict, List, Optional, Any
import json

# Assuming these imports work with your project structure
from src.view.recommendation_widget import RecommendationWidget
from src.view.stock_info_card import StockInfoCard


class ModernListWidget(QListWidget):
    """Enhanced list widget with modern styling and features"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlternatingRowColors(True)
        self.setWordWrap(True)
        self.setVerticalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def add_items_with_icons(self, items: List[str], icon_char: str = "•", color: str = "#ffffff"):
        """Add items with custom icons and colors"""
        for item in items:
            list_item = QListWidgetItem(f"{icon_char} {item}")
            list_item.setForeground(QColor(color))
            self.addItem(list_item)


class ModernTextDisplay(QTextEdit):
    """Enhanced text display with modern styling"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def set_placeholder_text(self, text: str):
        """Set placeholder text when no content is available"""
        self.setPlaceholderText(text)


class AnalysisDisplay(QWidget):
    """
    Modern, responsive three-column analysis display with beautiful styling
    and enhanced functionality that works seamlessly with the improved StockInfoCard
    """

    # Modern color scheme
    COLORS = {
        'bg_primary': '#6200d5',
        'bg_secondary': '#ffffff',
        'bg_card': '#f5f5f5',
        'border': '#d0d0d0',
        'primary': '#6200d5',
        'success': '#4caf50',
        'warning': '#ff9800',
        'danger': '#f44336',
        'text_primary': '#111111',
        'text_secondary': '#555555',
        'text_muted': '#757575'
    }

    # Data validation and formatting
    REQUIRED_STOCK_FIELDS = ['symbol', 'name', 'price', 'change']
    REQUIRED_ANALYSIS_FIELDS = ['recommendation', 'confidence', 'risk_level']

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loading_states = set()
        self._init_ui()
        self._apply_global_styling()
        self._setup_animations()

    def _init_ui(self):
        """Initialize the modern UI with responsive design"""
        # Main container with proper margins
        self.main_layout = QHBoxLayout(self)
        self.main_layout.setContentsMargins(16, 16, 16, 16)
        self.main_layout.setSpacing(16)

        # Create the three main columns
        self._create_left_column()
        self._create_center_column()
        self._create_right_column()

        # Add columns to main layout with equal weight
        self.main_layout.addLayout(self.left_column, 1)
        self.main_layout.addLayout(self.center_column, 1)
        self.main_layout.addLayout(self.right_column, 1)

    def _create_left_column(self):
        """Create the left column with stock information cards"""
        self.left_column = QVBoxLayout()
        self.left_column.setSpacing(12)

        # Stock Information Card
        self.info_card = StockInfoCard("📊 Stock Information", self._get_card_styles())

        # Key Metrics Card
        self.metrics_card = StockInfoCard("📈 Key Metrics", self._get_card_styles())

        # Technical Indicators Card
        self.technical_card = StockInfoCard("🔍 Technical Analysis", self._get_card_styles())

        # Add cards to layout
        cards = [self.info_card, self.metrics_card, self.technical_card]
        for card in cards:
            self.left_column.addWidget(card)

        # Add stretch to push cards to top
        self.left_column.addStretch()

    def _create_center_column(self):
        """Create the center column with recommendation and valuation"""
        self.center_column = QVBoxLayout()
        self.center_column.setSpacing(12)

        # Recommendation Widget (assuming it exists)
        try:
            self.rec_widget = RecommendationWidget()
            self.center_column.addWidget(self.rec_widget)
        except:
            # Fallback if RecommendationWidget is not available
            self.rec_widget = StockInfoCard("🎯 Recommendation", self._get_card_styles())
            self.center_column.addWidget(self.rec_widget)

        # Valuation Card
        self.valuation_card = StockInfoCard("💰 Valuation Analysis", self._get_card_styles())
        self.center_column.addWidget(self.valuation_card)

        # Summary Section
        self._create_summary_section()

        # Add stretch
        self.center_column.addStretch()

    def _create_summary_section(self):
        """Create the summary text section with modern styling"""
        # Summary title
        summary_title = QLabel("📝 Executive Summary")
        summary_title.setObjectName("sectionTitle")
        self.center_column.addWidget(summary_title)

        # Summary text display
        self.summary_text = ModernTextDisplay()
        self.summary_text.set_placeholder_text("Analysis summary will appear here...")
        self.summary_text.setMaximumHeight(120)
        self.summary_text.setMinimumHeight(80)

        self.center_column.addWidget(self.summary_text)

    def _create_right_column(self):
        """Create the right column with strengths and risks"""
        self.right_column = QVBoxLayout()
        self.right_column.setSpacing(12)

        # Strengths section
        strengths_title = QLabel("✅ Strengths")
        strengths_title.setObjectName("strengthsTitle")
        self.right_column.addWidget(strengths_title)

        self.strengths_list = ModernListWidget()
        self.strengths_list.setMaximumHeight(200)
        self.right_column.addWidget(self.strengths_list)

        # Risks section
        risks_title = QLabel("⚠️ Risks")
        risks_title.setObjectName("risksTitle")
        self.right_column.addWidget(risks_title)

        self.risks_list = ModernListWidget()
        self.risks_list.setMaximumHeight(200)
        self.right_column.addWidget(self.risks_list)

        # Add stretch
        self.right_column.addStretch()

    def _get_card_styles(self) -> Dict[str, Any]:
        """Get consistent styling for all cards"""
        return {
            'bg_color': self.COLORS['bg_card'],
            'border_color': self.COLORS['border'],
            'primary_color': self.COLORS['primary'],
            'text_primary': self.COLORS['text_primary'],
            'text_secondary': self.COLORS['text_secondary'],
            'hover_color': '#3a3a3a',
            'border_radius': 8,
            'padding': 12,
            'spacing': 8
        }

    def _apply_global_styling(self):
        """Apply consistent styling across the entire widget"""
        style = f"""
            AnalysisDisplay {{
                background-color: {self.COLORS['bg_primary']};
            }}
            
            QLabel#sectionTitle {{
                font-weight: bold;
                font-size: 14px;
                color: {self.COLORS['primary']};
                padding: 8px 0px 4px 0px;
                border-bottom: 1px solid {self.COLORS['border']};
                margin-bottom: 8px;
            }}
            
            QLabel#strengthsTitle {{
                font-weight: bold;
                font-size: 14px;
                color: {self.COLORS['success']};
                padding: 4px 0px;
            }}
            
            QLabel#risksTitle {{
                font-weight: bold;
                font-size: 14px;
                color: {self.COLORS['danger']};
                padding: 4px 0px;
            }}
            
            ModernListWidget {{
                background-color: {self.COLORS['bg_card']};
                border: 1px solid {self.COLORS['border']};
                border-radius: 8px;
                padding: 8px;
                font-size: 12px;
                color: {self.COLORS['text_primary']};
            }}
            
            ModernListWidget::item {{
                padding: 6px;
                border-radius: 4px;
                margin: 1px;
            }}
            
            ModernListWidget::item:hover {{
                background-color: rgba(255, 255, 255, 0.05);
            }}
            
            ModernListWidget::item:selected {{
                background-color: {self.COLORS['primary']};
                color: white;
            }}
            
            ModernTextDisplay {{
                background-color: {self.COLORS['bg_card']};
                border: 1px solid {self.COLORS['border']};
                border-radius: 8px;
                padding: 12px;
                font-size: 12px;
                color: {self.COLORS['text_primary']};
                line-height: 1.4;
            }}
            
            QScrollBar:vertical {{
                background: {self.COLORS['border']};
                width: 8px;
                border-radius: 4px;
            }}
            
            QScrollBar::handle:vertical {{
                background: {self.COLORS['primary']};
                border-radius: 4px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background: {self.COLORS['text_primary']};
            }}
        """
        self.setStyleSheet(style)

    def _setup_animations(self):
        """Setup smooth animations for transitions"""
        self.opacity_animation = QPropertyAnimation(self, b"windowOpacity")
        self.opacity_animation.setDuration(300)
        self.opacity_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def _validate_stock_data(self, data: Dict[str, Any]) -> bool:
        """Validate stock data structure"""
        if not isinstance(data, dict):
            return False

        for field in self.REQUIRED_STOCK_FIELDS:
            if field not in data:
                return False

        # Validate nested change data
        if 'change' in data and not isinstance(data['change'], dict):
            return False

        return True

    def _validate_analysis_data(self, data: Dict[str, Any]) -> bool:
        """Validate analysis data structure"""
        if not isinstance(data, dict):
            return False

        for field in self.REQUIRED_ANALYSIS_FIELDS:
            if field not in data:
                return False

        return True

    def _format_currency(self, value: float, symbol: str = "$") -> str:
        """Format currency values with proper formatting"""
        if value >= 1e12:
            return f"{symbol}{value/1e12:.2f}T"
        elif value >= 1e9:
            return f"{symbol}{value/1e9:.2f}B"
        elif value >= 1e6:
            return f"{symbol}{value/1e6:.2f}M"
        elif value >= 1e3:
            return f"{symbol}{value/1e3:.1f}K"
        else:
            return f"{symbol}{value:.2f}"

    def _format_percentage(self, value: float) -> str:
        """Format percentage values with proper sign"""
        sign = "+" if value >= 0 else ""
        return f"{sign}{value:.2f}%"

    def _get_color_for_value(self, value: float) -> str:
        """Get appropriate color for positive/negative values"""
        return self.COLORS['success'] if value >= 0 else self.COLORS['danger']

    def display_stock_data(self, stock_data: Dict[str, Any]):
        """
        Display stock information with enhanced formatting and error handling

        Args:
            stock_data: Dictionary containing stock information
        """
        if not self._validate_stock_data(stock_data):
            self._show_error("Invalid stock data provided")
            return

        try:
            # Clear previous data
            self.info_card.clear_items()
            self.metrics_card.clear_items()

            # Get price change color
            change_percent = stock_data.get('change', {}).get('percent', 0)
            price_color = self._get_color_for_value(change_percent)

            # Basic stock information
            info_items = [
                ("Symbol", stock_data.get('symbol', 'N/A')),
                ("Company", stock_data.get('name', 'N/A')),
                ("Sector", stock_data.get('sector', 'N/A')),
                ("Industry", stock_data.get('industry', 'N/A')),
                ("Price", self._format_currency(stock_data.get('price', 0))),
                ("Change", self._format_percentage(change_percent))
            ]

            for label, value in info_items:
                color = price_color if label in ('Price', 'Change') else None
                tooltip = f"Current {label.lower()}" if label in ('Price', 'Change') else None
                self.info_card.add_item(label, value, color, tooltip)

            # Key metrics with proper formatting
            if 'market_cap' in stock_data:
                self.metrics_card.add_item(
                    "Market Cap",
                    self._format_currency(stock_data['market_cap']),
                    tooltip="Total market value of all shares"
                )

            if 'pe_ratio' in stock_data:
                pe_ratio = stock_data['pe_ratio']
                pe_color = self.COLORS['success'] if 0 < pe_ratio < 25 else self.COLORS['warning']
                self.metrics_card.add_item("P/E Ratio", f"{pe_ratio:.2f}", pe_color)

            if 'volume' in stock_data:
                self.metrics_card.add_item("Volume", f"{stock_data['volume']:,}")

            if 'dividend_yield' in stock_data:
                div_yield = stock_data['dividend_yield']
                div_color = self.COLORS['success'] if div_yield > 2 else None
                self.metrics_card.add_item("Dividend Yield", f"{div_yield:.2f}%", div_color)

            # 52-week range
            if '52w_high' in stock_data and '52w_low' in stock_data:
                self.metrics_card.add_separator()
                self.metrics_card.add_item("52W High", self._format_currency(stock_data['52w_high']))
                self.metrics_card.add_item("52W Low", self._format_currency(stock_data['52w_low']))

        except Exception as e:
            self._show_error(f"Error displaying stock data: {str(e)}")

    def display_analysis(self, analysis_data: Dict[str, Any]):
        """
        Display analysis information with enhanced formatting

        Args:
            analysis_data: Dictionary containing analysis information
        """
        if not self._validate_analysis_data(analysis_data):
            self._show_error("Invalid analysis data provided")
            return

        try:
            # Update recommendation widget
            if hasattr(self.rec_widget, 'set_recommendation'):
                self.rec_widget.set_recommendation(
                    analysis_data.get('recommendation', 'HOLD'),
                    analysis_data.get('confidence', 0),
                    analysis_data.get('risk_level', 'MEDIUM')
                )
            else:
                # Fallback for custom recommendation display
                self._display_recommendation_fallback(analysis_data)

            # Technical indicators
            if 'technical_indicators' in analysis_data:
                self._display_technical_indicators(analysis_data['technical_indicators'])

            # Valuation analysis
            if 'valuation' in analysis_data:
                self._display_valuation_analysis(analysis_data)

            # Summary
            summary = analysis_data.get('summary', 'No summary available.')
            self.summary_text.setPlainText(summary)

            # Strengths and risks
            self._display_strengths_and_risks(analysis_data)

        except Exception as e:
            self._show_error(f"Error displaying analysis: {str(e)}")

    def _display_recommendation_fallback(self, analysis_data: Dict[str, Any]):
        """Fallback recommendation display if RecommendationWidget is not available"""
        self.rec_widget.clear_items()

        recommendation = analysis_data.get('recommendation', 'HOLD')
        confidence = analysis_data.get('confidence', 0)
        risk_level = analysis_data.get('risk_level', 'MEDIUM')

        # Color coding for recommendations
        rec_colors = {
            'BUY': self.COLORS['success'],
            'STRONG_BUY': self.COLORS['success'],
            'HOLD': self.COLORS['warning'],
            'SELL': self.COLORS['danger'],
            'STRONG_SELL': self.COLORS['danger']
        }

        self.rec_widget.add_item("Recommendation", recommendation, rec_colors.get(recommendation))
        self.rec_widget.add_item("Confidence", f"{confidence:.0f}%")
        self.rec_widget.add_item("Risk Level", risk_level)

    def _display_technical_indicators(self, technical_data: Dict[str, Any]):
        """Display technical indicators with color coding"""
        self.technical_card.clear_items()

        # Trend analysis
        if 'trend' in technical_data:
            trend = technical_data['trend']
            trend_colors = {
                'BULLISH': self.COLORS['success'],
                'BEARISH': self.COLORS['danger'],
                'NEUTRAL': self.COLORS['warning']
            }
            self.technical_card.add_item("Trend", trend, trend_colors.get(trend))

        # Support and resistance levels
        if 'support_level' in technical_data:
            self.technical_card.add_item(
                "Support",
                self._format_currency(technical_data['support_level'])
            )

        if 'resistance_level' in technical_data:
            self.technical_card.add_item(
                "Resistance",
                self._format_currency(technical_data['resistance_level'])
            )

    def _display_valuation_analysis(self, analysis_data: Dict[str, Any]):
        """Display valuation analysis with color coding"""
        self.valuation_card.clear_items()

        valuation = analysis_data.get('valuation', {})

        # Valuation assessment
        if 'assessment' in valuation:
            assessment = valuation['assessment']
            assessment_colors = {
                'UNDERVALUED': self.COLORS['success'],
                'OVERVALUED': self.COLORS['danger'],
                'FAIR': self.COLORS['warning']
            }
            self.valuation_card.add_item(
                "Assessment",
                assessment,
                assessment_colors.get(assessment)
            )

        # Price target
        if 'price_target' in analysis_data:
            self.valuation_card.add_item(
                "Target Price",
                self._format_currency(analysis_data['price_target'])
            )

        # Valuation reasoning
        if 'reasoning' in valuation:
            self.valuation_card.add_item("Reasoning", valuation['reasoning'])

    def _display_strengths_and_risks(self, analysis_data: Dict[str, Any]):
        """Display strengths and risks with proper formatting"""
        # Clear previous items
        self.strengths_list.clear()
        self.risks_list.clear()

        # Add strengths
        strengths = analysis_data.get('strengths', [])
        if strengths:
            self.strengths_list.add_items_with_icons(
                strengths, "✓", self.COLORS['success']
            )
        else:
            self.strengths_list.addItem("No strengths identified")

        # Add risks
        risks = analysis_data.get('risks', [])
        if risks:
            self.risks_list.add_items_with_icons(
                risks, "⚠", self.COLORS['danger']
            )
        else:
            self.risks_list.addItem("No risks identified")

    def _show_error(self, message: str):
        """Display error message in a user-friendly way"""
        print(f"AnalysisDisplay Error: {message}")  # For debugging
        # You could also show this in a status bar or notification

    def set_loading_state(self, section: str, loading: bool):
        """Set loading state for specific sections"""
        if loading:
            self._loading_states.add(section)
        else:
            self._loading_states.discard(section)

        # Apply loading state to relevant cards
        if section == "stock":
            self.info_card.set_loading_state(loading)
            self.metrics_card.set_loading_state(loading)
        elif section == "analysis":
            self.technical_card.set_loading_state(loading)
            self.valuation_card.set_loading_state(loading)

    def clear_all(self):
        """Clear all displayed information with smooth transition"""
        # Clear all cards
        for card in [self.info_card, self.metrics_card, self.technical_card, self.valuation_card]:
            card.clear_items()

        # Clear text and lists
        self.summary_text.clear()
        self.strengths_list.clear()
        self.risks_list.clear()

        # Reset recommendation widget
        if hasattr(self.rec_widget, 'set_recommendation'):
            self.rec_widget.set_recommendation("HOLD", 0, "MEDIUM")
        elif hasattr(self.rec_widget, 'clear_items'):
            self.rec_widget.clear_items()

    def export_data(self) -> Dict[str, Any]:
        """Export current display data as JSON-serializable dictionary"""
        # This could be used for saving/loading states
        return {
            "timestamp": QDateTime.currentDateTime().toString(),
            "has_data": len(self._loading_states) == 0,
            "loading_sections": list(self._loading_states)
        }


# Example usage and demo
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    # Create demo window
    window = QMainWindow()
    window.setWindowTitle("Enhanced Analysis Display Demo")
    window.setGeometry(100, 100, 1200, 800)

    # Create the analysis display
    analysis_display = AnalysisDisplay()
    window.setCentralWidget(analysis_display)

    # Demo data
    demo_stock_data = {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "price": 150.25,
        "change": {"percent": 2.5},
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "market_cap": 2.5e12,
        "pe_ratio": 25.5,
        "volume": 1234567,
        "dividend_yield": 0.5,
        "52w_high": 180.0,
        "52w_low": 120.0
    }

    demo_analysis_data = {
        "recommendation": "BUY",
        "confidence": 85,
        "risk_level": "MEDIUM",
        "technical_indicators": {
            "trend": "BULLISH",
            "support_level": 145.0,
            "resistance_level": 155.0
        },
        "valuation": {
            "assessment": "UNDERVALUED",
            "reasoning": "Strong fundamentals and growth prospects"
        },
        "price_target": 165.0,
        "summary": "Apple shows strong technical momentum with solid fundamentals. The stock is currently undervalued based on our analysis, presenting a good buying opportunity for long-term investors.",
        "strengths": [
            "Strong brand loyalty and ecosystem",
            "Consistent revenue growth",
            "Excellent cash flow generation",
            "Innovation in AI and services"
        ],
        "risks": [
            "Regulatory scrutiny in multiple markets",
            "High dependence on iPhone sales",
            "Supply chain vulnerabilities",
            "Intense competition in key markets"
        ]
    }

    # Display demo data
    QTimer.singleShot(100, lambda: analysis_display.display_stock_data(demo_stock_data))
    QTimer.singleShot(200, lambda: analysis_display.display_analysis(demo_analysis_data))

    window.show()
    sys.exit(app.exec())
