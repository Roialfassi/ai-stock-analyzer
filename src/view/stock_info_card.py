from typing import Optional, Dict, Any
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *


class StockInfoCard(QWidget):
    """
    A beautiful, reusable card widget for displaying key-value pairs
    with proper styling, animations, and robust functionality.
    """

    # Default styling constants (can be overridden)
    DEFAULT_STYLES = {
        'bg_color': '#2b2b2b',
        'border_color': '#404040',
        'primary_color': '#4a9eff',
        'text_primary': '#ffffff',
        'text_secondary': '#b0b0b0',
        'hover_color': '#3a3a3a',
        'border_radius': 8,
        'padding': 12,
        'spacing': 8
    }

    def __init__(self, title: str, styles: Optional[Dict[str, Any]] = None, parent=None):
        super().__init__(parent)

        # Merge custom styles with defaults
        self.styles = {**self.DEFAULT_STYLES, **(styles or {})}

        # Store references for proper cleanup
        self._item_widgets = []

        self._setup_ui(title)
        self._apply_styles()
        self._setup_animations()

    def _setup_ui(self, title: str):
        """Initialize the UI components"""
        # Main layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(
            self.styles['padding'], self.styles['padding'],
            self.styles['padding'], self.styles['padding']
        )
        self.main_layout.setSpacing(self.styles['spacing'])

        # Title label
        self.title_label = QLabel(title)
        self.title_label.setObjectName("cardTitle")
        self.main_layout.addWidget(self.title_label)

        # Content area with scroll support for many items
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)

        # Content widget inside scroll area
        self.content_widget = QWidget()
        self.content_layout = QVBoxLayout(self.content_widget)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.content_layout.setSpacing(4)

        self.scroll_area.setWidget(self.content_widget)
        self.main_layout.addWidget(self.scroll_area)

        # Add stretch to push content to top
        self.content_layout.addStretch()

    def _apply_styles(self):
        """Apply modern, beautiful styling"""
        card_style = f"""
            StockInfoCard {{
                background-color: {self.styles['bg_color']};
                border: 1px solid {self.styles['border_color']};
                border-radius: {self.styles['border_radius']}px;
            }}
            
            StockInfoCard:hover {{
                background-color: {self.styles['hover_color']};
                border-color: {self.styles['primary_color']};
            }}
            
            QLabel#cardTitle {{
                font-weight: bold;
                font-size: 14px;
                color: {self.styles['primary_color']};
                padding-bottom: 4px;
                border-bottom: 1px solid {self.styles['border_color']};
                margin-bottom: 8px;
            }}
            
            QLabel#itemKey {{
                color: {self.styles['text_secondary']};
                font-size: 12px;
            }}
            
            QLabel#itemValue {{
                font-weight: bold;
                font-size: 12px;
                color: {self.styles['text_primary']};
            }}
            
            QScrollArea {{
                background: transparent;
                border: none;
            }}
            
            QScrollBar:vertical {{
                background: {self.styles['border_color']};
                width: 8px;
                border-radius: 4px;
            }}
            
            QScrollBar::handle:vertical {{
                background: {self.styles['primary_color']};
                border-radius: 4px;
                min-height: 20px;
            }}
            
            QScrollBar::handle:vertical:hover {{
                background: {self.styles['text_primary']};
            }}
        """
        self.setStyleSheet(card_style)

    def _setup_animations(self):
        """Setup hover animations for smooth transitions"""
        self.hover_animation = QPropertyAnimation(self, b"windowOpacity")
        self.hover_animation.setDuration(200)
        self.hover_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

    def add_item(self, label: str, value: str, color: Optional[str] = None,
                 tooltip: Optional[str] = None) -> QWidget:
        """
        Add a key-value pair to the card

        Args:
            label: The key/label text
            value: The value text
            color: Optional custom color for the value
            tooltip: Optional tooltip text

        Returns:
            The created item widget for further customization
        """
        # Create container widget for the item
        item_widget = QWidget()
        item_layout = QHBoxLayout(item_widget)
        item_layout.setContentsMargins(4, 4, 4, 4)

        # Key label
        key_label = QLabel(label)
        key_label.setObjectName("itemKey")
        key_label.setMinimumWidth(120)
        key_label.setWordWrap(True)

        # Value label
        value_label = QLabel(str(value))
        value_label.setObjectName("itemValue")
        value_label.setWordWrap(True)
        value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

        # Apply custom color if provided
        if color:
            value_label.setStyleSheet(f"color: {color}; font-weight: bold; font-size: 12px;")

        # Add tooltip if provided
        if tooltip:
            item_widget.setToolTip(tooltip)

        # Add hover effect to individual items
        item_widget.setStyleSheet("""
            QWidget:hover {
                background-color: rgba(255, 255, 255, 0.05);
                border-radius: 4px;
            }
        """)

        # Layout the item
        item_layout.addWidget(key_label)
        item_layout.addStretch()
        item_layout.addWidget(value_label)

        # Insert before the stretch at the end
        insert_index = max(0, self.content_layout.count() - 1)
        self.content_layout.insertWidget(insert_index, item_widget)

        # Store reference for cleanup
        self._item_widgets.append(item_widget)

        return item_widget

    def add_separator(self):
        """Add a visual separator between items"""
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet(f"color: {self.styles['border_color']};")
        separator.setMaximumHeight(1)

        # Insert before the stretch
        insert_index = max(0, self.content_layout.count() - 1)
        self.content_layout.insertWidget(insert_index, separator)
        self._item_widgets.append(separator)

    def clear_items(self):
        """
        Properly clear all items from the card
        This fixes the original broken clear function
        """
        # Clear our reference list and properly delete widgets
        for widget in self._item_widgets:
            if widget and not widget.isHidden():
                widget.setParent(None)
                widget.deleteLater()

        self._item_widgets.clear()

        # Force layout update
        self.content_layout.update()

    def set_title(self, title: str):
        """Update the card title"""
        self.title_label.setText(title)

    def get_item_count(self) -> int:
        """Get the number of items in the card"""
        return len(self._item_widgets)

    def set_loading_state(self, loading: bool):
        """Show/hide loading indicator"""
        if loading:
            if not hasattr(self, '_loading_label'):
                self._loading_label = QLabel("Loading...")
                self._loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self._loading_label.setStyleSheet(f"color: {self.styles['text_secondary']};")

            self.clear_items()
            insert_index = max(0, self.content_layout.count() - 1)
            self.content_layout.insertWidget(insert_index, self._loading_label)
        else:
            if hasattr(self, '_loading_label'):
                self._loading_label.setParent(None)
                self._loading_label.deleteLater()
                delattr(self, '_loading_label')

    def enterEvent(self, event):
        """Handle mouse enter for hover effects"""
        self.hover_animation.setStartValue(1.0)
        self.hover_animation.setEndValue(0.95)
        self.hover_animation.start()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave for hover effects"""
        self.hover_animation.setStartValue(0.95)
        self.hover_animation.setEndValue(1.0)
        self.hover_animation.start()
        super().leaveEvent(event)


# Example usage and demo
if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)

    # Create demo window
    window = QWidget()
    window.setWindowTitle("StockInfoCard Demo")
    window.setGeometry(100, 100, 400, 600)
    window.setStyleSheet("background-color: #1e1e1e;")

    layout = QVBoxLayout(window)

    # Create card with default styling
    card1 = StockInfoCard("Stock Information")
    card1.add_item("Symbol", "AAPL", "#4CAF50")
    card1.add_item("Price", "$150.25", "#2196F3")
    card1.add_item("Change", "+2.50 (+1.69%)", "#4CAF50", "Daily change")
    card1.add_separator()
    card1.add_item("Volume", "1,234,567", tooltip="Trading volume")
    card1.add_item("Market Cap", "$2.5T")

    # Create card with custom styling
    custom_styles = {
        'bg_color': '#1a1a2e',
        'primary_color': '#00d4aa',
        'border_color': '#16213e',
        'hover_color': '#0f3460'
    }

    card2 = StockInfoCard("Portfolio Summary", custom_styles)
    card2.add_item("Total Value", "$50,000", "#00d4aa")
    card2.add_item("Today's Gain", "+$1,250", "#00d4aa")
    card2.add_item("Total Gain", "+$5,000", "#00d4aa")

    layout.addWidget(card1)
    layout.addWidget(card2)
    layout.addStretch()

    window.show()
    sys.exit(app.exec())
