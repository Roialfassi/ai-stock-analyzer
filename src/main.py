import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from src.view.main_window import StockAnalyzerWindow


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
