import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
# --- logging -----------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

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
