from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

# --- Stock Screener ---
from src.model.stock_data import StockData

# --- logging -----------------------------------------------------------
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class StockScreener:
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
