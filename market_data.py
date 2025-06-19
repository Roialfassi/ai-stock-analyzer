# market_data.py - Market Data Integration Layer

import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from contextlib import contextmanager
import time
import logging
from functools import lru_cache
import requests

from models import (
    StockData, TechnicalIndicators, FinancialMetrics,
    NewsItem, MarketOverview, CacheEntry
)

logger = logging.getLogger(__name__)


class RateLimiter:
    """Rate limiting for API calls"""

    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.calls = []

    async def acquire(self):
        now = time.time()
        self.calls = [call for call in self.calls if now - call < 60]

        if len(self.calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.calls[0])
            await asyncio.sleep(sleep_time)

        self.calls.append(time.time())


class DataCache:
    """SQLite-based data cache"""

    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    data TEXT,
                    timestamp REAL,
                    ttl_seconds INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS stock_history (
                    symbol TEXT,
                    date TEXT,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    PRIMARY KEY (symbol, date)
                )
            """)

    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def get(self, key: str) -> Optional[Any]:
        with self._get_connection() as conn:
            result = conn.execute(
                "SELECT data, timestamp, ttl_seconds FROM cache WHERE key = ?",
                (key,)
            ).fetchone()

            if result:
                data, timestamp, ttl = result
                if time.time() - timestamp <= ttl:
                    return json.loads(data)
                else:
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
        return None

    def set(self, key: str, data: Any, ttl_seconds: int):
        with self._get_connection() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, data, timestamp, ttl_seconds) VALUES (?, ?, ?, ?)",
                (key, json.dumps(data), time.time(), ttl_seconds)
            )


class MarketDataProvider:
    """Main market data provider with multiple sources"""

    def __init__(self, cache_dir: str = "cache"):
        self.cache = DataCache()
        self.rate_limiters = {
            'yfinance': RateLimiter(200),
            'alphavantage': RateLimiter(5),
            'newsapi': RateLimiter(100)
        }
        self.executor = ThreadPoolExecutor(max_workers=10)

    # Cache TTL settings (in seconds)
    CACHE_TTL = {
        'realtime': 60,  # 1 minute
        'daily': 86400,  # 24 hours
        'financials': 604800,  # 1 week
        'company_info': 2592000,  # 30 days
    }

    async def get_stock_data(self, symbol: str) -> Optional[StockData]:
        """Get comprehensive stock data"""
        cache_key = f"stock_data:{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return StockData(**cached)

        try:
            # Fetch from yfinance
            ticker = yf.Ticker(symbol)
            info = ticker.info

            stock_data = StockData(
                symbol=symbol,
                company_name=info.get('longName', symbol),
                sector=info.get('sector', 'Unknown'),
                industry=info.get('industry', 'Unknown'),
                current_price=info.get('currentPrice', 0),
                market_cap=info.get('marketCap', 0),
                pe_ratio=info.get('trailingPE'),
                forward_pe=info.get('forwardPE'),
                peg_ratio=info.get('pegRatio'),
                price_to_book=info.get('priceToBook'),
                dividend_yield=info.get('dividendYield'),
                eps=info.get('trailingEps'),
                revenue=info.get('totalRevenue'),
                revenue_growth=info.get('revenueGrowth'),
                profit_margin=info.get('profitMargins'),
                operating_margin=info.get('operatingMargins'),
                roe=info.get('returnOnEquity'),
                debt_to_equity=info.get('debtToEquity'),
                current_ratio=info.get('currentRatio'),
                beta=info.get('beta'),
                volume=info.get('volume'),
                avg_volume=info.get('averageVolume'),
                day_high=info.get('dayHigh'),
                day_low=info.get('dayLow'),
                year_high=info.get('fiftyTwoWeekHigh'),
                year_low=info.get('fiftyTwoWeekLow'),
                fifty_day_ma=info.get('fiftyDayAverage'),
                two_hundred_day_ma=info.get('twoHundredDayAverage')
            )

            # Cache the result
            self.cache.set(cache_key, stock_data.__dict__, self.CACHE_TTL['realtime'])
            return stock_data

        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None

    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical OHLCV data"""
        cache_key = f"history:{symbol}:{period}"
        cached = self.cache.get(cache_key)
        if cached:
            return pd.DataFrame(cached)

        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)

            # Cache as dict
            self.cache.set(cache_key, hist.to_dict(), self.CACHE_TTL['daily'])
            return hist

        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()

    async def get_financial_statements(self, symbol: str) -> FinancialMetrics:
        """Get detailed financial statements"""
        cache_key = f"financials:{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return FinancialMetrics(**cached)

        try:
            ticker = yf.Ticker(symbol)

            # Get financial data
            income_stmt = ticker.income_stmt.to_dict() if not ticker.income_stmt.empty else {}
            balance_sheet = ticker.balance_sheet.to_dict() if not ticker.balance_sheet.empty else {}
            cash_flow = ticker.cash_flow.to_dict() if not ticker.cash_flow.empty else {}

            # Calculate key ratios
            ratios = self._calculate_financial_ratios(ticker.info)

            # Get peer comparison data
            peers = await self._get_peer_comparison(symbol, ticker.info.get('sector'))

            metrics = FinancialMetrics(
                income_statement=income_stmt,
                balance_sheet=balance_sheet,
                cash_flow=cash_flow,
                ratios=ratios,
                growth_rates=self._calculate_growth_rates(income_stmt),
                peer_comparison=peers
            )

            self.cache.set(cache_key, metrics.__dict__, self.CACHE_TTL['financials'])
            return metrics

        except Exception as e:
            logger.error(f"Error fetching financials for {symbol}: {e}")
            return FinancialMetrics({}, {}, {}, {}, {}, {})

    async def get_technical_indicators(self, symbol: str) -> TechnicalIndicators:
        """Calculate technical indicators"""
        hist = await self.get_historical_data(symbol, "6mo")
        if hist.empty:
            return None

        close_prices = hist['Close']
        high_prices = hist['High']
        low_prices = hist['Low']
        volume = hist['Volume']

        # Calculate indicators
        rsi = self._calculate_rsi(close_prices)
        macd = self._calculate_macd(close_prices)

        # Moving averages
        ma_20 = close_prices.rolling(20).mean().iloc[-1]
        ma_50 = close_prices.rolling(50).mean().iloc[-1]
        ma_200 = close_prices.rolling(200).mean().iloc[-1] if len(close_prices) >= 200 else None

        # Bollinger Bands
        bb = self._calculate_bollinger_bands(close_prices)

        # Support/Resistance
        support, resistance = self._find_support_resistance(hist)

        # Trend detection
        trend = self._detect_trend(close_prices)

        # Momentum score
        momentum = self._calculate_momentum_score(close_prices, volume)

        return TechnicalIndicators(
            rsi=rsi,
            macd=macd,
            moving_averages={'MA20': ma_20, 'MA50': ma_50, 'MA200': ma_200},
            bollinger_bands=bb,
            volume_profile={'avg_volume': volume.mean(), 'volume_trend': self._volume_trend(volume)},
            support_levels=support,
            resistance_levels=resistance,
            trend_direction=trend,
            momentum_score=momentum
        )

    async def get_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        """Get recent news for a stock"""
        cache_key = f"news:{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return [NewsItem(**item) for item in cached]

        news_items = []

        # Use yfinance news
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news[:limit] if hasattr(ticker, 'news') else []

            for article in news:
                news_items.append(NewsItem(
                    title=article.get('title', ''),
                    source=article.get('publisher', ''),
                    published=datetime.fromtimestamp(article.get('providerPublishTime', 0)),
                    url=article.get('link', ''),
                    summary=article.get('summary', '')[:200],
                    sentiment_score=0.0,  # Would need NLP analysis
                    relevance_score=0.8
                ))

            self.cache.set(cache_key, [item.__dict__ for item in news_items],
                           self.CACHE_TTL['realtime'])

        except Exception as e:
            logger.error(f"Error fetching news for {symbol}: {e}")

        return news_items

    async def get_market_overview(self) -> MarketOverview:
        """Get overall market conditions"""
        cache_key = "market_overview"
        cached = self.cache.get(cache_key)
        if cached:
            return MarketOverview(**cached)

        try:
            # Get major indices
            indices = {}
            for symbol, name in [('SPY', 'S&P 500'), ('QQQ', 'Nasdaq'), ('DIA', 'Dow Jones')]:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                indices[name] = {
                    'price': info.get('currentPrice', 0),
                    'change': info.get('currentPrice', 0) - info.get('previousClose', 0),
                    'change_pct': ((info.get('currentPrice', 0) / info.get('previousClose', 1)) - 1) * 100
                }

            # Get VIX
            vix_ticker = yf.Ticker('^VIX')
            vix = vix_ticker.info.get('currentPrice', 0)

            # Placeholder for other market data
            overview = MarketOverview(
                indices=indices,
                sector_performance={},  # Would need sector ETF data
                market_breadth={'advances': 0, 'declines': 0, 'unchanged': 0},
                vix=vix,
                dollar_index=0,
                treasury_yields={'2Y': 0, '10Y': 0, '30Y': 0},
                crypto_prices={'BTC': 0, 'ETH': 0}
            )

            self.cache.set(cache_key, overview.__dict__, self.CACHE_TTL['realtime'])
            return overview

        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return MarketOverview({}, {}, {}, 0, 0, {}, {})

    async def batch_get_stocks(self, symbols: List[str]) -> Dict[str, StockData]:
        """Get multiple stocks efficiently"""
        tasks = [self.get_stock_data(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks)

        return {
            symbol: data
            for symbol, data in zip(symbols, results)
            if data is not None
        }

    async def get_options_chain(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get options chain data"""
        cache_key = f"options:{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return {k: pd.DataFrame(v) for k, v in cached.items()}

        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options

            if not expirations:
                return {}

            # Get options for nearest expiration
            options = ticker.option_chain(expirations[0])

            result = {
                'calls': options.calls,
                'puts': options.puts,
                'expirations': expirations
            }

            # Cache as dict
            cache_dict = {
                'calls': options.calls.to_dict(),
                'puts': options.puts.to_dict(),
                'expirations': expirations
            }
            self.cache.set(cache_key, cache_dict, self.CACHE_TTL['realtime'])

            return result

        except Exception as e:
            logger.error(f"Error fetching options for {symbol}: {e}")
            return {}

    # Helper methods
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1] if not rsi.empty else 50

    def _calculate_macd(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate MACD"""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line.iloc[-1],
            'signal': signal_line.iloc[-1],
            'histogram': histogram.iloc[-1]
        }

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()

        upper = sma + (std * 2)
        lower = sma - (std * 2)

        return {
            'upper': upper.iloc[-1],
            'middle': sma.iloc[-1],
            'lower': lower.iloc[-1]
        }

    def _find_support_resistance(self, hist: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels"""
        highs = hist['High'].values
        lows = hist['Low'].values

        # Simple peak/valley detection
        resistance = []
        support = []

        window = 5
        for i in range(window, len(highs) - window):
            if highs[i] == max(highs[i - window:i + window + 1]):
                resistance.append(highs[i])
            if lows[i] == min(lows[i - window:i + window + 1]):
                support.append(lows[i])

        # Get unique levels with some tolerance
        resistance = sorted(list(set(np.round(resistance, 2))))[-3:]
        support = sorted(list(set(np.round(support, 2))))[:3]

        return support, resistance

    def _detect_trend(self, prices: pd.Series) -> str:
        """Detect price trend"""
        if len(prices) < 50:
            return "neutral"

        ma20 = prices.rolling(20).mean()
        ma50 = prices.rolling(50).mean()

        current_price = prices.iloc[-1]
        ma20_current = ma20.iloc[-1]
        ma50_current = ma50.iloc[-1]

        if current_price > ma20_current > ma50_current:
            return "bullish"
        elif current_price < ma20_current < ma50_current:
            return "bearish"
        else:
            return "neutral"

    def _calculate_momentum_score(self, prices: pd.Series, volume: pd.Series) -> float:
        """Calculate momentum score (0-100)"""
        # Price momentum
        returns = prices.pct_change()
        momentum_1m = returns.tail(20).mean()
        momentum_3m = returns.tail(60).mean() if len(returns) >= 60 else momentum_1m

        # Volume momentum
        vol_avg = volume.mean()
        vol_recent = volume.tail(5).mean()
        vol_ratio = vol_recent / vol_avg if vol_avg > 0 else 1

        # RSI component
        rsi = self._calculate_rsi(prices)
        rsi_score = (rsi - 30) / 40 if rsi > 30 else 0  # Normalize RSI

        # Combine scores
        score = (
                momentum_1m * 30 +
                momentum_3m * 30 +
                (vol_ratio - 1) * 20 +
                rsi_score * 20
        )

        return max(0, min(100, score * 100))

    def _volume_trend(self, volume: pd.Series) -> str:
        """Determine volume trend"""
        if len(volume) < 20:
            return "normal"

        recent_avg = volume.tail(5).mean()
        historical_avg = volume.tail(20).mean()

        ratio = recent_avg / historical_avg if historical_avg > 0 else 1

        if ratio > 1.5:
            return "increasing"
        elif ratio < 0.7:
            return "decreasing"
        else:
            return "normal"

    def _calculate_financial_ratios(self, info: Dict) -> Dict[str, float]:
        """Calculate additional financial ratios"""
        ratios = {}

        # Profitability ratios
        ratios['gross_margin'] = info.get('grossMargins', 0)
        ratios['operating_margin'] = info.get('operatingMargins', 0)
        ratios['net_margin'] = info.get('profitMargins', 0)
        ratios['roe'] = info.get('returnOnEquity', 0)
        ratios['roa'] = info.get('returnOnAssets', 0)

        # Valuation ratios
        ratios['pe_ratio'] = info.get('trailingPE', 0)
        ratios['forward_pe'] = info.get('forwardPE', 0)
        ratios['peg_ratio'] = info.get('pegRatio', 0)
        ratios['price_to_book'] = info.get('priceToBook', 0)
        ratios['price_to_sales'] = info.get('priceToSalesTrailing12Months', 0)
        ratios['ev_to_ebitda'] = info.get('enterpriseToEbitda', 0)

        # Liquidity ratios
        ratios['current_ratio'] = info.get('currentRatio', 0)
        ratios['quick_ratio'] = info.get('quickRatio', 0)

        # Leverage ratios
        ratios['debt_to_equity'] = info.get('debtToEquity', 0)
        ratios['interest_coverage'] = info.get('interestCoverage', 0)

        return ratios

    def _calculate_growth_rates(self, income_stmt: Dict) -> Dict[str, float]:
        """Calculate growth rates from financial statements"""
        growth_rates = {}

        try:
            if income_stmt and len(income_stmt) > 0:
                # Get the most recent periods
                periods = sorted(income_stmt.keys(), reverse=True)
                if len(periods) >= 2:
                    current = income_stmt[periods[0]]
                    previous = income_stmt[periods[1]]

                    # Revenue growth
                    if 'Total Revenue' in current and 'Total Revenue' in previous:
                        rev_current = current['Total Revenue']
                        rev_previous = previous['Total Revenue']
                        if rev_previous != 0:
                            growth_rates['revenue_growth'] = (rev_current - rev_previous) / rev_previous

                    # Earnings growth
                    if 'Net Income' in current and 'Net Income' in previous:
                        ni_current = current['Net Income']
                        ni_previous = previous['Net Income']
                        if ni_previous != 0:
                            growth_rates['earnings_growth'] = (ni_current - ni_previous) / ni_previous

        except Exception as e:
            logger.error(f"Error calculating growth rates: {e}")

        return growth_rates

    async def _get_peer_comparison(self, symbol: str, sector: str) -> Dict[str, Dict[str, float]]:
        """Get peer comparison data"""
        # This would fetch data for sector peers
        # For now, return empty dict
        return {}

    async def get_insider_trading(self, symbol: str) -> List[Dict[str, Any]]:
        """Get insider trading data"""
        cache_key = f"insider:{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            ticker = yf.Ticker(symbol)
            insider_trades = ticker.insider_trades

            if insider_trades is not None and not insider_trades.empty:
                trades = insider_trades.to_dict('records')
                self.cache.set(cache_key, trades, self.CACHE_TTL['daily'])
                return trades

        except Exception as e:
            logger.error(f"Error fetching insider trading for {symbol}: {e}")

        return []

    async def get_analyst_ratings(self, symbol: str) -> Dict[str, Any]:
        """Get analyst ratings and price targets"""
        cache_key = f"analysts:{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            return cached

        try:
            ticker = yf.Ticker(symbol)
            recommendations = ticker.recommendations

            if recommendations is not None and not recommendations.empty:
                # Get latest recommendations
                latest = recommendations.tail(10)

                ratings = {
                    'current': latest.iloc[-1].to_dict() if not latest.empty else {},
                    'history': latest.to_dict('records'),
                    'consensus': self._calculate_consensus(latest)
                }

                self.cache.set(cache_key, ratings, self.CACHE_TTL['daily'])
                return ratings

        except Exception as e:
            logger.error(f"Error fetching analyst ratings for {symbol}: {e}")

        return {'current': {}, 'history': [], 'consensus': 'Hold'}

    def _calculate_consensus(self, recommendations: pd.DataFrame) -> str:
        """Calculate consensus recommendation"""
        if recommendations.empty:
            return "Hold"

        # Count recommendations
        rec_counts = recommendations['To Grade'].value_counts()

        # Simple scoring
        score = 0
        total = 0

        scoring = {
            'Strong Buy': 5, 'Buy': 4, 'Hold': 3,
            'Sell': 2, 'Strong Sell': 1,
            'Outperform': 4, 'Underperform': 2,
            'Overweight': 4, 'Underweight': 2
        }

        for grade, count in rec_counts.items():
            if grade in scoring:
                score += scoring[grade] * count
                total += count

        if total == 0:
            return "Hold"

        avg_score = score / total

        if avg_score >= 4.5:
            return "Strong Buy"
        elif avg_score >= 3.5:
            return "Buy"
        elif avg_score >= 2.5:
            return "Hold"
        elif avg_score >= 1.5:
            return "Sell"
        else:
            return "Strong Sell"

    async def search_stocks(self, query: str) -> List[Dict[str, str]]:
        """Search for stocks by name or symbol"""
        # This would use a stock symbol database
        # For now, return empty list
        return []

    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)# market_data.py - Market Data Integration Layer


