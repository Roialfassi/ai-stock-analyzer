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
        try:
            now = time.time()
            # Filter out calls older than 60 seconds
            self.calls = [call_ts for call_ts in self.calls if now - call_ts < 60]

            if len(self.calls) >= self.calls_per_minute:
                # Calculate sleep time ensuring it's not negative
                if self.calls: # Ensure list is not empty to prevent IndexError
                    sleep_for = 60.0 - (now - self.calls[0])
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                # If self.calls is empty here, it implies a logic flaw or race condition,
                # as len(self.calls) should not be >= self.calls_per_minute.
                # However, proceeding without sleep is safe.

            self.calls.append(time.time())
        except asyncio.CancelledError: # pragma: no cover
            logger.info("RateLimiter acquire task was cancelled.")
            raise # Re-raise CancelledError to allow task cleanup
        except Exception as e: # pragma: no cover
            logger.error(f"Unexpected error in RateLimiter.acquire: {e}", exc_info=True)
            # Fallback: sleep for a short period to prevent rapid retries if error is persistent
            await asyncio.sleep(1) # Sleep 1s to avoid busy loop on error


class DataCache:
    """SQLite-based data cache"""

    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        try:
            self._init_db()
        except sqlite3.Error as e: # pragma: no cover
            logger.critical(f"FATAL: Failed to initialize database at {db_path}: {e}", exc_info=True)
            # This is critical; depending on app requirements, might re-raise or exit.
            # For now, we allow the app to continue but cache will not work.
            # raise DatabaseInitializationError(f"Failed to initialize DB: {e}") from e # Example custom error

    def _init_db(self):
        try:
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
        except sqlite3.Error as e: # pragma: no cover
            logger.error(f"Error initializing database tables: {e}", exc_info=True)
            raise # Propagate error as DB init is critical for cache functionality

    @contextmanager
    def _get_connection(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=10) # Increased timeout for busy DBs
            yield conn
            conn.commit()
        except sqlite3.OperationalError as e: # More specific error for "database is locked"
            logger.error(f"SQLite operational error (e.g., database locked): {e}", exc_info=True)
            if conn:
                try:
                    conn.rollback()
                except sqlite3.Error as rb_err: # pragma: no cover
                    logger.error(f"SQLite error during rollback: {rb_err}", exc_info=True)
            raise # Re-raise to indicate failure to the caller
        except sqlite3.Error as e: # pragma: no cover
            logger.error(f"General SQLite error occurred: {e}", exc_info=True)
            if conn:
                try:
                    conn.rollback()
                except sqlite3.Error as rb_err: # pragma: no cover
                    logger.error(f"SQLite error during rollback: {rb_err}", exc_info=True)
            raise
        finally:
            if conn:
                try:
                    conn.close()
                except sqlite3.Error as close_err: # pragma: no cover
                     logger.error(f"SQLite error during connection close: {close_err}", exc_info=True)

    def get(self, key: str) -> Optional[Any]:
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT data, timestamp, ttl_seconds FROM cache WHERE key = ?",
                    (key,)
                )
                result = cursor.fetchone()

                if result:
                    data_str, timestamp, ttl = result
                    if time.time() - timestamp <= ttl:
                        try:
                            return json.loads(data_str)
                        except json.JSONDecodeError as e:
                            logger.error(f"Cache Corrupt: Failed to decode JSON from cache for key '{key}': {e}. Data snippet: {data_str[:200]}...", exc_info=True)
                            try:
                                cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                                logger.info(f"Deleted corrupted cache entry for key '{key}'.")
                            except sqlite3.Error as del_e: # pragma: no cover
                                logger.error(f"Failed to delete corrupted cache entry for key '{key}': {del_e}", exc_info=True)
                            return None
                    else:
                        logger.debug(f"Cache expired for key '{key}'. Deleting.")
                        try:
                            cursor.execute("DELETE FROM cache WHERE key = ?", (key,))
                        except sqlite3.Error as del_e: # pragma: no cover
                             logger.error(f"Failed to delete expired cache entry for key '{key}': {del_e}", exc_info=True)
                        return None
            return None
        except sqlite3.Error as e: # Errors from _get_connection or execute
            logger.error(f"SQLite error getting cache for key '{key}': {e}", exc_info=True)
            return None
        except Exception as e: # Catch any other unexpected errors # pragma: no cover
            logger.error(f"Unexpected error getting cache for key '{key}': {e}", exc_info=True)
            return None


    def set(self, key: str, data: Any, ttl_seconds: int):
        try:
            serialized_data = json.dumps(data) # Do serialization outside of DB transaction
        except TypeError as e: # Catch serialization errors early
            logger.error(f"Data Serialization Error: Failed to serialize data for caching (key: '{key}'). Data type: {type(data)}. Error: {e}", exc_info=True)
            return # Do not attempt to cache if serialization fails

        try:
            with self._get_connection() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO cache (key, data, timestamp, ttl_seconds) VALUES (?, ?, ?, ?)",
                    (key, serialized_data, time.time(), ttl_seconds)
                )
        except sqlite3.Error as e: # pragma: no cover
            logger.error(f"SQLite error setting cache for key '{key}': {e}", exc_info=True)
            # Data not cached, but function shouldn't crash the app
        except Exception as e: # Catch any other unexpected errors # pragma: no cover
            logger.error(f"Unexpected error setting cache for key '{key}': {e}", exc_info=True)


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
            try:
                cached_data = self.cache.get(cache_key)
                if cached_data and not (isinstance(cached_data, dict) and cached_data.get("error")):
                    return StockData(**cached_data)
            except TypeError as e:
                 logger.warning(f"Cached stock data for {symbol} is malformed or outdated, refetching: {e}", exc_info=True)
                 if self.cache: self.cache.set(cache_key, None, 0)

        try:
            await self.rate_limiters['yfinance'].acquire()
            ticker = yf.Ticker(symbol)
            info = await asyncio.to_thread(ticker.info)

            if not info or ("quoteType" in info and info["quoteType"] == "NONE") or info.get('isDelisted'):
                 logger.warning(f"No valid data returned from yfinance info for {symbol}. It might be delisted, an invalid symbol, or has no info.")
                 if self.cache: self.cache.set(cache_key, {"symbol": symbol, "error": "No data from yfinance info"}, self.CACHE_TTL['daily'])
                 return None

            stock_data = StockData(
                symbol=symbol,
                company_name=info.get('longName', info.get('shortName', symbol)),
                sector=info.get('sector', 'N/A'),
                industry=info.get('industry', 'N/A'),
                current_price=info.get('currentPrice', info.get('regularMarketPrice')),
                market_cap=info.get('marketCap'),
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
                volume=info.get('volume', info.get('regularMarketVolume')),
                avg_volume=info.get('averageVolume', info.get('averageDailyVolume10Day')),
                day_high=info.get('dayHigh', info.get('regularMarketDayHigh')),
                day_low=info.get('dayLow', info.get('regularMarketDayLow')),
                year_high=info.get('fiftyTwoWeekHigh'),
                year_low=info.get('fiftyTwoWeekLow'),
                fifty_day_ma=info.get('fiftyDayAverage'),
                two_hundred_day_ma=info.get('twoHundredDayAverage')
            )

            if self.cache: self.cache.set(cache_key, stock_data.__dict__, self.CACHE_TTL['realtime'])
            return stock_data

        except requests.exceptions.RequestException as e: # pragma: no cover
            logger.error(f"Network error fetching stock data for {symbol}: {e}", exc_info=True)
            return None
        except (AttributeError, KeyError) as e: # pragma: no cover
            logger.error(f"Data structure error processing yfinance data for {symbol}: {e}", exc_info=True)
            return None
        except Exception as e:
            err_str = str(e).lower()
            if "no data found" in err_str or "failed to get ticker" in err_str or "private assets" in err_str or "symbol may be delisted" in err_str:
                 logger.warning(f"yfinance could not find data for symbol {symbol} (get_stock_data): {e}")
                 if self.cache: self.cache.set(cache_key, {"symbol": symbol, "error": str(e)}, self.CACHE_TTL['daily'])
            else: # pragma: no cover
                 logger.error(f"Unexpected error fetching stock data for {symbol} with yfinance: {e.__class__.__name__} - {e}", exc_info=True)
            return None

    async def get_historical_data(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Get historical OHLCV data. Returns empty DataFrame on error."""
        cache_key = f"history:{symbol}:{period}"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                df = pd.DataFrame(cached)
                if not df.empty and 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date')
                elif not df.empty and df.index.name == 'Date' and not isinstance(df.index, pd.DatetimeIndex): # pragma: no cover
                     df.index = pd.to_datetime(df.index)
                return df
            except Exception as e: # pragma: no cover
                logger.warning(f"Failed to load/reconstruct historical data from cache for {symbol}, refetching: {e}", exc_info=True)
                if self.cache: self.cache.set(cache_key, None, 0)

        try:
            await self.rate_limiters['yfinance'].acquire()
            ticker = yf.Ticker(symbol)
            hist_df = await asyncio.to_thread(ticker.history, period=period)

            if hist_df.empty:
                logger.warning(f"No historical data returned by yfinance for {symbol} and period {period}.")
                if self.cache: self.cache.set(cache_key, [], self.CACHE_TTL['realtime'])
                return pd.DataFrame()

            if not isinstance(hist_df.index, pd.DatetimeIndex): # pragma: no cover
                hist_df.index = pd.to_datetime(hist_df.index)

            if self.cache: self.cache.set(cache_key, hist_df.reset_index().to_dict(orient='records'), self.CACHE_TTL['daily'])
            return hist_df

        except requests.exceptions.RequestException as e: # pragma: no cover
            logger.error(f"Network error fetching historical data for {symbol}: {e}", exc_info=True)
            return pd.DataFrame()
        except Exception as e:
            err_str = str(e).lower()
            if "no data found" in err_str or "no price data found" in err_str or "symbol may be delisted" in err_str:
                 logger.warning(f"yfinance could not find historical data for symbol {symbol} (get_historical_data): {e}")
                 if self.cache: self.cache.set(cache_key, [], self.CACHE_TTL['daily'])
            else: # pragma: no cover
                logger.error(f"Error fetching historical data for {symbol} with yfinance: {e.__class__.__name__} - {e}", exc_info=True)
            return pd.DataFrame()

    async def get_financial_statements(self, symbol: str) -> FinancialMetrics:
        """Get detailed financial statements. Returns a FinancialMetrics object, possibly with empty/default fields on error."""
        cache_key = f"financials:{symbol}"
        cached = self.cache.get(cache_key)
        if cached:
            try:
                return FinancialMetrics(**cached)
            except TypeError as e: # pragma: no cover
                logger.warning(f"Cached financial data for {symbol} is malformed or outdated, refetching: {e}", exc_info=True)
                if self.cache: self.cache.set(cache_key, None, 0)

        default_financials = FinancialMetrics(income_statement={}, balance_sheet={}, cash_flow={}, ratios={}, growth_rates={}, peer_comparison={})

        try:
            await self.rate_limiters['yfinance'].acquire()
            ticker = yf.Ticker(symbol)

            info_dict = await asyncio.to_thread(ticker.info)
            if not info_dict or ("quoteType" in info_dict and info_dict["quoteType"] == "NONE") or info_dict.get('isDelisted'):
                logger.warning(f"Could not retrieve valid yfinance info for {symbol}. Financials fetch aborted.")
                if self.cache: self.cache.set(cache_key, default_financials.__dict__, self.CACHE_TTL['daily'])
                return default_financials

            results = await asyncio.gather(
                asyncio.to_thread(lambda t: t.financials, ticker),
                asyncio.to_thread(lambda t: t.balance_sheet, ticker),
                asyncio.to_thread(lambda t: t.cashflow, ticker),
                return_exceptions=True
            )
            income_stmt_df, balance_sheet_df, cash_flow_df = results

            if isinstance(income_stmt_df, Exception): # pragma: no cover
                logger.error(f"Failed to fetch income statement for {symbol}: {income_stmt_df}", exc_info=income_stmt_df)
                income_stmt_df = pd.DataFrame()
            if isinstance(balance_sheet_df, Exception): # pragma: no cover
                logger.error(f"Failed to fetch balance sheet for {symbol}: {balance_sheet_df}", exc_info=balance_sheet_df)
                balance_sheet_df = pd.DataFrame()
            if isinstance(cash_flow_df, Exception): # pragma: no cover
                logger.error(f"Failed to fetch cash flow for {symbol}: {cash_flow_df}", exc_info=cash_flow_df)
                cash_flow_df = pd.DataFrame()

            income_stmt = income_stmt_df.to_dict(orient='dict') if not income_stmt_df.empty else {}
            balance_sheet = balance_sheet_df.to_dict(orient='dict') if not balance_sheet_df.empty else {}
            cash_flow = cash_flow_df.to_dict(orient='dict') if not cash_flow_df.empty else {}

            ratios = self._calculate_financial_ratios(info_dict)
            growth_rates = self._calculate_growth_rates(income_stmt_df)

            peers = await self._get_peer_comparison(symbol, info_dict.get('sector', 'N/A'))

            metrics = FinancialMetrics(
                income_statement=income_stmt,
                balance_sheet=balance_sheet,
                cash_flow=cash_flow,
                ratios=ratios,
                growth_rates=growth_rates,
                peer_comparison=peers
            )

            if self.cache: self.cache.set(cache_key, metrics.__dict__, self.CACHE_TTL['financials'])
            return metrics

        except requests.exceptions.RequestException as e: # pragma: no cover
            logger.error(f"Network error fetching financials for {symbol}: {e}", exc_info=True)
        except (AttributeError, KeyError) as e: # pragma: no cover
             logger.error(f"Data structure error processing yfinance financials for {symbol}: {e}", exc_info=True)
        except Exception as e: # pragma: no cover
            err_str = str(e).lower()
            if "no data found" in err_str or "symbol may be delisted" in err_str:
                 logger.warning(f"yfinance could not find financial data for symbol {symbol} (get_financial_statements): {e}")
            else:
                logger.error(f"Error fetching financials for {symbol} with yfinance: {e.__class__.__name__} - {e}", exc_info=True)

        return default_financials

    async def get_technical_indicators(self, symbol: str) -> TechnicalIndicators:
        """Calculate technical indicators. Returns default TechnicalIndicators on error."""
        hist_df = await self.get_historical_data(symbol, "1y")
        default_ti = TechnicalIndicators(
            rsi=50.0, macd={'macd': 0.0, 'signal': 0.0, 'histogram': 0.0},
            moving_averages={}, bollinger_bands={}, volume_profile={},
            support_levels=[], resistance_levels=[], trend_direction="neutral", momentum_score=50.0
        )

        if hist_df is None or hist_df.empty:
            logger.warning(f"Cannot calculate technical indicators for {symbol} due to missing historical data.")
            return default_ti

        try:
            required_cols = ['Close', 'High', 'Low', 'Volume']
            if not all(col in hist_df.columns for col in required_cols): # pragma: no cover
                logger.error(f"Historical data for {symbol} is missing one or more required columns: {required_cols}.")
                return default_ti

            close_prices = hist_df['Close']
            volume = hist_df['Volume']

            rsi = self._calculate_rsi(close_prices)
            macd_data = self._calculate_macd(close_prices)

            moving_averages = {}
            if len(close_prices) >= 20: moving_averages['MA20'] = close_prices.rolling(window=20).mean().iloc[-1]
            if len(close_prices) >= 50: moving_averages['MA50'] = close_prices.rolling(window=50).mean().iloc[-1]
            if len(close_prices) >= 200: moving_averages['MA200'] = close_prices.rolling(window=200).mean().iloc[-1]
            moving_averages = {k: (float(v) if pd.notna(v) else None) for k,v in moving_averages.items()}


            bb_data = self._calculate_bollinger_bands(close_prices)
            support, resistance = self._find_support_resistance(hist_df)
            trend = self._detect_trend(close_prices)
            momentum = self._calculate_momentum_score(close_prices, volume)

            avg_vol = volume.mean()
            volume_profile_data = {
                'avg_volume': float(avg_vol) if pd.notna(avg_vol) else 0.0,
                'volume_trend': self._volume_trend(volume)
            }

            return TechnicalIndicators(
                rsi=rsi,
                macd=macd_data,
                moving_averages=moving_averages,
                bollinger_bands=bb_data,
                volume_profile=volume_profile_data,
                support_levels=support,
                resistance_levels=resistance,
                trend_direction=trend,
                momentum_score=momentum
            )
        except KeyError as e: # pragma: no cover
            logger.error(f"Missing expected column in historical data for {symbol} when calculating TIs: {e}", exc_info=True)
        except IndexError as e: # pragma: no cover
            logger.error(f"Index error calculating TIs for {symbol}, likely insufficient data for iloc[-1]: {e}", exc_info=True)
        except Exception as e: # pragma: no cover
            logger.error(f"Unexpected error calculating technical indicators for {symbol}: {e}", exc_info=True)

        return default_ti


    async def get_news(self, symbol: str, limit: int = 10) -> List[NewsItem]:
        """Get recent news for a stock. Returns empty list on error."""
        cache_key = f"news:{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            try:
                return [NewsItem(**item) for item in cached]
            except TypeError as e: # pragma: no cover
                logger.warning(f"Cached news data for {symbol} is malformed or outdated, refetching: {e}", exc_info=True)
                if self.cache: self.cache.set(cache_key, None, 0)

        news_items = []
        try:
            await self.rate_limiters['yfinance'].acquire()
            ticker = yf.Ticker(symbol)
            news_data_list = await asyncio.to_thread(ticker.news)

            if not news_data_list:
                logger.info(f"No news returned by yfinance for {symbol}.")
                if self.cache: self.cache.set(cache_key, [], self.CACHE_TTL['daily'])
                return []

            for article_dict in news_data_list[:limit]:
                try:
                    published_ts = article_dict.get('providerPublishTime')
                    published_dt = datetime.fromtimestamp(published_ts) if pd.notna(published_ts) and isinstance(published_ts, (int, float)) else datetime.now()
                    summary = article_dict.get('title')

                    news_items.append(NewsItem(
                        title=article_dict.get('title', 'N/A'),
                        source=article_dict.get('publisher', 'N/A'),
                        published=published_dt,
                        url=article_dict.get('link', '#'),
                        summary=summary[:500] + "..." if summary else 'N/A',
                        sentiment_score=0.0,
                        relevance_score=0.8
                    ))
                except KeyError as e: # pragma: no cover
                    logger.warning(f"Missing key {e} in yfinance news article for {symbol}: {article_dict}")
                except Exception as e: # pragma: no cover
                    logger.error(f"Error processing individual yfinance news article for {symbol}: {e}. Article: {article_dict}", exc_info=True)

            if news_items and self.cache:
                self.cache.set(cache_key, [item.__dict__ for item in news_items], self.CACHE_TTL['realtime'])

        except requests.exceptions.RequestException as e: # pragma: no cover
            logger.error(f"Network-related error fetching news for {symbol} via yfinance: {e}", exc_info=True)
        except AttributeError as e: # pragma: no cover
            logger.error(f"Attribute error with yfinance Ticker object for news ({symbol}): {e}", exc_info=True)
        except Exception as e:
            err_str = str(e).lower()
            if "no news found" in err_str or "cannot find news" in err_str:  # pragma: no cover
                 logger.info(f"yfinance reported no news for {symbol}: {e}")
                 if self.cache: self.cache.set(cache_key, [], self.CACHE_TTL['daily'])
            else: # pragma: no cover
                logger.error(f"Error fetching news for {symbol} with yfinance: {e.__class__.__name__} - {e}", exc_info=True)

        return news_items


    async def get_market_overview(self) -> MarketOverview:
        """Get overall market conditions. Returns MarketOverview, possibly with partial data on errors."""
        cache_key = "market_overview"
        cached = self.cache.get(cache_key)
        if cached is not None:
            try:
                return MarketOverview(**cached)
            except TypeError as e: # pragma: no cover
                 logger.warning(f"Cached market overview data is malformed or outdated, refetching: {e}", exc_info=True)
                 if self.cache: self.cache.set(cache_key, None, 0)

        overview = MarketOverview(
            indices={},
            sector_performance={},
            market_breadth={'advances': 0, 'declines': 0, 'unchanged': 0},
            vix=0.0,
            dollar_index=0.0,
            treasury_yields={'2Y': 0.0, '10Y': 0.0, '30Y': 0.0},
            crypto_prices={'BTC': 0.0, 'ETH': 0.0}
        )

        async def fetch_index_info(idx_symbol: str, idx_name: str) -> Tuple[str, Dict[str, Any]]:
            try:
                await self.rate_limiters['yfinance'].acquire()
                ticker = yf.Ticker(idx_symbol)
                info = await asyncio.to_thread(ticker.info)

                price = info.get('regularMarketPrice', info.get('currentPrice', 0.0))
                prev_close = info.get('regularMarketPreviousClose', info.get('previousClose', price))

                change = (price - prev_close) if pd.notna(price) and pd.notna(prev_close) else 0.0
                change_pct = (change / prev_close) * 100 if prev_close and prev_close != 0 and pd.notna(change) else 0.0

                return idx_name, {
                    'price': float(price) if pd.notna(price) else 0.0,
                    'change': float(change) if pd.notna(change) else 0.0,
                    'change_pct': float(change_pct) if pd.notna(change_pct) else 0.0
                }
            except Exception as e: # pragma: no cover
                logger.error(f"Failed to fetch data for index {idx_name} ({idx_symbol}): {e}", exc_info=True)
                return idx_name, {'price': 0.0, 'change': 0.0, 'change_pct': 0.0, 'error': str(e)}

        index_symbols_to_fetch = [('^GSPC', 'S&P 500'), ('^IXIC', 'NASDAQ'), ('^DJI', 'Dow Jones')]
        vix_symbol_tuple = ('^VIX', 'VIX')

        tasks = [fetch_index_info(s, n) for s, n in index_symbols_to_fetch]
        tasks.append(fetch_index_info(vix_symbol_tuple[0], vix_symbol_tuple[1]))

        try:
            results = await asyncio.gather(*tasks)
            for name, data in results:
                if name == 'VIX':
                    overview.vix = data.get('price', 0.0)
                    if 'error' in data:
                        logger.warning(f"VIX data fetch failed: {data['error']}")
                else:
                    overview.indices[name] = data
        except Exception as e: # pragma: no cover
            logger.error(f"Critical error during market overview data aggregation: {e}", exc_info=True)

        if self.cache: self.cache.set(cache_key, overview.__dict__, self.CACHE_TTL['realtime'])
        return overview


    async def batch_get_stocks(self, symbols: List[str]) -> Dict[str, StockData]:
        """Get multiple stocks efficiently, returning a dict of successfully fetched StockData."""
        if not symbols:
            return {}

        tasks = []
        valid_symbols_for_tasks = []
        for sym_input in symbols:
            if not isinstance(sym_input, str) or not sym_input.strip():
                logger.warning(f"Invalid symbol '{sym_input}' in batch_get_stocks, skipping.")
                continue
            std_sym = sym_input.strip().upper()
            tasks.append(self.get_stock_data(std_sym))
            valid_symbols_for_tasks.append(std_sym)

        if not tasks:
            return {}

        fetched_data_list = await asyncio.gather(*tasks, return_exceptions=False)

        results_dict: Dict[str, StockData] = {}
        for i, data_item in enumerate(fetched_data_list):
            symbol_for_result = valid_symbols_for_tasks[i]
            if isinstance(data_item, StockData):
                results_dict[symbol_for_result] = data_item

        return results_dict

    async def get_options_chain(self, symbol: str) -> Dict[str, pd.DataFrame]:
        """Get options chain data. Returns a dict with 'calls', 'puts', 'expirations', or empty dict on error."""
        cache_key = f"options:{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            try:
                calls_df = pd.DataFrame(cached.get('calls', []))
                puts_df = pd.DataFrame(cached.get('puts', []))
                return {
                    'calls': calls_df,
                    'puts': puts_df,
                    'expirations': cached.get('expirations', [])
                }
            except Exception as e: # pragma: no cover
                logger.warning(f"Failed to load/reconstruct options chain from cache for {symbol}, refetching: {e}", exc_info=True)
                if self.cache: self.cache.set(cache_key, None, 0)

        empty_options_response = {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expirations': []}

        try:
            await self.rate_limiters['yfinance'].acquire()
            ticker = yf.Ticker(symbol)

            expirations_tuple = await asyncio.to_thread(ticker.options)

            if not expirations_tuple:
                logger.info(f"No option expirations found for {symbol} by yfinance.")
                if self.cache: self.cache.set(cache_key, {'calls': [], 'puts': [], 'expirations': []}, self.CACHE_TTL['daily'])
                return empty_options_response

            first_expiration = expirations_tuple[0]
            options_chain_obj = await asyncio.to_thread(ticker.option_chain, first_expiration)

            if options_chain_obj is None or (options_chain_obj.calls.empty and options_chain_obj.puts.empty):
                 logger.info(f"Options chain for {symbol} (exp: {first_expiration}) is empty or None from yfinance.")
                 if self.cache: self.cache.set(cache_key, {'calls': [], 'puts': [], 'expirations': list(expirations_tuple)}, self.CACHE_TTL['daily'])
                 return {'calls': pd.DataFrame(), 'puts': pd.DataFrame(), 'expirations': list(expirations_tuple)}

            cache_data = {
                'calls': options_chain_obj.calls.to_dict(orient='records') if not options_chain_obj.calls.empty else [],
                'puts': options_chain_obj.puts.to_dict(orient='records') if not options_chain_obj.puts.empty else [],
                'expirations': list(expirations_tuple)
            }
            if self.cache: self.cache.set(cache_key, cache_data, self.CACHE_TTL['realtime'])

            return {
                'calls': options_chain_obj.calls,
                'puts': options_chain_obj.puts,
                'expirations': list(expirations_tuple)
            }

        except requests.exceptions.RequestException as e: # pragma: no cover
            logger.error(f"Network-related error fetching options for {symbol}: {e}", exc_info=True)
        except AttributeError as e: # pragma: no cover
            logger.error(f"Attribute error with yfinance Ticker object for options ({symbol}): {e}", exc_info=True)
        except IndexError as e: # pragma: no cover
             logger.error(f"Index error accessing option expirations for {symbol}: {e}", exc_info=True)
        except Exception as e:
            err_str = str(e).lower()
            if "no options chain found" in err_str or "options chain is not available" in err_str :
                logger.info(f"yfinance reported no options chain found for {symbol}: {e}")
                if self.cache: self.cache.set(cache_key, {'calls': [], 'puts': [], 'expirations': []}, self.CACHE_TTL['daily'])
            else: # pragma: no cover
                logger.error(f"Error fetching options for {symbol} with yfinance: {e.__class__.__name__} - {e}", exc_info=True)

        return empty_options_response


    # Helper methods (ensure these are robust against bad data from yfinance)
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI. Returns RSI value or 50.0 on error/insufficient data."""
        if not isinstance(prices, pd.Series) or prices.empty or len(prices) < period + 1:
            logger.debug(f"RSI: Input is not a valid Series or not enough data (need {period+1}, got {len(prices) if isinstance(prices, pd.Series) else 0}). Default 50.0.")
            return 50.0
        try:
            delta = prices.diff()
            gain = delta.where(delta > 0, 0.0).astype(float)
            loss = (-delta.where(delta < 0, 0.0)).astype(float)

            avg_gain = gain.rolling(window=period, min_periods=1).mean()
            avg_loss = loss.rolling(window=period, min_periods=1).mean()

            rs = avg_gain / avg_loss.replace(0, 1e-9)
            rsi_val = 100.0 - (100.0 / (1.0 + rs)) # Renamed to avoid conflict

            last_rsi = rsi_val.iloc[-1]
            return float(last_rsi) if pd.notna(last_rsi) else 50.0
        except ZeroDivisionError: # pragma: no cover
            logger.warning("RSI: Division by zero encountered. Defaulting to 50.0.")
            return 50.0
        except IndexError: # pragma: no cover
            logger.warning("RSI: Index error, likely from .iloc[-1] on unexpectedly empty series. Defaulting to 50.0.")
            return 50.0
        except Exception as e: # pragma: no cover
            logger.error(f"RSI: Error calculating: {e}", exc_info=True)
            return 50.0


    def _calculate_macd(self, prices: pd.Series) -> Dict[str, float]:
        """Calculate MACD. Returns dict with MACD values or defaults on error."""
        default_macd = {'macd': 0.0, 'signal': 0.0, 'histogram': 0.0}
        if not isinstance(prices, pd.Series) or prices.empty or len(prices) < 26:
            logger.debug(f"MACD: Input is not a valid Series or not enough data (need ~26, got {len(prices) if isinstance(prices, pd.Series) else 0}). Returning defaults.")
            return default_macd
        try:
            exp1 = prices.ewm(span=12, adjust=False).mean()
            exp2 = prices.ewm(span=26, adjust=False).mean()
            macd_line = exp1 - exp2
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            histogram = macd_line - signal_line

            last_macd = macd_line.iloc[-1]
            last_signal = signal_line.iloc[-1]
            last_hist = histogram.iloc[-1]

            return {
                'macd': float(last_macd) if pd.notna(last_macd) else 0.0,
                'signal': float(last_signal) if pd.notna(last_signal) else 0.0,
                'histogram': float(last_hist) if pd.notna(last_hist) else 0.0,
            }
        except IndexError: # pragma: no cover
            logger.warning("MACD: Index error, likely from .iloc[-1] on unexpectedly empty series. Returning defaults.")
            return default_macd
        except Exception as e: # pragma: no cover
            logger.error(f"MACD: Error calculating: {e}", exc_info=True)
            return default_macd


    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20) -> Dict[str, float]:
        """Calculate Bollinger Bands. Returns dict with band values or defaults on error."""
        default_bands = {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}
        if not isinstance(prices, pd.Series) or prices.empty or len(prices) < period:
            logger.debug(f"Bollinger Bands: Input is not valid Series or not enough data (need {period}, got {len(prices) if isinstance(prices, pd.Series) else 0}). Returning defaults.")
            return default_bands
        try:
            sma = prices.rolling(window=period).mean()
            std = prices.rolling(window=period).std(ddof=0)

            upper_band = sma + (std * 2)
            lower_band = sma - (std * 2)

            last_upper = upper_band.iloc[-1]
            last_middle = sma.iloc[-1]
            last_lower = lower_band.iloc[-1]
            last_price = prices.iloc[-1]

            return {
                'upper': float(last_upper) if pd.notna(last_upper) else (float(last_price) if pd.notna(last_price) else 0.0),
                'middle': float(last_middle) if pd.notna(last_middle) else (float(last_price) if pd.notna(last_price) else 0.0),
                'lower': float(last_lower) if pd.notna(last_lower) else (float(last_price) if pd.notna(last_price) else 0.0),
            }
        except IndexError: # pragma: no cover
            logger.warning("Bollinger Bands: Index error. Returning defaults.")
            return default_bands
        except Exception as e: # pragma: no cover
            logger.error(f"Bollinger Bands: Error calculating: {e}", exc_info=True)
            return default_bands


    def _find_support_resistance(self, hist: pd.DataFrame) -> Tuple[List[float], List[float]]:
        """Find support and resistance levels. Returns (support_list, resistance_list)."""
        if not isinstance(hist, pd.DataFrame) or hist.empty or not all(col in hist.columns for col in ['High', 'Low']):
            logger.debug("S/R: Input is not valid DataFrame or missing High/Low columns.")
            return [], []

        try:
            highs = hist['High'].astype(float).values
            lows = hist['Low'].astype(float).values

            if len(highs) < 10:
                logger.debug("S/R: Not enough data points for calculation.")
                return [], []

            resistance_levels = []
            support_levels = []

            window = 5
            if len(highs) <= window * 2:
                window = max(1, (len(highs) // 2) -1 )
                if window % 2 == 0 and window > 1 : window -=1
                if window == 0: window = 1

            for i in range(window, len(highs) - window):
                if highs[i] == np.max(highs[i-window : i+window+1]):
                    resistance_levels.append(highs[i])

            for i in range(window, len(lows) - window):
                if lows[i] == np.min(lows[i-window : i+window+1]):
                    support_levels.append(lows[i])

            if resistance_levels:
                unique_res = sorted(list(set(np.round(resistance_levels, 2))), reverse=True)
                resistance_levels = unique_res[:min(3, len(unique_res))]
            if support_levels:
                unique_sup = sorted(list(set(np.round(support_levels, 2))))
                support_levels = unique_sup[:min(3, len(unique_sup))]

            return support_levels, resistance_levels
        except Exception as e: # pragma: no cover
            logger.error(f"S/R: Error finding support/resistance: {e}", exc_info=True)
            return [], []


    def _detect_trend(self, prices: pd.Series) -> str:
        """Detect price trend. Returns 'bullish', 'bearish', or 'neutral'."""
        min_len_short = 20
        min_len_long = 50

        if not isinstance(prices, pd.Series) or prices.empty or len(prices) < min_len_long:
            logger.debug(f"Trend Detection: Not enough data (need {min_len_long}, got {len(prices) if isinstance(prices, pd.Series) else 0}). Default neutral.")
            return "neutral"

        try:
            prices_float = prices.astype(float)
            ma_short = prices_float.rolling(window=min_len_short).mean()
            ma_long = prices_float.rolling(window=min_len_long).mean()

            last_price = prices_float.iloc[-1]
            last_ma_short = ma_short.iloc[-1]
            last_ma_long = ma_long.iloc[-1]

            if not all(pd.notna([last_price, last_ma_short, last_ma_long])): # pragma: no cover
                logger.warning("Trend Detection: NaN value encountered in MAs or price. Defaulting to neutral.")
                return "neutral"

            is_bullish = last_price > last_ma_short > last_ma_long
            is_bearish = last_price < last_ma_short < last_ma_long

            if is_bullish:
                return "bullish"
            elif is_bearish:
                return "bearish"
            else:
                return "neutral"
        except IndexError: # pragma: no cover
            logger.warning("Trend Detection: Index error. Defaulting to neutral.")
            return "neutral"
        except Exception as e: # pragma: no cover
            logger.error(f"Trend Detection: Error: {e}", exc_info=True)
            return "neutral"


    def _calculate_momentum_score(self, prices: pd.Series, volume: pd.Series) -> float:
        """Calculate momentum score (0-100). Returns 50.0 on error."""
        min_len_price_short = 20
        min_len_price_long = 60
        min_len_vol_short = 5
        min_len_vol_long = 20

        if not isinstance(prices, pd.Series) or prices.empty or \
           not isinstance(volume, pd.Series) or volume.empty or \
           len(prices) < min_len_price_long or len(volume) < min_len_vol_long: # pragma: no cover
            logger.debug(f"Momentum Score: Insufficient data. Prices len: {len(prices) if isinstance(prices,pd.Series) else 0}, Volume len: {len(volume) if isinstance(volume,pd.Series) else 0}.")
            return 50.0

        try:
            prices_float = prices.astype(float)
            volume_float = volume.astype(float)

            returns = prices_float.pct_change()
            mom_1m = returns.tail(min_len_price_short).mean() if len(returns) >= min_len_price_short else 0.0
            mom_3m = returns.tail(min_len_price_long).mean() if len(returns) >= min_len_price_long else mom_1m

            mom_1m = mom_1m if pd.notna(mom_1m) else 0.0
            mom_3m = mom_3m if pd.notna(mom_3m) else 0.0
            price_mom_component = 50 + ( (mom_1m * 0.6 + mom_3m * 0.4) / 0.005 * 50 )
            price_mom_component = max(0.0, min(100.0, price_mom_component))

            vol_hist_avg = volume_float.rolling(window=min_len_vol_long, min_periods=1).mean().iloc[-1]
            vol_recent_avg = volume_float.tail(min_len_vol_short).mean()

            vol_hist_avg = vol_hist_avg if pd.notna(vol_hist_avg) and vol_hist_avg > 1e-6 else 1.0
            vol_recent_avg = vol_recent_avg if pd.notna(vol_recent_avg) else vol_hist_avg
            vol_ratio = vol_recent_avg / vol_hist_avg
            volume_mom_component = 50 + (vol_ratio - 1.0) * 50
            volume_mom_component = max(0.0, min(100.0, volume_mom_component))

            rsi_val = self._calculate_rsi(prices_float)

            final_score = (price_mom_component * 0.5) + (rsi_val * 0.3) + (volume_mom_component * 0.2)

            return max(0.0, min(100.0, float(final_score)))
        except Exception as e: # pragma: no cover
            logger.error(f"Momentum Score: Error calculating: {e}", exc_info=True)
            return 50.0


    def _volume_trend(self, volume: pd.Series) -> str:
        """Determine volume trend. Returns 'increasing', 'decreasing', or 'normal'."""
        min_len_short = 5
        min_len_long = 20
        if not isinstance(volume, pd.Series) or volume.empty or len(volume) < min_len_long: # pragma: no cover
            logger.debug(f"Volume Trend: Insufficient data (need {min_len_long}, got {len(volume) if isinstance(volume,pd.Series) else 0}).")
            return "normal"

        try:
            volume_float = volume.astype(float)
            historical_avg = volume_float.rolling(window=min_len_long, min_periods=1).mean().iloc[-1]
            recent_avg = volume_float.tail(min_len_short).mean()

            if not (pd.notna(recent_avg) and pd.notna(historical_avg)): # pragma: no cover
                logger.debug("Volume Trend: NaN encountered in average calculations.")
                return "normal"

            if historical_avg < 1e-6:
                return "normal" if recent_avg < 1e-6 else "increasing"

            ratio = recent_avg / historical_avg

            if ratio > 1.5:
                return "increasing"
            elif ratio < 0.7:
                return "decreasing"
            else:
                return "normal"
        except Exception as e: # pragma: no cover
            logger.error(f"Volume Trend: Error determining: {e}", exc_info=True)
            return "normal"


    def _calculate_financial_ratios(self, info: Dict) -> Dict[str, Optional[float]]:
        """Calculate additional financial ratios from yfinance info dict. Returns dict with float or None values."""
        ratios: Dict[str, Optional[float]] = {}
        if not info:
            logger.warning("Financial Ratios: Input info dict is empty. Cannot calculate ratios.")
            ratio_keys = ['gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa',
                          'pe_ratio', 'forward_pe', 'peg_ratio', 'price_to_book',
                          'price_to_sales', 'ev_to_ebitda', 'current_ratio',
                          'quick_ratio', 'debt_to_equity', 'interest_coverage']
            return {key: None for key in ratio_keys}


        def get_ratio(key: str) -> Optional[float]:
            val = info.get(key)
            if isinstance(val, (int, float)) and pd.notna(val):
                return float(val)
            if isinstance(val, str) and val.lower() == 'none': # pragma: no cover
                 return None
            if val is None or (isinstance(val, (dict,list)) and not val) :
                 return None
            if isinstance(val, str): # pragma: no cover
                try: return float(val)
                except ValueError: return None
            return None

        try:
            ratios['gross_margin'] = get_ratio('grossMargins')
            ratios['operating_margin'] = get_ratio('operatingMargins')
            ratios['net_margin'] = get_ratio('profitMargins')
            ratios['roe'] = get_ratio('returnOnEquity')
            ratios['roa'] = get_ratio('returnOnAssets')
            ratios['pe_ratio'] = get_ratio('trailingPE')
            ratios['forward_pe'] = get_ratio('forwardPE')
            ratios['peg_ratio'] = get_ratio('pegRatio')
            ratios['price_to_book'] = get_ratio('priceToBook')
            ratios['price_to_sales'] = get_ratio('priceToSalesTrailing12Months')
            ratios['ev_to_ebitda'] = get_ratio('enterpriseToEbitda')
            ratios['current_ratio'] = get_ratio('currentRatio')
            ratios['quick_ratio'] = get_ratio('quickRatio')
            ratios['debt_to_equity'] = get_ratio('debtToEquity')
            ratios['interest_coverage'] = get_ratio('interestCoverage')

        except Exception as e: # pragma: no cover
            logger.error(f"Financial Ratios: Error calculating: {e}. Info keys: {list(info.keys())[:10]}...", exc_info=True)
        return ratios


    def _calculate_growth_rates(self, income_stmt_df: pd.DataFrame) -> Dict[str, Optional[float]]:
        """Calculate growth rates from financial statements DataFrame. Returns dict with float or None values."""
        growth_rates: Dict[str, Optional[float]] = {
            'revenue_growth_yoy': None, 'earnings_growth_yoy': None,
        }
        if not isinstance(income_stmt_df, pd.DataFrame) or income_stmt_df.empty or len(income_stmt_df.columns) < 2:
            logger.debug(f"Growth Rates: Income statement DataFrame is not valid or has insufficient periods. Columns: {income_stmt_df.columns if isinstance(income_stmt_df, pd.DataFrame) else 'Not a DF'}")
            return growth_rates

        try:
            if all(isinstance(col, pd.Timestamp) for col in income_stmt_df.columns):
                sorted_periods = sorted(income_stmt_df.columns, reverse=True)
            else:
                try: # pragma: no cover
                    date_cols = pd.to_datetime(income_stmt_df.columns, errors='coerce')
                    valid_date_cols = income_stmt_df.columns[pd.notna(date_cols)].tolist()
                    if len(valid_date_cols) < 2:
                        logger.debug(f"Growth Rates: Not enough valid date columns for sorting. Using original order if >1 col: {income_stmt_df.columns}")
                        sorted_periods = income_stmt_df.columns.tolist() if len(income_stmt_df.columns) >=2 else []
                    else:
                        sorted_periods = sorted(valid_date_cols, key=lambda x: pd.to_datetime(x), reverse=True)
                except Exception: # pragma: no cover
                     logger.warning(f"Growth Rates: Could not reliably sort income statement periods. Using first two available. Columns: {income_stmt_df.columns}")
                     sorted_periods = income_stmt_df.columns.tolist() if len(income_stmt_df.columns) >=2 else []


            if len(sorted_periods) < 2: # pragma: no cover
                logger.debug(f"Growth Rates: Not enough periods after sorting ({len(sorted_periods)} found).")
                return growth_rates

            current_period_col = sorted_periods[0]
            previous_period_col = sorted_periods[1]

            revenue_names = ['Total Revenue', 'Revenue', 'Total Operating Revenue', 'Net Revenue', 'Total Revenues']
            ni_names = ['Net Income', 'Net Income Common Stockholders', 'Net Earnings', 'Net Income From Continuing Ops', 'Net Income Applicable To Common Shares']

            def get_financial_value(df, period_col, item_names_list):
                for name in item_names_list:
                    if name in df.index: # Check if row exists
                        val = df.loc[name, period_col]
                        if pd.notna(val) and isinstance(val, (int, float)): return float(val) # Ensure it's a number
                return None

            rev_current = get_financial_value(income_stmt_df, current_period_col, revenue_names)
            rev_previous = get_financial_value(income_stmt_df, previous_period_col, revenue_names)

            if rev_current is not None and rev_previous is not None and rev_previous != 0:
                growth_rates['revenue_growth_yoy'] = (rev_current - rev_previous) / abs(rev_previous)

            ni_current = get_financial_value(income_stmt_df, current_period_col, ni_names)
            ni_previous = get_financial_value(income_stmt_df, previous_period_col, ni_names)

            if ni_current is not None and ni_previous is not None:
                if ni_previous != 0:
                    growth_rates['earnings_growth_yoy'] = (ni_current - ni_previous) / abs(ni_previous)
                elif ni_current != 0 :
                     growth_rates['earnings_growth_yoy'] = 1.0 if ni_current > 0 else -1.0

        except KeyError as e: # pragma: no cover
            logger.warning(f"Growth Rates: Missing expected financial data field '{e}'. Index: {income_stmt_df.index.tolist()[:5]}, Cols: {income_stmt_df.columns.tolist()[:5]}")
        except IndexError as e: # pragma: no cover
            logger.warning(f"Growth Rates: Index error, likely not enough columns/periods: {e}")
        except Exception as e: # pragma: no cover
            logger.error(f"Growth Rates: Error calculating: {e}", exc_info=True)

        return growth_rates


    async def _get_peer_comparison(self, symbol: str, sector: Optional[str]) -> Dict[str, Any]:
        """Get peer comparison data. Placeholder - returns empty dict."""
        if not sector or sector == 'N/A':
            logger.debug(f"No sector provided for {symbol} or sector is 'N/A', cannot fetch peer comparison.")
            return {'error': 'Sector information not available for peer comparison.'}

        logger.info(f"Peer comparison for {symbol} (sector: {sector}) is a placeholder and not currently implemented. Returning default structure.")
        return {
            'sector': sector,
            'message': 'Peer comparison is not implemented in this version.',
            'average_metrics': {
                'avg_pe_ratio': None, 'avg_roe': None, 'avg_revenue_growth_yoy': None,
            },
            'peers_data': []
        }


    async def get_insider_trading(self, symbol: str) -> List[Dict[str, Any]]:
        """Get insider trading data. Returns list of trades or empty list on error."""
        cache_key = f"insider:{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        try:
            await self.rate_limiters['yfinance'].acquire()
            ticker = yf.Ticker(symbol)

            insider_tx_df = await asyncio.to_thread(ticker.insider_transactions)

            if insider_tx_df is not None and not insider_tx_df.empty and isinstance(insider_tx_df, pd.DataFrame):
                for col_name in ['Start Date', 'End Date']:
                    if col_name in insider_tx_df.columns and pd.api.types.is_datetime64_any_dtype(insider_tx_df[col_name]): # pragma: no cover
                        insider_tx_df[col_name] = insider_tx_df[col_name].dt.strftime('%Y-%m-%d')

                trades = insider_tx_df.to_dict(orient='records')
                if self.cache: self.cache.set(cache_key, trades, self.CACHE_TTL['daily'])
                return trades
            else: # pragma: no cover
                logger.info(f"No 'insider_transactions' data found for {symbol} or data is not a DataFrame.")
                if self.cache: self.cache.set(cache_key, [], self.CACHE_TTL['daily'])
                return []

        except requests.exceptions.RequestException as e: # pragma: no cover
            logger.error(f"Network error fetching insider trading for {symbol}: {e}", exc_info=True)
        except AttributeError as e: # pragma: no cover
            logger.warning(f"The yfinance Ticker object for {symbol} may not have the 'insider_transactions' attribute: {e}")
        except Exception as e:
            err_str = str(e).lower()
            if "no data found" in err_str or "failed to get data" in err_str or "symbol may be delisted" in err_str: # pragma: no cover
                 logger.info(f"yfinance reported no insider trading data for {symbol}: {e}")
                 if self.cache: self.cache.set(cache_key, [], self.CACHE_TTL['daily'])
            else: # pragma: no cover
                logger.error(f"Error fetching insider trading for {symbol} with yfinance: {e.__class__.__name__} - {e}", exc_info=True)

        return []


    async def get_analyst_ratings(self, symbol: str) -> Dict[str, Any]:
        """Get analyst ratings and price targets. Returns dict with data or defaults on error."""
        cache_key = f"analysts:{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        default_ratings = {'current': {}, 'history': [], 'consensus': 'N/A', 'price_targets': {}}

        try:
            await self.rate_limiters['yfinance'].acquire()
            ticker = yf.Ticker(symbol)

            results = await asyncio.gather(
                asyncio.to_thread(getattr, ticker, 'recommendations', None),
                asyncio.to_thread(getattr, ticker, 'analyst_price_target', None),
                asyncio.to_thread(getattr, ticker, 'recommendations_summary', None),
                return_exceptions=True
            )
            recommendations_df, analyst_price_target_data, recommendations_summary_data = results

            ratings_history = []
            current_rating_dict = {}
            consensus_str = "N/A"
            price_target_info = {}

            if isinstance(recommendations_df, pd.DataFrame) and not recommendations_df.empty:
                latest_recs = recommendations_df.reset_index().sort_values(by='Date', ascending=False).head(20)

                if not latest_recs.empty:
                    current_raw = latest_recs.iloc[0]
                    current_rating_dict = {
                        'firm': current_raw.get('Firm', 'N/A'),
                        'to_grade': current_raw.get('To Grade', 'N/A'),
                        'from_grade': current_raw.get('From Grade', 'N/A'),
                        'action': current_raw.get('Action', 'N/A'),
                        'date': current_raw.get('Date', current_raw.get('index', pd.NaT)).strftime('%Y-%m-%d')
                                if pd.notna(current_raw.get('Date', current_raw.get('index'))) else 'N/A'
                    }
                ratings_history = [] # Initialize to ensure it's a list
                for _, row in latest_recs.iterrows():
                    item = row.to_dict()
                    date_val = item.get('Date', item.get('index')) # Prefer 'Date' column if exists
                    if pd.notna(date_val) and isinstance(date_val, pd.Timestamp):
                        item['Date'] = date_val.strftime('%Y-%m-%d')
                    elif pd.notna(date_val):
                        item['Date'] = str(date_val)
                    else: # pragma: no cover
                        item['Date'] = 'N/A'
                    if 'index' in item and isinstance(item['index'], pd.Timestamp):
                        del item['index']
                    ratings_history.append(item)

                consensus_str = self._calculate_consensus(recommendations_df)
            elif isinstance(recommendations_df, Exception): # pragma: no cover
                 logger.error(f"Failed to fetch recommendations for {symbol}: {recommendations_df}", exc_info=recommendations_df)

            pt_data_source = analyst_price_target_data
            if isinstance(pt_data_source, Exception): # pragma: no cover
                logger.error(f"Failed to fetch analyst price target for {symbol}: {pt_data_source}", exc_info=pt_data_source)
                pt_data_source = None

            if pt_data_source is not None:
                 pt_data_dict = {}
                 if isinstance(pt_data_source, pd.DataFrame) and not pt_data_source.empty:
                     pt_data_dict = pt_data_source.iloc[0].to_dict()
                 elif isinstance(pt_data_source, pd.Series) and not pt_data_source.empty: # pragma: no cover
                     pt_data_dict = pt_data_source.to_dict()

                 if pt_data_dict:
                     price_target_info = {
                         'mean': pt_data_dict.get('Mean Target', pt_data_dict.get('TargetMean', pt_data_dict.get('priceTargetMean'))),
                         'high': pt_data_dict.get('High Target', pt_data_dict.get('TargetHigh', pt_data_dict.get('priceTargetHigh'))),
                         'low': pt_data_dict.get('Low Target', pt_data_dict.get('TargetLow', pt_data_dict.get('priceTargetLow'))),
                         'median': pt_data_dict.get('Median Target', pt_data_dict.get('TargetMedian', pt_data_dict.get('priceTargetMedian'))),
                         'last_updated': pt_data_dict.get('Last Updated', getattr(pt_data_source, 'name', None))
                     }
                     if isinstance(price_target_info['last_updated'], pd.Timestamp):
                         price_target_info['last_updated'] = price_target_info['last_updated'].strftime('%Y-%m-%d')
                     elif price_target_info['last_updated'] is not None:
                          price_target_info['last_updated'] = str(price_target_info['last_updated'])

            if consensus_str == "N/A": # pragma: no cover
                if isinstance(recommendations_summary_data, pd.DataFrame) and not recommendations_summary_data.empty:
                    if 'recommendationKey' in recommendations_summary_data.columns:
                        key = recommendations_summary_data['recommendationKey'].iloc[0]
                        consensus_str = key.replace("_", " ").title() if isinstance(key, str) else "N/A"
                elif isinstance(recommendations_summary_data, Exception):
                    logger.error(f"Failed to fetch recommendations summary for {symbol}: {recommendations_summary_data}", exc_info=recommendations_summary_data)


            final_ratings_data = {
                'current': current_rating_dict, 'history': ratings_history,
                'consensus': consensus_str, 'price_targets': price_target_info
            }
            if self.cache: self.cache.set(cache_key, final_ratings_data, self.CACHE_TTL['daily'])
            return final_ratings_data

        except requests.exceptions.RequestException as e: # pragma: no cover
            logger.error(f"Network-related error during analyst ratings fetch for {symbol}: {e}", exc_info=True)
        except AttributeError as e: # pragma: no cover
            logger.error(f"Attribute error with yfinance objects for analyst ratings of {symbol}: {e}", exc_info=True)
        except Exception as e:
             err_str = str(e).lower()
             if "no data found" in err_str or "failed to get data" in err_str or "symbol may be delisted" in err_str: # pragma: no cover
                 logger.info(f"yfinance reported no analyst rating data for {symbol}: {e}")
             else: # pragma: no cover
                logger.error(f"Unexpected error fetching analyst ratings for {symbol}: {e.__class__.__name__} - {e}", exc_info=True)

        return default_ratings


    def _calculate_consensus(self, recommendations: pd.DataFrame) -> str:
        """Calculate consensus recommendation. Returns string like 'Buy', 'Hold', 'Sell', or 'N/A'."""
        if not isinstance(recommendations, pd.DataFrame) or recommendations.empty:
            logger.debug("Cannot calculate consensus: recommendations data is not a DataFrame or is empty.")
            return "N/A"

        grade_column_candidates = ['To Grade', 'Grade', 'Action', 'Recommendation']
        grade_column = None
        for col_candidate in grade_column_candidates:
            if col_candidate in recommendations.columns:
                grade_column = col_candidate
                break

        if not grade_column: # pragma: no cover
            logger.debug(f"No suitable grade column found in recommendations for consensus. Columns: {recommendations.columns.tolist()}")
            return "N/A"

        try:
            def standardize_grade(grade_obj):
                if not isinstance(grade_obj, str):
                    return 'hold'

                grade_lower = grade_obj.lower()
                if any(s in grade_lower for s in ['strong buy', 'buy', 'outperform', 'overweight', 'accumulate', 'long-term buy', 'top pick', 'add', 'positive', 'market outperform']):
                    return 'buy'
                if any(s in grade_lower for s in ['sell', 'strong sell', 'underperform', 'underweight', 'reduce', 'market underperform', 'negative']):
                    return 'sell'
                if any(s in grade_lower for s in ['hold', 'neutral', 'equal-weight', 'market perform', 'perform', 'fair value', 'peer perform', 'sector perform', 'in-line']):
                    return 'hold'

                if grade_column == 'Action': # pragma: no cover
                    if 'up' in grade_lower: return 'buy'
                    if 'down' in grade_lower: return 'sell'
                    if 'init' in grade_lower or 'main' in grade_lower or 'reit' in grade_lower: return 'hold'

                logger.debug(f"Unrecognized grade for consensus: '{grade_obj}' (column: {grade_column}) -> defaulting to hold.")
                return 'hold'

            standardized_grades = recommendations[grade_column].apply(standardize_grade)
            rec_counts = standardized_grades.value_counts()

            buy_count = rec_counts.get('buy', 0)
            sell_count = rec_counts.get('sell', 0)
            hold_count = rec_counts.get('hold', 0)

            total_valid_recs = buy_count + sell_count + hold_count
            if total_valid_recs == 0: # pragma: no cover
                logger.debug("No valid standardized recommendations found to calculate consensus.")
                return "N/A"

            score = (buy_count * 1) + (sell_count * -1)

            if score / total_valid_recs > 0.33:
                return "Buy"
            elif score / total_valid_recs < -0.33:
                return "Sell"
            else:
                return "Hold"

        except KeyError as e: # pragma: no cover
            logger.error(f"KeyError: '{e}' column missing in recommendations DataFrame for consensus calculation.", exc_info=True)
            return "N/A"
        except Exception as e: # pragma: no cover
            logger.error(f"Error calculating consensus recommendation: {e}", exc_info=True)
            return "N/A"


    async def search_stocks(self, query: str) -> List[Dict[str, str]]:
        """Search for stocks by name or symbol. Placeholder - not implemented."""
        if not query or not query.strip():
            logger.debug("Search stocks: received empty query.")
            return []

        logger.info(f"Stock search functionality for query '{query}' is a placeholder and not implemented in this version.")
        return []


    def cleanup(self):
        """Cleanup resources like the ThreadPoolExecutor."""
        try:
            logger.info("Shutting down ThreadPoolExecutor...")
            self.executor.shutdown(wait=True)
            logger.info("ThreadPoolExecutor shut down successfully.")
        except Exception as e: # pragma: no cover
            logger.error(f"Error shutting down ThreadPoolExecutor: {e}", exc_info=True)


