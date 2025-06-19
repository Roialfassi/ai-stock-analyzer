# screener.py - Natural Language Stock Screener

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
from datetime import datetime
import asyncio
import logging

from models import (
    StockData, ScreeningQuery, ScreeningResult
)
from market_data import MarketDataProvider
from llm_analyzer import NaturalLanguageScreener

logger = logging.getLogger(__name__)


class StockScreener:
    """Main stock screening engine"""

    def __init__(self, data_provider: MarketDataProvider, nl_screener: Optional[NaturalLanguageScreener]):
        if data_provider is None: # Should ideally be enforced by type hints in Python 3.9+ if using a linter
            logger.critical("StockScreener initialized with no DataProvider. Screener will not function.")
            raise ValueError("DataProvider cannot be None for StockScreener")

        self.data_provider = data_provider
        self.nl_screener = nl_screener # nl_screener can be None if LLM provider is not configured
        self.stock_universe: List[str] = []
        self.cache = ScreenerCache()

        try:
            self._load_stock_universe() # This can fail if Wikipedia is inaccessible
        except Exception as e: # pragma: no cover
            logger.error(f"Failed to load initial stock universe: {e}. Stock screener may have limited scope.", exc_info=True)
            # Fallback to a very basic list if loading fails entirely
            if not self.stock_universe:
                self.stock_universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"] # Minimal fallback

        self.preloaded_data: Dict[str, StockData] = {}
        try:
            # Using asyncio.create_task without storing it can sometimes hide exceptions
            # if not awaited or handled properly. For background tasks, ensure errors are logged within the task.
            self._preload_task = asyncio.create_task(self._preload_popular_stocks())
        except Exception as e: # pragma: no cover
            logger.error(f"Failed to create asyncio task for preloading stocks: {e}", exc_info=True)


    async def _preload_popular_stocks(self):
        """Preload data for popular stocks to speed up screening. Handles its own errors."""
        if not self.data_provider: return # Should not happen if __init__ check is in place

        # Use a slice of the universe; ensure universe is not empty
        popular_symbols = self.stock_universe[:50] if self.stock_universe else []
        if not popular_symbols:
            logger.info("No stock universe available for preloading.")
            return

        logger.info(f"Preloading data for {len(popular_symbols)} popular stocks...")
        try:
            for symbol in popular_symbols:
                try:
                    data = await self.data_provider.get_stock_data(symbol)
                    if data:
                        self.preloaded_data[symbol] = data
                except Exception as e_stock: # Catch error for a single stock
                    logger.warning(f"Error preloading stock data for {symbol}: {e_stock}", exc_info=True) # Log with traceback
            logger.info(f"Preloading complete. {len(self.preloaded_data)} stocks preloaded.")
        except Exception as e_main: # Catch any other unexpected error in the preload loop
            logger.error(f"Unexpected error during _preload_popular_stocks main loop: {e_main}", exc_info=True)


    async def _fetch_stock_data(self, symbols: List[str]) -> List[StockData]: # Note: This method seems unused, _fetch_stock_data_batch is used instead.
        """Fetch data for multiple stocks with preloaded cache (seems unused, keeping for now)"""
        stocks = []
        symbols_to_fetch = []

        # Use preloaded data where available
        for symbol in symbols:
            if symbol in self.preloaded_data:
                stocks.append(self.preloaded_data[symbol])
            else:
                symbols_to_fetch.append(symbol)

        # Fetch remaining stocks
        if symbols_to_fetch:
            fetched = await self._fetch_stock_data_batch(symbols_to_fetch)
            stocks.extend(fetched)

        return stocks

    def _load_stock_universe(self):
        """Load comprehensive list of stocks from multiple sources"""
        # Major US stocks - expanded list
        major_stocks = [
            # Tech giants
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "INTC", "AMD", "ORCL",
            "CRM", "ADBE", "NFLX", "CSCO", "AVGO", "QCOM", "TXN", "IBM", "MU", "AMAT",
            "LRCX", "ADI", "PYPL", "SQ", "SHOP", "UBER", "SNAP", "PINS", "ROKU", "DOCU",

            # Finance
            "JPM", "BAC", "WFC", "GS", "MS", "C", "USB", "PNC", "AXP", "BLK",
            "SCHW", "COF", "BK", "TFC", "SPGI", "CME", "ICE", "V", "MA", "PYPL",

            # Healthcare
            "JNJ", "UNH", "PFE", "ABBV", "TMO", "MRK", "ABT", "DHR", "CVS", "BMY",
            "AMGN", "GILD", "MDT", "ISRG", "VRTX", "REGN", "ZTS", "MRNA", "BIIB", "ILMN",

            # Consumer
            "WMT", "HD", "PG", "KO", "PEP", "COST", "MCD", "NKE", "SBUX", "TGT",
            "LOW", "CVX", "XOM", "DIS", "CMCSA", "NFLX", "T", "VZ", "TMUS", "CHTR",

            # Industrial
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "RTX", "LMT", "DE", "EMR",
            "ITW", "ETN", "NOC", "GD", "FDX", "NSC", "UNP", "CSX", "WM", "RSG",

            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "KMI",
            "WMB", "HAL", "BKR", "DVN", "HES", "MRO", "APA", "FANG", "CTRA", "OKE",

            # Real Estate
            "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "WELL", "AVB", "EQR", "DLR",
            "O", "SBAC", "WY", "VTR", "PEAK", "ARE", "MAA", "INVH", "UDR", "HST",

            # Materials
            "LIN", "APD", "SHW", "ECL", "DD", "NEM", "FCX", "DOW", "PPG", "CTVA",
            "ALB", "EMN", "CE", "VMC", "MLM", "NUE", "STLD", "CLF", "X", "RS",

            # Utilities
            "NEE", "DUK", "SO", "D", "SRE", "AEP", "EXC", "XEL", "PEG", "ED",
            "WEC", "ES", "AWK", "DTE", "ETR", "AEE", "CMS", "CNP", "ATO", "NI"
        ]

        # Get S&P 500 components from Wikipedia (using yfinance)
        try:
            sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            sp500_symbols = sp500_table['Symbol'].tolist()

            # Clean symbols (remove dots, special characters)
            sp500_symbols = [s.replace('.', '-') for s in sp500_symbols]

            # Combine with major stocks
            self.stock_universe = sorted(list(set(major_stocks + sp500_symbols))) # Sorted for consistency
        except requests.exceptions.RequestException as e: # More specific for network issues with read_html
            logger.error(f"Network error fetching S&P 500 list from Wikipedia: {e}. Falling back to predefined list.")
            self.stock_universe = sorted(list(set(major_stocks)))
        except ImportError: # If lxml or other parser for read_html is missing
             logger.error("Missing a parser library (like lxml) for pd.read_html. Falling back to predefined list.")
             self.stock_universe = sorted(list(set(major_stocks)))
        except Exception as e: # Catch any other error during Wikipedia fetch
            logger.error(f"Error fetching or parsing S&P 500 list from Wikipedia: {e}. Falling back to predefined list.", exc_info=True)
            self.stock_universe = sorted(list(set(major_stocks)))


        # Add popular ETFs for sector screening
        self.etf_universe = {
            'SPY': 'S&P 500',
            'QQQ': 'Nasdaq 100',
            'DIA': 'Dow Jones',
            'IWM': 'Russell 2000',
            'XLK': 'Technology',
            'XLV': 'Healthcare',
            'XLF': 'Financials',
            'XLE': 'Energy',
            'XLY': 'Consumer Discretionary',
            'XLP': 'Consumer Staples',
            'XLI': 'Industrials',
            'XLB': 'Materials',
            'XLRE': 'Real Estate',
            'XLU': 'Utilities',
            'GLD': 'Gold',
            'SLV': 'Silver',
            'USO': 'Oil',
            'UNG': 'Natural Gas',
            'VNQ': 'Real Estate',
            'AGG': 'Bonds'
        }

        logger.info(f"Loaded {len(self.stock_universe)} stocks in universe")

    async def screen_stocks(self, query: str, use_cache: bool = True) -> ScreeningResult:
        """Screen stocks based on natural language query with efficient data fetching"""
        start_time = datetime.now()

        # Check cache first
        start_time = datetime.now()
        cache_key = f"screen:{query}"
        default_empty_result = ScreeningResult(
            query=ScreeningQuery(raw_query=query, parsed_criteria={}, filters=[]),
            matches=[], total_count=0, execution_time=0, explanations={}
        )

        if use_cache:
            try:
                cached = self.cache.get(cache_key)
                if cached:
                    # Ensure cached data can be unpacked into ScreeningResult
                    return ScreeningResult(**cached)
            except Exception as e_cache: # pragma: no cover
                logger.warning(f"Error reading from screener cache for query '{query}': {e_cache}. Refetching.")

        if not self.nl_screener:
            logger.error("NaturalLanguageScreener is not available. Cannot parse query.")
            # Return an error or a result indicating failure
            default_empty_result.explanations = {"error": "NL Screener service not available."}
            return default_empty_result

        parsed_query_dict: Dict[str, Any] = {}
        try:
            parsed_query_dict = await self.nl_screener.parse_query(query)
            if not parsed_query_dict or parsed_query_dict.get("error"): # Check if nl_screener returned an error
                error_msg = parsed_query_dict.get("error", "NL query parsing failed, no specific error given.")
                logger.error(f"Failed to parse natural language query '{query}': {error_msg}")
                default_empty_result.explanations = {"error": f"Query parsing failed: {error_msg}"}
                return default_empty_result
        except Exception as e_parse: # pragma: no cover
            logger.exception(f"Exception during NL query parsing for '{query}': {e_parse}")
            default_empty_result.explanations = {"error": f"Exception during query parsing: {e_parse}"}
            return default_empty_result

        try:
            screening_query_obj = ScreeningQuery(
                raw_query=query,
                parsed_criteria=parsed_query_dict, # Use the validated dict
                filters=parsed_query_dict.get('filters', []),
                sort_by=parsed_query_dict.get('sort_by', 'market_cap'),
                sort_order=parsed_query_dict.get('sort_order', 'desc'),
                limit=int(parsed_query_dict.get('limit', 50)) # Ensure limit is int
            )
        except (TypeError, ValueError) as e_sq: # pragma: no cover
            logger.error(f"Error creating ScreeningQuery object from parsed query: {e_sq}. Parsed: {parsed_query_dict}", exc_info=True)
            default_empty_result.explanations = {"error": f"Internal error processing parsed query: {e_sq}"}
            return default_empty_result

        # Initialize explanations in case of early exit
        explanations_dict: Dict[str, str] = {}

        try:
            stocks_to_fetch = self._optimize_stock_selection(parsed_query_dict)
            if not stocks_to_fetch: # If universe is empty or no suitable stocks
                logger.warning(f"No stocks selected for fetching based on query: {query}")
                # Fallback to default empty result, but with the parsed query
                default_empty_result.query = screening_query_obj
                return default_empty_result


            stock_data_list = await self._fetch_stock_data_batch(stocks_to_fetch)
            if not stock_data_list:
                logger.info(f"No stock data fetched for query: {query}")
                # Return empty result, but with the parsed query
                default_empty_result.query = screening_query_obj
                return default_empty_result


            filtered_stocks = self._apply_filters(stock_data_list, screening_query_obj.filters)

            if 'qualitative' in parsed_query_dict and parsed_query_dict['qualitative']:
                filtered_stocks = self._apply_qualitative_filters(filtered_stocks, parsed_query_dict['qualitative'])

            if self._needs_advanced_filtering(parsed_query_dict):
                filtered_stocks = await self._apply_advanced_filters(filtered_stocks, parsed_query_dict)

            sorted_stocks = self._sort_stocks(filtered_stocks, screening_query_obj.sort_by, screening_query_obj.sort_order)
            final_stocks = sorted_stocks[:screening_query_obj.limit]

            if final_stocks and self.nl_screener: # Only generate explanations if there are stocks and screener exists
                explanations_dict = await self.nl_screener.explain_matches(query, final_stocks)

        except Exception as e_screen_flow: # Catch errors in the main screening flow # pragma: no cover
            logger.exception(f"Error during stock screening process for query '{query}': {e_screen_flow}")
            default_empty_result.query = screening_query_obj # Keep the parsed query
            default_empty_result.explanations = {"error": f"Screening process failed: {e_screen_flow}"}
            return default_empty_result

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        result = ScreeningResult(
            query=screening_query,
            matches=final_stocks,
            total_count=len(final_stocks),
            execution_time=execution_time,
            explanations=explanations
        )

        # Cache result (even if it's an error state from above, to avoid re-processing bad queries quickly)
        if use_cache and self.cache:
            try:
                self.cache.set(cache_key, result.__dict__, 300)  # 5 min cache
            except Exception as e_cache_set: # pragma: no cover
                 logger.error(f"Error trying to set screener cache for query '{query}': {e_cache_set}")
        return result

    def _optimize_stock_selection(self, parsed_query: Dict[str, Any]) -> List[str]:
        """Optimize which stocks to fetch based on query. Returns a list of symbols."""
        try:
            qualitative = parsed_query.get('qualitative', {}) if isinstance(parsed_query, dict) else {}
            sector = qualitative.get('sector', '').lower() if isinstance(qualitative, dict) else ""

            # If sector specified, attempt to narrow down. This is a very rough heuristic.
            if sector:
                # This sector_map is a placeholder for a more dynamic sector classification system.
                sector_map = {
                    'technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'INTC', 'AMD', 'CRM', 'ADBE', 'ORCL'],
                    'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'CVS', 'BMY', 'AMGN'],
                    # ... (add more sectors and representative stocks)
                }
                for map_sector, example_stocks in sector_map.items():
                    if sector in map_sector or map_sector in sector: # Basic matching
                        logger.debug(f"Optimizing selection for sector '{sector}', using example stocks.")
                        # Return a mix of example stocks and some from general universe for diversity
                        return list(set(example_stocks + self.stock_universe[:50]))[:100] # Limit size

            # Check for market cap filters to potentially narrow down universe (very coarse)
            filters = parsed_query.get('filters', []) if isinstance(parsed_query, dict) else []
            for filter_def in filters:
                if isinstance(filter_def, dict) and filter_def.get('field') == 'market_cap':
                    operator = filter_def.get('operator')
                    value = filter_def.get('value', 0)
                    try:
                        numeric_value = float(value)
                        if operator in ['<', '<='] and numeric_value <= 2e9: # Small cap
                            logger.debug("Optimizing for small cap query.")
                            return self.stock_universe[-100:] if self.stock_universe else [] # Example: last 100 might be smaller
                        elif operator in ['>', '>='] and numeric_value >= 200e9: # Mega cap
                            logger.debug("Optimizing for mega cap query.")
                            return self.stock_universe[:50] if self.stock_universe else [] # Example: first 50
                    except ValueError: # pragma: no cover
                        logger.warning(f"Market cap filter value '{value}' is not numeric, cannot optimize selection.")

            # Default: return a sizable chunk of the general universe if no strong optimizer.
            # Ensure stock_universe is not empty before slicing.
            return self.stock_universe[:200] if self.stock_universe else []
        except Exception as e: # pragma: no cover
            logger.exception(f"Error during stock selection optimization: {e}. Defaulting to full universe (limited).")
            return self.stock_universe[:200] if self.stock_universe else []


    async def _fetch_stock_data_batch(self, symbols: List[str], batch_size: int = 50) -> List[StockData]:
        """Fetch stock data in batches for efficiency"""
        if not self.data_provider: # Should be caught by __init__ but defensive
            logger.error("Fetch batch: MarketDataProvider not available.")
            return []
        if not symbols:
            return []

        all_fetched_stocks: List[StockData] = []
        try:
            for i in range(0, len(symbols), batch_size):
                current_batch_symbols = symbols[i:i + batch_size]

                # Create tasks for the current batch
                tasks = [self.data_provider.get_stock_data(s) for s in current_batch_symbols]

                # Gather results for the batch, allowing individual tasks to fail
                # get_stock_data itself returns None on error, so exceptions here are less likely
                # unless gather itself has an issue or get_stock_data re-raises something unexpected.
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                for symbol_idx, res_or_exc in enumerate(batch_results):
                    original_symbol = current_batch_symbols[symbol_idx]
                    if isinstance(res_or_exc, StockData):
                        all_fetched_stocks.append(res_or_exc)
                    elif isinstance(res_or_exc, Exception): # pragma: no cover
                        logger.error(f"Exception fetching data for {original_symbol} in batch: {res_or_exc}", exc_info=res_or_exc)
                    # If res_or_exc is None (get_stock_data handled its error), it's skipped.
            return all_fetched_stocks
        except Exception as e: # Catch errors related to batching or asyncio.gather itself
            logger.exception(f"Error during _fetch_stock_data_batch: {e}")
            return all_fetched_stocks # Return any stocks fetched so far


    def _needs_advanced_filtering(self, parsed_query: Dict[str, Any]) -> bool:
        """Check if query needs advanced filtering with historical data"""
        try:
            # Ensure parsed_query is a dict, and qualitative and characteristics are lists
            qualitative_data = parsed_query.get('qualitative', {}) if isinstance(parsed_query, dict) else {}
            characteristics = qualitative_data.get('characteristics', []) if isinstance(qualitative_data, dict) else []

            if not isinstance(characteristics, list): # pragma: no cover
                logger.warning(f"Advanced Filter Check: 'characteristics' is not a list: {characteristics}")
                return False

            advanced_keywords = [
                'momentum', 'trend', 'breakout', 'oversold', 'overbought',
                'volatility', 'beta', 'correlation', 'drawdown',
                'technical', 'rsi', 'macd', 'moving average', 'ma ', 'support', 'resistance' # Added more specific MA terms
            ]

            for char_item in characteristics:
                if not isinstance(char_item, str): continue # Skip non-string characteristics
                char_lower = char_item.lower()
                for keyword in advanced_keywords:
                    if keyword in char_lower:
                        return True
            return False
        except Exception as e: # pragma: no cover
            logger.exception(f"Error in _needs_advanced_filtering: {e}. Query: {parsed_query}")
            return False


    async def _apply_advanced_filters(self, stocks: List[StockData],
                                      parsed_query: Dict[str, Any]) -> List[StockData]:
        """Apply advanced filters that require historical data"""
        if not self.data_provider: # pragma: no cover
            logger.error("Advanced Filters: MarketDataProvider not available.")
            return stocks # Return original list if data provider is missing

        try:
            qualitative_data = parsed_query.get('qualitative', {}) if isinstance(parsed_query, dict) else {}
            characteristics = qualitative_data.get('characteristics', []) if isinstance(qualitative_data, dict) else []

            if not isinstance(characteristics, list): # pragma: no cover
                 logger.warning(f"Advanced Filter: 'characteristics' is not a list: {characteristics}")
                 return stocks

            current_filtered_stocks = list(stocks) # Work on a copy

            for char_item in characteristics:
                if not isinstance(char_item, str): continue
                char_lower = char_item.lower()

                next_batch_to_filter = [] # Stocks that passed previous characteristic filters

                if 'momentum' in char_lower or 'trending' in char_lower:
                    for stock in current_filtered_stocks:
                        try:
                            # get_technical_indicators often includes momentum scores or MAs
                            ti = await self.data_provider.get_technical_indicators(stock.symbol)
                            if ti and ti.momentum_score is not None and ti.momentum_score > 70: # Example threshold
                                next_batch_to_filter.append(stock)
                            elif ti and 'bullish' in ti.trend_direction.lower(): # Check trend
                                 next_batch_to_filter.append(stock)
                        except Exception as e_mom: # pragma: no cover
                            logger.warning(f"Error applying momentum/trend filter for {stock.symbol}: {e_mom}")
                    current_filtered_stocks = next_batch_to_filter

                elif 'oversold' in char_lower:
                    for stock in current_filtered_stocks:
                        try:
                            ti = await self.data_provider.get_technical_indicators(stock.symbol)
                            if ti and ti.rsi is not None and ti.rsi < 30:
                                next_batch_to_filter.append(stock)
                        except Exception as e_rsi: # pragma: no cover
                            logger.warning(f"Error applying oversold (RSI) filter for {stock.symbol}: {e_rsi}")
                    current_filtered_stocks = next_batch_to_filter

                elif 'overbought' in char_lower: # Added overbought
                    for stock in current_filtered_stocks:
                        try:
                            ti = await self.data_provider.get_technical_indicators(stock.symbol)
                            if ti and ti.rsi is not None and ti.rsi > 70:
                                next_batch_to_filter.append(stock)
                        except Exception as e_rsi_ob: # pragma: no cover
                             logger.warning(f"Error applying overbought (RSI) filter for {stock.symbol}: {e_rsi_ob}")
                    current_filtered_stocks = next_batch_to_filter


                elif 'breakout' in char_lower: # Near 52-week high and strong volume
                    for stock in current_filtered_stocks:
                        try:
                            # StockData should have year_high, current_price, volume, avg_volume
                            if stock.year_high is not None and stock.current_price is not None and \
                               stock.volume is not None and stock.avg_volume is not None and stock.avg_volume > 0:
                                if stock.current_price >= stock.year_high * 0.95 and \
                                   stock.volume > stock.avg_volume * 1.5: # Price near high and volume up
                                    next_batch_to_filter.append(stock)
                        except Exception as e_bo: # pragma: no cover
                            logger.warning(f"Error applying breakout filter for {stock.symbol}: {e_bo}")
                    current_filtered_stocks = next_batch_to_filter

            return current_filtered_stocks
        except Exception as e_adv: # pragma: no cover
            logger.exception(f"Unexpected error in _apply_advanced_filters: {e_adv}")
            return stocks # Return original list on major failure


    def _apply_filters(self, stocks: List[StockData], filters: List[Dict[str, Any]]) -> List[StockData]:
        """Apply quantitative filters to stocks"""
        if not filters:
            return stocks

        current_filtered_stocks = list(stocks) # Operate on a copy

        for filter_def in filters:
            if not isinstance(filter_def, dict): # pragma: no cover
                logger.warning(f"Skipping malformed filter definition (not a dict): {filter_def}")
                continue

            field = filter_def.get('field')
            operator = filter_def.get('operator')
            target_value = filter_def.get('value') # Renamed 'value' to 'target_value' to avoid conflict

            if not all([field, operator, target_value is not None]): # Ensure all parts are present
                logger.warning(f"Skipping incomplete filter definition: {filter_def}")
                continue

            # This list comprehension creates a new list in each iteration.
            # For very large stock lists and many filters, this could be inefficient.
            # Consider in-place removal or building a new list once if performance becomes an issue.
            current_filtered_stocks = [
                s for s in current_filtered_stocks
                if self._evaluate_filter(s, field, operator, target_value)
            ]
        return current_filtered_stocks


    def _evaluate_filter(self, stock: StockData, field: str, operator: str, target_value: Any) -> bool:
        """Evaluate a single filter condition"""
        # Map common field names to StockData attributes (case-insensitive)
        field_mapping = {
            'price': 'current_price', 'market_cap': 'market_cap',
            'pe': 'pe_ratio', 'pe_ratio': 'pe_ratio',
            'dividend': 'dividend_yield', 'dividend_yield': 'dividend_yield',
            'volume': 'volume', 'beta': 'beta',
            'profit_margin': 'profit_margin', 'debt_to_equity': 'debt_to_equity',
            'roe': 'roe', 'revenue': 'revenue', 'revenue_growth': 'revenue_growth',
            'eps': 'eps', 'eps_growth': 'eps_growth', # Added EPS fields
            'forward_pe': 'forward_pe', 'peg_ratio': 'peg_ratio', 'price_to_book': 'price_to_book'
        }

        actual_field_name = field_mapping.get(str(field).lower(), str(field).lower())

        try:
            stock_attr_value = getattr(stock, actual_field_name, None)
        except AttributeError: # Should not happen if StockData model is consistent # pragma: no cover
            logger.warning(f"Filter Eval: Attribute '{actual_field_name}' not found on StockData for {stock.symbol}. Skipping filter.")
            return False # Or True, depending on desired strictness (fail open vs fail closed)

        if stock_attr_value is None: # If attribute exists but its value is None
            return False # Cannot compare None with a value

        # Ensure target_value is of a comparable type, attempting conversion if needed
        try:
            if isinstance(stock_attr_value, (int, float)) and not isinstance(target_value, (int, float)):
                target_value_coerced = float(target_value)
            elif isinstance(stock_attr_value, str) and not isinstance(target_value, str): # pragma: no cover
                target_value_coerced = str(target_value)
            else:
                target_value_coerced = target_value # Assume types are compatible or target_value is already correct
        except ValueError: # pragma: no cover
            logger.warning(f"Filter Eval: Could not coerce target value '{target_value}' to compare with stock value '{stock_attr_value}' for field '{actual_field_name}'.")
            return False

        try:
            if operator == '>': return stock_attr_value > target_value_coerced
            elif operator == '<': return stock_attr_value < target_value_coerced
            elif operator == '>=': return stock_attr_value >= target_value_coerced
            elif operator == '<=': return stock_attr_value <= target_value_coerced
            elif operator == '==' or operator == '=': return stock_attr_value == target_value_coerced
            elif operator == '!=' or operator == '<>': return stock_attr_value != target_value_coerced
            # Add 'in' or 'contains' for string fields if needed
            elif operator.lower() == 'in' and isinstance(stock_attr_value, str) and isinstance(target_value_coerced, str):
                 return target_value_coerced.lower() in stock_attr_value.lower() # Case-insensitive substring
            else: # pragma: no cover
                logger.warning(f"Filter Eval: Unknown operator '{operator}' for field '{actual_field_name}'. Filter returns False.")
                return False # Default to False for unknown operators for safety
        except TypeError: # If comparison fails due to incompatible types after coercion attempt # pragma: no cover
            logger.warning(f"Filter Eval: TypeError comparing '{stock_attr_value}' ({type(stock_attr_value)}) "
                           f"with '{target_value_coerced}' ({type(target_value_coerced)}) for field '{actual_field_name}'.")
            return False
        except Exception as e: # Catch any other unexpected comparison error # pragma: no cover
            logger.exception(f"Filter Eval: Unexpected error evaluating filter for {stock.symbol} on field {actual_field_name}: {e}")
            return False


    def _apply_qualitative_filters(self, stocks: List[StockData],
                                   qualitative: Dict[str, Any]) -> List[StockData]:
        """Apply qualitative filters like sector, characteristics"""
        if not qualitative or not isinstance(qualitative, dict):
            return stocks # No qualitative filters to apply

        current_filtered_stocks = list(stocks)

        # Filter by sector
        sector_filter = qualitative.get('sector')
        if sector_filter and isinstance(sector_filter, str) and sector_filter.strip():
            sector_lower = sector_filter.strip().lower()
            current_filtered_stocks = [
                s for s in current_filtered_stocks
                if hasattr(s, 'sector') and isinstance(s.sector, str) and (
                    s.sector.lower() == sector_lower or sector_lower in s.sector.lower()
                )
            ]

        # Filter by characteristics
        characteristics_list = qualitative.get('characteristics')
        if characteristics_list and isinstance(characteristics_list, list):
            for char_item in characteristics_list:
                if isinstance(char_item, str) and char_item.strip():
                    # Each characteristic filter might reduce the list further
                    current_filtered_stocks = self._filter_by_characteristic(current_filtered_stocks, char_item.strip())

        return current_filtered_stocks


    def _filter_by_characteristic(self, stocks: List[StockData], characteristic: str) -> List[StockData]:
        """Filter stocks by qualitative characteristics"""
        char_lower = characteristic.lower() # Already lowercased by caller if called internally

        # Using a helper to safely get and check attributes
        def check_stock(stock: StockData, attr_name: str, condition_fn) -> bool:
            val = getattr(stock, attr_name, None)
            # Ensure val is not None and is a number (float/int) before applying condition
            return val is not None and isinstance(val, (float, int)) and pd.notna(val) and condition_fn(val)

        try:
            if 'growing earnings' in char_lower or 'profit growth' in char_lower:
                return [s for s in stocks if check_stock(s, 'eps_growth', lambda x: x > 0.05)] # e.g. >5%
            elif 'growing revenue' in char_lower:
                return [s for s in stocks if check_stock(s, 'revenue_growth', lambda x: x > 0.10)] # e.g. >10%
            elif 'strong dividend' in char_lower: # Example: yield > 3% and low payout (if payout was available)
                return [s for s in stocks if check_stock(s, 'dividend_yield', lambda x: x > 0.03)]
            elif 'dividend aristocrat' in char_lower: # Simplified: yield > 2%, low debt
                return [s for s in stocks if check_stock(s, 'dividend_yield', lambda x: x > 0.02) and \
                                             check_stock(s, 'debt_to_equity', lambda x: x < 0.75)]
            elif 'low debt' in char_lower:
                return [s for s in stocks if check_stock(s, 'debt_to_equity', lambda x: x < 0.5)]
            elif 'high roe' in char_lower:
                 return [s for s in stocks if check_stock(s, 'roe', lambda x: x > 0.15)] # ROE > 15%
            elif 'high profit margin' in char_lower:
                return [s for s in stocks if check_stock(s, 'profit_margin', lambda x: x > 0.20)] # Profit Margin > 20%
            elif 'undervalued' in char_lower: # Simple P/E and P/B based
                return [s for s in stocks if check_stock(s, 'pe_ratio', lambda x: x > 0 and x < 15) and \
                                             check_stock(s, 'price_to_book', lambda x: x > 0 and x < 1.5)]
            elif 'growth stock' in char_lower: # High revenue growth and potentially higher PE
                return [s for s in stocks if check_stock(s, 'revenue_growth', lambda x: x > 0.15) and \
                                             check_stock(s, 'forward_pe', lambda x: x > 10)] # Allow higher PE for growth
            elif 'value stock' in char_lower: # Low PE, Low P/B, possibly decent dividend
                return [s for s in stocks if check_stock(s, 'pe_ratio', lambda x: x > 0 and x < 12) and \
                                             check_stock(s, 'price_to_book', lambda x: x > 0 and x < 1) and \
                                             check_stock(s, 'dividend_yield', lambda x: x > 0.01)]
            else: # Unknown characteristic
                logger.debug(f"Unknown characteristic filter: '{characteristic}'. No stocks filtered by it.")
                return stocks
        except Exception as e: # pragma: no cover
            logger.exception(f"Error applying characteristic filter '{characteristic}': {e}")
            return stocks # Return original list on error to avoid losing all stocks


    def _sort_stocks(self, stocks: List[StockData], sort_by: str, order: str) -> List[StockData]:
        """Sort stocks by specified field"""
        if not stocks:
            return []

        # Define how to handle None values for sorting:
        # For numeric fields, None could be treated as very small (for asc) or very large (for desc)
        # or simply be placed at the end. Python's default sort places None at the beginning.
        # Using a large/small number for None helps keep numeric sort consistent.
        # For string fields, None can be converted to an empty string.

        # Map common sort fields to StockData attributes and provide default for None
        sort_field_map = {
            'market_cap': lambda s: getattr(s, 'market_cap', 0) or 0, # None market_cap treated as 0
            'price': lambda s: getattr(s, 'current_price', 0) or 0,
            'pe_ratio': lambda s: getattr(s, 'pe_ratio', float('inf')) or float('inf'), # High PE for None if ascending
            'dividend_yield': lambda s: getattr(s, 'dividend_yield', 0) or 0, # None yield as 0
            'volume': lambda s: getattr(s, 'volume', 0) or 0,
            'revenue_growth': lambda s: getattr(s, 'revenue_growth', -float('inf')) or -float('inf'), # Very low growth for None
            'profit_margin': lambda s: getattr(s, 'profit_margin', -float('inf')) or -float('inf'),
            'symbol': lambda s: getattr(s, 'symbol', '') or '' # Empty string for None symbol
        }

        sort_key_func = sort_field_map.get(sort_by.lower(), sort_field_map['market_cap']) # Default to market_cap
        is_reverse = (str(order).lower() == 'desc')

        try:
            return sorted(stocks, key=sort_key_func, reverse=is_reverse)
        except TypeError as te: # pragma: no cover
            # This can happen if a lambda tries to compare incompatible types (e.g. None with number, if not handled by key_func)
            logger.error(f"TypeError during sorting by '{sort_by}': {te}. Check None handling in sort_key_func.")
            return stocks # Return unsorted on error
        except Exception as e: # pragma: no cover
            logger.exception(f"Unexpected error during stock sorting: {e}")
            return stocks


class FilterBuilder:
    """Build filters from common screening patterns"""

    @staticmethod
    def parse_price_filter(text: str) -> Optional[List[Dict[str, Any]]]: # Return list for 'between' case
        """Parse price-related filters. Can return a list of filters."""
        try:
            # Pattern for "under $X" or "less than X"
            match_under = re.search(r'(?:under|less than|below|max)\s*\$?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if match_under:
                return [{'field': 'price', 'operator': '<', 'value': float(match_under.group(1))}]

            # Pattern for "over $X" or "more than X"
            match_over = re.search(r'(?:over|above|more than|greater than|min)\s*\$?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if match_over:
                return [{'field': 'price', 'operator': '>', 'value': float(match_over.group(1))}]

            # Pattern for "between $X and $Y"
            match_between = re.search(r'between\s*\$?(\d+(?:\.\d+)?)\s*and\s*\$?(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if match_between:
                val1 = float(match_between.group(1))
                val2 = float(match_between.group(2))
                return [
                    {'field': 'price', 'operator': '>=', 'value': min(val1, val2)},
                    {'field': 'price', 'operator': '<=', 'value': max(val1, val2)}
                ]
        except ValueError as ve: # pragma: no cover
            logger.error(f"FilterBuilder: Error parsing price value in '{text}': {ve}")
        except Exception as e: # pragma: no cover
             logger.exception(f"FilterBuilder: Unexpected error in parse_price_filter for '{text}': {e}")
        return None


    @staticmethod
    def parse_market_cap_filter(text: str) -> Optional[List[Dict[str, Any]]]: # Can return list for mid cap
        """Parse market cap filters"""
        try:
            multipliers = {'k': 1e3, 'm': 1e6, 'b': 1e9, 't': 1e12}

            # Generic pattern for over/under with multipliers
            # e.g., "market cap over 100m", "mkt cap < 2b"
            pattern_value = r'market cap\s*(?:is\s*)?(<|>|<=|>=|under|over|above|below|less than|more than)\s*\$?([\d\.]+)\s*([kmbt])?'
            match_value = re.search(pattern_value, text, re.IGNORECASE)

            if match_value:
                operator_str = match_value.group(1).lower()
                value_num = float(match_value.group(2))
                unit = match_value.group(3).lower() if match_value.group(3) else ''
                final_value = value_num * multipliers.get(unit, 1)

                operator_map = {
                    '<': '<', 'under': '<', 'below': '<', 'less than': '<',
                    '>': '>', 'over': '>', 'above': '>', 'more than': '>',
                    '<=': '<=', '>=': '>='
                }
                if operator_str not in operator_map: # Should not happen with regex # pragma: no cover
                    logger.warning(f"Market Cap Filter: Unknown operator string '{operator_str}'")
                    return None

                return [{'field': 'market_cap', 'operator': operator_map[operator_str], 'value': final_value}]

            # Predefined cap sizes (these are approximate ranges)
            if 'small cap' in text.lower():
                return [{'field': 'market_cap', 'operator': '<=', 'value': 2e9}] # e.g., < $2B
            elif 'mid cap' in text.lower():
                return [
                    {'field': 'market_cap', 'operator': '>', 'value': 2e9},  # > $2B
                    {'field': 'market_cap', 'operator': '<=', 'value': 10e9} # <= $10B
                ]
            elif 'large cap' in text.lower():
                return [{'field': 'market_cap', 'operator': '>', 'value': 10e9}] # > $10B
            elif 'mega cap' in text.lower(): # Added mega cap
                 return [{'field': 'market_cap', 'operator': '>', 'value': 200e9}] # > $200B
        except ValueError as ve: # pragma: no cover
            logger.error(f"FilterBuilder: Error parsing market cap value in '{text}': {ve}")
        except Exception as e: # pragma: no cover
            logger.exception(f"FilterBuilder: Unexpected error in parse_market_cap_filter for '{text}': {e}")
        return None


    @staticmethod
    def parse_pe_filter(text: str) -> Optional[Dict[str, Any]]:
        """Parse P/E ratio filters"""
        try:
            # Pattern for "P/E under X", "PE less than Y", "P/E ratio > Z"
            match = re.search(r'p/?e(?: ratio)?\s*(<|>|<=|>=|under|over|above|below|less than|more than)\s*(\d+(?:\.\d+)?)', text, re.IGNORECASE)
            if match:
                operator_str = match.group(1).lower()
                value = float(match.group(2))
                operator_map = {
                    '<': '<', 'under': '<', 'below': '<', 'less than': '<',
                    '>': '>', 'over': '>', 'above': '>', 'more than': '>',
                    '<=': '<=', '>=': '>='
                }
                if operator_str not in operator_map: return None # Should not happen # pragma: no cover

                return {'field': 'pe_ratio', 'operator': operator_map[operator_str], 'value': value}
        except ValueError as ve: # pragma: no cover
            logger.error(f"FilterBuilder: Error parsing P/E value in '{text}': {ve}")
        except Exception as e: # pragma: no cover
            logger.exception(f"FilterBuilder: Unexpected error in parse_pe_filter for '{text}': {e}")
        return None


    @staticmethod
    def parse_dividend_filter(text: str) -> Optional[Dict[str, Any]]:
        """Parse dividend yield filters"""
        try:
            # Pattern for "dividend yield over X%", "dividend > Y"
            match = re.search(r'dividend(?: yield)?\s*(<|>|<=|>=|over|under)\s*([\d\.]+)\s*(%)?', text, re.IGNORECASE)
            if match:
                operator_str = match.group(1).lower()
                value = float(match.group(2))
                is_percent = match.group(3) == '%'

                if is_percent or value > 1.0: # If specified as percent (e.g., 3%) or value implies percent (e.g. 3 for 3%)
                    value /= 100.0

                operator_map = {
                    '<': '<', 'under': '<',
                    '>': '>', 'over': '>',
                    '<=': '<=', '>=': '>='
                }
                if operator_str not in operator_map: return None # Should not happen # pragma: no cover

                return {'field': 'dividend_yield', 'operator': operator_map[operator_str], 'value': value}
        except ValueError as ve: # pragma: no cover
            logger.error(f"FilterBuilder: Error parsing dividend value in '{text}': {ve}")
        except Exception as e: # pragma: no cover
            logger.exception(f"FilterBuilder: Unexpected error in parse_dividend_filter for '{text}': {e}")
        return None


class ScreeningPresets:
    """Common screening presets"""

    PRESETS = {
        'dividend_aristocrats': {
            'filters': [
                {'field': 'dividend_yield', 'operator': '>', 'value': 0.02},
                {'field': 'market_cap', 'operator': '>', 'value': 10e9}
            ],
            'qualitative': {
                'characteristics': ['dividend aristocrat']
            },
            'sort_by': 'dividend_yield',
            'sort_order': 'desc'
        },

        'growth_stocks': {
            'filters': [
                {'field': 'revenue_growth', 'operator': '>', 'value': 0.15},
                {'field': 'pe_ratio', 'operator': '<', 'value': 50}
            ],
            'qualitative': {
                'characteristics': ['growing earnings', 'growing revenue']
            },
            'sort_by': 'revenue_growth',
            'sort_order': 'desc'
        },

        'value_stocks': {
            'filters': [
                {'field': 'pe_ratio', 'operator': '<', 'value': 15},
                {'field': 'price_to_book', 'operator': '<', 'value': 2}
            ],
            'qualitative': {
                'characteristics': ['undervalued']
            },
            'sort_by': 'pe_ratio',
            'sort_order': 'asc'
        },

        'tech_giants': {
            'filters': [
                {'field': 'market_cap', 'operator': '>', 'value': 100e9}
            ],
            'qualitative': {
                'sector': 'technology'
            },
            'sort_by': 'market_cap',
            'sort_order': 'desc'
        },

        'high_momentum': {
            'filters': [
                {'field': 'volume', 'operator': '>', 'value': 10e6}
            ],
            'qualitative': {
                'characteristics': ['high momentum']
            },
            'sort_by': 'volume',
            'sort_order': 'desc'
        }
    }

    @classmethod
    def get_preset(cls, name: str) -> Optional[Dict[str, Any]]:
        """Get a screening preset by name"""
        return cls.PRESETS.get(name.lower())

    @classmethod
    def detect_preset(cls, query: str) -> Optional[str]:
        """Detect if query matches a preset"""
        query_lower = query.lower()

        preset_keywords = {
            'dividend_aristocrats': ['dividend aristocrat', 'dividend stocks'],
            'growth_stocks': ['growth stocks', 'high growth'],
            'value_stocks': ['value stocks', 'undervalued', 'cheap stocks'],
            'tech_giants': ['tech giants', 'technology leaders', 'faang'],
            'high_momentum': ['high momentum', 'trending stocks']
        }

        for preset, keywords in preset_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    return preset

        return None


class ScreenerCache:
    """Cache screening results"""

    def __init__(self, ttl_seconds: int = 300):
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, query: str) -> Optional[ScreeningResult]:
        """Get cached result"""
        if query in self.cache:
            result, timestamp = self.cache[query]
            if (datetime.now() - timestamp).total_seconds() < self.ttl:
                return result
            else:
                del self.cache[query]
        return None

    def set(self, query: str, result: ScreeningResult):
        """Cache a result"""
        self.cache[query] = (result, datetime.now())

    def clear(self):
        """Clear all cached results"""
        self.cache.clear()


class RealTimeScreener:
    """Real-time stock screening with live updates"""

    def __init__(self, screener: StockScreener):
        self.screener = screener
        self.active_screens = {}
        self.update_interval = 60  # seconds

    async def start_live_screen(self, query: str, callback):
        """Start live screening with periodic updates"""
        screen_id = f"{query}_{datetime.now().timestamp()}"
        self.active_screens[screen_id] = {
            'query': query,
            'callback': callback,
            'active': True
        }

        # Run initial screen
        result = await self.screener.screen_stocks(query)
        callback(result)

        # Start update loop
        asyncio.create_task(self._update_loop(screen_id))

        return screen_id

    def stop_live_screen(self, screen_id: str):
        """Stop a live screening session"""
        if screen_id in self.active_screens:
            self.active_screens[screen_id]['active'] = False
            del self.active_screens[screen_id]

    async def _update_loop(self, screen_id: str):
        """Update loop for live screening"""
        while screen_id in self.active_screens and self.active_screens[screen_id]['active']:
            await asyncio.sleep(self.update_interval)

            if screen_id not in self.active_screens:
                break

            screen_data = self.active_screens[screen_id]
            result = await self.screener.screen_stocks(screen_data['query'])
            screen_data['callback'](result)


class CustomFilterValidator:
    """Validate custom filter expressions"""

    VALID_FIELDS = {
        'price', 'market_cap', 'pe_ratio', 'dividend_yield',
        'volume', 'beta', 'profit_margin', 'debt_to_equity',
        'roe', 'revenue', 'revenue_growth', 'eps', 'eps_growth'
    }

    VALID_OPERATORS = {'>', '<', '>=', '<=', '==', '=', '!=', '<>'}

    @classmethod
    def validate_filter(cls, filter_dict: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate a filter dictionary"""
        # Check required fields
        if 'field' not in filter_dict:
            return False, "Missing 'field' in filter"
        if 'operator' not in filter_dict:
            return False, "Missing 'operator' in filter"
        if 'value' not in filter_dict:
            return False, "Missing 'value' in filter"

        # Validate field
        if filter_dict['field'] not in cls.VALID_FIELDS:
            return False, f"Invalid field: {filter_dict['field']}"

        # Validate operator
        if filter_dict['operator'] not in cls.VALID_OPERATORS:
            return False, f"Invalid operator: {filter_dict['operator']}"

        # Validate value type
        try:
            float(filter_dict['value'])
        except (TypeError, ValueError):
            return False, f"Invalid value type: {filter_dict['value']}"

        return True, None

    @classmethod
    def validate_query(cls, query: ScreeningQuery) -> Tuple[bool, List[str]]:
        """Validate entire screening query"""
        errors = []

        for filter_dict in query.filters:
            valid, error = cls.validate_filter(filter_dict)
            if not valid:
                errors.append(error)

        # Validate sort field
        if query.sort_by not in cls.VALID_FIELDS and query.sort_by != 'symbol':
            errors.append(f"Invalid sort field: {query.sort_by}")

        # Validate limit
        if query.limit < 1 or query.limit > 1000:
            errors.append(f"Invalid limit: {query.limit} (must be 1-1000)")

        return len(errors) == 0, errors
