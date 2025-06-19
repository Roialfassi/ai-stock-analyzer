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

    def __init__(self, data_provider: MarketDataProvider, nl_screener: NaturalLanguageScreener):
        self.data_provider = data_provider
        self.nl_screener = nl_screener
        self.stock_universe = []
        self.cache = ScreenerCache()
        self._load_stock_universe()

        # Preload some data for faster screening
        self.preloaded_data = {}
        asyncio.create_task(self._preload_popular_stocks())

    async def _preload_popular_stocks(self):
        """Preload data for popular stocks to speed up screening"""
        popular = self.stock_universe[:50]  # Top 50 stocks

        try:
            for symbol in popular:
                data = await self.data_provider.get_stock_data(symbol)
                if data:
                    self.preloaded_data[symbol] = data
        except Exception as e:
            logger.error(f"Error preloading stocks: {e}")

    async def _fetch_stock_data(self, symbols: List[str]) -> List[StockData]:
        """Fetch data for multiple stocks with preloaded cache"""
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
            self.stock_universe = list(set(major_stocks + sp500_symbols))
        except:
            # Fallback to predefined list
            self.stock_universe = major_stocks

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
        if use_cache:
            cache_key = f"screen:{query}"
            cached = self.cache.get(cache_key)
            if cached:
                return ScreeningResult(**cached)

        # Parse the query
        parsed_query = await self.nl_screener.parse_query(query)

        # Create screening query object
        screening_query = ScreeningQuery(
            raw_query=query,
            parsed_criteria=parsed_query,
            filters=parsed_query.get('filters', []),
            sort_by=parsed_query.get('sort_by', 'market_cap'),
            sort_order=parsed_query.get('sort_order', 'desc'),
            limit=parsed_query.get('limit', 50)
        )

        # Determine which stocks to fetch based on query
        stocks_to_fetch = self._optimize_stock_selection(parsed_query)

        # Batch fetch stock data efficiently
        stock_data = await self._fetch_stock_data_batch(stocks_to_fetch)

        # Apply filters
        filtered_stocks = self._apply_filters(stock_data, screening_query.filters)

        # Apply qualitative filters
        if 'qualitative' in parsed_query:
            filtered_stocks = self._apply_qualitative_filters(
                filtered_stocks,
                parsed_query['qualitative']
            )

        # Apply advanced filters if needed
        if self._needs_advanced_filtering(parsed_query):
            filtered_stocks = await self._apply_advanced_filters(filtered_stocks, parsed_query)

        # Sort results
        sorted_stocks = self._sort_stocks(
            filtered_stocks,
            screening_query.sort_by,
            screening_query.sort_order
        )

        # Limit results
        final_stocks = sorted_stocks[:screening_query.limit]

        # Generate explanations
        explanations = await self.nl_screener.explain_matches(query, final_stocks)

        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()

        result = ScreeningResult(
            query=screening_query,
            matches=final_stocks,
            total_count=len(final_stocks),
            execution_time=execution_time,
            explanations=explanations
        )

        # Cache result
        if use_cache:
            self.cache.set(cache_key, result.__dict__, 300)  # 5 min cache

        return result

    def _optimize_stock_selection(self, parsed_query: Dict[str, Any]) -> List[str]:
        """Optimize which stocks to fetch based on query"""
        qualitative = parsed_query.get('qualitative', {})
        sector = qualitative.get('sector', '').lower()

        # If sector specified, filter universe first
        if sector:
            sector_map = {
                'technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'INTC', 'AMD', 'CRM', 'ADBE', 'ORCL', 'CSCO',
                               'AVGO', 'QCOM', 'TXN', 'MU'],
                'healthcare': ['JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'MRK', 'ABT', 'CVS', 'BMY', 'AMGN', 'GILD', 'MDT',
                               'ISRG', 'VRTX', 'MRNA'],
                'finance': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'AXP', 'BLK', 'SCHW', 'V', 'MA', 'PYPL', 'COF', 'USB',
                            'PNC'],
                'energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'PSX', 'VLO', 'OXY', 'KMI', 'HAL', 'BKR', 'DVN',
                           'FANG'],
                'consumer': ['WMT', 'HD', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'TGT', 'LOW', 'DIS', 'CMCSA',
                             'NFLX'],
                'industrial': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'RTX', 'LMT', 'DE', 'EMR', 'FDX', 'NSC', 'UNP',
                               'WM'],
                'realestate': ['AMT', 'PLD', 'CCI', 'EQIX', 'PSA', 'SPG', 'WELL', 'AVB', 'EQR', 'DLR', 'O', 'SBAC',
                               'WY', 'VTR'],
                'materials': ['LIN', 'APD', 'SHW', 'ECL', 'DD', 'NEM', 'FCX', 'DOW', 'PPG', 'ALB', 'NUE', 'CLF', 'VMC',
                              'MLM'],
                'utilities': ['NEE', 'DUK', 'SO', 'D', 'SRE', 'AEP', 'EXC', 'XEL', 'PEG', 'ED', 'WEC', 'ES', 'AWK',
                              'DTE']
            }

            # Find matching sector
            for key, stocks in sector_map.items():
                if sector in key or key in sector:
                    return stocks

        # Check for market cap filters to optimize
        filters = parsed_query.get('filters', [])
        for filter_def in filters:
            if filter_def.get('field') == 'market_cap':
                operator = filter_def.get('operator')
                value = filter_def.get('value', 0)

                # For small caps, use different universe
                if operator in ['<', '<='] and value <= 10e9:
                    # Return mix of smaller stocks
                    return self.stock_universe[-100:]  # Last 100 stocks tend to be smaller
                elif operator in ['>', '>='] and value >= 100e9:
                    # Return large caps
                    return self.stock_universe[:100]  # First 100 tend to be larger

        # For dividend queries, focus on dividend paying stocks
        characteristics = qualitative.get('characteristics', [])
        for char in characteristics:
            if 'dividend' in char.lower():
                # Known dividend payers
                return ['JNJ', 'PG', 'KO', 'PEP', 'ABBV', 'MRK', 'VZ', 'T', 'XOM', 'CVX',
                        'JPM', 'BAC', 'WFC', 'USB', 'PNC', 'HD', 'LOW', 'WMT', 'TGT', 'COST',
                        'MCD', 'SBUX', 'NKE', 'O', 'SPG', 'PSA', 'WELL', 'AMT', 'CCI', 'DLR']

        # Default: return top stocks by market cap
        return self.stock_universe[:200]  # Limit to 200 for performance

    async def _fetch_stock_data_batch(self, symbols: List[str], batch_size: int = 50) -> List[StockData]:
        """Fetch stock data in batches for efficiency"""
        all_stocks = []

        # Process in batches
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]

            # Use ThreadPoolExecutor for parallel fetching
            tasks = []
            for symbol in batch:
                task = self.data_provider.get_stock_data(symbol)
                tasks.append(task)

            # Gather results
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out failed fetches
            for result in results:
                if isinstance(result, StockData):
                    all_stocks.append(result)

        return all_stocks

    def _needs_advanced_filtering(self, parsed_query: Dict[str, Any]) -> bool:
        """Check if query needs advanced filtering with historical data"""
        characteristics = parsed_query.get('qualitative', {}).get('characteristics', [])

        advanced_keywords = [
            'momentum', 'trend', 'breakout', 'oversold', 'overbought',
            'volatility', 'beta', 'correlation', 'drawdown',
            'technical', 'rsi', 'macd', 'moving average'
        ]

        for char in characteristics:
            for keyword in advanced_keywords:
                if keyword in char.lower():
                    return True

        return False

    async def _apply_advanced_filters(self, stocks: List[StockData],
                                      parsed_query: Dict[str, Any]) -> List[StockData]:
        """Apply advanced filters that require historical data"""
        characteristics = parsed_query.get('qualitative', {}).get('characteristics', [])
        filtered = stocks

        for char in characteristics:
            char_lower = char.lower()

            if 'momentum' in char_lower or 'trending' in char_lower:
                # Filter by price momentum
                momentum_stocks = []
                for stock in filtered:
                    try:
                        hist = await self.data_provider.get_historical_data(stock.symbol, "3mo")
                        if not hist.empty and len(hist) > 20:
                            # Calculate 20-day momentum
                            momentum = (hist['Close'].iloc[-1] / hist['Close'].iloc[-20] - 1) * 100
                            if momentum > 10:  # 10% gain in 20 days
                                momentum_stocks.append(stock)
                    except:
                        pass
                filtered = momentum_stocks if momentum_stocks else filtered

            elif 'oversold' in char_lower:
                # Filter by RSI < 30
                oversold_stocks = []
                for stock in filtered:
                    try:
                        indicators = await self.data_provider.get_technical_indicators(stock.symbol)
                        if indicators and indicators.rsi < 30:
                            oversold_stocks.append(stock)
                    except:
                        pass
                filtered = oversold_stocks if oversold_stocks else filtered

            elif 'breakout' in char_lower:
                # Filter by stocks near 52-week high
                breakout_stocks = []
                for stock in filtered:
                    if stock.year_high and stock.current_price:
                        if stock.current_price >= stock.year_high * 0.95:  # Within 5% of high
                            breakout_stocks.append(stock)
                filtered = breakout_stocks if breakout_stocks else filtered

        return filtered

    def _apply_filters(self, stocks: List[StockData], filters: List[Dict[str, Any]]) -> List[StockData]:
        """Apply quantitative filters to stocks"""
        filtered = stocks

        for filter_def in filters:
            field = filter_def['field']
            operator = filter_def['operator']
            value = filter_def['value']

            filtered = [
                stock for stock in filtered
                if self._evaluate_filter(stock, field, operator, value)
            ]

        return filtered

    def _evaluate_filter(self, stock: StockData, field: str, operator: str, value: Any) -> bool:
        """Evaluate a single filter condition"""
        # Map common field names to StockData attributes
        field_mapping = {
            'price': 'current_price',
            'market_cap': 'market_cap',
            'pe': 'pe_ratio',
            'pe_ratio': 'pe_ratio',
            'dividend': 'dividend_yield',
            'dividend_yield': 'dividend_yield',
            'volume': 'volume',
            'beta': 'beta',
            'profit_margin': 'profit_margin',
            'debt_to_equity': 'debt_to_equity',
            'roe': 'roe',
            'revenue': 'revenue',
            'revenue_growth': 'revenue_growth'
        }

        # Get the actual field name
        actual_field = field_mapping.get(field.lower(), field.lower())

        # Get the stock value
        stock_value = getattr(stock, actual_field, None)
        if stock_value is None:
            return False

        # Apply operator
        try:
            if operator == '>':
                return stock_value > value
            elif operator == '<':
                return stock_value < value
            elif operator == '>=':
                return stock_value >= value
            elif operator == '<=':
                return stock_value <= value
            elif operator == '==' or operator == '=':
                return stock_value == value
            elif operator == '!=' or operator == '<>':
                return stock_value != value
            else:
                logger.warning(f"Unknown operator: {operator}")
                return True
        except:
            return False

    def _apply_qualitative_filters(self, stocks: List[StockData],
                                   qualitative: Dict[str, Any]) -> List[StockData]:
        """Apply qualitative filters like sector, characteristics"""
        filtered = stocks

        # Filter by sector
        if 'sector' in qualitative and qualitative['sector']:
            sector = qualitative['sector'].lower()
            filtered = [
                stock for stock in filtered
                if stock.sector.lower() == sector or sector in stock.sector.lower()
            ]

        # Filter by characteristics
        if 'characteristics' in qualitative:
            for characteristic in qualitative['characteristics']:
                filtered = self._filter_by_characteristic(filtered, characteristic)

        return filtered

    def _filter_by_characteristic(self, stocks: List[StockData], characteristic: str) -> List[StockData]:
        """Filter stocks by qualitative characteristics"""
        characteristic = characteristic.lower()

        if 'growing' in characteristic:
            if 'earnings' in characteristic or 'profit' in characteristic:
                return [s for s in stocks if s.eps_growth and s.eps_growth > 0]
            elif 'revenue' in characteristic:
                return [s for s in stocks if s.revenue_growth and s.revenue_growth > 0]

        elif 'dividend' in characteristic:
            if 'aristocrat' in characteristic:
                # Simplified - would need dividend history
                return [s for s in stocks if s.dividend_yield and s.dividend_yield > 0.02]
            else:
                return [s for s in stocks if s.dividend_yield and s.dividend_yield > 0]

        elif 'low debt' in characteristic:
            return [s for s in stocks if s.debt_to_equity and s.debt_to_equity < 1.0]

        elif 'high margin' in characteristic:
            return [s for s in stocks if s.profit_margin and s.profit_margin > 0.15]

        elif 'undervalued' in characteristic:
            # Simple P/E based valuation
            return [s for s in stocks if s.pe_ratio and s.pe_ratio < 20]

        elif 'growth' in characteristic:
            return [s for s in stocks if s.revenue_growth and s.revenue_growth > 0.1]

        elif 'value' in characteristic:
            return [s for s in stocks if s.pe_ratio and s.pe_ratio < 15 and s.price_to_book and s.price_to_book < 3]

        # Default - return all
        return stocks

    def _sort_stocks(self, stocks: List[StockData], sort_by: str, order: str) -> List[StockData]:
        """Sort stocks by specified field"""
        # Map sort fields
        sort_mapping = {
            'market_cap': lambda s: s.market_cap or 0,
            'price': lambda s: s.current_price or 0,
            'pe_ratio': lambda s: s.pe_ratio or float('inf'),
            'dividend_yield': lambda s: s.dividend_yield or 0,
            'volume': lambda s: s.volume or 0,
            'revenue_growth': lambda s: s.revenue_growth or 0,
            'profit_margin': lambda s: s.profit_margin or 0,
            'symbol': lambda s: s.symbol
        }

        key_func = sort_mapping.get(sort_by, lambda s: s.market_cap or 0)
        reverse = (order == 'desc')

        return sorted(stocks, key=key_func, reverse=reverse)


class FilterBuilder:
    """Build filters from common screening patterns"""

    @staticmethod
    def parse_price_filter(text: str) -> Optional[Dict[str, Any]]:
        """Parse price-related filters"""
        patterns = [
            (r'under \$?(\d+)', lambda m: {'field': 'price', 'operator': '<', 'value': float(m.group(1))}),
            (r'over \$?(\d+)', lambda m: {'field': 'price', 'operator': '>', 'value': float(m.group(1))}),
            (r'between \$?(\d+) and \$?(\d+)', lambda m: [
                {'field': 'price', 'operator': '>=', 'value': float(m.group(1))},
                {'field': 'price', 'operator': '<=', 'value': float(m.group(2))}
            ])
        ]

        for pattern, builder in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return builder(match)
        return None

    @staticmethod
    def parse_market_cap_filter(text: str) -> Optional[Dict[str, Any]]:
        """Parse market cap filters"""
        multipliers = {'k': 1e3, 'm': 1e6, 'b': 1e9, 't': 1e12}

        pattern = r'market cap (?:over|above|>) \$?(\d+(?:\.\d+)?)\s*([kmbt])?'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            value = float(match.group(1))
            multiplier = multipliers.get(match.group(2).lower(), 1) if match.group(2) else 1
            return {
                'field': 'market_cap',
                'operator': '>',
                'value': value * multiplier
            }

        # Small/mid/large cap
        if 'small cap' in text.lower():
            return {'field': 'market_cap', 'operator': '<', 'value': 2e9}
        elif 'mid cap' in text.lower():
            return [
                {'field': 'market_cap', 'operator': '>=', 'value': 2e9},
                {'field': 'market_cap', 'operator': '<=', 'value': 10e9}
            ]
        elif 'large cap' in text.lower():
            return {'field': 'market_cap', 'operator': '>', 'value': 10e9}

        return None

    @staticmethod
    def parse_pe_filter(text: str) -> Optional[Dict[str, Any]]:
        """Parse P/E ratio filters"""
        pattern = r'p/?e (?:ratio )?(?:under|below|<) (\d+(?:\.\d+)?)'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            return {
                'field': 'pe_ratio',
                'operator': '<',
                'value': float(match.group(1))
            }
        return None

    @staticmethod
    def parse_dividend_filter(text: str) -> Optional[Dict[str, Any]]:
        """Parse dividend yield filters"""
        pattern = r'dividend (?:yield )?(?:over|above|>) (\d+(?:\.\d+)?)\s*%?'
        match = re.search(pattern, text, re.IGNORECASE)

        if match:
            value = float(match.group(1))
            # Convert percentage to decimal if needed
            if value > 1:
                value /= 100
            return {
                'field': 'dividend_yield',
                'operator': '>',
                'value': value
            }
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
