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
        self._load_stock_universe()

    def _load_stock_universe(self):
        """Load list of available stocks"""
        # In production, this would load from a database or API
        # For now, use a sample list
        self.stock_universe = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "JPM",
            "JNJ", "V", "PG", "UNH", "HD", "MA", "DIS", "PYPL", "BAC", "ADBE",
            "NFLX", "CRM", "XOM", "VZ", "INTC", "WMT", "CVX", "KO", "PFE",
            "ABBV", "NKE", "TMO", "CSCO", "PEP", "ABT", "AVGO", "MRK", "ACN"
        ]

    async def screen_stocks(self, query: str) -> ScreeningResult:
        """Screen stocks based on natural language query"""
        start_time = datetime.now()

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

        # Get stock data for universe
        stock_data = await self._fetch_stock_data(self.stock_universe)

        # Apply filters
        filtered_stocks = self._apply_filters(stock_data, screening_query.filters)

        # Apply qualitative filters
        if 'qualitative' in parsed_query:
            filtered_stocks = self._apply_qualitative_filters(
                filtered_stocks,
                parsed_query['qualitative']
            )

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

        return ScreeningResult(
            query=screening_query,
            matches=final_stocks,
            total_count=len(final_stocks),
            execution_time=execution_time,
            explanations=explanations
        )

    async def _fetch_stock_data(self, symbols: List[str]) -> List[StockData]:
        """Fetch data for multiple stocks"""
        stocks_dict = await self.data_provider.batch_get_stocks(symbols)
        return list(stocks_dict.values())

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
