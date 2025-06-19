# models.py - Data Models and Market Structures

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum


class Recommendation(Enum):
    STRONG_BUY = "Strong Buy"
    BUY = "Buy"
    HOLD = "Hold"
    SELL = "Sell"
    STRONG_SELL = "Strong Sell"


class AnalysisType(Enum):
    FUNDAMENTAL = "Fundamental"
    TECHNICAL = "Technical"
    SENTIMENT = "Sentiment"
    COMPREHENSIVE = "Comprehensive"


@dataclass
class StockData:
    """Comprehensive stock information"""
    symbol: str
    company_name: str
    sector: str
    industry: str
    current_price: float
    market_cap: float
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    peg_ratio: Optional[float] = None
    price_to_book: Optional[float] = None
    dividend_yield: Optional[float] = None
    eps: Optional[float] = None
    eps_growth: Optional[float] = None
    revenue: Optional[float] = None
    revenue_growth: Optional[float] = None
    profit_margin: Optional[float] = None
    operating_margin: Optional[float] = None
    roe: Optional[float] = None
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    beta: Optional[float] = None
    volume: Optional[int] = None
    avg_volume: Optional[int] = None
    day_high: Optional[float] = None
    day_low: Optional[float] = None
    year_high: Optional[float] = None
    year_low: Optional[float] = None
    fifty_day_ma: Optional[float] = None
    two_hundred_day_ma: Optional[float] = None
    rsi: Optional[float] = None
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class TechnicalIndicators:
    """Technical analysis indicators"""
    rsi: float
    macd: Dict[str, float]  # signal, histogram, macd
    moving_averages: Dict[str, float]  # 20, 50, 200 day
    bollinger_bands: Dict[str, float]  # upper, middle, lower
    volume_profile: Dict[str, Any]
    support_levels: List[float]
    resistance_levels: List[float]
    trend_direction: str  # "bullish", "bearish", "neutral"
    momentum_score: float


@dataclass
class FinancialMetrics:
    """Detailed financial statement metrics"""
    income_statement: Dict[str, Dict[str, float]]  # quarterly data
    balance_sheet: Dict[str, Dict[str, float]]
    cash_flow: Dict[str, Dict[str, float]]
    ratios: Dict[str, float]
    growth_rates: Dict[str, float]
    peer_comparison: Dict[str, Dict[str, float]]


@dataclass
class NewsItem:
    """Individual news article"""
    title: str
    source: str
    published: datetime
    url: str
    summary: str
    sentiment_score: float  # -1 to 1
    relevance_score: float  # 0 to 1


@dataclass
class AnalysisResult:
    """Complete analysis result"""
    stock: StockData
    analysis_type: AnalysisType
    llm_summary: str
    bull_case: List[str]
    bear_case: List[str]
    technical_signals: Optional[TechnicalIndicators] = None
    fundamental_score: float = 0.0
    technical_score: float = 0.0
    sentiment_score: float = 0.0
    overall_score: float = 0.0
    recommendation: Recommendation = Recommendation.HOLD
    confidence: float = 0.0
    price_target: Optional[float] = None
    risk_factors: List[str] = field(default_factory=list)
    catalysts: List[str] = field(default_factory=list)
    key_metrics: Dict[str, Any] = field(default_factory=dict)
    peer_comparison: Dict[str, Dict[str, float]] = field(default_factory=dict)
    news_summary: List[NewsItem] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ScreeningQuery:
    """Natural language screening query"""
    raw_query: str
    parsed_criteria: Dict[str, Any]
    filters: List[Dict[str, Any]]
    sort_by: str = "market_cap"
    sort_order: str = "desc"
    limit: int = 50


@dataclass
class ScreeningResult:
    """Results from stock screening"""
    query: ScreeningQuery
    matches: List[StockData]
    total_count: int
    execution_time: float
    explanations: Dict[str, str]  # symbol -> why it matched


@dataclass
class ChainPromptTemplate:
    """Template for multi-step analysis"""
    name: str
    steps: List[Dict[str, Any]]
    context_keys: List[str]
    output_format: str


@dataclass
class PortfolioPosition:
    """Individual portfolio position"""
    symbol: str
    shares: float
    cost_basis: float
    current_value: float
    purchase_date: datetime
    pnl: float
    pnl_percent: float
    allocation_percent: float


@dataclass
class Portfolio:
    """Portfolio container"""
    name: str
    positions: List[PortfolioPosition]
    total_value: float
    cash_balance: float
    total_cost: float
    total_pnl: float
    total_pnl_percent: float
    last_updated: datetime


@dataclass
class MarketOverview:
    """Market conditions summary"""
    indices: Dict[str, Dict[str, float]]  # SPY, QQQ, DIA prices and changes
    sector_performance: Dict[str, float]
    market_breadth: Dict[str, int]  # advances, declines, unchanged
    vix: float
    dollar_index: float
    treasury_yields: Dict[str, float]  # 2Y, 10Y, 30Y
    crypto_prices: Dict[str, float]  # BTC, ETH


@dataclass
class Alert:
    """Price/event alert"""
    alert_id: str
    symbol: str
    alert_type: str  # price_above, price_below, volume_spike, news
    condition: Dict[str, Any]
    triggered: bool = False
    created_at: datetime = field(default_factory=datetime.now)
    triggered_at: Optional[datetime] = None


# Chain Prompt Templates
SCREENING_CHAIN_TEMPLATE = ChainPromptTemplate(
    name="stock_screening",
    steps=[
        {
            "name": "parse_query",
            "prompt": "Extract screening criteria from: {query}",
            "output": "criteria_list"
        },
        {
            "name": "generate_filters",
            "prompt": "Convert criteria to filters: {criteria_list}",
            "output": "filter_conditions"
        },
        {
            "name": "rank_results",
            "prompt": "Rank stocks by relevance: {filtered_stocks}",
            "output": "ranked_results"
        }
    ],
    context_keys=["query", "criteria_list", "filter_conditions", "filtered_stocks"],
    output_format="json"
)

FUNDAMENTAL_ANALYSIS_CHAIN = ChainPromptTemplate(
    name="fundamental_analysis",
    steps=[
        {
            "name": "analyze_financials",
            "prompt": "Analyze financial statements for {symbol}: {financial_data}",
            "output": "financial_analysis"
        },
        {
            "name": "peer_comparison",
            "prompt": "Compare {symbol} to peers {peer_list}: {peer_data}",
            "output": "competitive_position"
        },
        {
            "name": "growth_assessment",
            "prompt": "Evaluate growth prospects: {financial_analysis} {industry_trends}",
            "output": "growth_thesis"
        },
        {
            "name": "valuation_check",
            "prompt": "Assess valuation: {financial_analysis} {peer_comparison}",
            "output": "valuation_verdict"
        },
        {
            "name": "investment_thesis",
            "prompt": "Generate investment thesis: {all_analysis}",
            "output": "final_recommendation"
        }
    ],
    context_keys=["symbol", "financial_data", "peer_data", "industry_trends"],
    output_format="structured"
)

TECHNICAL_ANALYSIS_CHAIN = ChainPromptTemplate(
    name="technical_analysis",
    steps=[
        {
            "name": "pattern_recognition",
            "prompt": "Identify chart patterns in {price_data}",
            "output": "patterns"
        },
        {
            "name": "indicator_analysis",
            "prompt": "Analyze indicators: {technical_indicators}",
            "output": "signal_strength"
        },
        {
            "name": "volume_analysis",
            "prompt": "Assess volume patterns: {volume_data}",
            "output": "accumulation_distribution"
        },
        {
            "name": "entry_exit_points",
            "prompt": "Determine optimal entry/exit: {patterns} {signals}",
            "output": "trading_levels"
        }
    ],
    context_keys=["price_data", "technical_indicators", "volume_data"],
    output_format="json"
)

NEWS_SENTIMENT_CHAIN = ChainPromptTemplate(
    name="news_sentiment",
    steps=[
        {
            "name": "gather_news",
            "prompt": "Summarize recent news for {symbol}",
            "output": "news_summary"
        },
        {
            "name": "extract_events",
            "prompt": "Extract key events: {news_summary}",
            "output": "key_events"
        },
        {
            "name": "sentiment_analysis",
            "prompt": "Analyze sentiment: {news_summary}",
            "output": "sentiment_scores"
        },
        {
            "name": "impact_assessment",
            "prompt": "Assess market impact: {key_events} {sentiment_scores}",
            "output": "expected_impact"
        }
    ],
    context_keys=["symbol", "news_data"],
    output_format="structured"
)


@dataclass
class CacheEntry:
    """Data caching entry"""
    key: str
    data: Any
    timestamp: datetime
    ttl_seconds: int

    def is_expired(self) -> bool:
        age = (datetime.now() - self.timestamp).total_seconds()
        return age > self.ttl_seconds


@dataclass
class APIConfig:
    """API configuration"""
    provider: str
    api_key: str
    base_url: str
    rate_limit: int  # requests per minute
    timeout: int = 30


@dataclass
class UserPreferences:
    """User settings"""
    theme: str = "dark"
    default_watchlist: List[str] = field(default_factory=list)
    api_configs: Dict[str, APIConfig] = field(default_factory=dict)
    alert_settings: Dict[str, bool] = field(default_factory=dict)
    chart_preferences: Dict[str, Any] = field(default_factory=dict)
    export_format: str = "pdf"
