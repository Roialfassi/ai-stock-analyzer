# portfolio.py - Portfolio Management and Tracking

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json
import csv
from dataclasses import dataclass, field
import logging

from models import (
    Portfolio, PortfolioPosition, StockData, Alert
)
from market_data import MarketDataProvider

logger = logging.getLogger(__name__)


class PortfolioManager:
    """Main portfolio management class"""

    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
        self.portfolios: Dict[str, Portfolio] = {}
        self.alerts: List[Alert] = []
        self.transaction_history = []

    def create_portfolio(self, name: str, initial_cash: float = 0) -> Portfolio:
        """Create a new portfolio"""
        portfolio = Portfolio(
            name=name,
            positions=[],
            total_value=initial_cash,
            cash_balance=initial_cash,
            total_cost=0,
            total_pnl=0,
            total_pnl_percent=0,
            last_updated=datetime.now()
        )

        self.portfolios[name] = portfolio
        return portfolio

    def get_portfolio(self, name: str) -> Optional[Portfolio]:
        """Get portfolio by name"""
        return self.portfolios.get(name)

    def list_portfolios(self) -> List[str]:
        """List all portfolio names"""
        return list(self.portfolios.keys())

    async def add_position(self, portfolio_name: str, symbol: str,
                           shares: float, cost_per_share: float,
                           purchase_date: Optional[datetime] = None) -> PortfolioPosition:
        """Add a position to portfolio"""
        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        # Check if position already exists
        existing_position = self._find_position(portfolio, symbol)

        if existing_position:
            # Update existing position (average cost basis)
            total_shares = existing_position.shares + shares
            total_cost = (existing_position.cost_basis * existing_position.shares +
                          cost_per_share * shares)
            existing_position.shares = total_shares
            existing_position.cost_basis = total_cost / total_shares
            position = existing_position
        else:
            # Create new position
            stock_data = await self.data_provider.get_stock_data(symbol)
            if not stock_data:
                raise ValueError(f"Could not fetch data for {symbol}")

            position = PortfolioPosition(
                symbol=symbol,
                shares=shares,
                cost_basis=cost_per_share,
                current_value=stock_data.current_price * shares,
                purchase_date=purchase_date or datetime.now(),
                pnl=0,
                pnl_percent=0,
                allocation_percent=0
            )

            portfolio.positions.append(position)

        # Update portfolio
        portfolio.cash_balance -= cost_per_share * shares
        await self.update_portfolio(portfolio_name)

        # Record transaction
        self._record_transaction({
            'type': 'buy',
            'portfolio': portfolio_name,
            'symbol': symbol,
            'shares': shares,
            'price': cost_per_share,
            'date': datetime.now()
        })

        return position

    async def remove_position(self, portfolio_name: str, symbol: str,
                              shares: float, sell_price: float) -> float:
        """Remove or reduce a position"""
        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio:
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        position = self._find_position(portfolio, symbol)
        if not position:
            raise ValueError(f"Position {symbol} not found in portfolio")

        if shares > position.shares:
            raise ValueError(f"Cannot sell {shares} shares, only {position.shares} available")

        # Calculate realized P&L
        realized_pnl = (sell_price - position.cost_basis) * shares

        # Update position
        position.shares -= shares

        # Remove position if all shares sold
        if position.shares == 0:
            portfolio.positions.remove(position)

        # Update cash
        portfolio.cash_balance += sell_price * shares

        # Update portfolio
        await self.update_portfolio(portfolio_name)

        # Record transaction
        self._record_transaction({
            'type': 'sell',
            'portfolio': portfolio_name,
            'symbol': symbol,
            'shares': shares,
            'price': sell_price,
            'realized_pnl': realized_pnl,
            'date': datetime.now()
        })

        return realized_pnl

    async def update_portfolio(self, name: str):
        """Update portfolio values with current prices"""
        portfolio = self.get_portfolio(name)
        if not portfolio:
            return

        total_value = portfolio.cash_balance
        total_cost = 0

        # Update each position
        for position in portfolio.positions:
            stock_data = await self.data_provider.get_stock_data(position.symbol)
            if stock_data:
                current_price = stock_data.current_price
                position.current_value = current_price * position.shares
                position.pnl = (current_price - position.cost_basis) * position.shares
                position.pnl_percent = ((current_price / position.cost_basis) - 1) * 100

                total_value += position.current_value
                total_cost += position.cost_basis * position.shares

        # Calculate allocations
        for position in portfolio.positions:
            position.allocation_percent = (position.current_value / total_value) * 100

        # Update portfolio totals
        portfolio.total_value = total_value
        portfolio.total_cost = total_cost
        portfolio.total_pnl = total_value - total_cost - portfolio.cash_balance
        portfolio.total_pnl_percent = ((total_value / (
                    total_cost + portfolio.cash_balance)) - 1) * 100 if total_cost > 0 else 0
        portfolio.last_updated = datetime.now()

    async def get_performance_metrics(self, portfolio_name: str) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio:
            return {}

        metrics = {
            'total_return': portfolio.total_pnl_percent,
            'total_value': portfolio.total_value,
            'total_pnl': portfolio.total_pnl,
            'position_count': len(portfolio.positions),
            'cash_allocation': (portfolio.cash_balance / portfolio.total_value) * 100
        }

        # Calculate additional metrics
        if portfolio.positions:
            # Beta calculation (simplified)
            betas = []
            weights = []

            for position in portfolio.positions:
                stock_data = await self.data_provider.get_stock_data(position.symbol)
                if stock_data and stock_data.beta:
                    betas.append(stock_data.beta)
                    weights.append(position.allocation_percent / 100)

            if betas:
                metrics['portfolio_beta'] = sum(b * w for b, w in zip(betas, weights))

            # Concentration metrics
            allocations = [p.allocation_percent for p in portfolio.positions]
            metrics['largest_position'] = max(allocations)
            metrics['herfindahl_index'] = sum(a ** 2 for a in allocations) / 10000  # Concentration measure

            # P&L distribution
            pnl_values = [p.pnl_percent for p in portfolio.positions]
            metrics['best_performer'] = max(pnl_values)
            metrics['worst_performer'] = min(pnl_values)
            metrics['winners'] = sum(1 for p in pnl_values if p > 0)
            metrics['losers'] = sum(1 for p in pnl_values if p < 0)

        return metrics

    async def calculate_risk_metrics(self, portfolio_name: str,
                                     lookback_days: int = 252) -> Dict[str, float]:
        """Calculate risk metrics for portfolio"""
        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio or not portfolio.positions:
            return {}

        # Get historical data for positions
        returns_data = []
        weights = []

        for position in portfolio.positions:
            hist = await self.data_provider.get_historical_data(
                position.symbol,
                period=f"{lookback_days}d"
            )

            if not hist.empty:
                returns = hist['Close'].pct_change().dropna()
                returns_data.append(returns)
                weights.append(position.allocation_percent / 100)

        if not returns_data:
            return {}

        # Align returns data
        returns_df = pd.DataFrame(returns_data).T
        returns_df = returns_df.dropna()

        # Calculate portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)

        # Risk metrics
        metrics = {
            'volatility': portfolio_returns.std() * np.sqrt(252),  # Annualized
            'var_95': np.percentile(portfolio_returns, 5),  # 95% VaR
            'cvar_95': portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)].mean(),
            'max_drawdown': self._calculate_max_drawdown(portfolio_returns),
            'downside_deviation': portfolio_returns[portfolio_returns < 0].std() * np.sqrt(252)
        }

        # Sharpe ratio (assuming 2% risk-free rate)
        risk_free_rate = 0.02
        excess_returns = portfolio_returns.mean() * 252 - risk_free_rate
        metrics['sharpe_ratio'] = excess_returns / metrics['volatility'] if metrics['volatility'] > 0 else 0

        # Sortino ratio
        metrics['sortino_ratio'] = excess_returns / metrics['downside_deviation'] if metrics[
                                                                                         'downside_deviation'] > 0 else 0

        return metrics

    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()

    async def get_correlation_matrix(self, portfolio_name: str) -> pd.DataFrame:
        """Get correlation matrix for portfolio positions"""
        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio or len(portfolio.positions) < 2:
            return pd.DataFrame()

        # Get returns for all positions
        returns_dict = {}

        for position in portfolio.positions:
            hist = await self.data_provider.get_historical_data(position.symbol, period="1y")
            if not hist.empty:
                returns_dict[position.symbol] = hist['Close'].pct_change().dropna()

        if len(returns_dict) < 2:
            return pd.DataFrame()

        # Create returns dataframe and calculate correlation
        returns_df = pd.DataFrame(returns_dict)
        return returns_df.corr()

    async def rebalance_suggestions(self, portfolio_name: str,
                                    target_allocations: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate rebalancing suggestions"""
        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio:
            return []

        await self.update_portfolio(portfolio_name)

        suggestions = []
        current_allocations = {p.symbol: p.allocation_percent for p in portfolio.positions}

        # Check each target allocation
        for symbol, target_pct in target_allocations.items():
            current_pct = current_allocations.get(symbol, 0)
            diff_pct = target_pct - current_pct

            if abs(diff_pct) > 1:  # Only suggest if difference > 1%
                # Calculate shares to trade
                target_value = portfolio.total_value * (target_pct / 100)
                current_position = self._find_position(portfolio, symbol)
                current_value = current_position.current_value if current_position else 0

                value_diff = target_value - current_value

                # Get current price
                stock_data = await self.data_provider.get_stock_data(symbol)
                if stock_data:
                    shares_to_trade = value_diff / stock_data.current_price

                    suggestions.append({
                        'symbol': symbol,
                        'action': 'buy' if shares_to_trade > 0 else 'sell',
                        'shares': abs(shares_to_trade),
                        'current_allocation': current_pct,
                        'target_allocation': target_pct,
                        'value_change': value_diff
                    })

        return suggestions

    def set_alert(self, symbol: str, alert_type: str, condition: Dict[str, Any]) -> Alert:
        """Set a price or event alert"""
        alert = Alert(
            alert_id=f"{symbol}_{alert_type}_{datetime.now().timestamp()}",
            symbol=symbol,
            alert_type=alert_type,
            condition=condition
        )

        self.alerts.append(alert)
        return alert

    async def check_alerts(self) -> List[Alert]:
        """Check and trigger alerts"""
        triggered = []

        for alert in self.alerts:
            if alert.triggered:
                continue

            stock_data = await self.data_provider.get_stock_data(alert.symbol)
            if not stock_data:
                continue

            triggered_now = False

            if alert.alert_type == 'price_above':
                if stock_data.current_price > alert.condition['price']:
                    triggered_now = True

            elif alert.alert_type == 'price_below':
                if stock_data.current_price < alert.condition['price']:
                    triggered_now = True

            elif alert.alert_type == 'volume_spike':
                if stock_data.volume and stock_data.avg_volume:
                    if stock_data.volume > stock_data.avg_volume * alert.condition.get('multiplier', 2):
                        triggered_now = True

            if triggered_now:
                alert.triggered = True
                alert.triggered_at = datetime.now()
                triggered.append(alert)

        return triggered

    def export_portfolio(self, portfolio_name: str, format: str = 'csv') -> str:
        """Export portfolio to file"""
        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio:
            return ""

        if format == 'csv':
            return self._export_csv(portfolio)
        elif format == 'json':
            return self._export_json(portfolio)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _export_csv(self, portfolio: Portfolio) -> str:
        """Export portfolio as CSV"""
        output = []
        output.append("Symbol,Shares,Cost Basis,Current Value,P&L,P&L %,Allocation %")

        for position in portfolio.positions:
            output.append(
                f"{position.symbol},{position.shares},{position.cost_basis:.2f},"
                f"{position.current_value:.2f},{position.pnl:.2f},"
                f"{position.pnl_percent:.2f},{position.allocation_percent:.2f}"
            )

        output.append("")
        output.append(f"Total Value,{portfolio.total_value:.2f}")
        output.append(f"Cash Balance,{portfolio.cash_balance:.2f}")
        output.append(f"Total P&L,{portfolio.total_pnl:.2f}")
        output.append(f"Total P&L %,{portfolio.total_pnl_percent:.2f}")

        return "\n".join(output)

    def _export_json(self, portfolio: Portfolio) -> str:
        """Export portfolio as JSON"""
        data = {
            'name': portfolio.name,
            'total_value': portfolio.total_value,
            'cash_balance': portfolio.cash_balance,
            'total_pnl': portfolio.total_pnl,
            'total_pnl_percent': portfolio.total_pnl_percent,
            'last_updated': portfolio.last_updated.isoformat(),
            'positions': [
                {
                    'symbol': p.symbol,
                    'shares': p.shares,
                    'cost_basis': p.cost_basis,
                    'current_value': p.current_value,
                    'pnl': p.pnl,
                    'pnl_percent': p.pnl_percent,
                    'allocation_percent': p.allocation_percent,
                    'purchase_date': p.purchase_date.isoformat()
                }
                for p in portfolio.positions
            ]
        }

        return json.dumps(data, indent=2)

    def import_portfolio(self, name: str, data: str, format: str = 'csv') -> Portfolio:
        """Import portfolio from file"""
        if format == 'csv':
            return self._import_csv(name, data)
        elif format == 'json':
            return self._import_json(name, data)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _import_csv(self, name: str, data: str) -> Portfolio:
        """Import portfolio from CSV"""
        portfolio = self.create_portfolio(name)

        reader = csv.DictReader(data.strip().split('\n'))
        for row in reader:
            if 'Symbol' in row:  # Skip summary rows
                position = PortfolioPosition(
                    symbol=row['Symbol'],
                    shares=float(row['Shares']),
                    cost_basis=float(row['Cost Basis']),
                    current_value=0,  # Will be updated
                    purchase_date=datetime.now(),
                    pnl=0,
                    pnl_percent=0,
                    allocation_percent=0
                )
                portfolio.positions.append(position)

        return portfolio

    def _import_json(self, name: str, data: str) -> Portfolio:
        """Import portfolio from JSON"""
        json_data = json.loads(data)

        portfolio = self.create_portfolio(name, json_data.get('cash_balance', 0))

        for position_data in json_data.get('positions', []):
            position = PortfolioPosition(
                symbol=position_data['symbol'],
                shares=position_data['shares'],
                cost_basis=position_data['cost_basis'],
                current_value=0,  # Will be updated
                purchase_date=datetime.fromisoformat(position_data.get('purchase_date', datetime.now().isoformat())),
                pnl=0,
                pnl_percent=0,
                allocation_percent=0
            )
            portfolio.positions.append(position)

        return portfolio

    def _find_position(self, portfolio: Portfolio, symbol: str) -> Optional[PortfolioPosition]:
        """Find position in portfolio by symbol"""
        for position in portfolio.positions:
            if position.symbol == symbol:
                return position
        return None

    def _record_transaction(self, transaction: Dict[str, Any]):
        """Record a transaction in history"""
        self.transaction_history.append(transaction)

        # Keep only last 1000 transactions
        if len(self.transaction_history) > 1000:
            self.transaction_history = self.transaction_history[-1000:]

    def get_transaction_history(self, portfolio_name: Optional[str] = None,
                                days: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get transaction history"""
        transactions = self.transaction_history

        # Filter by portfolio
        if portfolio_name:
            transactions = [t for t in transactions if t.get('portfolio') == portfolio_name]

        # Filter by date
        if days:
            cutoff_date = datetime.now() - timedelta(days=days)
            transactions = [t for t in transactions if t.get('date', datetime.min) > cutoff_date]

        return transactions

    async def calculate_tax_implications(self, portfolio_name: str,
                                         tax_rate_short: float = 0.35,
                                         tax_rate_long: float = 0.15) -> Dict[str, Any]:
        """Calculate potential tax implications"""
        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio:
            return {}

        short_term_gains = 0
        long_term_gains = 0

        one_year_ago = datetime.now() - timedelta(days=365)

        for position in portfolio.positions:
            if position.pnl > 0:  # Only gains are taxed
                if position.purchase_date > one_year_ago:
                    short_term_gains += position.pnl
                else:
                    long_term_gains += position.pnl

        return {
            'short_term_gains': short_term_gains,
            'long_term_gains': long_term_gains,
            'short_term_tax': short_term_gains * tax_rate_short,
            'long_term_tax': long_term_gains * tax_rate_long,
            'total_tax': short_term_gains * tax_rate_short + long_term_gains * tax_rate_long
        }


class PortfolioOptimizer:
    """Portfolio optimization utilities"""

    @staticmethod
    async def optimize_portfolio(current_positions: List[PortfolioPosition],
                                 data_provider: MarketDataProvider,
                                 optimization_method: str = 'sharpe') -> Dict[str, float]:
        """Optimize portfolio allocation"""
        if not current_positions:
            return {}

        # Get historical returns
        symbols = [p.symbol for p in current_positions]
        returns_data = {}

        for symbol in symbols:
            hist = await data_provider.get_historical_data(symbol, period="2y")
            if not hist.empty:
                returns_data[symbol] = hist['Close'].pct_change().dropna()

        if not returns_data:
            return {}

        # Create returns matrix
        returns_df = pd.DataFrame(returns_data)
        returns_df = returns_df.dropna()

        # Calculate expected returns and covariance
        expected_returns = returns_df.mean() * 252  # Annualized
        cov_matrix = returns_df.cov() * 252  # Annualized

        # Optimize based on method
        if optimization_method == 'sharpe':
            weights = PortfolioOptimizer._maximize_sharpe(expected_returns, cov_matrix)
        elif optimization_method == 'min_variance':
            weights = PortfolioOptimizer._minimize_variance(cov_matrix)
        elif optimization_method == 'equal_weight':
            weights = np.array([1 / len(symbols)] * len(symbols))
        else:
            weights = np.array([1 / len(symbols)] * len(symbols))

        # Convert to allocation percentages
        allocations = {}
        for i, symbol in enumerate(symbols):
            allocations[symbol] = weights[i] * 100

        return allocations

    @staticmethod
    def _maximize_sharpe(expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                         risk_free_rate: float = 0.02) -> np.ndarray:
        """Maximize Sharpe ratio (simplified implementation)"""
        n = len(expected_returns)

        # Simple equal weight for now
        # In production, use scipy.optimize
        weights = np.array([1 / n] * n)

        return weights

    @staticmethod
    def _minimize_variance(cov_matrix: pd.DataFrame) -> np.ndarray:
        """Minimize portfolio variance (simplified)"""
        n = len(cov_matrix)

        # Simple equal weight for now
        # In production, use scipy.optimize
        weights = np.array([1 / n] * n)

        return weights


class DividendTracker:
    """Track and analyze dividends"""

    def __init__(self, data_provider: MarketDataProvider):
        self.data_provider = data_provider
        self.dividend_history = {}

    async def get_dividend_forecast(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Forecast dividend income"""
        monthly_dividends = [0] * 12
        annual_total = 0

        for position in portfolio.positions:
            stock_data = await self.data_provider.get_stock_data(position.symbol)

            if stock_data and stock_data.dividend_yield:
                # Estimate annual dividend
                annual_dividend = position.current_value * stock_data.dividend_yield
                annual_total += annual_dividend

                # Assume quarterly payments
                quarterly_payment = annual_dividend / 4
                for month in [0, 3, 6, 9]:  # Q1, Q2, Q3, Q4
                    monthly_dividends[month] += quarterly_payment

        return {
            'monthly_forecast': monthly_dividends,
            'annual_total': annual_total,
            'yield_on_cost': (annual_total / portfolio.total_cost * 100) if portfolio.total_cost > 0 else 0
        }

    async def analyze_dividend_safety(self, symbol: str) -> Dict[str, Any]:
        """Analyze dividend safety for a stock"""
        stock_data = await self.data_provider.get_stock_data(symbol)
        financials = await self.data_provider.get_financial_statements(symbol)

        if not stock_data or not financials:
            return {'safety_score': 0, 'analysis': 'Unable to analyze'}

        safety_score = 50  # Base score
        factors = []

        # Payout ratio check
        if stock_data.eps and stock_data.dividend_yield and stock_data.current_price:
            dividend_per_share = stock_data.current_price * stock_data.dividend_yield
            payout_ratio = dividend_per_share / stock_data.eps

            if payout_ratio < 0.6:
                safety_score += 20
                factors.append("Healthy payout ratio")
            elif payout_ratio > 0.9:
                safety_score -= 20
                factors.append("High payout ratio - risk")

        # Debt check
        if stock_data.debt_to_equity:
            if stock_data.debt_to_equity < 0.5:
                safety_score += 10
                factors.append("Low debt levels")
            elif stock_data.debt_to_equity > 2:
                safety_score -= 10
                factors.append("High debt levels")

        # Growth check
        if stock_data.revenue_growth and stock_data.revenue_growth > 0:
            safety_score += 10
            factors.append("Growing revenue")

        # Cap score
        safety_score = max(0, min(100, safety_score))

        return {
            'safety_score': safety_score,
            'factors': factors,
            'recommendation': self._get_dividend_recommendation(safety_score)
        }

    def _get_dividend_recommendation(self, score: float) -> str:
        """Get dividend safety recommendation"""
        if score >= 80:
            return "Very Safe - Low risk of dividend cut"
        elif score >= 60:
            return "Safe - Dividend appears sustainable"
        elif score >= 40:
            return "Moderate Risk - Monitor closely"
        elif score >= 20:
            return "High Risk - Dividend cut possible"
        else:
            return "Very High Risk - Dividend unsustainable"


class PortfolioComparison:
    """Compare multiple portfolios"""

    @staticmethod
    def compare_portfolios(portfolios: List[Portfolio]) -> pd.DataFrame:
        """Create comparison table of portfolios"""
        if not portfolios:
            return pd.DataFrame()

        comparison_data = []

        for portfolio in portfolios:
            comparison_data.append({
                'Name': portfolio.name,
                'Total Value': portfolio.total_value,
                'P&L': portfolio.total_pnl,
                'P&L %': portfolio.total_pnl_percent,
                'Positions': len(portfolio.positions),
                'Cash %': (portfolio.cash_balance / portfolio.total_value * 100) if portfolio.total_value > 0 else 0
            })

        return pd.DataFrame(comparison_data)

    @staticmethod
    async def compare_performance(portfolios: List[Portfolio],
                                  data_provider: MarketDataProvider,
                                  benchmark: str = "SPY") -> Dict[str, Any]:
        """Compare portfolio performance against benchmark"""
        # Get benchmark data
        benchmark_hist = await data_provider.get_historical_data(benchmark, period="1y")
        if benchmark_hist.empty:
            return {}

        benchmark_return = (benchmark_hist['Close'].iloc[-1] / benchmark_hist['Close'].iloc[0] - 1) * 100

        results = {
            'benchmark': benchmark,
            'benchmark_return': benchmark_return,
            'portfolios': {}
        }

        for portfolio in portfolios:
            results['portfolios'][portfolio.name] = {
                'return': portfolio.total_pnl_percent,
                'excess_return': portfolio.total_pnl_percent - benchmark_return,
                'outperformed': portfolio.total_pnl_percent > benchmark_return
            }

        return results
