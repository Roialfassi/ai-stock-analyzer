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
        if data_provider is None:
            logger.error("PortfolioManager initialized with no data_provider.")
            # Or raise ValueError("Data provider cannot be None")
            # For now, allow it but expect errors if methods are called.
        self.data_provider = data_provider
        self.portfolios: Dict[str, Portfolio] = {}
        self.alerts: List[Alert] = []
        self.transaction_history = []

    def create_portfolio(self, name: str, initial_cash: float = 0) -> Portfolio:
        """Create a new portfolio"""
        if not name or not isinstance(name, str):
            logger.error(f"Invalid portfolio name provided: {name}")
            raise ValueError("Portfolio name must be a non-empty string.")
        if not isinstance(initial_cash, (int, float)) or initial_cash < 0:
            logger.error(f"Invalid initial_cash value: {initial_cash}")
            raise ValueError("Initial cash must be a non-negative number.")

        if name in self.portfolios:
            logger.warning(f"Portfolio '{name}' already exists. Returning existing portfolio.")
            # Or raise ValueError(f"Portfolio '{name}' already exists.")
            return self.portfolios[name]

        try:
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
        except Exception as e: # Catch any error during Portfolio dataclass instantiation
            logger.exception(f"Error creating portfolio object for '{name}': {e}")
            # Depending on desired behavior, could re-raise or return None/raise custom error
            raise RuntimeError(f"Failed to instantiate portfolio: {e}") from e


    def get_portfolio(self, name: str) -> Optional[Portfolio]:
        """Get portfolio by name"""
        if not name: # Basic validation
            logger.warning("Attempted to get portfolio with empty name.")
            return None
        return self.portfolios.get(name)

    def list_portfolios(self) -> List[str]:
        """List all portfolio names"""
        try:
            return list(self.portfolios.keys())
        except Exception as e: # Should be rare for dict.keys()
            logger.exception(f"Error listing portfolios: {e}")
            return []
        return list(self.portfolios.keys())

    async def add_position(self, portfolio_name: str, symbol: str,
                           shares: float, cost_per_share: float,
                           purchase_date: Optional[datetime] = None) -> PortfolioPosition:
        """Add a position to portfolio"""
        if not self.data_provider:
            logger.error("Cannot add position: MarketDataProvider is not available.")
            raise RuntimeError("MarketDataProvider not initialized.")

        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio:
            logger.error(f"Portfolio '{portfolio_name}' not found for adding position.")
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string.")
        if not isinstance(shares, (int,float)) or shares <= 0:
            raise ValueError("Shares must be a positive number.")
        if not isinstance(cost_per_share, (int,float)) or cost_per_share < 0: # Cost can be 0 for grants etc.
            raise ValueError("Cost per share must be a non-negative number.")

        try:
            existing_position = self._find_position(portfolio, symbol)

            if existing_position:
                # Update existing position (average cost basis)
                new_total_shares = existing_position.shares + shares
                if new_total_shares == 0: # Should not happen if shares > 0
                     raise ValueError("Total shares cannot be zero after adding.")
                new_total_cost = (existing_position.cost_basis * existing_position.shares) + (cost_per_share * shares)

                existing_position.cost_basis = new_total_cost / new_total_shares
                existing_position.shares = new_total_shares
                # Potentially update purchase_date if strategy requires (e.g., to latest, or keep original)
                # For now, keeping original purchase_date of the first lot.
                position_to_return = existing_position
            else:
                stock_data = await self.data_provider.get_stock_data(symbol)
                if not stock_data or stock_data.current_price is None: # Ensure current_price is available
                    logger.error(f"Could not fetch valid stock data (or price) for {symbol} to add position.")
                    raise ValueError(f"Could not fetch valid market data for {symbol}")

                position_to_return = PortfolioPosition(
                    symbol=symbol,
                    shares=shares,
                    cost_basis=cost_per_share,
                    current_value=stock_data.current_price * shares, # Initial current value
                    purchase_date=purchase_date or datetime.now(),
                    pnl=0, # Initial P&L is 0 before first update
                    pnl_percent=0,
                    allocation_percent=0 # Will be calculated on update
                )
                portfolio.positions.append(position_to_return)

            portfolio.cash_balance -= cost_per_share * shares
            if portfolio.cash_balance < 0:
                logger.warning(f"Portfolio '{portfolio_name}' cash balance is negative after buying {symbol}.")
                # Depending on rules, this could raise an error or just be a warning.

            await self.update_portfolio(portfolio_name) # This method should handle its own errors
        except ValueError as ve: # Re-raise specific ValueErrors
            logger.error(f"ValueError adding position {symbol} to {portfolio_name}: {ve}")
            raise
        except Exception as e: # Catch other unexpected errors
            logger.exception(f"Unexpected error adding position {symbol} to {portfolio_name}: {e}")
            raise RuntimeError(f"Failed to add position {symbol}: {e}") from e

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
            logger.error(f"Portfolio '{portfolio_name}' not found for removing position.")
            raise ValueError(f"Portfolio '{portfolio_name}' not found")

        if not symbol or not isinstance(symbol, str):
            raise ValueError("Symbol must be a non-empty string.")
        if not isinstance(shares, (int,float)) or shares <= 0:
            raise ValueError("Shares to sell must be a positive number.")
        if not isinstance(sell_price, (int,float)) or sell_price < 0:
            raise ValueError("Sell price must be a non-negative number.")

        try:
            position = self._find_position(portfolio, symbol)
            if not position:
                logger.error(f"Position {symbol} not found in portfolio {portfolio_name} for removal.")
                raise ValueError(f"Position {symbol} not found in portfolio")

            if shares > position.shares:
                logger.error(f"Attempted to sell {shares} of {symbol}, but only {position.shares} available in {portfolio_name}.")
                raise ValueError(f"Cannot sell {shares} shares of {symbol}, only {position.shares} available.")

            realized_pnl = (sell_price - position.cost_basis) * shares
            position.shares -= shares

            if abs(position.shares) < 1e-9: # Effectively zero shares, remove position
                portfolio.positions.remove(position)

            portfolio.cash_balance += sell_price * shares
            await self.update_portfolio(portfolio_name) # Handles its own errors
        except ValueError as ve: # Re-raise specific ValueErrors
            logger.error(f"ValueError removing position {symbol} from {portfolio_name}: {ve}")
            raise
        except Exception as e: # Catch other unexpected errors
            logger.exception(f"Unexpected error removing position {symbol} from {portfolio_name}: {e}")
            raise RuntimeError(f"Failed to remove position {symbol}: {e}") from e

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
        if not self.data_provider:
            logger.error(f"Cannot update portfolio '{name}': MarketDataProvider is not available.")
            # Optionally, set portfolio to a state indicating data is stale if desired.
            return

        portfolio = self.get_portfolio(name)
        if not portfolio:
            logger.warning(f"Portfolio '{name}' not found for update.")
            return

        try:
            new_total_value = portfolio.cash_balance
            new_total_cost_of_positions = 0.0 # Cost of current holdings, not historical total cost

            for position in portfolio.positions:
                try:
                    stock_data = await self.data_provider.get_stock_data(position.symbol)
                    if stock_data and stock_data.current_price is not None:
                        current_price = stock_data.current_price
                        position.current_value = current_price * position.shares
                        position.pnl = (current_price - position.cost_basis) * position.shares
                        if position.cost_basis != 0: # Avoid ZeroDivisionError
                            position.pnl_percent = ((current_price / position.cost_basis) - 1) * 100.0
                        else: # Cost basis is 0 (e.g. gifted shares), P&L% is effectively infinite or undefined
                            position.pnl_percent = float('inf') if current_price > 0 else 0.0

                        new_total_value += position.current_value
                        new_total_cost_of_positions += position.cost_basis * position.shares
                    else: # Stock data not found or no price
                        logger.warning(f"Could not get current price for {position.symbol} in portfolio {name}. Using last known value for this position.")
                        # Keep existing position.current_value and P&L, but it will be stale.
                        # Add its last known current_value to total_value.
                        new_total_value += position.current_value
                        new_total_cost_of_positions += position.cost_basis * position.shares
                except Exception as e_pos: # Catch error for a single position update
                    logger.exception(f"Error updating position {position.symbol} in portfolio {name}: {e_pos}")
                    # If a position update fails, add its last known value to total to avoid skewing portfolio value too much
                    new_total_value += position.current_value
                    new_total_cost_of_positions += position.cost_basis * position.shares


            portfolio.total_value = new_total_value
            # portfolio.total_cost should represent the sum of initial investments for active positions
            # This might need adjustment based on how 'total_cost' is defined (e.g., including past closed trades for overall P&L)
            # For now, let's assume total_cost refers to the cost of current holdings.
            portfolio.total_cost = new_total_cost_of_positions

            # Overall P&L calculation
            # Total P&L = (Current Total Value of Positions + Cash) - (Total Cost of Positions + Initial Cash not part of positions)
            # Or, if total_cost is sum of all cash ever put in: Total P&L = Total Value - Total Cash In
            # The current model seems to imply total_pnl is based on current holdings:
            portfolio.total_pnl = portfolio.total_value - (portfolio.total_cost + portfolio.cash_balance) # If total_cost is cost of current positions

            if (portfolio.total_cost + portfolio.cash_balance) != 0: # Denominator for overall P&L %
                portfolio.total_pnl_percent = (portfolio.total_pnl / (portfolio.total_cost + portfolio.cash_balance)) * 100.0
            else:
                portfolio.total_pnl_percent = 0.0

            # Calculate allocations after total_value is finalized
            if portfolio.total_value != 0:
                for position in portfolio.positions:
                    position.allocation_percent = (position.current_value / portfolio.total_value) * 100.0
            else: # Avoid division by zero if total value is zero
                for position in portfolio.positions:
                    position.allocation_percent = 0.0

            portfolio.last_updated = datetime.now()
        except Exception as e:
            logger.exception(f"Unexpected error updating portfolio '{name}': {e}")
            # Decide if portfolio state should be partially updated or reverted/marked stale.


    async def get_performance_metrics(self, portfolio_name: str) -> Dict[str, float]:
        """Calculate portfolio performance metrics"""
        if not self.data_provider:
            logger.error("Cannot get performance metrics: MarketDataProvider is not available.")
            return {} # Or raise error

        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio:
            logger.warning(f"Portfolio '{portfolio_name}' not found for performance metrics.")
            return {}

        # Ensure portfolio is up-to-date before calculating metrics
        # await self.update_portfolio(portfolio_name) # This might be redundant if called frequently elsewhere

        metrics: Dict[str, Optional[float]] = {} # Allow None for metrics that can't be calculated
        try:
            metrics = {
                'total_return': portfolio.total_pnl_percent,
                'total_value': portfolio.total_value,
                'total_pnl': portfolio.total_pnl,
                'position_count': float(len(portfolio.positions)), # Ensure float for consistency
                'cash_allocation': (portfolio.cash_balance / portfolio.total_value) * 100.0 if portfolio.total_value != 0 else 0.0
            }

            if portfolio.positions:
                betas = []
                weights = []
                valid_positions_for_beta = 0
                for position in portfolio.positions:
                    try:
                        stock_data = await self.data_provider.get_stock_data(position.symbol)
                        if stock_data and stock_data.beta is not None and pd.notna(stock_data.beta):
                            betas.append(stock_data.beta)
                            # Use current value for weighting beta, not stored allocation_percent which might be stale
                            weights.append(position.current_value / portfolio.total_value if portfolio.total_value !=0 else 0)
                            valid_positions_for_beta +=1
                        elif stock_data and (stock_data.beta is None or pd.isna(stock_data.beta)):
                             logger.debug(f"Beta not available for {position.symbol}, skipping for portfolio beta.")
                        # else: stock_data is None, error already logged by get_stock_data or update_portfolio
                    except Exception as e_beta: # pragma: no cover
                        logger.error(f"Error fetching stock data for beta calculation of {position.symbol}: {e_beta}")

                if betas and weights and valid_positions_for_beta > 0: # Ensure we have data to calculate beta
                    # Normalize weights if only some positions had beta
                    current_sum_weights = sum(weights)
                    if current_sum_weights > 0 and current_sum_weights != 1.0: # Normalize
                        normalized_weights = [w / current_sum_weights for w in weights]
                        metrics['portfolio_beta'] = sum(b * w for b, w in zip(betas, normalized_weights))
                    elif current_sum_weights == 1.0: # Weights are already normalized
                         metrics['portfolio_beta'] = sum(b * w for b, w in zip(betas, weights))
                    else: # No valid weights or betas
                        metrics['portfolio_beta'] = None

                allocations = [p.allocation_percent for p in portfolio.positions if p.allocation_percent is not None]
                if allocations:
                    metrics['largest_position'] = max(allocations) if allocations else None
                    metrics['herfindahl_index'] = sum(a**2 for a in allocations) / 10000.0 # HHI usually on 0-1 or 0-10000 scale

                pnl_values = [p.pnl_percent for p in portfolio.positions if p.pnl_percent is not None and pd.notna(p.pnl_percent)]
                if pnl_values:
                    metrics['best_performer'] = max(pnl_values)
                    metrics['worst_performer'] = min(pnl_values)
                    metrics['winners'] = float(sum(1 for p in pnl_values if p > 0))
                    metrics['losers'] = float(sum(1 for p in pnl_values if p < 0))
        except ZeroDivisionError as zde: # pragma: no cover
            logger.warning(f"ZeroDivisionError calculating performance metrics for {portfolio_name}: {zde}")
            # Some metrics might be None or 0 due to this
        except Exception as e: # pragma: no cover
            logger.exception(f"Unexpected error calculating performance metrics for {portfolio_name}: {e}")
            # Return partially filled metrics or empty dict

        return metrics


    async def calculate_risk_metrics(self, portfolio_name: str,
                                     lookback_days: int = 252) -> Dict[str, float]:
        """Calculate risk metrics for portfolio. Returns dict with float values, defaults to 0.0 or NaN on error."""
        if not self.data_provider:
            logger.error("Cannot calculate risk metrics: MarketDataProvider is not available.")
            return {}

        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio or not portfolio.positions:
            logger.warning(f"Portfolio '{portfolio_name}' not found or has no positions for risk calculation.")
            return {}

        default_risk_metrics = {
            'volatility': 0.0, 'var_95': 0.0, 'cvar_95': 0.0,
            'max_drawdown': 0.0, 'downside_deviation': 0.0,
            'sharpe_ratio': 0.0, 'sortino_ratio': 0.0
        }

        try:
            returns_list = [] # Changed from returns_data to returns_list for clarity
            weights = []
            # Ensure portfolio total value is up-to-date for accurate weighting
            await self.update_portfolio(portfolio_name) # Recalculate current values and total_value
            if portfolio.total_value == 0: # Cannot calculate weights if total value is zero
                logger.warning(f"Portfolio '{portfolio_name}' total value is zero. Cannot calculate weighted risk metrics.")
                return default_risk_metrics


            for position in portfolio.positions:
                try:
                    hist_df = await self.data_provider.get_historical_data(
                        position.symbol, period=f"{lookback_days + 1}d" # Fetch 1 extra day for pct_change
                    )
                    if not hist_df.empty and 'Close' in hist_df.columns:
                        returns = hist_df['Close'].pct_change().dropna()
                        if not returns.empty:
                            returns_list.append(returns)
                            # Use current value for weighting, not stored allocation_percent
                            weights.append(position.current_value / portfolio.total_value)
                        else: logger.debug(f"No returns for {position.symbol} after pct_change/dropna.")
                    else: logger.debug(f"Empty or invalid historical data for {position.symbol}.")
                except Exception as e_hist: # pragma: no cover
                    logger.error(f"Error fetching/processing hist data for {position.symbol} in risk calc: {e_hist}")

            if not returns_list or len(returns_list) != len(weights) or not any(w > 0 for w in weights): # Ensure we have data
                logger.warning(f"Not enough valid returns data or weights to calculate risk metrics for {portfolio_name}.")
                return default_risk_metrics

            returns_df = pd.DataFrame(returns_list).T # Transpose to have symbols as columns, dates as index
            returns_df = returns_df.dropna(how='all') # Drop rows where all returns are NaN (e.g. market holidays)
            if returns_df.empty or len(returns_df) < 2: # Need at least 2 data points for std dev
                logger.warning(f"Returns DataFrame empty or too short after processing for {portfolio_name}.")
                return default_risk_metrics

            # Align weights with available returns columns
            aligned_weights = []
            final_returns_cols = []
            for i, col_name in enumerate(returns_df.columns):
                # This assumes returns_list order matches symbols in portfolio.positions
                # A more robust way would be to name series in returns_list by symbol
                # and then align weights using those symbols.
                # For now, direct mapping based on order.
                 if i < len(weights): # Check if weight exists for this column
                    aligned_weights.append(weights[i])
                    final_returns_cols.append(col_name)

            if not aligned_weights or not final_returns_cols: # pragma: no cover
                logger.warning(f"Could not align weights and returns for {portfolio_name}")
                return default_risk_metrics

            returns_df_aligned = returns_df[final_returns_cols]
            portfolio_returns = (returns_df_aligned * aligned_weights).sum(axis=1, skipna=False) # skipna=False to propagate NaNs
            portfolio_returns = portfolio_returns.dropna() # Drop any resulting NaNs in portfolio returns series

            if portfolio_returns.empty or len(portfolio_returns) < 2:
                logger.warning(f"Portfolio returns series empty or too short for risk calculation in {portfolio_name}.")
                return default_risk_metrics

            metrics = {}
            metrics['volatility'] = portfolio_returns.std() * np.sqrt(lookback_days) # Annualize based on lookback
            metrics['var_95'] = np.percentile(portfolio_returns, 5) if not portfolio_returns.empty else 0.0
            metrics['cvar_95'] = portfolio_returns[portfolio_returns <= metrics['var_95']].mean() if not portfolio_returns[portfolio_returns <= metrics['var_95']].empty else 0.0
            metrics['max_drawdown'] = self._calculate_max_drawdown(portfolio_returns)

            downside_returns = portfolio_returns[portfolio_returns < 0]
            metrics['downside_deviation'] = downside_returns.std() * np.sqrt(lookback_days) if not downside_returns.empty else 0.0

            risk_free_rate_annual = 0.02 # Annual RFR
            # Daily RFR for Sharpe/Sortino if returns are daily
            # For simplicity, using annualized returns vs annualized RFR
            annualized_return = portfolio_returns.mean() * lookback_days
            excess_returns_annualized = annualized_return - risk_free_rate_annual

            metrics['sharpe_ratio'] = excess_returns_annualized / metrics['volatility'] if metrics['volatility'] > 1e-9 else 0.0
            metrics['sortino_ratio'] = excess_returns_annualized / metrics['downside_deviation'] if metrics['downside_deviation'] > 1e-9 else 0.0

            # Ensure all are floats
            return {k: float(v) if pd.notna(v) else 0.0 for k, v in metrics.items()}

        except Exception as e: # pragma: no cover
            logger.exception(f"Unexpected error calculating risk metrics for {portfolio_name}: {e}")
            return default_risk_metrics


    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if not isinstance(returns, pd.Series) or returns.empty:
            return 0.0
        try:
            cumulative = (1 + returns).cumprod()
            if cumulative.empty: return 0.0
            running_max = cumulative.expanding(min_periods=1).max()
            drawdown = (cumulative - running_max) / running_max
            min_drawdown = drawdown.min()
            return float(min_drawdown) if pd.notna(min_drawdown) else 0.0
        except Exception as e: # pragma: no cover
            logger.error(f"Error calculating max drawdown: {e}", exc_info=True)
            return 0.0


    async def get_correlation_matrix(self, portfolio_name: str) -> pd.DataFrame:
        """Get correlation matrix for portfolio positions"""
        if not self.data_provider:
            logger.error("Cannot get correlation matrix: MarketDataProvider is not available.")
            return pd.DataFrame()

        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio or len(portfolio.positions) < 2:
            logger.debug(f"Portfolio '{portfolio_name}' has less than 2 positions, cannot compute correlation matrix.")
            return pd.DataFrame()

        returns_dict = {}
        try:
            for position in portfolio.positions:
                try:
                    hist_df = await self.data_provider.get_historical_data(position.symbol, period="1y")
                    if not hist_df.empty and 'Close' in hist_df.columns:
                        returns = hist_df['Close'].pct_change().dropna()
                        if not returns.empty:
                            returns_dict[position.symbol] = returns
                except Exception as e_hist: # pragma: no cover
                    logger.error(f"Error fetching hist data for {position.symbol} in correlation calc: {e_hist}")

            if len(returns_dict) < 2:
                logger.debug(f"Not enough valid historical data series to form correlation matrix for {portfolio_name}.")
                return pd.DataFrame()

            returns_df = pd.DataFrame(returns_dict)
            # Ensure consistent length by forward-filling NaNs then dropping rows with any NaNs left
            # This handles cases where stocks might have slightly different start dates within the period
            returns_df = returns_df.fillna(method='ffill').dropna(how='any')
            if len(returns_df) < 2 or len(returns_df.columns) < 2: # Need at least 2 data points and 2 assets
                 logger.debug(f"DataFrame for correlation for {portfolio_name} has insufficient dimensions after NaN handling.")
                 return pd.DataFrame()

            return returns_df.corr()
        except Exception as e: # pragma: no cover
            logger.exception(f"Unexpected error calculating correlation matrix for {portfolio_name}: {e}")
            return pd.DataFrame()


    async def rebalance_suggestions(self, portfolio_name: str,
                                    target_allocations: Dict[str, float]) -> List[Dict[str, Any]]:
        """Generate rebalancing suggestions"""
        if not self.data_provider:
            logger.error("Cannot suggest rebalance: MarketDataProvider is not available.")
            return []

        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio:
            logger.warning(f"Portfolio '{portfolio_name}' not found for rebalancing.")
            return []

        try:
            await self.update_portfolio(portfolio_name) # Ensure values are current
            if portfolio.total_value == 0: # Avoid division by zero if portfolio value is zero
                logger.info(f"Portfolio '{portfolio_name}' total value is zero. Cannot generate rebalance suggestions based on percentage targets.")
                return []


            suggestions = []
            current_allocations = {p.symbol: p.allocation_percent for p in portfolio.positions if p.allocation_percent is not None}

            for symbol, target_pct in target_allocations.items():
                if not (isinstance(target_pct, (int, float)) and 0 <= target_pct <= 100):
                    logger.warning(f"Invalid target allocation {target_pct}% for {symbol}. Skipping.")
                    continue

                current_pct = current_allocations.get(symbol, 0.0)
                diff_pct = target_pct - current_pct

                # Define a meaningful threshold for rebalancing, e.g., 1% of portfolio value or 5% of position
                # For now, using absolute percent difference.
                if abs(diff_pct) > 1.0:  # Suggest if difference > 1%
                    target_value_of_symbol = portfolio.total_value * (target_pct / 100.0)

                    current_pos_obj = self._find_position(portfolio, symbol)
                    current_value_of_symbol = current_pos_obj.current_value if current_pos_obj else 0.0

                    value_diff_to_trade = target_value_of_symbol - current_value_of_symbol

                    stock_data = await self.data_provider.get_stock_data(symbol)
                    if stock_data and stock_data.current_price is not None and stock_data.current_price > 0:
                        shares_to_trade = value_diff_to_trade / stock_data.current_price
                        suggestions.append({
                            'symbol': symbol,
                            'action': 'buy' if shares_to_trade > 0 else 'sell',
                            'shares': abs(shares_to_trade),
                            'current_allocation_pct': current_pct,
                            'target_allocation_pct': target_pct, # Corrected key
                            'value_change_needed': value_diff_to_trade # Corrected key
                        })
                    elif not stock_data or stock_data.current_price is None or stock_data.current_price <=0: # pragma: no cover
                        logger.warning(f"Could not get valid price for {symbol} to suggest rebalance. Skipping.")

            return suggestions
        except Exception as e: # pragma: no cover
            logger.exception(f"Error generating rebalance suggestions for {portfolio_name}: {e}")
            return []


    def set_alert(self, symbol: str, alert_type: str, condition: Dict[str, Any]) -> Alert:
        """Set a price or event alert"""
        if not all([symbol, alert_type, isinstance(condition, dict)]):
            raise ValueError("Invalid parameters for creating alert.")

        # Basic validation of condition based on alert_type
        if alert_type in ['price_above', 'price_below']:
            if 'price' not in condition or not isinstance(condition['price'], (int, float)):
                raise ValueError(f"Invalid condition for {alert_type}: 'price' must be a number.")
        elif alert_type == 'volume_spike':
            if 'multiplier' not in condition or not isinstance(condition['multiplier'], (int, float)): # pragma: no cover
                logger.warning("Volume spike alert condition missing 'multiplier', defaulting to 2.")
                condition.setdefault('multiplier', 2.0)
        # Add more validation for other alert types as needed

        try:
            alert = Alert(
                alert_id=f"{symbol}_{alert_type}_{datetime.now().timestamp()}", # Timestamp makes it unique
                symbol=symbol,
                alert_type=alert_type,
                condition=condition
            )
            self.alerts.append(alert)
            return alert
        except Exception as e: # Catch potential errors from Alert dataclass instantiation
            logger.exception(f"Error creating alert for {symbol}: {e}")
            raise RuntimeError(f"Could not create alert: {e}") from e


    async def check_alerts(self) -> List[Alert]:
        """Check and trigger alerts"""
        if not self.data_provider:
            logger.error("Cannot check alerts: MarketDataProvider is not available.")
            return []

        triggered_alerts = []
        for alert in self.alerts:
            if alert.triggered:
                continue

            try:
                stock_data = await self.data_provider.get_stock_data(alert.symbol)
                if not stock_data or stock_data.current_price is None: # Ensure price is available
                    logger.warning(f"Could not get stock data or price for alert on {alert.symbol}. Skipping check.")
                    continue

                alert_triggered_flag = False
                condition_price = alert.condition.get('price')
                condition_multiplier = alert.condition.get('multiplier', 2.0) # Default multiplier

                if alert.alert_type == 'price_above' and condition_price is not None:
                    if stock_data.current_price > condition_price:
                        alert_triggered_flag = True
                elif alert.alert_type == 'price_below' and condition_price is not None:
                    if stock_data.current_price < condition_price:
                        alert_triggered_flag = True
                elif alert.alert_type == 'volume_spike': # pragma: no cover
                    if stock_data.volume is not None and stock_data.avg_volume is not None and stock_data.avg_volume > 0:
                        if stock_data.volume > stock_data.avg_volume * condition_multiplier:
                            alert_triggered_flag = True
                    else:
                         logger.debug(f"Insufficient volume data to check volume spike for {alert.symbol}")
                # Add more alert types here

                if alert_triggered_flag:
                    alert.triggered = True
                    alert.triggered_at = datetime.now()
                    triggered_alerts.append(alert)
            except KeyError as ke: # If condition dict is missing expected keys
                logger.error(f"Alert {alert.alert_id} for {alert.symbol} has malformed condition (missing key {ke}). Skipping.", exc_info=True)
            except Exception as e: # Catch any other error during single alert check
                logger.exception(f"Error checking alert {alert.alert_id} for {alert.symbol}: {e}")

        return triggered_alerts


    def export_portfolio(self, portfolio_name: str, format: str = 'csv') -> str:
        """Export portfolio to file"""
        portfolio = self.get_portfolio(portfolio_name)
        if not portfolio:
            logger.error(f"Portfolio '{portfolio_name}' not found for export.")
            return "" # Or raise ValueError

        try:
            if format == 'csv':
                return self._export_csv(portfolio)
            elif format == 'json':
                return self._export_json(portfolio)
            else:
                logger.error(f"Unsupported export format: {format} for portfolio {portfolio_name}")
                raise ValueError(f"Unsupported export format: {format}. Must be 'csv' or 'json'.")
        except Exception as e:
            logger.exception(f"Error exporting portfolio {portfolio_name} to {format}: {e}")
            return "" # Or re-raise


    def _export_csv(self, portfolio: Portfolio) -> str:
        """Export portfolio as CSV"""
        # Using io.StringIO to build CSV in memory, then get value.
        import io
        output = io.StringIO()
        try:
            writer = csv.writer(output)
            writer.writerow(["Symbol", "Shares", "Cost Basis", "Current Value", "P&L", "P&L %", "Allocation %", "Purchase Date"])

            for p in portfolio.positions:
                writer.writerow([
                    p.symbol,
                    f"{p.shares:.4f}" if p.shares is not None else 'N/A', # Format numbers
                    f"{p.cost_basis:.2f}" if p.cost_basis is not None else 'N/A',
                    f"{p.current_value:.2f}" if p.current_value is not None else 'N/A',
                    f"{p.pnl:.2f}" if p.pnl is not None else 'N/A',
                    f"{p.pnl_percent:.2f}" if p.pnl_percent is not None and p.pnl_percent != float('inf') else 'N/A',
                    f"{p.allocation_percent:.2f}" if p.allocation_percent is not None else 'N/A',
                    p.purchase_date.strftime('%Y-%m-%d') if p.purchase_date else 'N/A'
                ])

            # Add summary rows if desired, or handle them in the main export function
            writer.writerow([]) # Blank line
            writer.writerow(["Summary Metric", "Value"])
            writer.writerow(["Total Value", f"{portfolio.total_value:.2f}" if portfolio.total_value is not None else 'N/A'])
            writer.writerow(["Cash Balance", f"{portfolio.cash_balance:.2f}" if portfolio.cash_balance is not None else 'N/A'])
            writer.writerow(["Total P&L", f"{portfolio.total_pnl:.2f}" if portfolio.total_pnl is not None else 'N/A'])
            writer.writerow(["Total P&L %", f"{portfolio.total_pnl_percent:.2f}" if portfolio.total_pnl_percent is not None else 'N/A'])

            return output.getvalue()
        finally:
            output.close()


    def _export_json(self, portfolio: Portfolio) -> str:
        """Export portfolio as JSON"""
        try:
            data_to_dump = {
                'name': portfolio.name,
                'total_value': portfolio.total_value if pd.notna(portfolio.total_value) else None,
                'cash_balance': portfolio.cash_balance if pd.notna(portfolio.cash_balance) else None,
                'total_pnl': portfolio.total_pnl if pd.notna(portfolio.total_pnl) else None,
                'total_pnl_percent': portfolio.total_pnl_percent if pd.notna(portfolio.total_pnl_percent) and portfolio.total_pnl_percent != float('inf') else None,
                'last_updated': portfolio.last_updated.isoformat() if portfolio.last_updated else None,
                'positions': [
                    {
                        'symbol': p.symbol,
                        'shares': p.shares if pd.notna(p.shares) else None,
                        'cost_basis': p.cost_basis if pd.notna(p.cost_basis) else None,
                        'current_value': p.current_value if pd.notna(p.current_value) else None,
                        'pnl': p.pnl if pd.notna(p.pnl) else None,
                        'pnl_percent': p.pnl_percent if pd.notna(p.pnl_percent) and p.pnl_percent != float('inf') else None,
                        'allocation_percent': p.allocation_percent if pd.notna(p.allocation_percent) else None,
                        'purchase_date': p.purchase_date.isoformat() if p.purchase_date else None
                    }
                    for p in portfolio.positions
                ]
            }
            return json.dumps(data_to_dump, indent=2)
        except TypeError as te: # Handle potential issues with data types not being JSON serializable
            logger.exception(f"TypeError during JSON export for portfolio {portfolio.name}: {te}")
            raise ValueError(f"Data in portfolio {portfolio.name} is not JSON serializable.") from te
        except Exception as e: # pragma: no cover
            logger.exception(f"Unexpected error during JSON export for portfolio {portfolio.name}: {e}")
            raise


    def import_portfolio(self, name: str, data: str, format: str = 'csv') -> Portfolio:
        """Import portfolio from file"""
        try:
            if format == 'csv':
                return self._import_csv(name, data)
            elif format == 'json':
                return self._import_json(name, data)
            else:
                logger.error(f"Unsupported import format: {format}")
                raise ValueError(f"Unsupported format: {format}. Must be 'csv' or 'json'.")
        except ValueError as ve: # Re-raise ValueErrors (e.g. from create_portfolio or format error)
            raise
        except (json.JSONDecodeError, csv.Error) as parse_error:
            logger.error(f"Error parsing {format} data for portfolio '{name}': {parse_error}", exc_info=True)
            raise ValueError(f"Invalid {format} format: {parse_error}") from parse_error
        except KeyError as ke:
            logger.error(f"Missing expected field '{ke}' in {format} data for portfolio '{name}'.", exc_info=True)
            raise ValueError(f"Missing field '{ke}' in {format} data.") from ke
        except Exception as e:
            logger.exception(f"Unexpected error importing portfolio '{name}' from {format}: {e}")
            raise RuntimeError(f"Failed to import portfolio: {e}") from e


    def _import_csv(self, name: str, data: str) -> Portfolio:
        """Import portfolio from CSV"""
        # create_portfolio already handles if name exists, can raise ValueError
        portfolio = self.create_portfolio(name)

        # Use io.StringIO to treat string data as a file for csv.DictReader
        data_io = io.StringIO(data.strip())
        reader = csv.DictReader(data_io)

        for i, row in enumerate(reader):
            # Check for summary rows or rows without essential data like 'Symbol'
            if 'Symbol' not in row or not row['Symbol'] or row['Symbol'].lower() == 'summary metric':
                logger.debug(f"Skipping CSV row {i+1} (summary or no symbol): {row}")
                continue
            try:
                # Validate and convert data types
                shares = float(row['Shares'])
                cost_basis = float(row['Cost Basis'])
                purchase_date_str = row.get('Purchase Date') # Optional column
                purchase_date = datetime.strptime(purchase_date_str, '%Y-%m-%d') if purchase_date_str else datetime.now()

                if shares <= 0 or cost_basis < 0: # Basic validation
                    logger.warning(f"Skipping row {i+1} with invalid shares/cost_basis: {row}")
                    continue

                position = PortfolioPosition(
                    symbol=str(row['Symbol']).upper(),
                    shares=shares,
                    cost_basis=cost_basis,
                    current_value=0,  # To be updated later by update_portfolio
                    purchase_date=purchase_date,
                    pnl=0, pnl_percent=0, allocation_percent=0 # Initial values
                )
                portfolio.positions.append(position)
            except KeyError as ke:
                raise ValueError(f"Missing column in CSV on row {i+1}: {ke}. Expected columns: Symbol, Shares, Cost Basis.") from ke
            except ValueError as ve:
                raise ValueError(f"Invalid data type in CSV on row {i+1}: {ve}. Row data: {row}") from ve
        return portfolio


    def _import_json(self, name: str, data: str) -> Portfolio:
        """Import portfolio from JSON"""
        json_data = json.loads(data) # Can raise json.JSONDecodeError

        # create_portfolio handles if name exists
        portfolio = self.create_portfolio(name, float(json_data.get('cash_balance', 0.0)))

        for i, pos_data in enumerate(json_data.get('positions', [])):
            try:
                shares = float(pos_data['shares'])
                cost_basis = float(pos_data['cost_basis'])
                if shares <= 0 or cost_basis < 0: # Basic validation
                    logger.warning(f"Skipping JSON position {i+1} with invalid shares/cost_basis: {pos_data}")
                    continue

                purchase_date_str = pos_data.get('purchase_date', datetime.now().isoformat())
                purchase_date = datetime.fromisoformat(purchase_date_str) if purchase_date_str else datetime.now()


                position = PortfolioPosition(
                    symbol=str(pos_data['symbol']).upper(),
                    shares=shares,
                    cost_basis=cost_basis,
                    current_value=0,  # To be updated
                    purchase_date=purchase_date,
                    pnl=0, pnl_percent=0, allocation_percent=0
                )
                portfolio.positions.append(position)
            except KeyError as ke:
                raise ValueError(f"Missing key '{ke}' in position data in JSON (position {i+1}).") from ke
            except (TypeError, ValueError) as ve:
                raise ValueError(f"Invalid data type for position {i+1} in JSON: {ve}. Data: {pos_data}") from ve
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
            logger.warning(f"Portfolio '{portfolio_name}' not found for tax calculation.")
            return {}

        # Ensure portfolio is updated for current P&L values
        # await self.update_portfolio(portfolio_name) # Consider if this should be forced here or assumed up-to-date

        short_term_gains = 0.0
        long_term_gains = 0.0
        one_year_ago = datetime.now() - timedelta(days=365.25) # Use 365.25 for better leap year handling

        try:
            for position in portfolio.positions:
                if position.pnl is not None and position.pnl > 0:  # Only consider gains
                    # Ensure purchase_date is valid datetime
                    if not isinstance(position.purchase_date, datetime): # pragma: no cover
                        logger.warning(f"Position {position.symbol} in {portfolio_name} has invalid purchase_date type. Skipping for tax calc.")
                        continue

                    if position.purchase_date > one_year_ago:
                        short_term_gains += position.pnl
                    else:
                        long_term_gains += position.pnl

            short_term_tax = short_term_gains * tax_rate_short
            long_term_tax = long_term_gains * tax_rate_long

            return {
                'short_term_gains': short_term_gains,
                'long_term_gains': long_term_gains,
                'short_term_tax': short_term_tax,
                'long_term_tax': long_term_tax,
                'total_estimated_tax': short_term_tax + long_term_tax # Renamed for clarity
            }
        except Exception as e: # pragma: no cover
            logger.exception(f"Error calculating tax implications for {portfolio_name}: {e}")
            return {'error': f"Could not calculate tax implications: {e}"}


class PortfolioOptimizer:
    """Portfolio optimization utilities"""

    @staticmethod
    async def optimize_portfolio(current_positions: List[PortfolioPosition],
                                 data_provider: MarketDataProvider,
                                 optimization_method: str = 'sharpe') -> Dict[str, float]:
        """Optimize portfolio allocation"""
        if not data_provider: # pragma: no cover
            logger.error("PortfolioOptimizer: MarketDataProvider is not available.")
            return {}
        if not current_positions:
            logger.debug("PortfolioOptimizer: No current positions provided for optimization.")
            return {}

        symbols = [p.symbol for p in current_positions if isinstance(p, PortfolioPosition)]
        if not symbols: # pragma: no cover
            logger.debug("PortfolioOptimizer: No valid symbols in current positions.")
            return {}

        returns_data = {}
        try:
            for symbol in symbols:
                try:
                    hist = await data_provider.get_historical_data(symbol, period="2y") # Longer period for better stats
                    if not hist.empty and 'Close' in hist.columns:
                        returns = hist['Close'].pct_change().dropna()
                        if not returns.empty:
                             returns_data[symbol] = returns
                except Exception as e_hist: # pragma: no cover
                    logger.error(f"Optimizer: Error fetching historical data for {symbol}: {e_hist}")

            if not returns_data or len(returns_data) < 1 : # Need at least one asset with returns
                logger.warning("Optimizer: Not enough historical returns data for optimization.")
                return {s: 100.0/len(symbols) if symbols else 0 for s in symbols} # Fallback to equal weight if possible

            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.fillna(method='ffill').dropna(how='any', axis=0) # Handle NaNs robustly

            if returns_df.empty or len(returns_df) < 2 or len(returns_df.columns) == 0: # Need data points and assets
                logger.warning("Optimizer: Returns DataFrame is empty or insufficient after processing for optimization.")
                return {s: 100.0/len(symbols) if symbols else 0 for s in symbols}


            expected_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252      # Annualized

            num_assets = len(expected_returns)
            if num_assets == 0: return {}


            if optimization_method == 'sharpe':
                weights = PortfolioOptimizer._maximize_sharpe(expected_returns, cov_matrix)
            elif optimization_method == 'min_variance':
                weights = PortfolioOptimizer._minimize_variance(cov_matrix)
            else: # Default to equal weight
                weights = np.array([1.0 / num_assets] * num_assets)

            # Ensure symbols used for allocation match those in expected_returns (which come from returns_df.columns)
            final_symbols = expected_returns.index.tolist()
            allocations = {symbol: weights[i] * 100.0 for i, symbol in enumerate(final_symbols)}
            return allocations

        except Exception as e: # pragma: no cover
            logger.exception(f"Error during portfolio optimization: {e}")
            # Fallback to equal weight among original symbols if optimization fails badly
            return {s: 100.0/len(symbols) if symbols else 0 for s in symbols}


    @staticmethod
    def _maximize_sharpe(expected_returns: pd.Series, cov_matrix: pd.DataFrame,
                         risk_free_rate: float = 0.02) -> np.ndarray:
        """Maximize Sharpe ratio (simplified implementation)"""
        num_assets = len(expected_returns)
        if num_assets == 0: return np.array([])

        # Placeholder: In a real scenario, use scipy.optimize.minimize
        # For this example, returning equal weights as a robust fallback.
        # This part requires a numerical optimization library (e.g., scipy)
        # which is a large dependency and complex to implement correctly here.
        logger.info("Sharpe ratio maximization is using simplified equal weight as placeholder.")
        try:
            # Example of how it might look with scipy (NOT EXECUTED, just for illustration)
            # def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
            #     p_ret = np.sum(expected_returns * weights)
            #     p_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            #     return -(p_ret - risk_free_rate) / p_vol if p_vol != 0 else -np.inf
            # constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1}) # Sum of weights = 1
            # bounds = tuple((0,1) for asset in range(num_assets)) # Weights between 0 and 1
            # initial_guess = np.array(num_assets * [1./num_assets,])
            # result = scipy.optimize.minimize(neg_sharpe_ratio, initial_guess,
            #                                  args=(expected_returns, cov_matrix, risk_free_rate),
            #                                  method='SLSQP', bounds=bounds, constraints=constraints)
            # if result.success: return result.x
            return np.array([1.0 / num_assets] * num_assets)
        except Exception as e: # pragma: no cover
            logger.error(f"Error in Sharpe maximization placeholder: {e}")
            return np.array([1.0 / num_assets] * num_assets)


    @staticmethod
    def _minimize_variance(cov_matrix: pd.DataFrame) -> np.ndarray:
        """Minimize portfolio variance (simplified)"""
        num_assets = len(cov_matrix)
        if num_assets == 0: return np.array([])

        logger.info("Minimum variance optimization is using simplified equal weight as placeholder.")
        try:
            # Placeholder: Similar to Sharpe, would use scipy.optimize.minimize
            # Objective function would be portfolio variance: w^T * Cov * w
            return np.array([1.0 / num_assets] * num_assets)
        except Exception as e: # pragma: no cover
            logger.error(f"Error in Min Variance placeholder: {e}")
            return np.array([1.0 / num_assets] * num_assets)


class DividendTracker:
    """Track and analyze dividends"""

    def __init__(self, data_provider: MarketDataProvider):
        if data_provider is None: # pragma: no cover
            logger.error("DividendTracker initialized with no data_provider.")
            # Allow initialization but expect errors if methods are called.
        self.data_provider = data_provider
        self.dividend_history: Dict[str, Any] = {} # symbol -> history data

    async def get_dividend_forecast(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Forecast dividend income"""
        if not self.data_provider: # pragma: no cover
            logger.error("DividendTracker: MarketDataProvider not available for forecast.")
            return {'monthly_forecast': [0.0]*12, 'annual_total': 0.0, 'yield_on_cost': 0.0, 'error': "Data provider missing."}
        if not portfolio or not isinstance(portfolio, Portfolio): # pragma: no cover
            logger.error("Invalid portfolio object provided to get_dividend_forecast.")
            return {'monthly_forecast': [0.0]*12, 'annual_total': 0.0, 'yield_on_cost': 0.0, 'error': "Invalid portfolio."}

        monthly_dividends = [0.0] * 12 # Ensure float
        annual_total = 0.0

        try:
            for position in portfolio.positions:
                try:
                    stock_data = await self.data_provider.get_stock_data(position.symbol)
                    if stock_data and stock_data.dividend_yield is not None and \
                       position.current_value is not None and pd.notna(stock_data.dividend_yield) and pd.notna(position.current_value):

                        annual_dividend_for_position = position.current_value * stock_data.dividend_yield
                        annual_total += annual_dividend_for_position

                        # Simplified: assume quarterly payments, spread from current month or typical ex-dividend months
                        # A more accurate forecast would use actual ex-dividend dates and payment schedules.
                        # This is a placeholder for such logic.
                        quarterly_payment = annual_dividend_for_position / 4.0
                        # Example: distribute based on typical ex-div months (e.g., Mar, Jun, Sep, Dec for many US stocks)
                        # This is highly simplified.
                        payment_months = [2, 5, 8, 11] # 0-indexed for March, June, etc.
                        for month_idx in payment_months:
                            monthly_dividends[month_idx] += quarterly_payment
                    else: # pragma: no cover
                        logger.debug(f"No dividend yield or current value for {position.symbol} in forecast.")
                except Exception as e_pos: # pragma: no cover
                    logger.error(f"Error processing position {position.symbol} for dividend forecast: {e_pos}")

            yield_on_cost_val = 0.0
            if portfolio.total_cost is not None and portfolio.total_cost > 0:
                yield_on_cost_val = (annual_total / portfolio.total_cost) * 100.0

            return {
                'monthly_forecast': monthly_dividends,
                'annual_total': annual_total,
                'yield_on_cost': yield_on_cost_val
            }
        except Exception as e: # pragma: no cover
            logger.exception(f"Error calculating dividend forecast for portfolio {portfolio.name}: {e}")
            return {'monthly_forecast': [0.0]*12, 'annual_total': 0.0, 'yield_on_cost': 0.0, 'error': str(e)}


    async def analyze_dividend_safety(self, symbol: str) -> Dict[str, Any]:
        """Analyze dividend safety for a stock"""
        default_result = {'safety_score': 0.0, 'analysis': 'Unable to analyze due to missing data or error.', 'factors': []}
        if not self.data_provider: # pragma: no cover
            logger.error(f"DividendSafety: MarketDataProvider not available for {symbol}.")
            return {**default_result, 'analysis': 'Data provider service unavailable.'}
        if not symbol: # pragma: no cover
            logger.warning("DividendSafety: No symbol provided.")
            return {**default_result, 'analysis': 'No symbol provided.'}

        try:
            # Gather data concurrently
            results = await asyncio.gather(
                self.data_provider.get_stock_data(symbol),
                self.data_provider.get_financial_statements(symbol),
                return_exceptions=True
            )
            stock_data, financials = results

            if isinstance(stock_data, Exception) or stock_data is None: # pragma: no cover
                logger.error(f"Failed to get stock data for {symbol} for dividend safety: {stock_data}")
                return {**default_result, 'analysis': f"Could not fetch stock data: {stock_data}"}
            if isinstance(financials, Exception) or financials is None: # pragma: no cover
                logger.error(f"Failed to get financials for {symbol} for dividend safety: {financials}")
                # Can proceed with partial analysis if financials are missing but stock_data is present
                financials = FinancialMetrics({},{},{},{},{},{}) # Use default empty


            safety_score = 50.0  # Base score
            factors = []

            # Payout ratio from EPS
            if stock_data.eps is not None and pd.notna(stock_data.eps) and stock_data.eps > 0 and \
               stock_data.dividend_yield is not None and pd.notna(stock_data.dividend_yield) and \
               stock_data.current_price is not None and pd.notna(stock_data.current_price):

                dividend_per_share = stock_data.current_price * stock_data.dividend_yield
                payout_ratio_eps = dividend_per_share / stock_data.eps
                factors.append(f"EPS Payout Ratio: {payout_ratio_eps:.2%}")
                if payout_ratio_eps < 0.10: safety_score += 5 # Very low might mean room to grow or just started
                elif payout_ratio_eps < 0.60: safety_score += 20
                elif payout_ratio_eps > 0.90: safety_score -= 30 # High is risky
                else: safety_score += 5 # Moderate is okay
            else: factors.append("Payout ratio (EPS) not determinable.")


            # Payout ratio from Free Cash Flow (FCF) - more robust
            # This requires FCF per share, which might need to be calculated or fetched.
            # Assuming 'cash_flow' dict in financials and 'sharesOutstanding' in stock_data.info (via ratios)
            # This is a simplified placeholder, real FCF payout needs careful calculation.
            if financials.cash_flow and stock_data.market_cap and stock_data.current_price and stock_data.current_price > 0:
                # Try to get recent FCF from cash_flow statement
                # This is highly dependent on yfinance data structure.
                # Placeholder: if 'Free Cash Flow' is an item and we have shares outstanding
                # For now, this part is too complex to implement robustly without knowing exact yf output.
                pass


            if stock_data.debt_to_equity is not None and pd.notna(stock_data.debt_to_equity):
                factors.append(f"Debt/Equity: {stock_data.debt_to_equity:.2f}")
                if stock_data.debt_to_equity < 0.5: safety_score += 15
                elif stock_data.debt_to_equity > 1.5: safety_score -= 15 # High D/E can strain cash for divs

            if stock_data.revenue_growth is not None and pd.notna(stock_data.revenue_growth) and stock_data.revenue_growth > 0.05:
                safety_score += 10; factors.append("Positive revenue growth.")
            elif stock_data.revenue_growth is not None and pd.notna(stock_data.revenue_growth) and stock_data.revenue_growth < -0.05:
                safety_score -= 10; factors.append("Negative revenue growth.")

            # Dividend growth history (placeholder, would need historical dividend data)
            # if has_consistent_dividend_growth_5yr: safety_score += 15; factors.append("Consistent 5yr dividend growth.")

            safety_score = max(0.0, min(100.0, safety_score)) # Clamp score

            return {
                'safety_score': safety_score,
                'factors': factors,
                'recommendation': self._get_dividend_recommendation(safety_score)
            }
        except Exception as e: # pragma: no cover
            logger.exception(f"Error analyzing dividend safety for {symbol}: {e}")
            return {**default_result, 'analysis': f"Unexpected error: {e}"}


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
            logger.debug("PortfolioComparison: No portfolios provided for comparison.")
            return pd.DataFrame()

        comparison_data = []
        try:
            for portfolio in portfolios:
                if not isinstance(portfolio, Portfolio): # pragma: no cover
                    logger.warning(f"Skipping invalid item in portfolio list for comparison: {type(portfolio)}")
                    continue

                cash_alloc_pct = 0.0
                if portfolio.total_value is not None and portfolio.total_value > 0 and portfolio.cash_balance is not None:
                    cash_alloc_pct = (portfolio.cash_balance / portfolio.total_value) * 100.0

                comparison_data.append({
                    'Name': portfolio.name,
                    'Total Value': portfolio.total_value if pd.notna(portfolio.total_value) else 0.0,
                    'P&L': portfolio.total_pnl if pd.notna(portfolio.total_pnl) else 0.0,
                    'P&L %': portfolio.total_pnl_percent if pd.notna(portfolio.total_pnl_percent) else 0.0,
                    'Positions': len(portfolio.positions),
                    'Cash %': cash_alloc_pct
                })
            return pd.DataFrame(comparison_data)
        except Exception as e: # pragma: no cover
            logger.exception(f"Error creating portfolio comparison table: {e}")
            return pd.DataFrame() # Return empty on error


    @staticmethod
    async def compare_performance(portfolios: List[Portfolio],
                                  data_provider: MarketDataProvider,
                                  benchmark: str = "SPY") -> Dict[str, Any]:
        """Compare portfolio performance against benchmark"""
        if not data_provider: # pragma: no cover
            logger.error("PortfolioComparison: MarketDataProvider not available for performance comparison.")
            return {'error': "Data provider missing."}
        if not portfolios: # pragma: no cover
            logger.debug("PortfolioComparison: No portfolios provided for performance comparison.")
            return {'benchmark': benchmark, 'benchmark_return': None, 'portfolios': {}, 'error': "No portfolios."}

        results: Dict[str, Any] = { 'benchmark': benchmark, 'benchmark_return': None, 'portfolios': {} }

        try:
            benchmark_hist = await data_provider.get_historical_data(benchmark, period="1y")
            if benchmark_hist.empty or 'Close' not in benchmark_hist.columns or len(benchmark_hist['Close']) < 2: # pragma: no cover
                logger.warning(f"Could not get valid historical data for benchmark {benchmark}.")
            else:
                first_close = benchmark_hist['Close'].iloc[0]
                last_close = benchmark_hist['Close'].iloc[-1]
                if pd.notna(first_close) and pd.notna(last_close) and first_close != 0:
                    results['benchmark_return'] = ((last_close / first_close) - 1) * 100.0
                else: # pragma: no cover
                     logger.warning(f"Invalid close prices for benchmark {benchmark} for return calculation.")


            for portfolio in portfolios:
                if not isinstance(portfolio, Portfolio): continue # pragma: no cover

                pnl_pct = portfolio.total_pnl_percent if pd.notna(portfolio.total_pnl_percent) else 0.0
                excess_ret = None
                if results['benchmark_return'] is not None and pd.notna(results['benchmark_return']):
                    excess_ret = pnl_pct - results['benchmark_return']

                results['portfolios'][portfolio.name] = {
                    'return': pnl_pct,
                    'excess_return': excess_ret,
                    'outperformed': excess_ret > 0 if excess_ret is not None else None
                }
            return results
        except Exception as e: # pragma: no cover
            logger.exception(f"Error comparing portfolio performance: {e}")
            results['error'] = str(e)
            return results
