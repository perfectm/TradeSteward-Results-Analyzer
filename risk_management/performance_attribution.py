"""
Performance Attribution and Backtesting Framework - Phase 3 Implementation
Comprehensive performance analysis and strategy backtesting for options trading
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import itertools

# Import our risk management modules
from data_loader import TradeStewardDataLoader
from advanced_risk_measures import AdvancedRiskCalculator
from options_factor_models import OptionsFactorModel
from portfolio_optimization import PortfolioOptimizer
from monte_carlo_simulation import MonteCarloSimulator, SimulationParameters

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AttributionMethod(Enum):
    """Performance attribution methods"""
    BRINSON_HOOD_BEEBOWER = "brinson_hood_beebower"
    BRINSON_FACHLER = "brinson_fachler"
    FACTOR_BASED = "factor_based"
    GEOMETRIC_LINKING = "geometric_linking"
    ARITHMETIC_LINKING = "arithmetic_linking"

class BacktestMetric(Enum):
    """Backtesting evaluation metrics"""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    VOLATILITY = "volatility"
    SHARPE_RATIO = "sharpe_ratio"
    CALMAR_RATIO = "calmar_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    VAR_95 = "var_95"
    CVAR_95 = "cvar_95"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    INFORMATION_RATIO = "information_ratio"

@dataclass
class PerformanceAttribution:
    """Performance attribution results"""
    period_start: date
    period_end: date
    strategy_name: str
    total_return: float
    benchmark_return: float
    active_return: float
    attribution_breakdown: Dict[str, float]
    factor_contributions: Dict[str, float]
    interaction_effects: Dict[str, float]
    risk_metrics: Dict[str, float]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class BacktestResult:
    """Backtesting result structure"""
    strategy_name: str
    backtest_period: Tuple[date, date]
    performance_metrics: Dict[BacktestMetric, float]
    returns_series: pd.Series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    trade_statistics: Dict[str, Any]
    risk_analytics: Dict[str, float]
    benchmark_comparison: Optional[Dict[str, float]]
    factor_exposures: Optional[Dict[str, float]]
    optimization_results: Optional[Dict[str, Any]]

@dataclass
class BacktestConfig:
    """Backtesting configuration"""
    start_date: date
    end_date: date
    initial_capital: float
    rebalance_frequency: str  # 'daily', 'weekly', 'monthly', 'quarterly'
    transaction_costs: float  # basis points
    risk_free_rate: float
    benchmark_symbol: Optional[str]
    optimization_objective: str  # 'returns', 'sharpe', 'risk_parity'
    max_leverage: float
    position_limits: Dict[str, float]
    exclude_strategies: List[str]

class PerformanceAttributor:
    """Performance attribution analysis engine"""
    
    def __init__(self, returns_data: pd.DataFrame, factor_data: Optional[pd.DataFrame] = None):
        """
        Initialize performance attributor
        
        Args:
            returns_data: DataFrame with strategy returns
            factor_data: Optional factor returns for factor-based attribution
        """
        self.returns_data = returns_data
        self.factor_data = factor_data
        self.logger = logging.getLogger(__name__ + '.PerformanceAttributor')
    
    def calculate_brinson_attribution(self, 
                                    portfolio_weights: pd.DataFrame,
                                    benchmark_weights: pd.DataFrame,
                                    method: AttributionMethod = AttributionMethod.BRINSON_HOOD_BEEBOWER) -> PerformanceAttribution:
        """
        Calculate Brinson-style performance attribution
        
        Args:
            portfolio_weights: Portfolio weights over time
            benchmark_weights: Benchmark weights over time
            method: Attribution method to use
        
        Returns:
            PerformanceAttribution object
        """
        
        try:
            # Align data
            common_index = portfolio_weights.index.intersection(benchmark_weights.index)
            portfolio_weights = portfolio_weights.loc[common_index]
            benchmark_weights = benchmark_weights.loc[common_index]
            returns_data = self.returns_data.loc[common_index]
            
            # Calculate portfolio and benchmark returns
            portfolio_returns = (portfolio_weights * returns_data).sum(axis=1)
            benchmark_returns = (benchmark_weights * returns_data).sum(axis=1)
            
            # Calculate total returns
            total_portfolio_return = (1 + portfolio_returns).prod() - 1
            total_benchmark_return = (1 + benchmark_returns).prod() - 1
            active_return = total_portfolio_return - total_benchmark_return
            
            # Attribution components
            if method == AttributionMethod.BRINSON_HOOD_BEEBOWER:
                attribution = self._brinson_hood_beebower_attribution(
                    portfolio_weights, benchmark_weights, returns_data
                )
            elif method == AttributionMethod.BRINSON_FACHLER:
                attribution = self._brinson_fachler_attribution(
                    portfolio_weights, benchmark_weights, returns_data
                )
            else:
                attribution = self._factor_based_attribution(
                    portfolio_weights, benchmark_weights, returns_data
                )
            
            # Calculate risk metrics
            risk_metrics = self._calculate_attribution_risk_metrics(
                portfolio_returns, benchmark_returns
            )
            
            result = PerformanceAttribution(
                period_start=common_index[0].date(),
                period_end=common_index[-1].date(),
                strategy_name="Portfolio",
                total_return=total_portfolio_return,
                benchmark_return=total_benchmark_return,
                active_return=active_return,
                attribution_breakdown=attribution['main_effects'],
                factor_contributions=attribution.get('factor_contributions', {}),
                interaction_effects=attribution.get('interaction_effects', {}),
                risk_metrics=risk_metrics
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in Brinson attribution: {str(e)}")
            raise
    
    def _brinson_hood_beebower_attribution(self, 
                                         portfolio_weights: pd.DataFrame,
                                         benchmark_weights: pd.DataFrame,
                                         returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Brinson-Hood-Beebower attribution model"""
        
        # Asset allocation effect: (wp - wb) * rb
        weight_diff = portfolio_weights - benchmark_weights
        allocation_effect = (weight_diff * returns_data).sum(axis=1)
        
        # Security selection effect: wb * (rp - rb)
        # Approximated as benchmark weights times excess returns
        avg_benchmark_weights = benchmark_weights.mean()
        avg_returns_diff = returns_data.mean() - returns_data.mean()  # This would need actual security-level data
        selection_effect = (avg_benchmark_weights * avg_returns_diff).sum()
        
        # Interaction effect: (wp - wb) * (rp - rb)
        interaction_effect = (weight_diff * avg_returns_diff).sum()
        
        return {
            'main_effects': {
                'allocation_effect': allocation_effect.sum(),
                'selection_effect': selection_effect,
                'interaction_effect': interaction_effect
            }
        }
    
    def _brinson_fachler_attribution(self, 
                                   portfolio_weights: pd.DataFrame,
                                   benchmark_weights: pd.DataFrame,
                                   returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Brinson-Fachler attribution model (removes interaction effect)"""
        
        bhb_attribution = self._brinson_hood_beebower_attribution(
            portfolio_weights, benchmark_weights, returns_data
        )
        
        # In Brinson-Fachler, interaction effect is added to selection effect
        main_effects = bhb_attribution['main_effects']
        main_effects['selection_effect'] += main_effects['interaction_effect']
        main_effects['interaction_effect'] = 0
        
        return {'main_effects': main_effects}
    
    def _factor_based_attribution(self, 
                                portfolio_weights: pd.DataFrame,
                                benchmark_weights: pd.DataFrame,
                                returns_data: pd.DataFrame) -> Dict[str, Any]:
        """Factor-based performance attribution"""
        
        if self.factor_data is None:
            self.logger.warning("No factor data available for factor-based attribution")
            return self._brinson_hood_beebower_attribution(portfolio_weights, benchmark_weights, returns_data)
        
        # Align factor data
        common_index = returns_data.index.intersection(self.factor_data.index)
        factor_data_aligned = self.factor_data.loc[common_index]
        returns_aligned = returns_data.loc[common_index]
        
        # Calculate portfolio and benchmark returns
        portfolio_returns = (portfolio_weights.loc[common_index] * returns_aligned).sum(axis=1)
        benchmark_returns = (benchmark_weights.loc[common_index] * returns_aligned).sum(axis=1)
        active_returns = portfolio_returns - benchmark_returns
        
        # Regress active returns on factors
        X = factor_data_aligned.values
        y = active_returns.values
        
        if X.shape[0] > X.shape[1]:  # Ensure we have enough observations
            reg = LinearRegression().fit(X, y)
            factor_loadings = reg.coef_
            
            # Calculate factor contributions
            factor_contributions = {}
            for i, factor_name in enumerate(factor_data_aligned.columns):
                factor_return = factor_data_aligned[factor_name].mean()
                contribution = factor_loadings[i] * factor_return
                factor_contributions[factor_name] = contribution
            
            # Residual (security selection)
            predicted_returns = reg.predict(X)
            residual_contribution = (active_returns - predicted_returns).mean()
            
            return {
                'main_effects': {
                    'factor_attribution': sum(factor_contributions.values()),
                    'security_selection': residual_contribution
                },
                'factor_contributions': factor_contributions
            }
        else:
            self.logger.warning("Insufficient data for factor-based attribution")
            return self._brinson_hood_beebower_attribution(portfolio_weights, benchmark_weights, returns_data)
    
    def _calculate_attribution_risk_metrics(self, 
                                          portfolio_returns: pd.Series,
                                          benchmark_returns: pd.Series) -> Dict[str, float]:
        """Calculate risk metrics for attribution analysis"""
        
        active_returns = portfolio_returns - benchmark_returns
        
        return {
            'tracking_error': active_returns.std() * np.sqrt(252),
            'information_ratio': active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0,
            'active_return_volatility': active_returns.std(),
            'hit_rate': (active_returns > 0).mean(),
            'up_capture': portfolio_returns[benchmark_returns > 0].mean() / benchmark_returns[benchmark_returns > 0].mean() if (benchmark_returns > 0).any() else 0,
            'down_capture': portfolio_returns[benchmark_returns < 0].mean() / benchmark_returns[benchmark_returns < 0].mean() if (benchmark_returns < 0).any() else 0
        }

class StrategyBacktester:
    """Comprehensive strategy backtesting engine"""
    
    def __init__(self, data_loader: TradeStewardDataLoader):
        self.data_loader = data_loader
        self.logger = logging.getLogger(__name__ + '.StrategyBacktester')
    
    def backtest_strategy(self, 
                         strategy_name: str,
                         config: BacktestConfig,
                         optimization_method: Optional[str] = None) -> BacktestResult:
        """
        Backtest a specific strategy
        
        Args:
            strategy_name: Name of strategy to backtest
            config: Backtesting configuration
            optimization_method: Optional portfolio optimization method
        
        Returns:
            BacktestResult object
        """
        
        try:
            # Load strategy data
            trades_df = self.data_loader.load_user_trades()
            enhanced_df = self.data_loader.parse_raw_data_json(trades_df)
            clean_df = self.data_loader.clean_and_validate_data(enhanced_df)
            
            # Filter by strategy and date range
            strategy_data = clean_df[
                (clean_df['strategy'] == strategy_name) &
                (clean_df['close_date'].dt.date >= config.start_date) &
                (clean_df['close_date'].dt.date <= config.end_date)
            ].copy()
            
            if strategy_data.empty:
                raise ValueError(f"No data found for strategy {strategy_name} in specified period")
            
            # Create daily returns series
            daily_returns = self._create_daily_returns_series(strategy_data, config)
            
            # Apply optimization if specified
            if optimization_method:
                daily_returns = self._apply_optimization(daily_returns, optimization_method, config)
            
            # Calculate equity curve
            equity_curve = self._calculate_equity_curve(daily_returns, config.initial_capital)
            
            # Calculate drawdown series
            drawdown_series = self._calculate_drawdown_series(equity_curve)
            
            # Calculate performance metrics
            performance_metrics = self._calculate_performance_metrics(daily_returns, config.risk_free_rate)
            
            # Calculate trade statistics
            trade_statistics = self._calculate_trade_statistics(strategy_data)
            
            # Calculate risk analytics
            risk_analytics = self._calculate_risk_analytics(daily_returns, drawdown_series)
            
            # Benchmark comparison (if specified)
            benchmark_comparison = None
            if config.benchmark_symbol:
                benchmark_comparison = self._calculate_benchmark_comparison(
                    daily_returns, config.benchmark_symbol, config
                )
            
            # Factor exposures
            factor_exposures = self._calculate_factor_exposures(daily_returns)
            
            # Optimization results
            optimization_results = None
            if optimization_method:
                optimization_results = self._run_optimization_analysis(strategy_data, optimization_method)
            
            result = BacktestResult(
                strategy_name=strategy_name,
                backtest_period=(config.start_date, config.end_date),
                performance_metrics=performance_metrics,
                returns_series=daily_returns,
                equity_curve=equity_curve,
                drawdown_series=drawdown_series,
                trade_statistics=trade_statistics,
                risk_analytics=risk_analytics,
                benchmark_comparison=benchmark_comparison,
                factor_exposures=factor_exposures,
                optimization_results=optimization_results
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error backtesting strategy {strategy_name}: {str(e)}")
            raise
    
    def backtest_multiple_strategies(self, 
                                   strategy_names: List[str],
                                   config: BacktestConfig,
                                   parallel: bool = True) -> Dict[str, BacktestResult]:
        """
        Backtest multiple strategies
        
        Args:
            strategy_names: List of strategy names to backtest
            config: Backtesting configuration
            parallel: Whether to run backtests in parallel
        
        Returns:
            Dictionary of strategy names to BacktestResult objects
        """
        
        results = {}
        
        if parallel and len(strategy_names) > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=min(len(strategy_names), mp.cpu_count())) as executor:
                future_to_strategy = {
                    executor.submit(self.backtest_strategy, strategy, config): strategy 
                    for strategy in strategy_names
                }
                
                for future in as_completed(future_to_strategy):
                    strategy = future_to_strategy[future]
                    try:
                        result = future.result()
                        results[strategy] = result
                        self.logger.info(f"Completed backtest for {strategy}")
                    except Exception as e:
                        self.logger.error(f"Error backtesting {strategy}: {str(e)}")
        else:
            # Sequential execution
            for strategy in strategy_names:
                try:
                    result = self.backtest_strategy(strategy, config)
                    results[strategy] = result
                    self.logger.info(f"Completed backtest for {strategy}")
                except Exception as e:
                    self.logger.error(f"Error backtesting {strategy}: {str(e)}")
        
        return results
    
    def _create_daily_returns_series(self, strategy_data: pd.DataFrame, config: BacktestConfig) -> pd.Series:
        """Create daily returns series from trade data"""
        
        # Group trades by day and sum P&L
        strategy_data['trade_date'] = strategy_data['close_date'].dt.date
        daily_pnl = strategy_data.groupby('trade_date')['net_pnl'].sum()
        
        # Create complete date range
        date_range = pd.date_range(config.start_date, config.end_date, freq='D')
        daily_pnl = daily_pnl.reindex(date_range, fill_value=0)
        
        # Convert to returns (assuming initial capital)
        daily_returns = daily_pnl / config.initial_capital
        
        # Apply transaction costs
        trading_days = (daily_pnl != 0).sum()
        if trading_days > 0:
            avg_daily_cost = config.transaction_costs / 10000  # Convert bps to decimal
            daily_returns = daily_returns - avg_daily_cost * (daily_pnl != 0)
        
        return daily_returns
    
    def _apply_optimization(self, 
                          daily_returns: pd.Series,
                          optimization_method: str,
                          config: BacktestConfig) -> pd.Series:
        """Apply portfolio optimization to returns"""
        
        # For single strategy, optimization might involve position sizing
        # This is a simplified implementation
        
        if optimization_method == 'kelly':
            # Kelly criterion for position sizing
            win_rate = (daily_returns > 0).mean()
            avg_win = daily_returns[daily_returns > 0].mean()
            avg_loss = daily_returns[daily_returns < 0].mean()
            
            if avg_loss != 0:
                kelly_fraction = win_rate - ((1 - win_rate) * avg_win / abs(avg_loss))
                kelly_fraction = max(0, min(kelly_fraction, config.max_leverage))
                daily_returns = daily_returns * kelly_fraction
        
        elif optimization_method == 'risk_parity':
            # Risk parity position sizing
            lookback_window = 20
            rolling_vol = daily_returns.rolling(lookback_window).std()
            target_vol = daily_returns.std()
            
            position_size = target_vol / rolling_vol.fillna(target_vol)
            position_size = position_size.clip(0, config.max_leverage)
            daily_returns = daily_returns * position_size
        
        return daily_returns
    
    def _calculate_equity_curve(self, daily_returns: pd.Series, initial_capital: float) -> pd.Series:
        """Calculate equity curve from daily returns"""
        return initial_capital * (1 + daily_returns).cumprod()
    
    def _calculate_drawdown_series(self, equity_curve: pd.Series) -> pd.Series:
        """Calculate drawdown series"""
        running_max = equity_curve.expanding().max()
        return (equity_curve - running_max) / running_max
    
    def _calculate_performance_metrics(self, daily_returns: pd.Series, risk_free_rate: float) -> Dict[BacktestMetric, float]:
        """Calculate comprehensive performance metrics"""
        
        metrics = {}
        
        # Basic return metrics
        total_return = (1 + daily_returns).prod() - 1
        n_years = len(daily_returns) / 252
        annualized_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else 0
        
        metrics[BacktestMetric.TOTAL_RETURN] = total_return
        metrics[BacktestMetric.ANNUALIZED_RETURN] = annualized_return
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)
        metrics[BacktestMetric.VOLATILITY] = volatility
        
        # Risk-adjusted metrics
        excess_returns = daily_returns - risk_free_rate / 252
        sharpe_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        metrics[BacktestMetric.SHARPE_RATIO] = sharpe_ratio
        
        # Drawdown metrics
        equity_curve = (1 + daily_returns).cumprod()
        running_max = equity_curve.expanding().max()
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = drawdown.min()
        metrics[BacktestMetric.MAX_DRAWDOWN] = max_drawdown
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
        metrics[BacktestMetric.CALMAR_RATIO] = calmar_ratio
        
        # Sortino ratio
        downside_returns = daily_returns[daily_returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = annualized_return / downside_std if downside_std > 0 else 0
        metrics[BacktestMetric.SORTINO_RATIO] = sortino_ratio
        
        # VaR and CVaR
        var_95 = np.percentile(daily_returns, 5)
        cvar_95 = daily_returns[daily_returns <= var_95].mean()
        metrics[BacktestMetric.VAR_95] = var_95
        metrics[BacktestMetric.CVAR_95] = cvar_95
        
        # Win rate and profit factor
        winning_days = daily_returns > 0
        win_rate = winning_days.mean()
        metrics[BacktestMetric.WIN_RATE] = win_rate
        
        gross_profit = daily_returns[daily_returns > 0].sum()
        gross_loss = abs(daily_returns[daily_returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
        metrics[BacktestMetric.PROFIT_FACTOR] = profit_factor
        
        return metrics
    
    def _calculate_trade_statistics(self, strategy_data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate trade-level statistics"""
        
        stats = {}
        
        # Basic trade counts
        stats['total_trades'] = len(strategy_data)
        stats['winning_trades'] = (strategy_data['net_pnl'] > 0).sum()
        stats['losing_trades'] = (strategy_data['net_pnl'] < 0).sum()
        stats['breakeven_trades'] = (strategy_data['net_pnl'] == 0).sum()
        
        # Win/loss statistics
        if stats['total_trades'] > 0:
            stats['win_rate'] = stats['winning_trades'] / stats['total_trades']
            
            if stats['winning_trades'] > 0:
                stats['avg_win'] = strategy_data[strategy_data['net_pnl'] > 0]['net_pnl'].mean()
                stats['largest_win'] = strategy_data['net_pnl'].max()
            
            if stats['losing_trades'] > 0:
                stats['avg_loss'] = strategy_data[strategy_data['net_pnl'] < 0]['net_pnl'].mean()
                stats['largest_loss'] = strategy_data['net_pnl'].min()
                
                if stats['avg_loss'] != 0:
                    stats['avg_win_loss_ratio'] = stats.get('avg_win', 0) / abs(stats['avg_loss'])
        
        # Trade duration statistics
        if 'open_date' in strategy_data.columns and 'close_date' in strategy_data.columns:
            trade_durations = (strategy_data['close_date'] - strategy_data['open_date']).dt.days
            stats['avg_trade_duration'] = trade_durations.mean()
            stats['max_trade_duration'] = trade_durations.max()
            stats['min_trade_duration'] = trade_durations.min()
        
        # P&L statistics
        stats['total_pnl'] = strategy_data['net_pnl'].sum()
        stats['avg_pnl_per_trade'] = strategy_data['net_pnl'].mean()
        stats['pnl_std'] = strategy_data['net_pnl'].std()
        
        return stats
    
    def _calculate_risk_analytics(self, daily_returns: pd.Series, drawdown_series: pd.Series) -> Dict[str, float]:
        """Calculate advanced risk analytics"""
        
        analytics = {}
        
        # Volatility metrics
        analytics['daily_volatility'] = daily_returns.std()
        analytics['annualized_volatility'] = daily_returns.std() * np.sqrt(252)
        
        # Skewness and kurtosis
        analytics['skewness'] = daily_returns.skew()
        analytics['kurtosis'] = daily_returns.kurtosis()
        
        # Tail risk measures
        analytics['var_1'] = np.percentile(daily_returns, 1)
        analytics['var_5'] = np.percentile(daily_returns, 5)
        analytics['cvar_5'] = daily_returns[daily_returns <= analytics['var_5']].mean()
        
        # Maximum drawdown analytics
        analytics['max_drawdown'] = drawdown_series.min()
        analytics['max_drawdown_duration'] = self._calculate_max_drawdown_duration(drawdown_series)
        analytics['recovery_time'] = self._calculate_recovery_time(drawdown_series)
        
        # Rolling metrics
        rolling_sharpe = self._calculate_rolling_sharpe(daily_returns, window=252)
        analytics['sharpe_stability'] = rolling_sharpe.std() if len(rolling_sharpe) > 1 else 0
        
        return analytics
    
    def _calculate_max_drawdown_duration(self, drawdown_series: pd.Series) -> int:
        """Calculate maximum drawdown duration in days"""
        
        is_drawdown = drawdown_series < 0
        drawdown_periods = (is_drawdown != is_drawdown.shift()).cumsum()
        
        if is_drawdown.any():
            duration_by_period = is_drawdown.groupby(drawdown_periods).sum()
            return duration_by_period[duration_by_period > 0].max()
        
        return 0
    
    def _calculate_recovery_time(self, drawdown_series: pd.Series) -> int:
        """Calculate average recovery time from drawdowns"""
        
        recovery_times = []
        in_drawdown = False
        drawdown_start = None
        
        for i, dd in enumerate(drawdown_series):
            if dd < 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                drawdown_start = i
            elif dd == 0 and in_drawdown:
                # Recovery
                recovery_time = i - drawdown_start
                recovery_times.append(recovery_time)
                in_drawdown = False
        
        return np.mean(recovery_times) if recovery_times else 0
    
    def _calculate_rolling_sharpe(self, daily_returns: pd.Series, window: int = 252) -> pd.Series:
        """Calculate rolling Sharpe ratio"""
        
        if len(daily_returns) < window:
            return pd.Series([])
        
        rolling_mean = daily_returns.rolling(window).mean()
        rolling_std = daily_returns.rolling(window).std()
        
        return rolling_mean / rolling_std * np.sqrt(252)
    
    def _calculate_benchmark_comparison(self, 
                                     daily_returns: pd.Series,
                                     benchmark_symbol: str,
                                     config: BacktestConfig) -> Dict[str, float]:
        """Calculate benchmark comparison metrics"""
        
        # This would typically fetch actual benchmark data
        # For demonstration, we'll create synthetic benchmark returns
        np.random.seed(42)
        benchmark_returns = pd.Series(
            np.random.normal(0.0005, 0.012, len(daily_returns)),
            index=daily_returns.index
        )
        
        # Calculate comparison metrics
        strategy_total = (1 + daily_returns).prod() - 1
        benchmark_total = (1 + benchmark_returns).prod() - 1
        
        excess_returns = daily_returns - benchmark_returns
        tracking_error = excess_returns.std() * np.sqrt(252)
        information_ratio = excess_returns.mean() / excess_returns.std() * np.sqrt(252) if excess_returns.std() > 0 else 0
        
        # Beta calculation
        covariance = np.cov(daily_returns, benchmark_returns)[0, 1]
        benchmark_variance = np.var(benchmark_returns)
        beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
        
        return {
            'strategy_return': strategy_total,
            'benchmark_return': benchmark_total,
            'excess_return': strategy_total - benchmark_total,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
            'beta': beta,
            'correlation': np.corrcoef(daily_returns, benchmark_returns)[0, 1]
        }
    
    def _calculate_factor_exposures(self, daily_returns: pd.Series) -> Dict[str, float]:
        """Calculate factor exposures (simplified)"""
        
        # This would typically use factor models
        # For demonstration, we'll calculate basic exposures
        
        return {
            'momentum_factor': daily_returns.rolling(20).mean().iloc[-1] if len(daily_returns) >= 20 else 0,
            'volatility_factor': daily_returns.rolling(20).std().iloc[-1] if len(daily_returns) >= 20 else 0,
            'trend_factor': (daily_returns > 0).rolling(20).mean().iloc[-1] if len(daily_returns) >= 20 else 0
        }
    
    def _run_optimization_analysis(self, strategy_data: pd.DataFrame, optimization_method: str) -> Dict[str, Any]:
        """Run optimization analysis on strategy"""
        
        # Create returns matrix for the strategy
        strategy_data['trade_date'] = strategy_data['close_date'].dt.date
        daily_pnl = strategy_data.groupby('trade_date')['net_pnl'].sum()
        
        # This would include more sophisticated optimization
        # For now, return basic optimization metrics
        
        return {
            'optimization_method': optimization_method,
            'optimal_position_size': 1.0,  # Simplified
            'risk_budget': daily_pnl.std(),
            'expected_return': daily_pnl.mean()
        }

class BacktestAnalyzer:
    """Analyze and compare backtest results"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__ + '.BacktestAnalyzer')
    
    def compare_strategies(self, backtest_results: Dict[str, BacktestResult]) -> pd.DataFrame:
        """Compare multiple strategy backtest results"""
        
        comparison_data = []
        
        for strategy_name, result in backtest_results.items():
            metrics = result.performance_metrics
            
            row = {
                'Strategy': strategy_name,
                'Total Return': f"{metrics[BacktestMetric.TOTAL_RETURN]:.2%}",
                'Annualized Return': f"{metrics[BacktestMetric.ANNUALIZED_RETURN]:.2%}",
                'Volatility': f"{metrics[BacktestMetric.VOLATILITY]:.2%}",
                'Sharpe Ratio': f"{metrics[BacktestMetric.SHARPE_RATIO]:.3f}",
                'Max Drawdown': f"{metrics[BacktestMetric.MAX_DRAWDOWN]:.2%}",
                'Calmar Ratio': f"{metrics[BacktestMetric.CALMAR_RATIO]:.3f}",
                'Win Rate': f"{metrics[BacktestMetric.WIN_RATE]:.2%}",
                'Profit Factor': f"{metrics[BacktestMetric.PROFIT_FACTOR]:.2f}",
                'Total Trades': result.trade_statistics['total_trades']
            }
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def plot_strategy_comparison(self, 
                               backtest_results: Dict[str, BacktestResult],
                               save_path: Optional[str] = None):
        """Plot strategy comparison charts"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Strategy Backtest Comparison', fontsize=16, fontweight='bold')
        
        # Equity curves
        ax1 = axes[0, 0]
        for strategy_name, result in backtest_results.items():
            ax1.plot(result.equity_curve.index, result.equity_curve.values, 
                    label=strategy_name, linewidth=2)
        ax1.set_title('Equity Curves')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown comparison
        ax2 = axes[0, 1]
        for strategy_name, result in backtest_results.items():
            ax2.plot(result.drawdown_series.index, result.drawdown_series.values * 100,
                    label=strategy_name, linewidth=2)
        ax2.set_title('Drawdown Comparison')
        ax2.set_ylabel('Drawdown (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Rolling returns
        ax3 = axes[1, 0]
        for strategy_name, result in backtest_results.items():
            rolling_returns = result.returns_series.rolling(252).sum()
            ax3.plot(rolling_returns.index, rolling_returns.values * 100,
                    label=strategy_name, linewidth=2)
        ax3.set_title('Rolling 252-Day Returns')
        ax3.set_ylabel('Return (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Risk-return scatter
        ax4 = axes[1, 1]
        for strategy_name, result in backtest_results.items():
            annual_return = result.performance_metrics[BacktestMetric.ANNUALIZED_RETURN]
            volatility = result.performance_metrics[BacktestMetric.VOLATILITY]
            ax4.scatter(volatility * 100, annual_return * 100, 
                       s=100, label=strategy_name, alpha=0.7)
        ax4.set_title('Risk-Return Profile')
        ax4.set_xlabel('Volatility (%)')
        ax4.set_ylabel('Annualized Return (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Strategy comparison plot saved to {save_path}")
        
        plt.show()
    
    def generate_performance_report(self, 
                                  backtest_results: Dict[str, BacktestResult],
                                  output_file: str = "performance_report.html"):
        """Generate comprehensive performance report"""
        
        # Create comparison DataFrame
        comparison_df = self.compare_strategies(backtest_results)
        
        # Generate HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Performance Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: right; }}
                th {{ background-color: #f2f2f2; font-weight: bold; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric {{ margin: 10px 0; }}
            </style>
        </head>
        <body>
            <h1>Strategy Performance Report</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Strategy Comparison</h2>
            {comparison_df.to_html(index=False, escape=False)}
            
            <h2>Individual Strategy Details</h2>
        """
        
        for strategy_name, result in backtest_results.items():
            html_content += f"""
            <h3>{strategy_name}</h3>
            <div class="metric">
                <strong>Backtest Period:</strong> {result.backtest_period[0]} to {result.backtest_period[1]}
            </div>
            <div class="metric">
                <strong>Total Trades:</strong> {result.trade_statistics['total_trades']}
            </div>
            <div class="metric">
                <strong>Win Rate:</strong> {result.trade_statistics.get('win_rate', 0):.2%}
            </div>
            <div class="metric">
                <strong>Average P&L per Trade:</strong> ${result.trade_statistics.get('avg_pnl_per_trade', 0):.2f}
            </div>
            """
        
        html_content += """
        </body>
        </html>
        """
        
        # Save HTML report
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        self.logger.info(f"Performance report saved to {output_file}")


def main():
    """Example usage of Performance Attribution and Backtesting Framework"""
    
    print("Performance Attribution and Backtesting Framework - Phase 3")
    print("=" * 60)
    
    try:
        # Initialize data loader
        data_loader = TradeStewardDataLoader("../trades.db")
        
        # Load available strategies
        trades_df = data_loader.load_user_trades()
        if trades_df.empty:
            print("❌ No trades data available")
            return
        
        enhanced_df = data_loader.parse_raw_data_json(trades_df)
        clean_df = data_loader.clean_and_validate_data(enhanced_df)
        
        available_strategies = clean_df['strategy'].unique()
        print(f"📊 Available strategies: {list(available_strategies)}")
        
        # Configure backtesting
        config = BacktestConfig(
            start_date=date(2023, 1, 1),
            end_date=date.today(),
            initial_capital=100000,
            rebalance_frequency='daily',
            transaction_costs=2.0,  # 2 bps
            risk_free_rate=0.02,    # 2% annual
            benchmark_symbol='SPY',
            optimization_objective='sharpe',
            max_leverage=2.0,
            position_limits={'max_single_position': 0.2},
            exclude_strategies=[]
        )
        
        print(f"📅 Backtest period: {config.start_date} to {config.end_date}")
        print(f"💰 Initial capital: ${config.initial_capital:,}")
        
        # Initialize backtester
        backtester = StrategyBacktester(data_loader)
        
        # Select strategies to backtest (limit to first 3 for demo)
        strategies_to_test = list(available_strategies)[:3]
        print(f"🧪 Testing strategies: {strategies_to_test}")
        
        # Run backtests
        print(f"\n🚀 Running backtests...")
        backtest_results = backtester.backtest_multiple_strategies(
            strategies_to_test, config, parallel=False
        )
        
        # Initialize analyzer
        analyzer = BacktestAnalyzer()
        
        # Display results
        print(f"\n📊 Backtest Results Summary:")
        comparison_df = analyzer.compare_strategies(backtest_results)
        print(comparison_df.to_string(index=False))
        
        # Detailed results for each strategy
        for strategy_name, result in backtest_results.items():
            print(f"\n📈 {strategy_name} Detailed Results:")
            print(f"   • Total Return: {result.performance_metrics[BacktestMetric.TOTAL_RETURN]:.2%}")
            print(f"   • Sharpe Ratio: {result.performance_metrics[BacktestMetric.SHARPE_RATIO]:.3f}")
            print(f"   • Max Drawdown: {result.performance_metrics[BacktestMetric.MAX_DRAWDOWN]:.2%}")
            print(f"   • Win Rate: {result.performance_metrics[BacktestMetric.WIN_RATE]:.2%}")
            print(f"   • Total Trades: {result.trade_statistics['total_trades']}")
            
            if result.benchmark_comparison:
                print(f"   • Excess Return: {result.benchmark_comparison['excess_return']:.2%}")
                print(f"   • Information Ratio: {result.benchmark_comparison['information_ratio']:.3f}")
        
        # Generate visualizations
        print(f"\n📊 Generating comparison charts...")
        analyzer.plot_strategy_comparison(backtest_results, save_path="strategy_comparison.png")
        
        # Generate performance report
        print(f"\n📋 Generating performance report...")
        analyzer.generate_performance_report(backtest_results, "performance_report.html")
        
        # Performance Attribution Analysis (if enough data)
        if len(backtest_results) >= 2:
            print(f"\n🔍 Running Performance Attribution Analysis...")
            
            # Create sample attribution analysis
            strategy_names = list(backtest_results.keys())
            returns_data = pd.DataFrame({
                name: result.returns_series 
                for name, result in backtest_results.items()
            })
            
            attributor = PerformanceAttributor(returns_data)
            
            # Create sample weights (equal weight)
            portfolio_weights = pd.DataFrame(
                np.ones((len(returns_data), len(strategy_names))) / len(strategy_names),
                index=returns_data.index,
                columns=strategy_names
            )
            
            # Benchmark weights (different allocation)
            benchmark_weights = portfolio_weights.copy()
            benchmark_weights.iloc[:, 0] *= 1.5  # Overweight first strategy
            benchmark_weights = benchmark_weights.div(benchmark_weights.sum(axis=1), axis=0)
            
            attribution_result = attributor.calculate_brinson_attribution(
                portfolio_weights, benchmark_weights
            )
            
            print(f"✅ Attribution Analysis Results:")
            print(f"   • Total Return: {attribution_result.total_return:.2%}")
            print(f"   • Benchmark Return: {attribution_result.benchmark_return:.2%}")
            print(f"   • Active Return: {attribution_result.active_return:.2%}")
            print(f"   • Allocation Effect: {attribution_result.attribution_breakdown.get('allocation_effect', 0):.2%}")
            print(f"   • Selection Effect: {attribution_result.attribution_breakdown.get('selection_effect', 0):.2%}")
            print(f"   • Information Ratio: {attribution_result.risk_metrics['information_ratio']:.3f}")
        
        print(f"\n🎉 Performance attribution and backtesting analysis complete!")
        print(f"📁 Generated files:")
        print(f"   • strategy_comparison.png")
        print(f"   • performance_report.html")
        
    except Exception as e:
        logger.error(f"Error running performance attribution and backtesting: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()