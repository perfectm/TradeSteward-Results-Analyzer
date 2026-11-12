"""
CVaR Risk Calculator - Phase 1 Implementation
Calculates Conditional Value at Risk (CVaR) for options trades and converts to standardized Risk Units
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

class CVaRMethod(str, Enum):
    HISTORICAL_SIMULATION = "historical_simulation"
    PARAMETRIC_NORMAL = "parametric_normal"
    PARAMETRIC_T_DIST = "parametric_t_dist"
    MONTE_CARLO = "monte_carlo"
    CORNISH_FISHER = "cornish_fisher"

class RiskUnitBasis(str, Enum):
    CVAR_95 = "cvar_95"
    CVAR_99 = "cvar_99"
    VAR_95 = "var_95"
    VAR_99 = "var_99"

@dataclass
class TradeRiskData:
    trade_id: str
    strategy: str
    entry_date: datetime
    exit_date: Optional[datetime]
    underlying_symbol: str
    trade_pnl: float
    max_loss: float
    days_to_expiration: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float
    premium_paid: float
    premium_received: float
    
@dataclass
class CVaRResult:
    trade_id: str
    cvar_amount: float
    var_amount: float
    confidence_level: float
    method: CVaRMethod
    calculation_date: datetime
    market_conditions: Dict[str, float]
    
@dataclass
class RiskUnitsResult:
    trade_id: str
    risk_units: float
    cvar_amount: float
    unit_size: float
    basis: RiskUnitBasis
    calculation_date: datetime
    risk_efficiency: float  # PnL per risk unit

class CVaRRiskCalculator:
    """
    Calculate CVaR-based risk units for options trades
    
    Features:
    - Historical simulation CVaR
    - Parametric CVaR (normal/t-distribution)
    - Monte Carlo CVaR simulation
    - Dynamic confidence level adjustment
    """
    
    def __init__(self, default_unit_size: float = 1000.0, default_confidence: float = 0.95):
        self.default_unit_size = default_unit_size
        self.default_confidence = default_confidence
        self.historical_lookback_days = 252  # 1 year of trading days
        self.monte_carlo_simulations = 10000
        self.risk_free_rate = 0.045  # 4.5% risk-free rate
        
        # Market regime detection parameters
        self.volatility_regimes = {
            'low': (0.0, 0.15),
            'normal': (0.15, 0.30),
            'high': (0.30, 0.60),
            'extreme': (0.60, float('inf'))
        }
        
        logging.info("CVaR Risk Calculator initialized")
    
    def calculate_trade_cvar(self, 
                           trade_data: TradeRiskData, 
                           method: CVaRMethod = CVaRMethod.HISTORICAL_SIMULATION,
                           confidence_level: Optional[float] = None) -> CVaRResult:
        """
        Calculate CVaR for a specific trade using the specified method
        """
        if confidence_level is None:
            confidence_level = self.default_confidence
            
        try:
            if method == CVaRMethod.HISTORICAL_SIMULATION:
                cvar, var = self._calculate_historical_cvar(trade_data, confidence_level)
            elif method == CVaRMethod.PARAMETRIC_NORMAL:
                cvar, var = self._calculate_parametric_cvar_normal(trade_data, confidence_level)
            elif method == CVaRMethod.PARAMETRIC_T_DIST:
                cvar, var = self._calculate_parametric_cvar_t(trade_data, confidence_level)
            elif method == CVaRMethod.MONTE_CARLO:
                cvar, var = self._calculate_monte_carlo_cvar(trade_data, confidence_level)
            elif method == CVaRMethod.CORNISH_FISHER:
                cvar, var = self._calculate_cornish_fisher_cvar(trade_data, confidence_level)
            else:
                raise ValueError(f"Unsupported CVaR method: {method}")
            
            market_conditions = self._assess_market_conditions(trade_data)
            
            return CVaRResult(
                trade_id=trade_data.trade_id,
                cvar_amount=abs(cvar),  # Always positive for risk measurement
                var_amount=abs(var),
                confidence_level=confidence_level,
                method=method,
                calculation_date=datetime.now(),
                market_conditions=market_conditions
            )
            
        except Exception as e:
            logging.error(f"CVaR calculation failed for trade {trade_data.trade_id}: {e}")
            # Fallback to maximum loss
            return CVaRResult(
                trade_id=trade_data.trade_id,
                cvar_amount=abs(trade_data.max_loss),
                var_amount=abs(trade_data.max_loss * 0.8),
                confidence_level=confidence_level,
                method=method,
                calculation_date=datetime.now(),
                market_conditions={}
            )
    
    def _calculate_historical_cvar(self, trade_data: TradeRiskData, confidence_level: float) -> Tuple[float, float]:
        """
        Calculate CVaR using historical simulation method
        """
        # Generate historical P&L scenarios based on trade characteristics
        scenarios = self._generate_historical_scenarios(trade_data)
        
        # Calculate VaR (percentile)
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(scenarios, var_percentile)
        
        # Calculate CVaR (expected value of losses beyond VaR)
        tail_losses = scenarios[scenarios <= var]
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        return cvar, var
    
    def _calculate_parametric_cvar_normal(self, trade_data: TradeRiskData, confidence_level: float) -> Tuple[float, float]:
        """
        Calculate CVaR assuming normal distribution of returns
        """
        # Estimate parameters from trade data
        expected_return = trade_data.trade_pnl
        volatility = self._estimate_trade_volatility(trade_data)
        
        # Calculate VaR
        z_score = stats.norm.ppf(1 - confidence_level)
        var = expected_return + z_score * volatility
        
        # Calculate CVaR
        phi_z = stats.norm.pdf(z_score)
        cvar = expected_return - (volatility * phi_z) / (1 - confidence_level)
        
        return cvar, var
    
    def _calculate_parametric_cvar_t(self, trade_data: TradeRiskData, confidence_level: float) -> Tuple[float, float]:
        """
        Calculate CVaR assuming t-distribution of returns (better for fat tails)
        """
        # Estimate parameters
        expected_return = trade_data.trade_pnl
        volatility = self._estimate_trade_volatility(trade_data)
        
        # Estimate degrees of freedom (higher for more normal-like, lower for fatter tails)
        df = self._estimate_degrees_of_freedom(trade_data)
        
        # Calculate VaR
        t_score = stats.t.ppf(1 - confidence_level, df)
        var = expected_return + t_score * volatility
        
        # Calculate CVaR for t-distribution
        pdf_t = stats.t.pdf(t_score, df)
        cvar = expected_return - volatility * pdf_t * (df + t_score**2) / ((df - 1) * (1 - confidence_level))
        
        return cvar, var
    
    def _calculate_monte_carlo_cvar(self, trade_data: TradeRiskData, confidence_level: float) -> Tuple[float, float]:
        """
        Calculate CVaR using Monte Carlo simulation
        """
        scenarios = self._generate_monte_carlo_scenarios(trade_data, self.monte_carlo_simulations)
        
        # Calculate VaR and CVaR from simulated scenarios
        var_percentile = (1 - confidence_level) * 100
        var = np.percentile(scenarios, var_percentile)
        
        tail_losses = scenarios[scenarios <= var]
        cvar = np.mean(tail_losses) if len(tail_losses) > 0 else var
        
        return cvar, var
    
    def _calculate_cornish_fisher_cvar(self, trade_data: TradeRiskData, confidence_level: float) -> Tuple[float, float]:
        """
        Calculate CVaR using Cornish-Fisher expansion (accounts for skewness and kurtosis)
        """
        expected_return = trade_data.trade_pnl
        volatility = self._estimate_trade_volatility(trade_data)
        
        # Estimate skewness and kurtosis from trade characteristics
        skewness = self._estimate_skewness(trade_data)
        kurtosis = self._estimate_kurtosis(trade_data)
        
        # Cornish-Fisher VaR adjustment
        z = stats.norm.ppf(1 - confidence_level)
        cf_adjustment = (z**2 - 1) * skewness / 6 + (z**3 - 3*z) * (kurtosis - 3) / 24
        cf_z = z + cf_adjustment
        
        var = expected_return + cf_z * volatility
        
        # CVaR approximation for Cornish-Fisher
        phi_z = stats.norm.pdf(cf_z)
        cvar = expected_return - (volatility * phi_z) / (1 - confidence_level)
        
        return cvar, var
    
    def _generate_historical_scenarios(self, trade_data: TradeRiskData) -> np.ndarray:
        """
        Generate P&L scenarios based on historical market data patterns
        """
        # Base scenarios on strategy type and market conditions
        strategy_vol = self._get_strategy_volatility(trade_data.strategy)
        market_vol = trade_data.implied_volatility
        
        # Generate scenarios with realistic distribution
        n_scenarios = 1000
        base_return = trade_data.trade_pnl
        
        # Create fat-tailed distribution for options
        normal_component = np.random.normal(0, strategy_vol * 0.7, n_scenarios)
        fat_tail_component = np.random.laplace(0, strategy_vol * 0.3, n_scenarios)
        
        scenarios = base_return + (normal_component + fat_tail_component) * abs(trade_data.max_loss)
        
        # Add strategy-specific adjustments
        scenarios = self._apply_strategy_adjustments(scenarios, trade_data)
        
        return scenarios
    
    def _generate_monte_carlo_scenarios(self, trade_data: TradeRiskData, n_simulations: int) -> np.ndarray:
        """
        Generate Monte Carlo scenarios for P&L distribution
        """
        # Advanced Monte Carlo with multiple risk factors
        scenarios = []
        
        for _ in range(n_simulations):
            # Simulate underlying price movement
            underlying_return = np.random.normal(0, trade_data.implied_volatility / np.sqrt(252))
            
            # Calculate option P&L using Greeks
            delta_pnl = trade_data.delta * underlying_return * 100  # Assuming $100 underlying move
            gamma_pnl = 0.5 * trade_data.gamma * (underlying_return * 100) ** 2
            theta_pnl = trade_data.theta * (1/252)  # Daily theta decay
            vega_pnl = trade_data.vega * np.random.normal(0, 0.01)  # IV change
            
            total_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl
            scenarios.append(total_pnl)
        
        return np.array(scenarios)
    
    def _estimate_trade_volatility(self, trade_data: TradeRiskData) -> float:
        """
        Estimate volatility for the trade based on multiple factors
        """
        # Base volatility from implied volatility
        base_vol = trade_data.implied_volatility
        
        # Adjust for time to expiration (volatility smile effect)
        time_adjustment = 1 + (30 - trade_data.days_to_expiration) / 100
        
        # Adjust for strategy type
        strategy_multiplier = self._get_strategy_volatility_multiplier(trade_data.strategy)
        
        # Adjust for Greeks concentration
        greeks_adjustment = 1 + abs(trade_data.delta) * 0.1 + abs(trade_data.gamma) * 0.05
        
        return base_vol * time_adjustment * strategy_multiplier * greeks_adjustment
    
    def _estimate_degrees_of_freedom(self, trade_data: TradeRiskData) -> float:
        """
        Estimate degrees of freedom for t-distribution based on trade characteristics
        """
        # Base degrees of freedom
        base_df = 6.0
        
        # Adjust based on strategy (more complex strategies have fatter tails)
        strategy_adjustment = {
            'straddle': -1.0,
            'strangle': -1.0,
            'iron_condor': -0.5,
            'credit_spread': 1.0,
            'covered_call': 2.0,
            'protective_put': 1.5
        }
        
        strategy_adj = strategy_adjustment.get(trade_data.strategy.lower(), 0.0)
        
        # Adjust based on time to expiration (shorter time = fatter tails)
        time_adj = max(-2.0, (trade_data.days_to_expiration - 30) / 15)
        
        return max(3.0, base_df + strategy_adj + time_adj)
    
    def _estimate_skewness(self, trade_data: TradeRiskData) -> float:
        """
        Estimate skewness based on trade characteristics
        """
        # Options strategies typically have negative skewness (fat left tail)
        base_skewness = -0.5
        
        # Adjust based on strategy
        if 'call' in trade_data.strategy.lower():
            return base_skewness * 0.8  # Calls less negatively skewed
        elif 'put' in trade_data.strategy.lower():
            return base_skewness * 1.2  # Puts more negatively skewed
        
        return base_skewness
    
    def _estimate_kurtosis(self, trade_data: TradeRiskData) -> float:
        """
        Estimate excess kurtosis based on trade characteristics
        """
        # Options typically have high kurtosis (fat tails)
        base_kurtosis = 6.0
        
        # Adjust based on strategy complexity
        strategy_kurtosis = {
            'straddle': 8.0,
            'strangle': 7.5,
            'iron_condor': 5.0,
            'credit_spread': 4.0,
            'covered_call': 3.0
        }
        
        return strategy_kurtosis.get(trade_data.strategy.lower(), base_kurtosis)
    
    def _get_strategy_volatility(self, strategy: str) -> float:
        """
        Get typical volatility for different strategy types
        """
        strategy_volatilities = {
            'straddle': 0.25,
            'strangle': 0.22,
            'iron_condor': 0.15,
            'credit_spread': 0.12,
            'covered_call': 0.08,
            'protective_put': 0.10,
            'collar': 0.09,
            'ratio_spread': 0.18
        }
        
        return strategy_volatilities.get(strategy.lower(), 0.15)
    
    def _get_strategy_volatility_multiplier(self, strategy: str) -> float:
        """
        Get volatility multiplier for strategy risk calculation
        """
        multipliers = {
            'straddle': 1.3,
            'strangle': 1.25,
            'iron_condor': 1.1,
            'credit_spread': 1.0,
            'covered_call': 0.8,
            'protective_put': 0.9
        }
        
        return multipliers.get(strategy.lower(), 1.0)
    
    def _apply_strategy_adjustments(self, scenarios: np.ndarray, trade_data: TradeRiskData) -> np.ndarray:
        """
        Apply strategy-specific adjustments to P&L scenarios
        """
        # Adjust for maximum loss constraints
        scenarios = np.maximum(scenarios, -abs(trade_data.max_loss))
        
        # Apply strategy-specific distribution characteristics
        if 'credit' in trade_data.strategy.lower():
            # Credit strategies have limited upside, higher probability of small gains
            scenarios = np.where(scenarios > trade_data.premium_received, 
                                trade_data.premium_received, scenarios)
        
        return scenarios
    
    def _assess_market_conditions(self, trade_data: TradeRiskData) -> Dict[str, float]:
        """
        Assess current market conditions for risk calculation context
        """
        iv = trade_data.implied_volatility
        
        # Determine volatility regime
        regime = 'normal'
        for regime_name, (low, high) in self.volatility_regimes.items():
            if low <= iv < high:
                regime = regime_name
                break
        
        return {
            'implied_volatility': iv,
            'volatility_regime': regime,
            'time_to_expiration': trade_data.days_to_expiration,
            'delta_exposure': abs(trade_data.delta),
            'gamma_exposure': abs(trade_data.gamma),
            'vega_exposure': abs(trade_data.vega)
        }
    
    def convert_to_risk_units(self, 
                            cvar_result: CVaRResult, 
                            unit_size: Optional[float] = None,
                            basis: RiskUnitBasis = RiskUnitBasis.CVAR_95) -> RiskUnitsResult:
        """
        Convert CVaR amount to standardized risk units
        """
        if unit_size is None:
            unit_size = self.default_unit_size
        
        # Select appropriate risk measure based on basis
        if basis == RiskUnitBasis.CVAR_95:
            risk_amount = cvar_result.cvar_amount
        elif basis == RiskUnitBasis.VAR_95:
            risk_amount = cvar_result.var_amount
        else:
            risk_amount = cvar_result.cvar_amount  # Default to CVaR
        
        # Calculate risk units
        risk_units = risk_amount / unit_size
        
        # Calculate risk efficiency (would need actual P&L data)
        risk_efficiency = 0.0  # Placeholder - will be calculated with actual trade results
        
        return RiskUnitsResult(
            trade_id=cvar_result.trade_id,
            risk_units=risk_units,
            cvar_amount=cvar_result.cvar_amount,
            unit_size=unit_size,
            basis=basis,
            calculation_date=cvar_result.calculation_date,
            risk_efficiency=risk_efficiency
        )
    
    def update_risk_parameters(self, market_data: Dict[str, float]) -> None:
        """
        Update risk calculation parameters based on current market conditions
        """
        if 'vix' in market_data:
            vix = market_data['vix']
            
            # Adjust Monte Carlo simulations based on market volatility
            if vix > 30:  # High volatility
                self.monte_carlo_simulations = 15000
            elif vix > 20:  # Medium volatility
                self.monte_carlo_simulations = 12000
            else:  # Low volatility
                self.monte_carlo_simulations = 10000
        
        if 'risk_free_rate' in market_data:
            self.risk_free_rate = market_data['risk_free_rate']
        
        logging.info(f"Risk parameters updated: MC sims={self.monte_carlo_simulations}, RF rate={self.risk_free_rate}")
    
    def validate_risk_calculation(self, trade_id: str, cvar_result: CVaRResult) -> Dict[str, bool]:
        """
        Validate the risk calculation results for reasonableness
        """
        validation_results = {
            'cvar_positive': cvar_result.cvar_amount >= 0,
            'cvar_greater_than_var': cvar_result.cvar_amount >= cvar_result.var_amount,
            'confidence_level_valid': 0.5 <= cvar_result.confidence_level <= 0.999,
            'calculation_recent': (datetime.now() - cvar_result.calculation_date).days <= 1
        }
        
        # Log any validation failures
        for check, passed in validation_results.items():
            if not passed:
                logging.warning(f"Validation failed for trade {trade_id}: {check}")
        
        return validation_results
    
    def get_strategy_risk_profile(self, strategy: str) -> Dict[str, float]:
        """
        Get risk profile characteristics for a specific strategy
        """
        risk_profiles = {
            'straddle': {
                'typical_risk_units': 3.2,
                'volatility_sensitivity': 0.8,
                'time_decay_impact': 0.6,
                'max_loss_ratio': 1.0
            },
            'strangle': {
                'typical_risk_units': 2.8,
                'volatility_sensitivity': 0.7,
                'time_decay_impact': 0.5,
                'max_loss_ratio': 1.0
            },
            'iron_condor': {
                'typical_risk_units': 1.8,
                'volatility_sensitivity': 0.4,
                'time_decay_impact': 0.3,
                'max_loss_ratio': 0.8
            },
            'credit_spread': {
                'typical_risk_units': 1.5,
                'volatility_sensitivity': 0.3,
                'time_decay_impact': 0.2,
                'max_loss_ratio': 0.7
            },
            'covered_call': {
                'typical_risk_units': 1.0,
                'volatility_sensitivity': 0.2,
                'time_decay_impact': 0.1,
                'max_loss_ratio': 0.3
            }
        }
        
        return risk_profiles.get(strategy.lower(), {
            'typical_risk_units': 2.0,
            'volatility_sensitivity': 0.5,
            'time_decay_impact': 0.4,
            'max_loss_ratio': 0.8
        })

# Example usage and testing
def main():
    # Initialize calculator
    calculator = CVaRRiskCalculator()
    
    # Create sample trade data
    sample_trade = TradeRiskData(
        trade_id="TRADE_001",
        strategy="iron_condor",
        entry_date=datetime(2024, 1, 15),
        exit_date=None,
        underlying_symbol="SPY",
        trade_pnl=0.0,  # Not yet closed
        max_loss=-2500.0,
        days_to_expiration=45,
        implied_volatility=0.18,
        delta=-0.05,
        gamma=0.02,
        theta=-15.0,
        vega=25.0,
        premium_paid=2500.0,
        premium_received=500.0
    )
    
    print("CVaR Risk Calculator - Phase 1 Implementation")
    print("=" * 50)
    
    # Test different CVaR calculation methods
    methods = [
        CVaRMethod.HISTORICAL_SIMULATION,
        CVaRMethod.PARAMETRIC_NORMAL,
        CVaRMethod.PARAMETRIC_T_DIST,
        CVaRMethod.MONTE_CARLO
    ]
    
    results = {}
    for method in methods:
        print(f"\nTesting {method.value}...")
        cvar_result = calculator.calculate_trade_cvar(sample_trade, method)
        risk_units_result = calculator.convert_to_risk_units(cvar_result)
        
        results[method.value] = {
            'cvar': cvar_result.cvar_amount,
            'var': cvar_result.var_amount,
            'risk_units': risk_units_result.risk_units
        }
        
        print(f"  CVaR (95%): ${cvar_result.cvar_amount:,.2f}")
        print(f"  VaR (95%): ${cvar_result.var_amount:,.2f}")
        print(f"  Risk Units: {risk_units_result.risk_units:.2f}")
        
        # Validate calculation
        validation = calculator.validate_risk_calculation(sample_trade.trade_id, cvar_result)
        print(f"  Validation: {all(validation.values())}")
    
    # Compare methods
    print(f"\nMethod Comparison:")
    print(f"{'Method':<25} {'CVaR':<12} {'Risk Units':<12}")
    print("-" * 50)
    for method, values in results.items():
        print(f"{method:<25} ${values['cvar']:<11,.0f} {values['risk_units']:<11.2f}")
    
    # Test strategy risk profiles
    print(f"\nStrategy Risk Profiles:")
    strategies = ['straddle', 'iron_condor', 'credit_spread', 'covered_call']
    for strategy in strategies:
        profile = calculator.get_strategy_risk_profile(strategy)
        print(f"{strategy.title()}: {profile['typical_risk_units']:.1f} risk units (typical)")

if __name__ == "__main__":
    main()