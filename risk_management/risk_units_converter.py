"""
Risk Units Conversion Framework - Phase 1 Implementation
Standardizes risk measurement across all trading strategies using CVaR-based Risk Units
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import json
from pathlib import Path

from cvar_risk_calculator import CVaRRiskCalculator, CVaRResult, RiskUnitsResult, TradeRiskData, RiskUnitBasis

class RiskUnitStandard(str, Enum):
    STANDARD_1K = "standard_1k"      # $1,000 per risk unit
    INSTITUTIONAL_10K = "inst_10k"   # $10,000 per risk unit  
    RETAIL_100 = "retail_100"        # $100 per risk unit
    CUSTOM = "custom"                # Custom unit size

class RiskScalingMethod(str, Enum):
    LINEAR = "linear"                # Simple linear scaling
    LOG_SCALING = "log_scaling"      # Logarithmic scaling for large positions
    SQRT_SCALING = "sqrt_scaling"    # Square root scaling
    POSITION_BASED = "position_based" # Based on position size

@dataclass
class RiskUnitConfiguration:
    standard: RiskUnitStandard
    unit_size: float
    basis: RiskUnitBasis
    scaling_method: RiskScalingMethod
    confidence_level: float
    currency: str = "USD"
    active: bool = True
    created_date: datetime = None
    
    def __post_init__(self):
        if self.created_date is None:
            self.created_date = datetime.now()

@dataclass
class ConversionResult:
    original_trade_id: str
    risk_units: float
    original_cvar: float
    standardized_cvar: float
    unit_size: float
    confidence_level: float
    conversion_method: str
    scaling_applied: float
    conversion_date: datetime
    metadata: Dict[str, any]

@dataclass
class PortfolioRiskUnits:
    portfolio_id: str
    total_risk_units: float
    strategy_breakdown: Dict[str, float]
    concentration_risk: float
    diversification_benefit: float
    effective_risk_units: float
    last_updated: datetime

class RiskUnitsConverter:
    """
    Framework for converting CVaR calculations to standardized Risk Units
    
    Features:
    - Multiple risk unit standards (retail, standard, institutional)
    - Flexible scaling methods
    - Portfolio-level aggregation
    - Risk concentration analysis
    - Historical risk unit tracking
    """
    
    def __init__(self, 
                 default_config: Optional[RiskUnitConfiguration] = None,
                 cvar_calculator: Optional[CVaRRiskCalculator] = None):
        
        if default_config is None:
            default_config = RiskUnitConfiguration(
                standard=RiskUnitStandard.STANDARD_1K,
                unit_size=1000.0,
                basis=RiskUnitBasis.CVAR_95,
                scaling_method=RiskScalingMethod.LINEAR,
                confidence_level=0.95
            )
        
        self.config = default_config
        self.cvar_calculator = cvar_calculator or CVaRRiskCalculator()
        
        # Risk unit standards
        self.standards = {
            RiskUnitStandard.STANDARD_1K: 1000.0,
            RiskUnitStandard.INSTITUTIONAL_10K: 10000.0,
            RiskUnitStandard.RETAIL_100: 100.0
        }
        
        # Conversion history
        self.conversion_history: List[ConversionResult] = []
        
        # Portfolio tracking
        self.portfolio_risk_units: Dict[str, PortfolioRiskUnits] = {}
        
        logging.info(f"Risk Units Converter initialized with {self.config.standard.value} standard")
    
    def convert_trade_to_risk_units(self, 
                                  trade_data: TradeRiskData,
                                  custom_config: Optional[RiskUnitConfiguration] = None) -> ConversionResult:
        """
        Convert a single trade to risk units using specified configuration
        """
        config = custom_config or self.config
        
        try:
            # Calculate CVaR for the trade
            cvar_result = self.cvar_calculator.calculate_trade_cvar(
                trade_data, 
                confidence_level=config.confidence_level
            )
            
            # Convert to risk units
            risk_units_result = self._convert_cvar_to_units(cvar_result, config)
            
            # Apply scaling if needed
            scaled_risk_units, scaling_factor = self._apply_scaling(
                risk_units_result.risk_units, 
                trade_data, 
                config.scaling_method
            )
            
            # Create conversion result
            conversion_result = ConversionResult(
                original_trade_id=trade_data.trade_id,
                risk_units=scaled_risk_units,
                original_cvar=cvar_result.cvar_amount,
                standardized_cvar=cvar_result.cvar_amount,
                unit_size=config.unit_size,
                confidence_level=config.confidence_level,
                conversion_method=f"{config.basis.value}_{config.scaling_method.value}",
                scaling_applied=scaling_factor,
                conversion_date=datetime.now(),
                metadata={
                    'strategy': trade_data.strategy,
                    'underlying': trade_data.underlying_symbol,
                    'days_to_expiration': trade_data.days_to_expiration,
                    'implied_volatility': trade_data.implied_volatility,
                    'max_loss': trade_data.max_loss,
                    'cvar_method': cvar_result.method.value,
                    'market_conditions': cvar_result.market_conditions
                }
            )
            
            # Store in history
            self.conversion_history.append(conversion_result)
            
            return conversion_result
            
        except Exception as e:
            logging.error(f"Risk units conversion failed for trade {trade_data.trade_id}: {e}")
            raise
    
    def _convert_cvar_to_units(self, cvar_result: CVaRResult, config: RiskUnitConfiguration) -> RiskUnitsResult:
        """
        Internal method to convert CVaR to risk units based on configuration
        """
        return self.cvar_calculator.convert_to_risk_units(
            cvar_result, 
            unit_size=config.unit_size,
            basis=config.basis
        )
    
    def _apply_scaling(self, 
                      base_risk_units: float, 
                      trade_data: TradeRiskData, 
                      scaling_method: RiskScalingMethod) -> Tuple[float, float]:
        """
        Apply scaling adjustments to risk units based on trade characteristics
        """
        if scaling_method == RiskScalingMethod.LINEAR:
            return base_risk_units, 1.0
        
        elif scaling_method == RiskScalingMethod.LOG_SCALING:
            # Logarithmic scaling for large positions
            if base_risk_units > 10:
                scaling_factor = np.log10(base_risk_units) / np.log10(10)
                scaled_units = base_risk_units * scaling_factor
                return scaled_units, scaling_factor
            return base_risk_units, 1.0
        
        elif scaling_method == RiskScalingMethod.SQRT_SCALING:
            # Square root scaling (reduces impact of large positions)
            scaling_factor = np.sqrt(base_risk_units) / base_risk_units if base_risk_units > 1 else 1.0
            scaled_units = base_risk_units * scaling_factor
            return scaled_units, scaling_factor
        
        elif scaling_method == RiskScalingMethod.POSITION_BASED:
            # Scale based on position size relative to typical trade
            position_value = abs(trade_data.max_loss)
            typical_position = 2000.0  # Typical $2,000 max loss
            
            if position_value > typical_position * 2:
                scaling_factor = 0.8  # Reduce for large positions
            elif position_value < typical_position * 0.5:
                scaling_factor = 1.2  # Increase for small positions
            else:
                scaling_factor = 1.0
                
            scaled_units = base_risk_units * scaling_factor
            return scaled_units, scaling_factor
        
        return base_risk_units, 1.0
    
    def convert_portfolio_to_risk_units(self, 
                                      trades: List[TradeRiskData],
                                      portfolio_id: str = "default") -> PortfolioRiskUnits:
        """
        Convert entire portfolio to risk units with diversification analysis
        """
        strategy_risk_units = {}
        total_risk_units = 0.0
        
        # Convert each trade
        trade_conversions = []
        for trade in trades:
            conversion = self.convert_trade_to_risk_units(trade)
            trade_conversions.append(conversion)
            
            # Aggregate by strategy
            strategy = trade.strategy
            if strategy not in strategy_risk_units:
                strategy_risk_units[strategy] = 0.0
            strategy_risk_units[strategy] += conversion.risk_units
            total_risk_units += conversion.risk_units
        
        # Calculate risk concentration
        concentration_risk = self._calculate_concentration_risk(strategy_risk_units, total_risk_units)
        
        # Calculate diversification benefit
        diversification_benefit = self._calculate_diversification_benefit(trade_conversions)
        
        # Calculate effective risk units (adjusted for diversification)
        effective_risk_units = total_risk_units * (1 - diversification_benefit)
        
        portfolio_risk = PortfolioRiskUnits(
            portfolio_id=portfolio_id,
            total_risk_units=total_risk_units,
            strategy_breakdown=strategy_risk_units,
            concentration_risk=concentration_risk,
            diversification_benefit=diversification_benefit,
            effective_risk_units=effective_risk_units,
            last_updated=datetime.now()
        )
        
        # Store portfolio risk
        self.portfolio_risk_units[portfolio_id] = portfolio_risk
        
        return portfolio_risk
    
    def _calculate_concentration_risk(self, strategy_breakdown: Dict[str, float], total_risk: float) -> float:
        """
        Calculate risk concentration using Herfindahl-Hirschman Index
        """
        if total_risk == 0:
            return 0.0
        
        # Calculate concentration percentages
        concentrations = [risk / total_risk for risk in strategy_breakdown.values()]
        
        # HHI calculation
        hhi = sum(conc**2 for conc in concentrations)
        
        # Normalize to 0-1 scale (1 = maximum concentration, 0 = perfect diversification)
        n_strategies = len(strategy_breakdown)
        if n_strategies <= 1:
            return 1.0
        
        # Normalize HHI
        min_hhi = 1.0 / n_strategies  # Perfect diversification
        max_hhi = 1.0  # Complete concentration
        
        normalized_concentration = (hhi - min_hhi) / (max_hhi - min_hhi)
        return max(0.0, min(1.0, normalized_concentration))
    
    def _calculate_diversification_benefit(self, conversions: List[ConversionResult]) -> float:
        """
        Calculate diversification benefit based on strategy correlations
        """
        if len(conversions) < 2:
            return 0.0
        
        # Strategy correlation matrix (simplified)
        strategy_correlations = {
            ('straddle', 'strangle'): 0.8,
            ('iron_condor', 'credit_spread'): 0.6,
            ('covered_call', 'protective_put'): -0.3,
            ('straddle', 'iron_condor'): 0.4,
            ('credit_spread', 'covered_call'): 0.2
        }
        
        # Get unique strategies
        strategies = list(set(conv.metadata['strategy'] for conv in conversions))
        
        if len(strategies) < 2:
            return 0.0
        
        # Calculate weighted average correlation
        total_risk = sum(conv.risk_units for conv in conversions)
        weighted_correlation = 0.0
        total_weight = 0.0
        
        for i, strategy1 in enumerate(strategies):
            for j, strategy2 in enumerate(strategies):
                if i >= j:
                    continue
                
                # Get correlation (default to 0.3 if not specified)
                corr_key = tuple(sorted([strategy1, strategy2]))
                correlation = strategy_correlations.get(corr_key, 0.3)
                
                # Get weights for this pair
                weight1 = sum(conv.risk_units for conv in conversions if conv.metadata['strategy'] == strategy1)
                weight2 = sum(conv.risk_units for conv in conversions if conv.metadata['strategy'] == strategy2)
                
                pair_weight = (weight1 * weight2) / (total_risk ** 2)
                weighted_correlation += correlation * pair_weight
                total_weight += pair_weight
        
        if total_weight > 0:
            avg_correlation = weighted_correlation / total_weight
            # Diversification benefit is inverse of correlation
            return max(0.0, (1 - avg_correlation) * 0.3)  # Max 30% benefit
        
        return 0.0
    
    def get_risk_units_summary(self, 
                             portfolio_id: str = "default",
                             period_days: int = 30) -> Dict[str, any]:
        """
        Get comprehensive risk units summary for portfolio
        """
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_conversions = [
            conv for conv in self.conversion_history 
            if conv.conversion_date >= cutoff_date
        ]
        
        if not recent_conversions:
            return {"error": "No recent conversions found"}
        
        # Calculate summary statistics
        risk_units_values = [conv.risk_units for conv in recent_conversions]
        
        summary = {
            "portfolio_id": portfolio_id,
            "period_days": period_days,
            "total_trades": len(recent_conversions),
            "total_risk_units": sum(risk_units_values),
            "average_risk_units": np.mean(risk_units_values),
            "median_risk_units": np.median(risk_units_values),
            "std_risk_units": np.std(risk_units_values),
            "min_risk_units": min(risk_units_values),
            "max_risk_units": max(risk_units_values),
            "risk_unit_config": asdict(self.config)
        }
        
        # Strategy breakdown
        strategy_breakdown = {}
        for conv in recent_conversions:
            strategy = conv.metadata['strategy']
            if strategy not in strategy_breakdown:
                strategy_breakdown[strategy] = {
                    'count': 0,
                    'total_risk_units': 0.0,
                    'avg_risk_units': 0.0
                }
            strategy_breakdown[strategy]['count'] += 1
            strategy_breakdown[strategy]['total_risk_units'] += conv.risk_units
        
        # Calculate averages
        for strategy_data in strategy_breakdown.values():
            strategy_data['avg_risk_units'] = strategy_data['total_risk_units'] / strategy_data['count']
        
        summary['strategy_breakdown'] = strategy_breakdown
        
        # Portfolio risk if available
        if portfolio_id in self.portfolio_risk_units:
            portfolio_risk = self.portfolio_risk_units[portfolio_id]
            summary['portfolio_metrics'] = {
                'concentration_risk': portfolio_risk.concentration_risk,
                'diversification_benefit': portfolio_risk.diversification_benefit,
                'effective_risk_units': portfolio_risk.effective_risk_units,
                'risk_efficiency': portfolio_risk.effective_risk_units / portfolio_risk.total_risk_units if portfolio_risk.total_risk_units > 0 else 0
            }
        
        return summary
    
    def compare_strategies_by_risk_units(self, period_days: int = 30) -> Dict[str, Dict[str, float]]:
        """
        Compare different strategies by their risk unit characteristics
        """
        cutoff_date = datetime.now() - timedelta(days=period_days)
        recent_conversions = [
            conv for conv in self.conversion_history 
            if conv.conversion_date >= cutoff_date
        ]
        
        strategy_stats = {}
        
        for conv in recent_conversions:
            strategy = conv.metadata['strategy']
            if strategy not in strategy_stats:
                strategy_stats[strategy] = []
            strategy_stats[strategy].append(conv.risk_units)
        
        # Calculate statistics for each strategy
        comparison = {}
        for strategy, risk_units_list in strategy_stats.items():
            if len(risk_units_list) > 0:
                comparison[strategy] = {
                    'count': len(risk_units_list),
                    'avg_risk_units': np.mean(risk_units_list),
                    'median_risk_units': np.median(risk_units_list),
                    'std_risk_units': np.std(risk_units_list),
                    'min_risk_units': min(risk_units_list),
                    'max_risk_units': max(risk_units_list),
                    'coefficient_of_variation': np.std(risk_units_list) / np.mean(risk_units_list) if np.mean(risk_units_list) > 0 else 0
                }
        
        return comparison
    
    def set_configuration(self, new_config: RiskUnitConfiguration) -> None:
        """
        Update the risk units configuration
        """
        old_config = self.config
        self.config = new_config
        
        logging.info(f"Risk units configuration updated from {old_config.standard.value} to {new_config.standard.value}")
    
    def export_conversion_history(self, filepath: str) -> None:
        """
        Export conversion history to JSON file
        """
        export_data = []
        for conv in self.conversion_history:
            export_data.append({
                'trade_id': conv.original_trade_id,
                'risk_units': conv.risk_units,
                'original_cvar': conv.original_cvar,
                'unit_size': conv.unit_size,
                'confidence_level': conv.confidence_level,
                'conversion_method': conv.conversion_method,
                'scaling_applied': conv.scaling_applied,
                'conversion_date': conv.conversion_date.isoformat(),
                'metadata': conv.metadata
            })
        
        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)
        
        logging.info(f"Conversion history exported to {filepath}")
    
    def validate_risk_units(self, conversion_result: ConversionResult) -> Dict[str, bool]:
        """
        Validate risk units calculation for reasonableness
        """
        validations = {
            'risk_units_positive': conversion_result.risk_units > 0,
            'risk_units_reasonable': 0.1 <= conversion_result.risk_units <= 50.0,  # Reasonable range
            'scaling_reasonable': 0.5 <= conversion_result.scaling_applied <= 2.0,
            'cvar_positive': conversion_result.original_cvar > 0,
            'unit_size_valid': conversion_result.unit_size > 0
        }
        
        # Log validation failures
        for check, passed in validations.items():
            if not passed:
                logging.warning(f"Risk units validation failed for trade {conversion_result.original_trade_id}: {check}")
        
        return validations

# Utility functions for risk units analysis
class RiskUnitsAnalyzer:
    """
    Additional analysis tools for risk units
    """
    
    @staticmethod
    def calculate_risk_adjusted_return(pnl: float, risk_units: float) -> float:
        """
        Calculate return per risk unit
        """
        return pnl / risk_units if risk_units > 0 else 0.0
    
    @staticmethod
    def calculate_risk_efficiency_ratio(total_pnl: float, total_risk_units: float) -> float:
        """
        Calculate overall risk efficiency
        """
        return total_pnl / total_risk_units if total_risk_units > 0 else 0.0
    
    @staticmethod
    def risk_unit_sharpe_ratio(returns_per_risk_unit: List[float]) -> float:
        """
        Calculate Sharpe ratio for risk-adjusted returns
        """
        if len(returns_per_risk_unit) < 2:
            return 0.0
        
        mean_return = np.mean(returns_per_risk_unit)
        std_return = np.std(returns_per_risk_unit)
        
        return mean_return / std_return if std_return > 0 else 0.0

# Example usage and testing
def main():
    # Initialize converter
    converter = RiskUnitsConverter()
    
    print("Risk Units Conversion Framework - Phase 1 Implementation")
    print("=" * 60)
    
    # Create sample trades
    sample_trades = [
        TradeRiskData(
            trade_id="TRADE_001",
            strategy="iron_condor",
            entry_date=datetime(2024, 1, 15),
            exit_date=None,
            underlying_symbol="SPY",
            trade_pnl=150.0,
            max_loss=-2500.0,
            days_to_expiration=45,
            implied_volatility=0.18,
            delta=-0.05,
            gamma=0.02,
            theta=-15.0,
            vega=25.0,
            premium_paid=2500.0,
            premium_received=500.0
        ),
        TradeRiskData(
            trade_id="TRADE_002",
            strategy="straddle",
            entry_date=datetime(2024, 1, 16),
            exit_date=None,
            underlying_symbol="AAPL",
            trade_pnl=-200.0,
            max_loss=-3500.0,
            days_to_expiration=30,
            implied_volatility=0.25,
            delta=0.0,
            gamma=0.08,
            theta=-25.0,
            vega=45.0,
            premium_paid=3500.0,
            premium_received=0.0
        ),
        TradeRiskData(
            trade_id="TRADE_003",
            strategy="credit_spread",
            entry_date=datetime(2024, 1, 17),
            exit_date=None,
            underlying_symbol="QQQ",
            trade_pnl=75.0,
            max_loss=-1500.0,
            days_to_expiration=60,
            implied_volatility=0.20,
            delta=-0.15,
            gamma=0.01,
            theta=-8.0,
            vega=15.0,
            premium_paid=1500.0,
            premium_received=300.0
        )
    ]
    
    # Convert individual trades
    print("Individual Trade Conversions:")
    print("-" * 40)
    conversions = []
    for trade in sample_trades:
        conversion = converter.convert_trade_to_risk_units(trade)
        conversions.append(conversion)
        
        print(f"Trade ID: {conversion.original_trade_id}")
        print(f"  Strategy: {trade.strategy}")
        print(f"  CVaR: ${conversion.original_cvar:,.2f}")
        print(f"  Risk Units: {conversion.risk_units:.2f}")
        print(f"  Risk/PnL Ratio: {abs(conversion.original_cvar/trade.trade_pnl):.1f}" if trade.trade_pnl != 0 else "  Risk/PnL Ratio: N/A")
        print()
    
    # Convert portfolio
    print("Portfolio Risk Analysis:")
    print("-" * 40)
    portfolio_risk = converter.convert_portfolio_to_risk_units(sample_trades, "test_portfolio")
    
    print(f"Total Risk Units: {portfolio_risk.total_risk_units:.2f}")
    print(f"Effective Risk Units: {portfolio_risk.effective_risk_units:.2f}")
    print(f"Concentration Risk: {portfolio_risk.concentration_risk:.2%}")
    print(f"Diversification Benefit: {portfolio_risk.diversification_benefit:.2%}")
    
    print("\nStrategy Breakdown:")
    for strategy, risk_units in portfolio_risk.strategy_breakdown.items():
        percentage = (risk_units / portfolio_risk.total_risk_units) * 100
        print(f"  {strategy}: {risk_units:.2f} units ({percentage:.1f}%)")
    
    # Get summary statistics
    print("\nRisk Units Summary:")
    print("-" * 40)
    summary = converter.get_risk_units_summary("test_portfolio")
    
    print(f"Average Risk Units per Trade: {summary['average_risk_units']:.2f}")
    print(f"Standard Deviation: {summary['std_risk_units']:.2f}")
    print(f"Range: {summary['min_risk_units']:.2f} - {summary['max_risk_units']:.2f}")
    
    # Strategy comparison
    print("\nStrategy Risk Comparison:")
    print("-" * 40)
    strategy_comparison = converter.compare_strategies_by_risk_units()
    
    for strategy, stats in strategy_comparison.items():
        print(f"{strategy.title()}:")
        print(f"  Average: {stats['avg_risk_units']:.2f} units")
        print(f"  Volatility: {stats['coefficient_of_variation']:.2f}")
        print(f"  Count: {stats['count']} trades")

if __name__ == "__main__":
    main()