"""
Strategy Risk Profiler - Phase 1 Implementation
Analyzes and profiles risk characteristics by options trading strategy
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from scipy import stats
import json

from cvar_risk_calculator import TradeRiskData, CVaRResult
from risk_units_converter import ConversionResult

class StrategyCategory(str, Enum):
    DIRECTIONAL = "directional"
    NEUTRAL = "neutral"
    VOLATILITY = "volatility"
    INCOME = "income"
    HEDGE = "hedge"

class RiskProfile(str, Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    SPECULATIVE = "speculative"

@dataclass
class StrategyCharacteristics:
    strategy_name: str
    category: StrategyCategory
    risk_profile: RiskProfile
    typical_dte_range: Tuple[int, int]  # Days to expiration range
    typical_iv_range: Tuple[float, float]  # Implied volatility range
    delta_neutrality: float  # 0 = delta neutral, 1 = highly directional
    gamma_sensitivity: float  # 0-1 scale
    theta_dependency: float  # 0-1 scale (income strategies high)
    vega_sensitivity: float  # 0-1 scale
    max_loss_predictability: float  # 0-1 scale (1 = fully predictable)
    win_rate_expectation: float  # Expected win rate
    profit_loss_ratio: float  # Average winner / average loser
    capital_efficiency: float  # 0-1 scale

@dataclass
class StrategyRiskMetrics:
    strategy_name: str
    sample_size: int
    avg_risk_units: float
    std_risk_units: float
    min_risk_units: float
    max_risk_units: float
    risk_unit_percentiles: Dict[str, float]  # 25th, 50th, 75th, 90th, 95th
    avg_cvar: float
    avg_max_loss: float
    risk_efficiency: float  # Return per risk unit
    consistency_score: float  # 1 - coefficient of variation
    risk_skewness: float
    risk_kurtosis: float
    last_updated: datetime

@dataclass
class StrategyComparison:
    base_strategy: str
    comparison_strategy: str
    risk_ratio: float  # comparison / base risk units
    efficiency_ratio: float  # comparison / base efficiency
    consistency_ratio: float  # comparison / base consistency
    correlation: float  # Risk correlation between strategies
    recommendation: str  # Diversification recommendation

class StrategyRiskProfiler:
    """
    Analyze and profile risk characteristics by strategy
    
    Features:
    - Strategy risk fingerprinting
    - Dynamic risk profiling
    - Cross-strategy risk correlation
    - Risk efficiency scoring
    """
    
    def __init__(self):
        # Predefined strategy characteristics
        self.strategy_characteristics = self._initialize_strategy_characteristics()
        
        # Dynamic risk profiles
        self.dynamic_profiles: Dict[str, StrategyRiskMetrics] = {}
        
        # Risk correlation matrix
        self.risk_correlations: Dict[Tuple[str, str], float] = {}
        
        # Historical data storage
        self.strategy_history: Dict[str, List[ConversionResult]] = {}
        
        logging.info("Strategy Risk Profiler initialized")
    
    def _initialize_strategy_characteristics(self) -> Dict[str, StrategyCharacteristics]:
        """
        Initialize predefined characteristics for common options strategies
        """
        characteristics = {
            'straddle': StrategyCharacteristics(
                strategy_name='straddle',
                category=StrategyCategory.VOLATILITY,
                risk_profile=RiskProfile.AGGRESSIVE,
                typical_dte_range=(14, 60),
                typical_iv_range=(0.15, 0.40),
                delta_neutrality=0.9,  # High delta neutrality
                gamma_sensitivity=0.8,
                theta_dependency=0.3,
                vega_sensitivity=0.9,
                max_loss_predictability=0.9,  # Premium paid
                win_rate_expectation=0.35,
                profit_loss_ratio=2.5,
                capital_efficiency=0.6
            ),
            'strangle': StrategyCharacteristics(
                strategy_name='strangle',
                category=StrategyCategory.VOLATILITY,
                risk_profile=RiskProfile.AGGRESSIVE,
                typical_dte_range=(14, 60),
                typical_iv_range=(0.15, 0.40),
                delta_neutrality=0.85,
                gamma_sensitivity=0.7,
                theta_dependency=0.3,
                vega_sensitivity=0.85,
                max_loss_predictability=0.9,
                win_rate_expectation=0.40,
                profit_loss_ratio=2.2,
                capital_efficiency=0.65
            ),
            'iron_condor': StrategyCharacteristics(
                strategy_name='iron_condor',
                category=StrategyCategory.NEUTRAL,
                risk_profile=RiskProfile.MODERATE,
                typical_dte_range=(30, 60),
                typical_iv_range=(0.12, 0.30),
                delta_neutrality=0.95,
                gamma_sensitivity=0.4,
                theta_dependency=0.8,
                vega_sensitivity=0.5,
                max_loss_predictability=0.95,  # Spread width - credit
                win_rate_expectation=0.65,
                profit_loss_ratio=0.8,
                capital_efficiency=0.75
            ),
            'credit_spread': StrategyCharacteristics(
                strategy_name='credit_spread',
                category=StrategyCategory.INCOME,
                risk_profile=RiskProfile.MODERATE,
                typical_dte_range=(20, 45),
                typical_iv_range=(0.15, 0.35),
                delta_neutrality=0.3,  # More directional
                gamma_sensitivity=0.3,
                theta_dependency=0.9,
                vega_sensitivity=0.4,
                max_loss_predictability=0.95,
                win_rate_expectation=0.70,
                profit_loss_ratio=0.6,
                capital_efficiency=0.8
            ),
            'covered_call': StrategyCharacteristics(
                strategy_name='covered_call',
                category=StrategyCategory.INCOME,
                risk_profile=RiskProfile.CONSERVATIVE,
                typical_dte_range=(20, 60),
                typical_iv_range=(0.10, 0.25),
                delta_neutrality=0.2,  # Highly directional (long stock)
                gamma_sensitivity=0.2,
                theta_dependency=0.7,
                vega_sensitivity=0.3,
                max_loss_predictability=0.3,  # Stock can fall significantly
                win_rate_expectation=0.75,
                profit_loss_ratio=0.4,
                capital_efficiency=0.4
            ),
            'protective_put': StrategyCharacteristics(
                strategy_name='protective_put',
                category=StrategyCategory.HEDGE,
                risk_profile=RiskProfile.CONSERVATIVE,
                typical_dte_range=(30, 90),
                typical_iv_range=(0.15, 0.40),
                delta_neutrality=0.1,  # Highly directional
                gamma_sensitivity=0.4,
                theta_dependency=0.2,
                vega_sensitivity=0.6,
                max_loss_predictability=0.8,
                win_rate_expectation=0.80,
                profit_loss_ratio=0.3,
                capital_efficiency=0.3
            ),
            'collar': StrategyCharacteristics(
                strategy_name='collar',
                category=StrategyCategory.HEDGE,
                risk_profile=RiskProfile.CONSERVATIVE,
                typical_dte_range=(30, 90),
                typical_iv_range=(0.12, 0.30),
                delta_neutrality=0.2,
                gamma_sensitivity=0.3,
                theta_dependency=0.5,
                vega_sensitivity=0.4,
                max_loss_predictability=0.9,
                win_rate_expectation=0.85,
                profit_loss_ratio=0.25,
                capital_efficiency=0.5
            ),
            'ratio_spread': StrategyCharacteristics(
                strategy_name='ratio_spread',
                category=StrategyCategory.VOLATILITY,
                risk_profile=RiskProfile.SPECULATIVE,
                typical_dte_range=(20, 45),
                typical_iv_range=(0.15, 0.35),
                delta_neutrality=0.6,
                gamma_sensitivity=0.9,
                theta_dependency=0.4,
                vega_sensitivity=0.7,
                max_loss_predictability=0.2,  # Unlimited risk potential
                win_rate_expectation=0.45,
                profit_loss_ratio=1.8,
                capital_efficiency=0.7
            )
        }
        
        return characteristics
    
    def profile_strategy_risk(self, 
                            strategy_name: str, 
                            conversions: List[ConversionResult]) -> StrategyRiskMetrics:
        """
        Analyze risk characteristics for a specific strategy based on historical data
        """
        if not conversions:
            raise ValueError(f"No conversion data provided for strategy {strategy_name}")
        
        # Extract risk units data
        risk_units = [conv.risk_units for conv in conversions]
        cvar_amounts = [conv.original_cvar for conv in conversions]
        max_losses = [conv.metadata.get('max_loss', 0) for conv in conversions]
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(risk_units, cvar_amounts, max_losses)
        
        # Calculate risk efficiency (if P&L data available)
        risk_efficiency = self._calculate_risk_efficiency(conversions)
        
        # Calculate consistency score
        consistency_score = self._calculate_consistency_score(risk_units)
        
        # Calculate percentiles
        percentiles = self._calculate_percentiles(risk_units)
        
        # Create risk profile
        profile = StrategyRiskMetrics(
            strategy_name=strategy_name,
            sample_size=len(conversions),
            avg_risk_units=risk_metrics['mean'],
            std_risk_units=risk_metrics['std'],
            min_risk_units=risk_metrics['min'],
            max_risk_units=risk_metrics['max'],
            risk_unit_percentiles=percentiles,
            avg_cvar=np.mean(cvar_amounts),
            avg_max_loss=np.mean(max_losses),
            risk_efficiency=risk_efficiency,
            consistency_score=consistency_score,
            risk_skewness=risk_metrics['skewness'],
            risk_kurtosis=risk_metrics['kurtosis'],
            last_updated=datetime.now()
        )
        
        # Store in dynamic profiles
        self.dynamic_profiles[strategy_name] = profile
        
        # Store historical data
        if strategy_name not in self.strategy_history:
            self.strategy_history[strategy_name] = []
        self.strategy_history[strategy_name].extend(conversions)
        
        return profile
    
    def _calculate_risk_metrics(self, 
                              risk_units: List[float], 
                              cvar_amounts: List[float], 
                              max_losses: List[float]) -> Dict[str, float]:
        """
        Calculate comprehensive risk statistics
        """
        ru_array = np.array(risk_units)
        
        return {
            'mean': np.mean(ru_array),
            'std': np.std(ru_array),
            'min': np.min(ru_array),
            'max': np.max(ru_array),
            'median': np.median(ru_array),
            'skewness': stats.skew(ru_array),
            'kurtosis': stats.kurtosis(ru_array),
            'var_95': np.percentile(ru_array, 95),
            'var_99': np.percentile(ru_array, 99)
        }
    
    def _calculate_risk_efficiency(self, conversions: List[ConversionResult]) -> float:
        """
        Calculate return per risk unit (if P&L data available)
        """
        total_pnl = 0.0
        total_risk_units = 0.0
        
        for conv in conversions:
            # Check if P&L data is available in metadata
            if 'realized_pnl' in conv.metadata:
                total_pnl += conv.metadata['realized_pnl']
                total_risk_units += conv.risk_units
        
        return total_pnl / total_risk_units if total_risk_units > 0 else 0.0
    
    def _calculate_consistency_score(self, risk_units: List[float]) -> float:
        """
        Calculate consistency score (1 - coefficient of variation)
        """
        if len(risk_units) < 2:
            return 0.0
        
        mean_ru = np.mean(risk_units)
        std_ru = np.std(risk_units)
        
        if mean_ru == 0:
            return 0.0
        
        cv = std_ru / mean_ru
        return max(0.0, 1.0 - cv)  # Higher score = more consistent
    
    def _calculate_percentiles(self, risk_units: List[float]) -> Dict[str, float]:
        """
        Calculate risk unit percentiles
        """
        return {
            '25th': np.percentile(risk_units, 25),
            '50th': np.percentile(risk_units, 50),
            '75th': np.percentile(risk_units, 75),
            '90th': np.percentile(risk_units, 90),
            '95th': np.percentile(risk_units, 95)
        }
    
    def calculate_risk_efficiency(self, strategy_data: List[ConversionResult]) -> float:
        """
        Calculate risk efficiency score for a strategy
        """
        if not strategy_data:
            return 0.0
        
        # Risk efficiency components
        consistency = self._calculate_consistency_score([conv.risk_units for conv in strategy_data])
        
        # Predictability (based on how close actual risk matches expected)
        predictability = self._calculate_predictability_score(strategy_data)
        
        # Capital efficiency (risk units per dollar at risk)
        capital_efficiency = self._calculate_capital_efficiency(strategy_data)
        
        # Weighted risk efficiency score
        efficiency_score = (
            consistency * 0.4 +
            predictability * 0.3 +
            capital_efficiency * 0.3
        )
        
        return min(1.0, max(0.0, efficiency_score))
    
    def _calculate_predictability_score(self, conversions: List[ConversionResult]) -> float:
        """
        Calculate how predictable the risk calculation is
        """
        risk_units = [conv.risk_units for conv in conversions]
        strategy_name = conversions[0].metadata.get('strategy', 'unknown')
        
        # Get expected risk range for strategy
        if strategy_name in self.strategy_characteristics:
            char = self.strategy_characteristics[strategy_name]
            expected_predictability = char.max_loss_predictability
        else:
            expected_predictability = 0.5
        
        # Calculate actual predictability based on variance
        if len(risk_units) < 2:
            return expected_predictability
        
        cv = np.std(risk_units) / np.mean(risk_units)
        actual_predictability = max(0.0, 1.0 - cv)
        
        # Blend expected and actual
        return (expected_predictability * 0.6 + actual_predictability * 0.4)
    
    def _calculate_capital_efficiency(self, conversions: List[ConversionResult]) -> float:
        """
        Calculate capital efficiency (lower risk units per dollar at risk is better)
        """
        efficiencies = []
        
        for conv in conversions:
            max_loss = abs(conv.metadata.get('max_loss', conv.original_cvar))
            if max_loss > 0:
                efficiency = conv.risk_units / (max_loss / 1000)  # Risk units per $1000 at risk
                efficiencies.append(efficiency)
        
        if not efficiencies:
            return 0.5
        
        avg_efficiency = np.mean(efficiencies)
        
        # Normalize to 0-1 scale (lower is better, so invert)
        # Typical range is 0.5 to 3.0 risk units per $1000
        normalized = max(0.0, min(1.0, (3.0 - avg_efficiency) / 2.5))
        
        return normalized
    
    def generate_risk_benchmarks(self, strategy_type: str) -> Dict[str, float]:
        """
        Generate risk benchmarks for a strategy type
        """
        if strategy_type in self.strategy_characteristics:
            char = self.strategy_characteristics[strategy_type]
            
            # Generate benchmarks based on strategy characteristics
            base_risk_units = self._estimate_base_risk_units(char)
            
            benchmarks = {
                'conservative_risk_units': base_risk_units * 0.7,
                'typical_risk_units': base_risk_units,
                'aggressive_risk_units': base_risk_units * 1.5,
                'max_recommended_risk_units': base_risk_units * 2.0,
                'expected_win_rate': char.win_rate_expectation,
                'expected_profit_loss_ratio': char.profit_loss_ratio,
                'risk_efficiency_target': char.capital_efficiency
            }
        else:
            # Default benchmarks for unknown strategies
            benchmarks = {
                'conservative_risk_units': 1.0,
                'typical_risk_units': 2.0,
                'aggressive_risk_units': 3.5,
                'max_recommended_risk_units': 5.0,
                'expected_win_rate': 0.5,
                'expected_profit_loss_ratio': 1.0,
                'risk_efficiency_target': 0.5
            }
        
        return benchmarks
    
    def _estimate_base_risk_units(self, characteristics: StrategyCharacteristics) -> float:
        """
        Estimate base risk units for a strategy based on its characteristics
        """
        # Start with risk profile base
        risk_profile_base = {
            RiskProfile.CONSERVATIVE: 1.0,
            RiskProfile.MODERATE: 2.0,
            RiskProfile.AGGRESSIVE: 3.0,
            RiskProfile.SPECULATIVE: 4.0
        }
        
        base = risk_profile_base[characteristics.risk_profile]
        
        # Adjust for strategy characteristics
        volatility_adjustment = characteristics.vega_sensitivity * 0.5
        gamma_adjustment = characteristics.gamma_sensitivity * 0.3
        predictability_adjustment = (1 - characteristics.max_loss_predictability) * 0.5
        
        adjusted_base = base + volatility_adjustment + gamma_adjustment + predictability_adjustment
        
        return max(0.5, min(5.0, adjusted_base))
    
    def compare_strategy_risks(self, strategy_list: List[str]) -> Dict[str, StrategyComparison]:
        """
        Compare risk characteristics between multiple strategies
        """
        if len(strategy_list) < 2:
            raise ValueError("Need at least 2 strategies for comparison")
        
        comparisons = {}
        base_strategy = strategy_list[0]
        
        for i in range(1, len(strategy_list)):
            comparison_strategy = strategy_list[i]
            comparison = self._compare_two_strategies(base_strategy, comparison_strategy)
            comparisons[f"{base_strategy}_vs_{comparison_strategy}"] = comparison
        
        return comparisons
    
    def _compare_two_strategies(self, strategy1: str, strategy2: str) -> StrategyComparison:
        """
        Compare two strategies in detail
        """
        # Get profiles (use predefined if dynamic not available)
        profile1 = self.dynamic_profiles.get(strategy1)
        profile2 = self.dynamic_profiles.get(strategy2)
        
        if profile1 and profile2:
            # Use dynamic profiles
            risk_ratio = profile2.avg_risk_units / profile1.avg_risk_units
            efficiency_ratio = profile2.risk_efficiency / profile1.risk_efficiency if profile1.risk_efficiency > 0 else 1.0
            consistency_ratio = profile2.consistency_score / profile1.consistency_score if profile1.consistency_score > 0 else 1.0
            
            # Calculate correlation if both strategies have history
            correlation = self._calculate_strategy_correlation(strategy1, strategy2)
        else:
            # Use predefined characteristics
            char1 = self.strategy_characteristics.get(strategy1)
            char2 = self.strategy_characteristics.get(strategy2)
            
            if char1 and char2:
                base1 = self._estimate_base_risk_units(char1)
                base2 = self._estimate_base_risk_units(char2)
                
                risk_ratio = base2 / base1
                efficiency_ratio = char2.capital_efficiency / char1.capital_efficiency
                consistency_ratio = char2.max_loss_predictability / char1.max_loss_predictability
                correlation = self._estimate_strategy_correlation(char1, char2)
            else:
                # Default comparison
                risk_ratio = 1.0
                efficiency_ratio = 1.0
                consistency_ratio = 1.0
                correlation = 0.3
        
        # Generate recommendation
        recommendation = self._generate_diversification_recommendation(
            risk_ratio, efficiency_ratio, consistency_ratio, correlation
        )
        
        return StrategyComparison(
            base_strategy=strategy1,
            comparison_strategy=strategy2,
            risk_ratio=risk_ratio,
            efficiency_ratio=efficiency_ratio,
            consistency_ratio=consistency_ratio,
            correlation=correlation,
            recommendation=recommendation
        )
    
    def _calculate_strategy_correlation(self, strategy1: str, strategy2: str) -> float:
        """
        Calculate correlation between two strategies based on historical risk units
        """
        if strategy1 not in self.strategy_history or strategy2 not in self.strategy_history:
            return 0.3  # Default correlation
        
        hist1 = self.strategy_history[strategy1]
        hist2 = self.strategy_history[strategy2]
        
        if len(hist1) < 5 or len(hist2) < 5:
            return 0.3  # Insufficient data
        
        # Align data by date (simplified - use most recent overlapping period)
        risk_units1 = [conv.risk_units for conv in hist1[-10:]]
        risk_units2 = [conv.risk_units for conv in hist2[-10:]]
        
        min_length = min(len(risk_units1), len(risk_units2))
        if min_length < 3:
            return 0.3
        
        ru1 = np.array(risk_units1[-min_length:])
        ru2 = np.array(risk_units2[-min_length:])
        
        correlation = np.corrcoef(ru1, ru2)[0, 1]
        
        return correlation if not np.isnan(correlation) else 0.3
    
    def _estimate_strategy_correlation(self, char1: StrategyCharacteristics, char2: StrategyCharacteristics) -> float:
        """
        Estimate correlation between strategies based on their characteristics
        """
        # Strategies in same category have higher correlation
        category_correlation = 0.6 if char1.category == char2.category else 0.2
        
        # Similar Greek sensitivities increase correlation
        vega_similarity = 1 - abs(char1.vega_sensitivity - char2.vega_sensitivity)
        gamma_similarity = 1 - abs(char1.gamma_sensitivity - char2.gamma_sensitivity)
        delta_similarity = 1 - abs(char1.delta_neutrality - char2.delta_neutrality)
        
        greek_correlation = (vega_similarity + gamma_similarity + delta_similarity) / 3 * 0.4
        
        estimated_correlation = category_correlation + greek_correlation
        
        return max(0.0, min(0.9, estimated_correlation))
    
    def _generate_diversification_recommendation(self, 
                                               risk_ratio: float, 
                                               efficiency_ratio: float, 
                                               consistency_ratio: float, 
                                               correlation: float) -> str:
        """
        Generate diversification recommendation based on comparison metrics
        """
        if correlation > 0.7:
            return "High correlation - limited diversification benefit"
        elif correlation < 0.3:
            return "Low correlation - excellent diversification opportunity"
        elif efficiency_ratio > 1.2 and risk_ratio < 1.5:
            return "Good risk-adjusted diversification - consider allocation"
        elif risk_ratio > 2.0:
            return "High risk differential - use for small allocation only"
        elif consistency_ratio > 1.2:
            return "More consistent strategy - good for stable allocation"
        else:
            return "Moderate diversification benefit - standard allocation appropriate"
    
    def get_strategy_profile_summary(self, strategy_name: str) -> Dict[str, any]:
        """
        Get comprehensive profile summary for a strategy
        """
        summary = {"strategy_name": strategy_name}
        
        # Add predefined characteristics if available
        if strategy_name in self.strategy_characteristics:
            char = self.strategy_characteristics[strategy_name]
            summary["predefined_characteristics"] = asdict(char)
        
        # Add dynamic profile if available
        if strategy_name in self.dynamic_profiles:
            profile = self.dynamic_profiles[strategy_name]
            summary["dynamic_profile"] = asdict(profile)
        
        # Add benchmarks
        summary["benchmarks"] = self.generate_risk_benchmarks(strategy_name)
        
        # Add risk efficiency if calculated
        if strategy_name in self.strategy_history:
            summary["risk_efficiency"] = self.calculate_risk_efficiency(
                self.strategy_history[strategy_name]
            )
        
        return summary

# Example usage and testing
def main():
    # Initialize profiler
    profiler = StrategyRiskProfiler()
    
    print("Strategy Risk Profiler - Phase 1 Implementation")
    print("=" * 55)
    
    # Display predefined strategy characteristics
    print("Predefined Strategy Characteristics:")
    print("-" * 40)
    
    for strategy_name, char in profiler.strategy_characteristics.items():
        print(f"\n{strategy_name.upper()}:")
        print(f"  Category: {char.category.value}")
        print(f"  Risk Profile: {char.risk_profile.value}")
        print(f"  Expected Win Rate: {char.win_rate_expectation:.1%}")
        print(f"  Vega Sensitivity: {char.vega_sensitivity:.1f}")
        print(f"  Capital Efficiency: {char.capital_efficiency:.1f}")
    
    # Generate benchmarks for strategies
    print(f"\nRisk Unit Benchmarks:")
    print("-" * 40)
    
    strategies = ['iron_condor', 'straddle', 'credit_spread', 'covered_call']
    for strategy in strategies:
        benchmarks = profiler.generate_risk_benchmarks(strategy)
        print(f"\n{strategy.upper()}:")
        print(f"  Typical Risk Units: {benchmarks['typical_risk_units']:.1f}")
        print(f"  Conservative: {benchmarks['conservative_risk_units']:.1f}")
        print(f"  Aggressive: {benchmarks['aggressive_risk_units']:.1f}")
        print(f"  Expected Win Rate: {benchmarks['expected_win_rate']:.1%}")
    
    # Compare strategies
    print(f"\nStrategy Risk Comparisons:")
    print("-" * 40)
    
    comparison_strategies = ['iron_condor', 'straddle', 'credit_spread']
    comparisons = profiler.compare_strategy_risks(comparison_strategies)
    
    for comparison_name, comparison in comparisons.items():
        print(f"\n{comparison_name}:")
        print(f"  Risk Ratio: {comparison.risk_ratio:.2f}")
        print(f"  Efficiency Ratio: {comparison.efficiency_ratio:.2f}")
        print(f"  Correlation: {comparison.correlation:.2f}")
        print(f"  Recommendation: {comparison.recommendation}")
    
    # Display strategy profiles
    print(f"\nDetailed Strategy Profiles:")
    print("-" * 40)
    
    for strategy in ['iron_condor', 'straddle']:
        profile_summary = profiler.get_strategy_profile_summary(strategy)
        char = profile_summary["predefined_characteristics"]
        benchmarks = profile_summary["benchmarks"]
        
        print(f"\n{strategy.upper()} PROFILE:")
        print(f"  Delta Neutrality: {char['delta_neutrality']:.1f}")
        print(f"  Gamma Sensitivity: {char['gamma_sensitivity']:.1f}")
        print(f"  Theta Dependency: {char['theta_dependency']:.1f}")
        print(f"  Vega Sensitivity: {char['vega_sensitivity']:.1f}")
        print(f"  Max Loss Predictability: {char['max_loss_predictability']:.1f}")
        print(f"  Typical Risk Units: {benchmarks['typical_risk_units']:.1f}")

if __name__ == "__main__":
    main()