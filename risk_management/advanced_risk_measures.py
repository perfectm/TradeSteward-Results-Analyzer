"""
Advanced Risk Measures Module - Phase 2 Implementation
Implements riskfolio-lib's exotic risk measures for options trading portfolios
"""

import numpy as np
import pandas as pd
import riskfolio as rp
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AdvancedRiskCalculator:
    """
    Advanced risk measure calculator using riskfolio-lib's sophisticated metrics
    Specializes in options trading portfolio risk assessment
    """
    
    def __init__(self, returns_data: pd.DataFrame, alpha: float = 0.05):
        """
        Initialize advanced risk calculator
        
        Args:
            returns_data: DataFrame with strategy returns (date index)
            alpha: Confidence level for risk measures (default 5%)
        """
        self.returns_data = returns_data
        self.alpha = alpha
        self.portfolio = None
        self.risk_measures = {}
        
        # Initialize riskfolio Portfolio object
        self._initialize_portfolio()
    
    def _initialize_portfolio(self):
        """Initialize riskfolio-lib Portfolio object"""
        try:
            # Clean returns data
            clean_returns = self.returns_data.dropna()
            
            if len(clean_returns) < 30:
                logger.warning("Insufficient data for robust risk analysis")
                return
            
            # Create Portfolio object
            self.portfolio = rp.Portfolio(returns=clean_returns)
            
            # Calculate basic statistics
            self.portfolio.assets_stats(method_mu='hist', method_cov='hist')
            
            logger.info(f"Portfolio initialized with {len(clean_returns)} observations and {len(clean_returns.columns)} assets")
            
        except Exception as e:
            logger.error(f"Error initializing portfolio: {str(e)}")
            self.portfolio = None
    
    def calculate_var_measures(self, weights: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate Value at Risk based measures
        
        Args:
            weights: Portfolio weights, None for equal weights
            
        Returns:
            Dictionary of VaR measures
        """
        logger.info("Calculating VaR-based risk measures...")
        
        if self.portfolio is None:
            return {}
        
        if weights is None:
            n_assets = len(self.portfolio.returns.columns)
            weights = np.ones(n_assets) / n_assets
        
        portfolio_returns = (self.portfolio.returns * weights).sum(axis=1)
        
        # Standard VaR (Historical)
        var_hist = np.percentile(portfolio_returns, self.alpha * 100)
        
        # Conditional VaR (Expected Shortfall)
        cvar_hist = portfolio_returns[portfolio_returns <= var_hist].mean()
        
        # Parametric VaR (assuming normal distribution)
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()
        var_param = mu + sigma * stats.norm.ppf(self.alpha)
        
        # Modified VaR (Cornish-Fisher expansion)
        skew = stats.skew(portfolio_returns)
        kurt = stats.kurtosis(portfolio_returns)
        z_alpha = stats.norm.ppf(self.alpha)
        
        cf_adjustment = (z_alpha**2 - 1) * skew / 6 + (z_alpha**3 - 3*z_alpha) * kurt / 24
        var_modified = mu + sigma * (z_alpha + cf_adjustment)
        
        var_measures = {
            'VaR_Historical': abs(var_hist),
            'CVaR_Historical': abs(cvar_hist),
            'VaR_Parametric': abs(var_param),
            'VaR_Modified': abs(var_modified),
            'Expected_Shortfall': abs(cvar_hist)
        }
        
        logger.info(f"VaR measures calculated: VaR={var_measures['VaR_Historical']:.4f}, CVaR={var_measures['CVaR_Historical']:.4f}")
        
        return var_measures
    
    def calculate_evar(self, weights: np.ndarray = None, alpha: float = None) -> float:
        """
        Calculate Entropic Value at Risk (EVaR)
        
        Args:
            weights: Portfolio weights
            alpha: Confidence level
            
        Returns:
            EVaR value
        """
        if alpha is None:
            alpha = self.alpha
        
        if self.portfolio is None:
            return 0.0
        
        try:
            if weights is None:
                n_assets = len(self.portfolio.returns.columns)
                weights = np.ones(n_assets) / n_assets
            
            # Use riskfolio's EVaR calculation
            evar = rp.EVaR(self.portfolio.returns, w=weights, alpha=alpha)
            
            logger.info(f"EVaR calculated: {evar:.4f}")
            return float(evar)
            
        except Exception as e:
            logger.warning(f"EVaR calculation failed: {str(e)}")
            # Fallback to manual calculation
            portfolio_returns = (self.portfolio.returns * weights).sum(axis=1)
            return self._manual_evar(portfolio_returns, alpha)
    
    def _manual_evar(self, returns: pd.Series, alpha: float) -> float:
        """Manual EVaR calculation as fallback"""
        try:
            # EVaR is related to the exponential of the log returns
            log_returns = np.log(1 + returns)
            
            # Calculate exponential expectation
            exp_returns = np.exp(log_returns)
            
            # Use optimization to find EVaR
            def evar_objective(t):
                return t + np.log(np.mean(np.exp((log_returns - t) / alpha))) / alpha
            
            result = minimize(evar_objective, x0=np.percentile(log_returns, alpha * 100))
            
            return abs(float(result.fun)) if result.success else 0.0
            
        except Exception as e:
            logger.warning(f"Manual EVaR calculation failed: {str(e)}")
            return 0.0
    
    def calculate_rlvar(self, weights: np.ndarray = None, alpha: float = None, kappa: float = 0.3) -> float:
        """
        Calculate Relativistic Value at Risk (RLVaR)
        
        Args:
            weights: Portfolio weights
            alpha: Confidence level
            kappa: Relativistic parameter
            
        Returns:
            RLVaR value
        """
        if alpha is None:
            alpha = self.alpha
        
        if self.portfolio is None:
            return 0.0
        
        try:
            if weights is None:
                n_assets = len(self.portfolio.returns.columns)
                weights = np.ones(n_assets) / n_assets
            
            # Use riskfolio's RLVaR calculation
            rlvar = rp.RLVaR(self.portfolio.returns, w=weights, alpha=alpha, kappa=kappa)
            
            logger.info(f"RLVaR calculated: {rlvar:.4f}")
            return float(rlvar)
            
        except Exception as e:
            logger.warning(f"RLVaR calculation failed: {str(e)}")
            # Fallback calculation
            portfolio_returns = (self.portfolio.returns * weights).sum(axis=1)
            return self._manual_rlvar(portfolio_returns, alpha, kappa)
    
    def _manual_rlvar(self, returns: pd.Series, alpha: float, kappa: float) -> float:
        """Manual RLVaR calculation as fallback"""
        try:
            # RLVaR uses relativistic entropy
            def relativistic_moment(x, kappa):
                return np.sqrt(1 + (kappa * x)**2) - 1
            
            rel_returns = relativistic_moment(returns.values, kappa)
            
            # Use CVaR on transformed returns
            var_level = np.percentile(rel_returns, alpha * 100)
            rlvar = rel_returns[rel_returns <= var_level].mean()
            
            return abs(float(rlvar))
            
        except Exception as e:
            logger.warning(f"Manual RLVaR calculation failed: {str(e)}")
            return 0.0
    
    def calculate_drawdown_measures(self, weights: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate advanced drawdown-based risk measures
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary of drawdown measures
        """
        logger.info("Calculating drawdown-based risk measures...")
        
        if self.portfolio is None:
            return {}
        
        if weights is None:
            n_assets = len(self.portfolio.returns.columns)
            weights = np.ones(n_assets) / n_assets
        
        # Calculate portfolio returns and cumulative returns
        portfolio_returns = (self.portfolio.returns * weights).sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        # Calculate drawdowns
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        
        # Basic drawdown measures
        max_drawdown = abs(drawdowns.min())
        avg_drawdown = abs(drawdowns.mean())
        
        # Ulcer Index
        ulcer_index = np.sqrt(np.mean(drawdowns**2))
        
        # Conditional Drawdown at Risk (CDaR)
        cdar_level = np.percentile(drawdowns, self.alpha * 100)
        cdar = abs(drawdowns[drawdowns <= cdar_level].mean())
        
        # Calculate EDaR and RLDaR using riskfolio if available
        try:
            edar = abs(float(rp.EDaR(self.portfolio.returns, w=weights, alpha=self.alpha)))
        except:
            edar = cdar  # Fallback to CDaR
        
        try:
            rldar = abs(float(rp.RLDaR(self.portfolio.returns, w=weights, alpha=self.alpha, kappa=0.3)))
        except:
            rldar = cdar  # Fallback to CDaR
        
        drawdown_measures = {
            'Maximum_Drawdown': max_drawdown,
            'Average_Drawdown': avg_drawdown,
            'Ulcer_Index': ulcer_index,
            'CDaR': cdar,
            'EDaR': edar,
            'RLDaR': rldar
        }
        
        logger.info(f"Drawdown measures calculated: MDD={max_drawdown:.4f}, CDaR={cdar:.4f}")
        
        return drawdown_measures
    
    def calculate_dispersion_measures(self, weights: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate dispersion-based risk measures
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary of dispersion measures
        """
        logger.info("Calculating dispersion-based risk measures...")
        
        if self.portfolio is None:
            return {}
        
        if weights is None:
            n_assets = len(self.portfolio.returns.columns)
            weights = np.ones(n_assets) / n_assets
        
        portfolio_returns = (self.portfolio.returns * weights).sum(axis=1)
        
        # Standard deviation
        std_dev = portfolio_returns.std()
        
        # Mean Absolute Deviation (MAD)
        mad = np.mean(np.abs(portfolio_returns - portfolio_returns.mean()))
        
        # Gini Mean Difference
        n = len(portfolio_returns)
        returns_array = portfolio_returns.values
        gmd = 0.0
        
        for i in range(n):
            for j in range(n):
                gmd += abs(returns_array[i] - returns_array[j])
        
        gmd = gmd / (2 * n**2)
        
        # Semi-deviation (downside deviation)
        mean_return = portfolio_returns.mean()
        downside_returns = portfolio_returns[portfolio_returns < mean_return]
        semi_deviation = np.sqrt(np.mean((downside_returns - mean_return)**2))
        
        # Range
        returns_range = portfolio_returns.max() - portfolio_returns.min()
        
        dispersion_measures = {
            'Standard_Deviation': std_dev,
            'Mean_Absolute_Deviation': mad,
            'Gini_Mean_Difference': gmd,
            'Semi_Deviation': semi_deviation,
            'Range': returns_range
        }
        
        logger.info(f"Dispersion measures calculated: Std={std_dev:.4f}, MAD={mad:.4f}")
        
        return dispersion_measures
    
    def calculate_tail_measures(self, weights: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate tail risk measures
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary of tail measures
        """
        logger.info("Calculating tail risk measures...")
        
        if self.portfolio is None:
            return {}
        
        if weights is None:
            n_assets = len(self.portfolio.returns.columns)
            weights = np.ones(n_assets) / n_assets
        
        portfolio_returns = (self.portfolio.returns * weights).sum(axis=1)
        
        # Skewness and Kurtosis
        skewness = stats.skew(portfolio_returns)
        kurtosis = stats.kurtosis(portfolio_returns)
        
        # Tail Gini coefficient
        sorted_returns = np.sort(portfolio_returns.values)
        n = len(sorted_returns)
        tail_threshold = int(self.alpha * n)
        
        if tail_threshold > 0:
            tail_returns = sorted_returns[:tail_threshold]
            tail_gini = 0.0
            
            for i in range(len(tail_returns)):
                for j in range(len(tail_returns)):
                    tail_gini += abs(tail_returns[i] - tail_returns[j])
            
            tail_gini = tail_gini / (2 * len(tail_returns)**2) if len(tail_returns) > 0 else 0.0
        else:
            tail_gini = 0.0
        
        # Lower Partial Moments
        mean_return = portfolio_returns.mean()
        threshold = 0.0  # Use 0 as threshold for downside risk
        
        downside_returns = portfolio_returns[portfolio_returns < threshold]
        
        # First Lower Partial Moment (Omega ratio denominator)
        lpm1 = np.mean(np.maximum(threshold - portfolio_returns, 0))
        
        # Second Lower Partial Moment (Sortino ratio denominator)
        lpm2 = np.sqrt(np.mean(np.maximum(threshold - portfolio_returns, 0)**2))
        
        tail_measures = {
            'Skewness': skewness,
            'Kurtosis': kurtosis,
            'Tail_Gini': tail_gini,
            'Lower_Partial_Moment_1': lpm1,
            'Lower_Partial_Moment_2': lpm2
        }
        
        logger.info(f"Tail measures calculated: Skewness={skewness:.4f}, Kurtosis={kurtosis:.4f}")
        
        return tail_measures
    
    def comprehensive_risk_analysis(self, weights: np.ndarray = None) -> Dict[str, Dict[str, float]]:
        """
        Perform comprehensive risk analysis using all advanced measures
        
        Args:
            weights: Portfolio weights
            
        Returns:
            Dictionary containing all risk measures
        """
        logger.info("Performing comprehensive advanced risk analysis...")
        
        if self.portfolio is None:
            logger.error("Portfolio not initialized")
            return {}
        
        # Calculate all risk measure categories
        risk_analysis = {
            'VaR_Measures': self.calculate_var_measures(weights),
            'Drawdown_Measures': self.calculate_drawdown_measures(weights),
            'Dispersion_Measures': self.calculate_dispersion_measures(weights),
            'Tail_Measures': self.calculate_tail_measures(weights)
        }
        
        # Add exotic measures
        risk_analysis['Exotic_Measures'] = {
            'EVaR': self.calculate_evar(weights),
            'RLVaR': self.calculate_rlvar(weights)
        }
        
        # Calculate risk-adjusted return ratios
        if weights is None:
            n_assets = len(self.portfolio.returns.columns)
            weights = np.ones(n_assets) / n_assets
        
        portfolio_returns = (self.portfolio.returns * weights).sum(axis=1)
        annual_return = portfolio_returns.mean() * 252
        annual_vol = portfolio_returns.std() * np.sqrt(252)
        
        risk_analysis['Performance_Ratios'] = {
            'Sharpe_Ratio': annual_return / annual_vol if annual_vol > 0 else 0,
            'Sortino_Ratio': annual_return / risk_analysis['Dispersion_Measures']['Semi_Deviation'] if risk_analysis['Dispersion_Measures']['Semi_Deviation'] > 0 else 0,
            'Calmar_Ratio': annual_return / risk_analysis['Drawdown_Measures']['Maximum_Drawdown'] if risk_analysis['Drawdown_Measures']['Maximum_Drawdown'] > 0 else 0,
            'Omega_Ratio': (portfolio_returns[portfolio_returns > 0].sum()) / abs(portfolio_returns[portfolio_returns < 0].sum()) if abs(portfolio_returns[portfolio_returns < 0].sum()) > 0 else float('inf')
        }
        
        self.risk_measures = risk_analysis
        logger.info("Comprehensive risk analysis completed")
        
        return risk_analysis
    
    def plot_risk_measures(self, save_path: str = None) -> None:
        """
        Create comprehensive visualization of risk measures
        
        Args:
            save_path: Path to save the plot
        """
        if not self.risk_measures:
            logger.error("No risk measures calculated. Run comprehensive_risk_analysis() first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Advanced Risk Measures Analysis', fontsize=16, fontweight='bold')
        
        # 1. VaR Measures
        var_measures = self.risk_measures['VaR_Measures']
        axes[0, 0].bar(var_measures.keys(), var_measures.values(), color='red', alpha=0.7)
        axes[0, 0].set_title('Value at Risk Measures', fontweight='bold')
        axes[0, 0].set_ylabel('Risk Value')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Drawdown Measures
        drawdown_measures = self.risk_measures['Drawdown_Measures']
        axes[0, 1].bar(drawdown_measures.keys(), drawdown_measures.values(), color='orange', alpha=0.7)
        axes[0, 1].set_title('Drawdown Measures', fontweight='bold')
        axes[0, 1].set_ylabel('Drawdown Value')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Dispersion Measures
        dispersion_measures = self.risk_measures['Dispersion_Measures']
        axes[0, 2].bar(dispersion_measures.keys(), dispersion_measures.values(), color='blue', alpha=0.7)
        axes[0, 2].set_title('Dispersion Measures', fontweight='bold')
        axes[0, 2].set_ylabel('Dispersion Value')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Performance Ratios
        ratios = self.risk_measures['Performance_Ratios']
        # Filter out infinite values for plotting
        finite_ratios = {k: v for k, v in ratios.items() if np.isfinite(v)}
        axes[1, 0].bar(finite_ratios.keys(), finite_ratios.values(), color='green', alpha=0.7)
        axes[1, 0].set_title('Risk-Adjusted Performance Ratios', fontweight='bold')
        axes[1, 0].set_ylabel('Ratio Value')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        # 5. Exotic Measures
        exotic_measures = self.risk_measures['Exotic_Measures']
        axes[1, 1].bar(exotic_measures.keys(), exotic_measures.values(), color='purple', alpha=0.7)
        axes[1, 1].set_title('Exotic Risk Measures', fontweight='bold')
        axes[1, 1].set_ylabel('Risk Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # 6. Risk Comparison Radar Chart
        if self.portfolio is not None:
            portfolio_returns = self.portfolio.returns.mean(axis=1)
            
            # Create data for radar chart
            measures_for_radar = [
                var_measures.get('VaR_Historical', 0),
                drawdown_measures.get('Maximum_Drawdown', 0),
                dispersion_measures.get('Standard_Deviation', 0),
                exotic_measures.get('EVaR', 0),
                abs(self.risk_measures['Tail_Measures'].get('Skewness', 0))
            ]
            
            categories = ['VaR', 'Max DD', 'Vol', 'EVaR', 'Skew']
            
            # Normalize for radar chart (0-1 scale)
            max_val = max(measures_for_radar) if max(measures_for_radar) > 0 else 1
            normalized_measures = [x / max_val for x in measures_for_radar]
            
            angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False)
            normalized_measures += normalized_measures[:1]  # Complete the circle
            angles = np.concatenate((angles, [angles[0]]))
            
            axes[1, 2].plot(angles, normalized_measures, 'o-', linewidth=2, color='red')
            axes[1, 2].fill(angles, normalized_measures, alpha=0.25, color='red')
            axes[1, 2].set_xticks(angles[:-1])
            axes[1, 2].set_xticklabels(categories)
            axes[1, 2].set_title('Risk Profile Radar', fontweight='bold')
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risk measures plot saved to {save_path}")
        
        plt.show()
    
    def generate_risk_report(self) -> Dict:
        """
        Generate detailed risk report with all advanced measures
        
        Returns:
            Comprehensive risk report dictionary
        """
        if not self.risk_measures:
            logger.warning("No risk measures available. Running comprehensive analysis...")
            self.comprehensive_risk_analysis()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'analysis_period': {
                'start': self.returns_data.index.min().strftime('%Y-%m-%d'),
                'end': self.returns_data.index.max().strftime('%Y-%m-%d'),
                'observations': len(self.returns_data)
            },
            'confidence_level': f"{(1-self.alpha)*100:.1f}%",
            'risk_measures': self.risk_measures,
            'summary': {
                'highest_var': max(self.risk_measures.get('VaR_Measures', {}).values()) if self.risk_measures.get('VaR_Measures') else 0,
                'highest_drawdown': max(self.risk_measures.get('Drawdown_Measures', {}).values()) if self.risk_measures.get('Drawdown_Measures') else 0,
                'best_ratio': max([v for v in self.risk_measures.get('Performance_Ratios', {}).values() if np.isfinite(v)]) if self.risk_measures.get('Performance_Ratios') else 0
            }
        }
        
        return report


def main():
    """Example usage of Advanced Risk Measures"""
    
    try:
        # Load data from Phase 1 results
        returns_matrix = pd.read_csv('returns_matrix.csv', index_col=0, parse_dates=True)
        
        print("Advanced Risk Measures Analysis - Phase 2")
        print("=" * 60)
        
        # Initialize calculator
        risk_calc = AdvancedRiskCalculator(returns_matrix, alpha=0.05)
        
        # Run comprehensive analysis
        risk_analysis = risk_calc.comprehensive_risk_analysis()
        
        # Display results
        for category, measures in risk_analysis.items():
            print(f"\n{category.replace('_', ' ').title()}:")
            for measure, value in measures.items():
                if np.isfinite(value):
                    print(f"  • {measure.replace('_', ' ')}: {value:.4f}")
                else:
                    print(f"  • {measure.replace('_', ' ')}: {value}")
        
        # Create visualizations
        risk_calc.plot_risk_measures('advanced_risk_analysis.png')
        
        # Generate report
        report = risk_calc.generate_risk_report()
        
        # Save report
        import json
        with open('advanced_risk_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\n✅ Advanced risk analysis complete!")
        print("📁 Results saved to:")
        print("  • advanced_risk_analysis.png")
        print("  • advanced_risk_report.json")
        
    except Exception as e:
        logger.error(f"Error in advanced risk analysis: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()