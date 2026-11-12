"""
Portfolio Optimization Module - Phase 2 Implementation  
Advanced portfolio optimization using riskfolio-lib for options trading strategies
"""

import numpy as np
import pandas as pd
import riskfolio as rp
from scipy.optimize import minimize, differential_evolution
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
import logging
from datetime import datetime
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class PortfolioOptimizer:
    """
    Advanced portfolio optimization for options trading strategies
    Uses riskfolio-lib for sophisticated optimization techniques
    """
    
    def __init__(self, returns_data: pd.DataFrame, factor_data: pd.DataFrame = None):
        """
        Initialize portfolio optimizer
        
        Args:
            returns_data: Strategy returns data
            factor_data: Factor returns data (optional)
        """
        self.returns_data = returns_data
        self.factor_data = factor_data
        self.portfolio = None
        self.optimized_weights = {}
        self.efficient_frontier = None
        
        # Initialize riskfolio Portfolio
        self._initialize_portfolio()
    
    def _initialize_portfolio(self):
        """Initialize riskfolio-lib Portfolio object"""
        try:
            # Clean returns data
            clean_returns = self.returns_data.dropna()
            
            if len(clean_returns) < 30:
                logger.warning("Insufficient data for optimization")
                return
            
            # Create Portfolio object
            self.portfolio = rp.Portfolio(returns=clean_returns)
            
            # Calculate asset statistics
            self.portfolio.assets_stats(method_mu='hist', method_cov='hist')
            
            logger.info(f"Portfolio optimizer initialized with {len(clean_returns)} observations")
            
        except Exception as e:
            logger.error(f"Error initializing portfolio optimizer: {str(e)}")
            self.portfolio = None
    
    def optimize_mean_variance(self, obj='Sharpe', rf=0.0, constraints=None) -> Dict:
        """
        Perform mean-variance optimization
        
        Args:
            obj: Objective function ('Sharpe', 'MinRisk', 'MaxRet', 'Utility')
            rf: Risk-free rate
            constraints: Additional constraints dictionary
            
        Returns:
            Optimization results
        """
        logger.info(f"Running mean-variance optimization with objective: {obj}")
        
        if self.portfolio is None:
            return {}
        
        try:
            # Set optimization parameters
            self.portfolio.alpha = 0.05  # Significance level for risk measures
            
            # Apply constraints if provided
            if constraints:
                self._apply_constraints(constraints)
            
            # Optimize portfolio
            if obj == 'Sharpe':
                weights = self.portfolio.optimization(model='Classic', rm='MV', obj='Sharpe', rf=rf)
            elif obj == 'MinRisk':
                weights = self.portfolio.optimization(model='Classic', rm='MV', obj='MinRisk')
            elif obj == 'MaxRet':
                weights = self.portfolio.optimization(model='Classic', rm='MV', obj='MaxRet')
            elif obj == 'Utility':
                weights = self.portfolio.optimization(model='Classic', rm='MV', obj='Utility', l=2)
            else:
                logger.error(f"Unknown objective: {obj}")
                return {}
            
            # Calculate portfolio metrics
            portfolio_return = (weights.T @ self.portfolio.mu_)[0, 0] * 252
            portfolio_risk = np.sqrt(weights.T @ self.portfolio.cov @ weights)[0, 0] * np.sqrt(252)
            sharpe_ratio = (portfolio_return - rf) / portfolio_risk if portfolio_risk > 0 else 0
            
            optimization_result = {
                'objective': obj,
                'weights': weights.to_dict()[weights.columns[0]],
                'expected_return': portfolio_return,
                'volatility': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'risk_free_rate': rf
            }
            
            self.optimized_weights[obj] = optimization_result
            
            logger.info(f"Optimization complete - Return: {portfolio_return:.3f}, Risk: {portfolio_risk:.3f}, Sharpe: {sharpe_ratio:.3f}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in mean-variance optimization: {str(e)}")
            return {}
    
    def optimize_risk_parity(self, risk_measure='MV') -> Dict:
        """
        Perform risk parity optimization
        
        Args:
            risk_measure: Risk measure ('MV', 'MAD', 'CVaR', 'EVaR', 'CDaR')
            
        Returns:
            Risk parity optimization results
        """
        logger.info(f"Running risk parity optimization with risk measure: {risk_measure}")
        
        if self.portfolio is None:
            return {}
        
        try:
            # Risk parity optimization
            weights = self.portfolio.rp_optimization(model='Classic', rm=risk_measure, rf=0.0)
            
            # Calculate portfolio metrics
            portfolio_return = (weights.T @ self.portfolio.mu_)[0, 0] * 252
            
            if risk_measure == 'MV':
                portfolio_risk = np.sqrt(weights.T @ self.portfolio.cov @ weights)[0, 0] * np.sqrt(252)
            else:
                # For other risk measures, use riskfolio's calculation
                portfolio_risk = rp.Sharpe(self.portfolio.returns, weights)[0]
            
            # Calculate risk contributions
            risk_contributions = self._calculate_risk_contributions(weights, risk_measure)
            
            optimization_result = {
                'objective': 'RiskParity',
                'risk_measure': risk_measure,
                'weights': weights.to_dict()[weights.columns[0]],
                'expected_return': portfolio_return,
                'risk_metric': portfolio_risk,
                'risk_contributions': risk_contributions
            }
            
            self.optimized_weights['RiskParity'] = optimization_result
            
            logger.info(f"Risk parity optimization complete - Risk measure: {portfolio_risk:.3f}")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in risk parity optimization: {str(e)}")
            return {}
    
    def optimize_advanced_risk_measures(self, risk_measures=['CVaR', 'EVaR', 'CDaR']) -> Dict:
        """
        Optimize using advanced risk measures
        
        Args:
            risk_measures: List of advanced risk measures to optimize
            
        Returns:
            Dictionary of optimization results for each risk measure
        """
        logger.info(f"Running optimization with advanced risk measures: {risk_measures}")
        
        if self.portfolio is None:
            return {}
        
        results = {}
        
        for rm in risk_measures:
            try:
                logger.info(f"Optimizing with risk measure: {rm}")
                
                # Optimize for minimum risk
                weights = self.portfolio.optimization(model='Classic', rm=rm, obj='MinRisk')
                
                # Calculate portfolio metrics
                portfolio_return = (weights.T @ self.portfolio.mu_)[0, 0] * 252
                
                # Calculate risk measure value
                if rm == 'CVaR':
                    risk_value = rp.CVaR(self.portfolio.returns, weights, alpha=self.portfolio.alpha)[0]
                elif rm == 'EVaR':
                    risk_value = rp.EVaR(self.portfolio.returns, weights, alpha=self.portfolio.alpha)[0]
                elif rm == 'CDaR':
                    risk_value = rp.CDaR(self.portfolio.returns, weights, alpha=self.portfolio.alpha)[0]
                else:
                    risk_value = 0.0
                
                results[rm] = {
                    'risk_measure': rm,
                    'weights': weights.to_dict()[weights.columns[0]],
                    'expected_return': portfolio_return,
                    'risk_value': abs(risk_value),
                    'alpha': self.portfolio.alpha
                }
                
                logger.info(f"{rm} optimization complete - Risk: {abs(risk_value):.4f}")
                
            except Exception as e:
                logger.warning(f"Error optimizing {rm}: {str(e)}")
                continue
        
        self.optimized_weights['AdvancedRisk'] = results
        return results
    
    def optimize_black_litterman(self, views=None, tau=0.05) -> Dict:
        """
        Perform Black-Litterman optimization
        
        Args:
            views: Market views dictionary
            tau: Scaling factor for uncertainty
            
        Returns:
            Black-Litterman optimization results
        """
        logger.info("Running Black-Litterman optimization")
        
        if self.portfolio is None:
            return {}
        
        try:
            # Market capitalization weights (equal for simplicity)
            w_market = np.ones((len(self.portfolio.returns.columns), 1)) / len(self.portfolio.returns.columns)
            
            # Black-Litterman optimization
            if views is None:
                # Default views - assuming slight positive bias for historical performers
                P = np.eye(len(self.portfolio.returns.columns))  # View matrix
                Q = self.portfolio.mu_.values * 1.2  # 20% more optimistic than historical
                Omega = np.diag(np.diag(self.portfolio.cov.values)) * 0.1  # View uncertainty
            else:
                P, Q, Omega = self._construct_views_matrices(views)
            
            # Calculate Black-Litterman inputs
            bl_mu, bl_cov = rp.black_litterman(
                self.portfolio.returns, w_market, P, Q, Omega, tau=tau
            )
            
            # Update portfolio with BL estimates
            self.portfolio.mu = pd.DataFrame(bl_mu, index=self.portfolio.returns.columns)
            self.portfolio.cov = pd.DataFrame(bl_cov, 
                                            index=self.portfolio.returns.columns,
                                            columns=self.portfolio.returns.columns)
            
            # Optimize with updated estimates
            weights = self.portfolio.optimization(model='Classic', rm='MV', obj='Sharpe', rf=0.0)
            
            # Calculate metrics
            portfolio_return = (weights.T @ bl_mu)[0, 0] * 252
            portfolio_risk = np.sqrt(weights.T @ bl_cov @ weights)[0, 0] * np.sqrt(252)
            sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
            
            bl_result = {
                'objective': 'BlackLitterman',
                'weights': weights.to_dict()[weights.columns[0]],
                'expected_return': portfolio_return,
                'volatility': portfolio_risk,
                'sharpe_ratio': sharpe_ratio,
                'tau': tau,
                'views_applied': views is not None
            }
            
            self.optimized_weights['BlackLitterman'] = bl_result
            
            logger.info(f"Black-Litterman optimization complete - Sharpe: {sharpe_ratio:.3f}")
            
            return bl_result
            
        except Exception as e:
            logger.error(f"Error in Black-Litterman optimization: {str(e)}")
            return {}
    
    def generate_efficient_frontier(self, n_points=50, risk_range=None) -> pd.DataFrame:
        """
        Generate efficient frontier
        
        Args:
            n_points: Number of points on the frontier
            risk_range: Tuple of (min_risk, max_risk), None for automatic
            
        Returns:
            DataFrame with efficient frontier points
        """
        logger.info(f"Generating efficient frontier with {n_points} points")
        
        if self.portfolio is None:
            return pd.DataFrame()
        
        try:
            # Calculate minimum risk portfolio
            w_min_risk = self.portfolio.optimization(model='Classic', rm='MV', obj='MinRisk')
            min_risk = np.sqrt(w_min_risk.T @ self.portfolio.cov @ w_min_risk)[0, 0] * np.sqrt(252)
            
            # Calculate maximum return portfolio
            w_max_ret = self.portfolio.optimization(model='Classic', rm='MV', obj='MaxRet')
            max_risk = np.sqrt(w_max_ret.T @ self.portfolio.cov @ w_max_ret)[0, 0] * np.sqrt(252)
            
            # Set risk range
            if risk_range is None:
                risk_range = (min_risk, max_risk * 1.5)
            
            # Generate frontier points
            risk_levels = np.linspace(risk_range[0], risk_range[1], n_points)
            frontier_points = []
            
            for target_risk in risk_levels:
                try:
                    # Optimize for maximum return given risk constraint
                    self.portfolio.upperdev = target_risk / np.sqrt(252)
                    weights = self.portfolio.optimization(model='Classic', rm='MV', obj='MaxRet')
                    
                    portfolio_return = (weights.T @ self.portfolio.mu_)[0, 0] * 252
                    portfolio_risk = np.sqrt(weights.T @ self.portfolio.cov @ weights)[0, 0] * np.sqrt(252)
                    sharpe_ratio = portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
                    
                    frontier_points.append({
                        'Risk': portfolio_risk,
                        'Return': portfolio_return,
                        'Sharpe': sharpe_ratio,
                        'Weights': weights.to_dict()[weights.columns[0]]
                    })
                    
                except Exception as e:
                    logger.warning(f"Failed to optimize for risk level {target_risk:.3f}: {str(e)}")
                    continue
            
            # Reset constraint
            self.portfolio.upperdev = None
            
            frontier_df = pd.DataFrame(frontier_points)
            self.efficient_frontier = frontier_df
            
            logger.info(f"Efficient frontier generated with {len(frontier_df)} valid points")
            
            return frontier_df
            
        except Exception as e:
            logger.error(f"Error generating efficient frontier: {str(e)}")
            return pd.DataFrame()
    
    def _apply_constraints(self, constraints: Dict):
        """Apply optimization constraints"""
        if 'min_weights' in constraints:
            self.portfolio.lowerret = constraints['min_weights']
        
        if 'max_weights' in constraints:
            self.portfolio.upperret = constraints['max_weights']
        
        if 'budget_constraint' in constraints:
            self.portfolio.budget = constraints['budget_constraint']
        
        if 'group_constraints' in constraints:
            # Implementation depends on specific group constraint structure
            pass
    
    def _calculate_risk_contributions(self, weights: pd.DataFrame, risk_measure: str) -> Dict:
        """Calculate risk contributions for each asset"""
        try:
            if risk_measure == 'MV':
                # Variance risk contributions
                portfolio_var = weights.T @ self.portfolio.cov @ weights
                marginal_contributions = 2 * self.portfolio.cov @ weights
                risk_contributions = weights.values.flatten() * marginal_contributions.values.flatten()
                risk_contributions = risk_contributions / portfolio_var.iloc[0, 0]
            else:
                # For other risk measures, use simplified calculation
                risk_contributions = weights.values.flatten()
            
            return dict(zip(self.portfolio.returns.columns, risk_contributions))
            
        except Exception as e:
            logger.warning(f"Error calculating risk contributions: {str(e)}")
            return {}
    
    def _construct_views_matrices(self, views: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Construct Black-Litterman views matrices"""
        assets = self.portfolio.returns.columns
        n_assets = len(assets)
        n_views = len(views)
        
        P = np.zeros((n_views, n_assets))
        Q = np.zeros((n_views, 1))
        Omega = np.zeros((n_views, n_views))
        
        for i, (view_name, view_data) in enumerate(views.items()):
            # P matrix - which assets the view applies to
            for asset, weight in view_data.get('assets', {}).items():
                if asset in assets:
                    P[i, assets.get_loc(asset)] = weight
            
            # Q matrix - expected returns
            Q[i, 0] = view_data.get('expected_return', 0)
            
            # Omega matrix - view uncertainty
            Omega[i, i] = view_data.get('confidence', 0.1)
        
        return P, Q, Omega
    
    def compare_strategies(self) -> pd.DataFrame:
        """
        Compare different optimization strategies
        
        Returns:
            DataFrame comparing optimization results
        """
        logger.info("Comparing optimization strategies")
        
        if not self.optimized_weights:
            logger.warning("No optimization results available")
            return pd.DataFrame()
        
        comparison_data = []
        
        for strategy_name, results in self.optimized_weights.items():
            if strategy_name == 'AdvancedRisk':
                # Handle advanced risk measures
                for risk_measure, result in results.items():
                    comparison_data.append({
                        'Strategy': f"{strategy_name}_{risk_measure}",
                        'Expected_Return': result.get('expected_return', 0),
                        'Risk_Measure': result.get('risk_value', 0),
                        'Max_Weight': max(result['weights'].values()) if result['weights'] else 0,
                        'Min_Weight': min(result['weights'].values()) if result['weights'] else 0,
                        'Number_of_Assets': sum(1 for w in result['weights'].values() if w > 0.001)
                    })
            else:
                comparison_data.append({
                    'Strategy': strategy_name,
                    'Expected_Return': results.get('expected_return', 0),
                    'Risk_Measure': results.get('volatility', results.get('risk_metric', 0)),
                    'Sharpe_Ratio': results.get('sharpe_ratio', 0),
                    'Max_Weight': max(results['weights'].values()) if results['weights'] else 0,
                    'Min_Weight': min(results['weights'].values()) if results['weights'] else 0,
                    'Number_of_Assets': sum(1 for w in results['weights'].values() if w > 0.001)
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df
    
    def plot_optimization_results(self, save_path: str = None) -> None:
        """
        Create comprehensive visualization of optimization results
        
        Args:
            save_path: Path to save the plot
        """
        if not self.optimized_weights:
            logger.error("No optimization results available for plotting")
            return
        
        # Determine the number of subplots needed
        n_strategies = sum(len(v) if isinstance(v, dict) and 'AdvancedRisk' in k else 1 
                          for k, v in self.optimized_weights.items())
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Portfolio Optimization Results', fontsize=16, fontweight='bold')
        
        # 1. Weight comparison across strategies
        ax1 = axes[0, 0]
        strategies_to_plot = []
        weights_to_plot = []
        
        for strategy_name, results in self.optimized_weights.items():
            if strategy_name == 'AdvancedRisk':
                for risk_measure, result in results.items():
                    strategies_to_plot.append(f"{strategy_name}_{risk_measure}")
                    weights_to_plot.append(list(result['weights'].values()))
            else:
                strategies_to_plot.append(strategy_name)
                weights_to_plot.append(list(results['weights'].values()))
        
        if weights_to_plot:
            weights_df = pd.DataFrame(weights_to_plot, 
                                    index=strategies_to_plot,
                                    columns=self.returns_data.columns)
            
            # Plot as heatmap
            sns.heatmap(weights_df, annot=True, cmap='RdYlBu', center=0, ax=ax1, fmt='.3f')
            ax1.set_title('Portfolio Weights by Strategy', fontweight='bold')
            ax1.set_ylabel('Optimization Strategy')
        
        # 2. Risk-Return scatter
        ax2 = axes[0, 1]
        returns = []
        risks = []
        strategy_names = []
        
        for strategy_name, results in self.optimized_weights.items():
            if strategy_name == 'AdvancedRisk':
                for risk_measure, result in results.items():
                    returns.append(result.get('expected_return', 0))
                    risks.append(result.get('risk_value', 0))
                    strategy_names.append(f"{strategy_name}_{risk_measure}")
            else:
                returns.append(results.get('expected_return', 0))
                risks.append(results.get('volatility', results.get('risk_metric', 0)))
                strategy_names.append(strategy_name)
        
        if returns and risks:
            scatter = ax2.scatter(risks, returns, s=100, alpha=0.7)
            
            # Add labels
            for i, name in enumerate(strategy_names):
                ax2.annotate(name, (risks[i], returns[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            ax2.set_title('Risk-Return Profile', fontweight='bold')
            ax2.set_xlabel('Risk')
            ax2.set_ylabel('Expected Return')
            ax2.grid(True, alpha=0.3)
        
        # 3. Efficient Frontier (if available)
        ax3 = axes[0, 2]
        if self.efficient_frontier is not None and not self.efficient_frontier.empty:
            ax3.plot(self.efficient_frontier['Risk'], self.efficient_frontier['Return'], 
                    'b-', linewidth=2, label='Efficient Frontier')
            
            # Plot optimized portfolios
            if returns and risks:
                ax3.scatter(risks, returns, s=100, alpha=0.8, c='red', label='Optimized Portfolios')
            
            ax3.set_title('Efficient Frontier', fontweight='bold')
            ax3.set_xlabel('Risk (Volatility)')
            ax3.set_ylabel('Expected Return')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Efficient Frontier\nGenerated', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('Efficient Frontier', fontweight='bold')
        
        # 4. Weight concentration
        ax4 = axes[1, 0]
        if weights_to_plot:
            concentrations = []
            for weights in weights_to_plot:
                # Calculate Herfindahl index (concentration measure)
                herfindahl = sum(w**2 for w in weights)
                concentrations.append(herfindahl)
            
            ax4.bar(strategies_to_plot, concentrations, alpha=0.7)
            ax4.set_title('Portfolio Concentration (Herfindahl Index)', fontweight='bold')
            ax4.set_ylabel('Concentration')
            ax4.tick_params(axis='x', rotation=45)
        
        # 5. Sharpe Ratios
        ax5 = axes[1, 1]
        sharpe_ratios = []
        sharpe_names = []
        
        for strategy_name, results in self.optimized_weights.items():
            if strategy_name != 'AdvancedRisk' and 'sharpe_ratio' in results:
                sharpe_ratios.append(results['sharpe_ratio'])
                sharpe_names.append(strategy_name)
        
        if sharpe_ratios:
            ax5.bar(sharpe_names, sharpe_ratios, alpha=0.7, color='green')
            ax5.set_title('Sharpe Ratios by Strategy', fontweight='bold')
            ax5.set_ylabel('Sharpe Ratio')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Number of active positions
        ax6 = axes[1, 2]
        if weights_to_plot:
            active_positions = []
            for weights in weights_to_plot:
                active_count = sum(1 for w in weights if w > 0.001)  # Count non-trivial weights
                active_positions.append(active_count)
            
            ax6.bar(strategies_to_plot, active_positions, alpha=0.7, color='orange')
            ax6.set_title('Number of Active Positions', fontweight='bold')
            ax6.set_ylabel('Count')
            ax6.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Optimization results plot saved to {save_path}")
        
        plt.show()
    
    def generate_optimization_report(self) -> Dict:
        """
        Generate comprehensive optimization report
        
        Returns:
            Detailed optimization report
        """
        logger.info("Generating portfolio optimization report")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'optimization_results': self.optimized_weights,
            'comparison_table': self.compare_strategies().to_dict() if not self.compare_strategies().empty else {},
            'efficient_frontier': self.efficient_frontier.to_dict() if self.efficient_frontier is not None else {},
            'summary': {
                'strategies_optimized': len(self.optimized_weights),
                'assets_available': len(self.returns_data.columns),
                'optimization_period': {
                    'start': self.returns_data.index.min().strftime('%Y-%m-%d'),
                    'end': self.returns_data.index.max().strftime('%Y-%m-%d')
                }
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization-based recommendations"""
        recommendations = []
        
        if not self.optimized_weights:
            return ["No optimization results available for recommendations"]
        
        # Compare Sharpe ratios
        sharpe_ratios = {}
        for strategy, results in self.optimized_weights.items():
            if strategy != 'AdvancedRisk' and 'sharpe_ratio' in results:
                sharpe_ratios[strategy] = results['sharpe_ratio']
        
        if sharpe_ratios:
            best_sharpe = max(sharpe_ratios, key=sharpe_ratios.get)
            recommendations.append(f"Best risk-adjusted performance: {best_sharpe} strategy (Sharpe: {sharpe_ratios[best_sharpe]:.3f})")
        
        # Check concentration
        comparison_df = self.compare_strategies()
        if not comparison_df.empty:
            high_concentration = comparison_df[comparison_df['Max_Weight'] > 0.5]
            if not high_concentration.empty:
                recommendations.append("High concentration detected in some strategies. Consider diversification.")
            
            # Check number of active positions
            low_diversification = comparison_df[comparison_df['Number_of_Assets'] < 3]
            if not low_diversification.empty:
                recommendations.append("Some strategies have very few active positions. Consider broader diversification.")
        
        if not recommendations:
            recommendations.append("Portfolio optimization results appear balanced across all strategies.")
        
        return recommendations


def main():
    """Example usage of Portfolio Optimizer"""
    
    try:
        # Load data from Phase 1
        returns_matrix = pd.read_csv('returns_matrix.csv', index_col=0, parse_dates=True)
        
        print("Portfolio Optimization - Phase 2")
        print("=" * 60)
        
        # Initialize optimizer
        optimizer = PortfolioOptimizer(returns_matrix)
        
        # Run different optimization strategies
        print("🎯 Running optimization strategies...")
        
        # 1. Mean-Variance Optimization
        mv_result = optimizer.optimize_mean_variance(obj='Sharpe')
        print(f"  • Mean-Variance (Sharpe): {mv_result.get('sharpe_ratio', 0):.3f}")
        
        # 2. Risk Parity
        rp_result = optimizer.optimize_risk_parity(risk_measure='MV')
        print(f"  • Risk Parity: {rp_result.get('expected_return', 0):.3f} return")
        
        # 3. Advanced Risk Measures
        adv_results = optimizer.optimize_advanced_risk_measures(['CVaR', 'EVaR'])
        for measure, result in adv_results.items():
            print(f"  • {measure} optimization: {result.get('risk_value', 0):.4f} risk")
        
        # 4. Black-Litterman
        bl_result = optimizer.optimize_black_litterman()
        print(f"  • Black-Litterman: {bl_result.get('sharpe_ratio', 0):.3f} Sharpe")
        
        # Generate efficient frontier
        print("\n📈 Generating efficient frontier...")
        frontier = optimizer.generate_efficient_frontier(n_points=30)
        print(f"  • Efficient frontier: {len(frontier)} points generated")
        
        # Strategy comparison
        comparison = optimizer.compare_strategies()
        print(f"\n📊 Strategy Comparison:")
        print(comparison.round(4))
        
        # Create visualizations
        optimizer.plot_optimization_results('portfolio_optimization_analysis.png')
        
        # Generate report
        report = optimizer.generate_optimization_report()
        
        # Save report
        with open('portfolio_optimization_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display recommendations
        print(f"\n💡 Optimization Recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print("\n✅ Portfolio optimization complete!")
        print("📁 Results saved to:")
        print("  • portfolio_optimization_analysis.png")
        print("  • portfolio_optimization_report.json")
        
    except Exception as e:
        logger.error(f"Error in portfolio optimization: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()