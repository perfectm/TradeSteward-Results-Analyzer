"""
Monte Carlo Simulation Framework - Phase 2 Implementation
Advanced portfolio simulation for options trading risk assessment
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import multiprocessing as mp
from functools import partial
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class SimulationParameters:
    """Parameters for Monte Carlo simulation"""
    n_simulations: int = 10000
    time_horizon: int = 252  # Trading days (1 year)
    confidence_levels: List[float] = None
    random_seed: int = 42
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.01, 0.05, 0.10, 0.90, 0.95, 0.99]

class MonteCarloSimulator:
    """
    Advanced Monte Carlo simulation framework for options portfolio risk assessment
    """
    
    def __init__(self, returns_data: pd.DataFrame, factor_data: pd.DataFrame = None):
        """
        Initialize Monte Carlo simulator
        
        Args:
            returns_data: Historical returns data
            factor_data: Factor returns data (optional)
        """
        self.returns_data = returns_data
        self.factor_data = factor_data
        self.simulation_results = None
        self.risk_metrics = None
        
        # Fit return distributions
        self._fit_return_distributions()
    
    def _fit_return_distributions(self):
        """Fit statistical distributions to historical returns"""
        logger.info("Fitting return distributions...")
        
        self.return_stats = {}
        self.distribution_params = {}
        
        for asset in self.returns_data.columns:
            returns = self.returns_data[asset].dropna()
            
            if len(returns) < 30:
                logger.warning(f"Insufficient data for {asset}")
                continue
            
            # Basic statistics
            self.return_stats[asset] = {
                'mean': returns.mean(),
                'std': returns.std(),
                'skew': stats.skew(returns),
                'kurtosis': stats.kurtosis(returns),
                'min': returns.min(),
                'max': returns.max()
            }
            
            # Fit multiple distributions and select best
            distributions = [
                ('normal', stats.norm),
                ('t', stats.t),
                ('skewnorm', stats.skewnorm),
                ('laplace', stats.laplace)
            ]
            
            best_dist = None
            best_params = None
            best_aic = np.inf
            
            for dist_name, dist in distributions:
                try:
                    # Fit distribution
                    params = dist.fit(returns)
                    
                    # Calculate AIC
                    log_likelihood = np.sum(dist.logpdf(returns, *params))
                    aic = 2 * len(params) - 2 * log_likelihood
                    
                    if aic < best_aic:
                        best_aic = aic
                        best_dist = dist_name
                        best_params = params
                        
                except Exception as e:
                    logger.warning(f"Failed to fit {dist_name} for {asset}: {str(e)}")
                    continue
            
            self.distribution_params[asset] = {
                'distribution': best_dist,
                'params': best_params,
                'aic': best_aic
            }
        
        logger.info(f"Fitted distributions for {len(self.distribution_params)} assets")
    
    def generate_correlated_returns(self, n_simulations: int, time_horizon: int, 
                                  correlation_method: str = 'historical') -> np.ndarray:
        """
        Generate correlated random returns using Monte Carlo
        
        Args:
            n_simulations: Number of simulation paths
            time_horizon: Number of time steps
            correlation_method: Method for correlation ('historical', 'exponential', 'shrinkage')
            
        Returns:
            Array of simulated returns [n_simulations, time_horizon, n_assets]
        """
        logger.info(f"Generating {n_simulations} correlated return paths...")
        
        assets = list(self.returns_data.columns)
        n_assets = len(assets)
        
        # Calculate correlation matrix
        if correlation_method == 'historical':
            corr_matrix = self.returns_data.corr().values
        elif correlation_method == 'exponential':
            corr_matrix = self._exponential_correlation()
        elif correlation_method == 'shrinkage':
            corr_matrix = self._shrinkage_correlation()
        else:
            corr_matrix = np.eye(n_assets)
        
        # Ensure positive definite
        corr_matrix = self._make_positive_definite(corr_matrix)
        
        # Cholesky decomposition for correlation
        try:
            L = np.linalg.cholesky(corr_matrix)
        except np.linalg.LinAlgError:
            logger.warning("Cholesky decomposition failed, using eigenvalue decomposition")
            L = self._robust_cholesky(corr_matrix)
        
        # Generate simulations
        simulated_returns = np.zeros((n_simulations, time_horizon, n_assets))
        
        for sim in range(n_simulations):
            # Generate independent random variables
            random_vars = np.zeros((time_horizon, n_assets))
            
            for i, asset in enumerate(assets):
                dist_info = self.distribution_params.get(asset)
                
                if dist_info and dist_info['distribution']:
                    # Use fitted distribution
                    if dist_info['distribution'] == 'normal':
                        random_vars[:, i] = stats.norm.rvs(*dist_info['params'], size=time_horizon)
                    elif dist_info['distribution'] == 't':
                        random_vars[:, i] = stats.t.rvs(*dist_info['params'], size=time_horizon)
                    elif dist_info['distribution'] == 'skewnorm':
                        random_vars[:, i] = stats.skewnorm.rvs(*dist_info['params'], size=time_horizon)
                    elif dist_info['distribution'] == 'laplace':
                        random_vars[:, i] = stats.laplace.rvs(*dist_info['params'], size=time_horizon)
                    else:
                        # Fallback to normal
                        stats_info = self.return_stats[asset]
                        random_vars[:, i] = np.random.normal(stats_info['mean'], stats_info['std'], time_horizon)
                else:
                    # Fallback to normal distribution
                    random_vars[:, i] = np.random.normal(0, 0.01, time_horizon)
            
            # Apply correlation structure
            correlated_returns = random_vars @ L.T
            simulated_returns[sim, :, :] = correlated_returns
        
        logger.info("Correlated returns generation completed")
        return simulated_returns
    
    def _exponential_correlation(self, lambda_param: float = 0.94) -> np.ndarray:
        """Calculate exponentially weighted correlation matrix"""
        returns = self.returns_data.dropna()
        n_assets = len(returns.columns)
        n_obs = len(returns)
        
        # Calculate exponential weights
        weights = np.array([lambda_param**(n_obs-1-i) for i in range(n_obs)])
        weights = weights / weights.sum()
        
        # Weighted covariance
        weighted_returns = returns * np.sqrt(weights.reshape(-1, 1))
        cov_matrix = weighted_returns.T @ weighted_returns
        
        # Convert to correlation
        stds = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(stds, stds)
        
        return corr_matrix
    
    def _shrinkage_correlation(self, shrinkage: float = 0.2) -> np.ndarray:
        """Apply shrinkage to correlation matrix"""
        sample_corr = self.returns_data.corr().values
        n_assets = len(sample_corr)
        target_corr = np.eye(n_assets)  # Identity matrix
        
        shrunk_corr = (1 - shrinkage) * sample_corr + shrinkage * target_corr
        return shrunk_corr
    
    def _make_positive_definite(self, matrix: np.ndarray, min_eigenvalue: float = 1e-8) -> np.ndarray:
        """Ensure matrix is positive definite"""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, min_eigenvalue)
        return eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    def _robust_cholesky(self, matrix: np.ndarray) -> np.ndarray:
        """Robust Cholesky decomposition using eigenvalue decomposition"""
        eigenvals, eigenvecs = np.linalg.eigh(matrix)
        eigenvals = np.maximum(eigenvals, 1e-8)
        return eigenvecs @ np.diag(np.sqrt(eigenvals))
    
    def simulate_portfolio(self, weights: np.ndarray, params: SimulationParameters) -> Dict:
        """
        Simulate portfolio performance using Monte Carlo
        
        Args:
            weights: Portfolio weights
            params: Simulation parameters
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Starting Monte Carlo simulation with {params.n_simulations} paths...")
        
        np.random.seed(params.random_seed)
        
        # Generate return paths
        simulated_returns = self.generate_correlated_returns(
            params.n_simulations, 
            params.time_horizon
        )
        
        # Calculate portfolio returns
        portfolio_returns = np.tensordot(simulated_returns, weights, axes=([2], [0]))
        
        # Calculate cumulative returns
        cumulative_returns = np.cumprod(1 + portfolio_returns, axis=1)
        final_returns = cumulative_returns[:, -1] - 1
        
        # Calculate paths statistics
        simulation_results = {
            'portfolio_returns': portfolio_returns,
            'cumulative_returns': cumulative_returns,
            'final_returns': final_returns,
            'params': params
        }
        
        # Calculate risk metrics
        risk_metrics = self._calculate_simulation_metrics(simulation_results)
        
        self.simulation_results = simulation_results
        self.risk_metrics = risk_metrics
        
        logger.info("Monte Carlo simulation completed")
        return {**simulation_results, 'risk_metrics': risk_metrics}
    
    def _calculate_simulation_metrics(self, simulation_results: Dict) -> Dict:
        """Calculate risk metrics from simulation results"""
        final_returns = simulation_results['final_returns']
        portfolio_returns = simulation_results['portfolio_returns']
        cumulative_returns = simulation_results['cumulative_returns']
        params = simulation_results['params']
        
        # Basic statistics
        metrics = {
            'expected_return': np.mean(final_returns),
            'volatility': np.std(final_returns),
            'skewness': stats.skew(final_returns),
            'kurtosis': stats.kurtosis(final_returns),
            'min_return': np.min(final_returns),
            'max_return': np.max(final_returns)
        }
        
        # VaR and CVaR at different confidence levels
        for confidence in params.confidence_levels:
            var_level = np.percentile(final_returns, confidence * 100)
            cvar_level = np.mean(final_returns[final_returns <= var_level])
            
            metrics[f'VaR_{int(confidence*100)}'] = abs(var_level)
            metrics[f'CVaR_{int(confidence*100)}'] = abs(cvar_level)
        
        # Drawdown analysis
        drawdowns = self._calculate_drawdowns(cumulative_returns)
        metrics.update({
            'max_drawdown_mean': np.mean(np.min(drawdowns, axis=1)),
            'max_drawdown_95': np.percentile(np.min(drawdowns, axis=1), 95),
            'max_drawdown_99': np.percentile(np.min(drawdowns, axis=1), 99),
            'avg_drawdown': np.mean(drawdowns[drawdowns < 0]),
            'drawdown_duration_avg': self._calculate_avg_drawdown_duration(drawdowns)
        })
        
        # Probability metrics
        metrics.update({
            'prob_positive': np.mean(final_returns > 0),
            'prob_loss_5': np.mean(final_returns < -0.05),
            'prob_loss_10': np.mean(final_returns < -0.10),
            'prob_loss_20': np.mean(final_returns < -0.20),
            'prob_gain_10': np.mean(final_returns > 0.10),
            'prob_gain_20': np.mean(final_returns > 0.20)
        })
        
        # Risk-adjusted metrics
        if metrics['volatility'] > 0:
            metrics['sharpe_ratio'] = metrics['expected_return'] / metrics['volatility']
        
        if metrics.get('max_drawdown_mean', 0) > 0:
            metrics['calmar_ratio'] = metrics['expected_return'] / metrics['max_drawdown_mean']
        
        return metrics
    
    def _calculate_drawdowns(self, cumulative_returns: np.ndarray) -> np.ndarray:
        """Calculate drawdowns for all simulation paths"""
        running_max = np.maximum.accumulate(cumulative_returns, axis=1)
        drawdowns = (cumulative_returns - running_max) / running_max
        return drawdowns
    
    def _calculate_avg_drawdown_duration(self, drawdowns: np.ndarray) -> float:
        """Calculate average drawdown duration across simulations"""
        durations = []
        
        for path in drawdowns:
            in_drawdown = False
            current_duration = 0
            path_durations = []
            
            for dd in path:
                if dd < 0:
                    if not in_drawdown:
                        in_drawdown = True
                        current_duration = 1
                    else:
                        current_duration += 1
                else:
                    if in_drawdown:
                        path_durations.append(current_duration)
                        in_drawdown = False
                        current_duration = 0
            
            if in_drawdown:
                path_durations.append(current_duration)
            
            if path_durations:
                durations.extend(path_durations)
        
        return np.mean(durations) if durations else 0
    
    def stress_test(self, weights: np.ndarray, stress_scenarios: Dict[str, Dict]) -> Dict:
        """
        Run stress testing scenarios
        
        Args:
            weights: Portfolio weights
            stress_scenarios: Dictionary of stress scenarios
            
        Returns:
            Stress test results
        """
        logger.info("Running stress testing scenarios...")
        
        stress_results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            logger.info(f"Running stress scenario: {scenario_name}")
            
            # Modify return parameters for stress scenario
            stressed_returns = self._apply_stress_scenario(scenario_params)
            
            # Run simulation with stressed parameters
            original_returns = self.returns_data
            self.returns_data = stressed_returns
            self._fit_return_distributions()
            
            # Simulate with stressed distributions
            params = SimulationParameters(n_simulations=5000, time_horizon=63)  # 3 months
            stress_result = self.simulate_portfolio(weights, params)
            
            stress_results[scenario_name] = {
                'scenario_params': scenario_params,
                'risk_metrics': stress_result['risk_metrics'],
                'final_returns': stress_result['final_returns']
            }
            
            # Restore original data
            self.returns_data = original_returns
            self._fit_return_distributions()
        
        logger.info("Stress testing completed")
        return stress_results
    
    def _apply_stress_scenario(self, scenario_params: Dict) -> pd.DataFrame:
        """Apply stress scenario to returns data"""
        stressed_returns = self.returns_data.copy()
        
        # Apply shifts to mean returns
        if 'mean_shift' in scenario_params:
            for asset, shift in scenario_params['mean_shift'].items():
                if asset in stressed_returns.columns:
                    stressed_returns[asset] += shift
        
        # Apply volatility scaling
        if 'vol_scaling' in scenario_params:
            for asset, scaling in scenario_params['vol_scaling'].items():
                if asset in stressed_returns.columns:
                    mean_return = stressed_returns[asset].mean()
                    stressed_returns[asset] = mean_return + (stressed_returns[asset] - mean_return) * scaling
        
        # Apply correlation stress
        if 'correlation_stress' in scenario_params:
            # Increase correlations during stress
            correlation_factor = scenario_params['correlation_stress']
            # This is a simplified implementation - in practice, would use more sophisticated methods
            
        return stressed_returns
    
    def scenario_analysis(self, weights: np.ndarray, scenarios: Dict[str, np.ndarray]) -> Dict:
        """
        Run specific scenario analysis
        
        Args:
            weights: Portfolio weights
            scenarios: Dictionary of return scenarios
            
        Returns:
            Scenario analysis results
        """
        logger.info("Running scenario analysis...")
        
        scenario_results = {}
        
        for scenario_name, scenario_returns in scenarios.items():
            if len(scenario_returns.shape) == 1:
                # Single period scenario
                portfolio_return = np.dot(scenario_returns, weights)
                scenario_results[scenario_name] = {
                    'portfolio_return': portfolio_return,
                    'individual_returns': dict(zip(self.returns_data.columns, scenario_returns))
                }
            else:
                # Multi-period scenario
                portfolio_returns = np.dot(scenario_returns, weights)
                cumulative_return = np.prod(1 + portfolio_returns) - 1
                
                scenario_results[scenario_name] = {
                    'portfolio_returns': portfolio_returns,
                    'cumulative_return': cumulative_return,
                    'individual_returns': scenario_returns
                }
        
        return scenario_results
    
    def plot_simulation_results(self, save_path: str = None) -> None:
        """
        Create comprehensive visualization of simulation results
        
        Args:
            save_path: Path to save the plot
        """
        if self.simulation_results is None:
            logger.error("No simulation results available. Run simulation first.")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Monte Carlo Simulation Results', fontsize=16, fontweight='bold')
        
        final_returns = self.simulation_results['final_returns']
        cumulative_returns = self.simulation_results['cumulative_returns']
        portfolio_returns = self.simulation_results['portfolio_returns']
        
        # 1. Final Returns Distribution
        axes[0, 0].hist(final_returns, bins=50, alpha=0.7, color='steelblue', density=True)
        axes[0, 0].axvline(np.mean(final_returns), color='red', linestyle='--', label=f'Mean: {np.mean(final_returns):.3f}')
        axes[0, 0].axvline(np.percentile(final_returns, 5), color='orange', linestyle='--', label='5% VaR')
        axes[0, 0].set_title('Final Returns Distribution', fontweight='bold')
        axes[0, 0].set_xlabel('Final Return')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].legend()
        
        # 2. Cumulative Return Paths (sample)
        n_paths_to_plot = min(1000, len(cumulative_returns))
        sample_indices = np.random.choice(len(cumulative_returns), n_paths_to_plot, replace=False)
        
        for i in sample_indices[:100]:  # Plot first 100 for visibility
            axes[0, 1].plot(cumulative_returns[i], alpha=0.1, color='blue')
        
        axes[0, 1].plot(np.mean(cumulative_returns, axis=0), color='red', linewidth=2, label='Mean Path')
        axes[0, 1].plot(np.percentile(cumulative_returns, 5, axis=0), color='orange', linestyle='--', label='5th Percentile')
        axes[0, 1].plot(np.percentile(cumulative_returns, 95, axis=0), color='green', linestyle='--', label='95th Percentile')
        axes[0, 1].set_title('Cumulative Return Paths', fontweight='bold')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Cumulative Return')
        axes[0, 1].legend()
        
        # 3. VaR Confidence Intervals
        confidence_levels = self.simulation_results['params'].confidence_levels
        var_values = [self.risk_metrics[f'VaR_{int(c*100)}'] for c in confidence_levels]
        
        axes[0, 2].bar([f'{int(c*100)}%' for c in confidence_levels], var_values, alpha=0.7, color='red')
        axes[0, 2].set_title('Value at Risk by Confidence Level', fontweight='bold')
        axes[0, 2].set_ylabel('VaR')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Drawdown Distribution
        drawdowns = self._calculate_drawdowns(cumulative_returns)
        max_drawdowns = np.min(drawdowns, axis=1)
        
        axes[1, 0].hist(max_drawdowns, bins=50, alpha=0.7, color='red', density=True)
        axes[1, 0].axvline(np.mean(max_drawdowns), color='black', linestyle='--', label=f'Mean: {np.mean(max_drawdowns):.3f}')
        axes[1, 0].set_title('Maximum Drawdown Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Maximum Drawdown')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        
        # 5. Q-Q Plot for Normality Check
        stats.probplot(final_returns, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot vs Normal Distribution', fontweight='bold')
        
        # 6. Risk-Return Scatter (by simulation)
        sim_returns = np.mean(portfolio_returns, axis=1)
        sim_vols = np.std(portfolio_returns, axis=1)
        
        axes[1, 2].scatter(sim_vols, sim_returns, alpha=0.6, s=20)
        axes[1, 2].set_title('Risk-Return by Simulation Path', fontweight='bold')
        axes[1, 2].set_xlabel('Volatility')
        axes[1, 2].set_ylabel('Average Return')
        
        # 7. Rolling Volatility
        rolling_vol = pd.DataFrame(portfolio_returns).rolling(window=21, axis=1).std()
        
        for i in range(min(100, len(rolling_vol))):
            axes[2, 0].plot(rolling_vol.iloc[i], alpha=0.1, color='blue')
        
        axes[2, 0].plot(rolling_vol.mean(), color='red', linewidth=2, label='Mean Rolling Vol')
        axes[2, 0].set_title('Rolling Volatility (21-day)', fontweight='bold')
        axes[2, 0].set_xlabel('Time Steps')
        axes[2, 0].set_ylabel('Rolling Volatility')
        axes[2, 0].legend()
        
        # 8. Tail Risk Analysis
        tail_returns = final_returns[final_returns <= np.percentile(final_returns, 5)]
        
        axes[2, 1].hist(tail_returns, bins=30, alpha=0.7, color='darkred', density=True)
        axes[2, 1].set_title('Left Tail Distribution (5% worst outcomes)', fontweight='bold')
        axes[2, 1].set_xlabel('Tail Returns')
        axes[2, 1].set_ylabel('Density')
        
        # 9. Risk Metrics Summary
        key_metrics = [
            f"Expected Return: {self.risk_metrics['expected_return']:.3f}",
            f"Volatility: {self.risk_metrics['volatility']:.3f}",
            f"VaR (5%): {self.risk_metrics['VaR_5']:.3f}",
            f"CVaR (5%): {self.risk_metrics['CVaR_5']:.3f}",
            f"Max DD (95%): {self.risk_metrics['max_drawdown_95']:.3f}",
            f"Prob(Loss>10%): {self.risk_metrics['prob_loss_10']:.3f}",
            f"Sharpe Ratio: {self.risk_metrics.get('sharpe_ratio', 0):.3f}"
        ]
        
        axes[2, 2].text(0.1, 0.9, '\n'.join(key_metrics), transform=axes[2, 2].transAxes,
                       fontsize=12, verticalalignment='top', fontfamily='monospace',
                       bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[2, 2].set_title('Key Risk Metrics', fontweight='bold')
        axes[2, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Simulation results plot saved to {save_path}")
        
        plt.show()
    
    def generate_simulation_report(self) -> Dict:
        """
        Generate comprehensive simulation report
        
        Returns:
            Detailed simulation report
        """
        if self.simulation_results is None or self.risk_metrics is None:
            logger.error("No simulation results available")
            return {}
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'simulation_parameters': {
                'n_simulations': self.simulation_results['params'].n_simulations,
                'time_horizon': self.simulation_results['params'].time_horizon,
                'confidence_levels': self.simulation_results['params'].confidence_levels,
                'random_seed': self.simulation_results['params'].random_seed
            },
            'risk_metrics': self.risk_metrics,
            'distribution_fits': {
                asset: {
                    'distribution': info['distribution'],
                    'aic': info['aic']
                } for asset, info in self.distribution_params.items()
            },
            'summary_statistics': {
                'assets_analyzed': len(self.returns_data.columns),
                'historical_observations': len(self.returns_data),
                'simulation_paths': self.simulation_results['params'].n_simulations
            }
        }
        
        return report


def main():
    """Example usage of Monte Carlo Simulator"""
    
    try:
        # Load data from Phase 1
        returns_matrix = pd.read_csv('returns_matrix.csv', index_col=0, parse_dates=True)
        
        print("Monte Carlo Simulation Framework - Phase 2")
        print("=" * 60)
        
        # Initialize simulator
        simulator = MonteCarloSimulator(returns_matrix)
        
        # Set up simulation parameters
        params = SimulationParameters(
            n_simulations=10000,
            time_horizon=252,  # 1 year
            confidence_levels=[0.01, 0.05, 0.10, 0.90, 0.95, 0.99]
        )
        
        # Equal weight portfolio
        n_assets = len(returns_matrix.columns)
        weights = np.ones(n_assets) / n_assets
        
        print(f"Running simulation with {params.n_simulations} paths...")
        
        # Run simulation
        results = simulator.simulate_portfolio(weights, params)
        
        # Display key results
        print(f"\n📊 Simulation Results:")
        print(f"  • Expected Return: {results['risk_metrics']['expected_return']:.3f}")
        print(f"  • Volatility: {results['risk_metrics']['volatility']:.3f}")
        print(f"  • VaR (5%): {results['risk_metrics']['VaR_5']:.3f}")
        print(f"  • CVaR (5%): {results['risk_metrics']['CVaR_5']:.3f}")
        print(f"  • Max Drawdown (95%): {results['risk_metrics']['max_drawdown_95']:.3f}")
        print(f"  • Probability of Loss > 10%: {results['risk_metrics']['prob_loss_10']:.3f}")
        
        # Stress testing
        stress_scenarios = {
            'market_crash': {
                'mean_shift': {asset: -0.002 for asset in returns_matrix.columns},
                'vol_scaling': {asset: 2.0 for asset in returns_matrix.columns}
            },
            'volatility_spike': {
                'vol_scaling': {asset: 3.0 for asset in returns_matrix.columns}
            }
        }
        
        print(f"\n🔥 Running stress tests...")
        stress_results = simulator.stress_test(weights, stress_scenarios)
        
        for scenario, result in stress_results.items():
            print(f"  • {scenario.title()}:")
            print(f"    - VaR (5%): {result['risk_metrics']['VaR_5']:.3f}")
            print(f"    - Expected Return: {result['risk_metrics']['expected_return']:.3f}")
        
        # Create visualizations
        simulator.plot_simulation_results('monte_carlo_analysis.png')
        
        # Generate report
        report = simulator.generate_simulation_report()
        
        # Save report
        with open('monte_carlo_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print("\n✅ Monte Carlo simulation complete!")
        print("📁 Results saved to:")
        print("  • monte_carlo_analysis.png")
        print("  • monte_carlo_report.json")
        
    except Exception as e:
        logger.error(f"Error in Monte Carlo simulation: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()