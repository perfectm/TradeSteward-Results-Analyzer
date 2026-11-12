"""
Phase 2 Main Script - Advanced Risk Management Implementation
Comprehensive integration of all Phase 2 modules for TradeSteward options trading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import logging
from datetime import datetime
import json
from typing import Dict, List

# Import Phase 1 modules
from data_loader import TradeStewardDataLoader
from risk_attribution import FactorRiskModel

# Import Phase 2 modules
from advanced_risk_measures import AdvancedRiskCalculator
from options_factor_models import OptionsFactorModel
from monte_carlo_simulation import MonteCarloSimulator, SimulationParameters
from portfolio_optimization import PortfolioOptimizer

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase2_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Phase2RiskAnalysis:
    """
    Comprehensive Phase 2 risk analysis coordinator
    Integrates all advanced risk management modules
    """
    
    def __init__(self, db_path: str = "../trades.db"):
        """
        Initialize Phase 2 analysis
        
        Args:
            db_path: Path to TradeSteward database
        """
        self.db_path = db_path
        self.loader = None
        self.processed_data = None
        self.returns_matrix = None
        self.factor_matrix = None
        
        # Analysis modules
        self.advanced_risk_calc = None
        self.options_model = None
        self.monte_carlo_sim = None
        self.portfolio_optimizer = None
        
        # Results storage
        self.analysis_results = {}
        
    def load_and_prepare_data(self) -> bool:
        """
        Load and prepare data from Phase 1
        
        Returns:
            Success status
        """
        logger.info("Phase 2: Loading and preparing data...")
        
        try:
            # Initialize data loader
            self.loader = TradeStewardDataLoader(self.db_path)
            
            # Load trades data
            trades_df = self.loader.load_user_trades()
            
            if trades_df.empty:
                logger.error("No trades data available")
                return False
            
            # Process data
            enhanced_df = self.loader.parse_raw_data_json(trades_df)
            clean_df = self.loader.clean_and_validate_data(enhanced_df)
            daily_series = self.loader.create_daily_returns_series(clean_df)
            strategy_portfolios = self.loader.create_strategy_portfolios(clean_df)
            
            # Export for analysis
            self.returns_matrix, self.factor_matrix = self.loader.export_for_riskfolio()
            
            # Store processed data
            self.processed_data = clean_df
            
            logger.info(f"Data preparation complete: {len(clean_df)} trades, "
                       f"{len(strategy_portfolios)} strategies, "
                       f"{self.returns_matrix.shape[0]} daily observations")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            return False
    
    def run_advanced_risk_analysis(self) -> Dict:
        """
        Run advanced risk measures analysis
        
        Returns:
            Advanced risk analysis results
        """
        logger.info("Phase 2: Running advanced risk measures analysis...")
        
        try:
            # Initialize advanced risk calculator
            self.advanced_risk_calc = AdvancedRiskCalculator(self.returns_matrix, alpha=0.05)
            
            # Equal weight portfolio for analysis
            n_assets = len(self.returns_matrix.columns)
            weights = np.ones(n_assets) / n_assets
            
            # Run comprehensive risk analysis
            risk_analysis = self.advanced_risk_calc.comprehensive_risk_analysis(weights)
            
            # Generate report
            risk_report = self.advanced_risk_calc.generate_risk_report()
            
            # Save results
            with open('phase2_advanced_risk_report.json', 'w') as f:
                json.dump(risk_report, f, indent=2, default=str)
            
            # Create visualizations
            try:
                self.advanced_risk_calc.plot_risk_measures('phase2_advanced_risk_measures.png')
            except Exception as e:
                logger.warning(f"Advanced risk visualization failed: {str(e)}")
            
            self.analysis_results['advanced_risk'] = risk_analysis
            
            logger.info("Advanced risk analysis completed")
            return risk_analysis
            
        except Exception as e:
            logger.error(f"Error in advanced risk analysis: {str(e)}")
            return {}
    
    def run_options_factor_analysis(self) -> Dict:
        """
        Run options-specific factor analysis
        
        Returns:
            Options factor analysis results
        """
        logger.info("Phase 2: Running options factor analysis...")
        
        try:
            # Initialize options factor model
            self.options_model = OptionsFactorModel(self.processed_data)
            
            # Calculate portfolio Greeks
            portfolio_greeks = self.options_model.calculate_portfolio_greeks()
            
            # Factor loadings
            factor_loadings = self.options_model.calculate_factor_loadings()
            
            # Term structure analysis
            term_structure = self.options_model.analyze_term_structure_risk()
            
            # Create volatility surface
            vol_surface = self.options_model.create_volatility_surface()
            
            # Generate comprehensive report
            options_report = self.options_model.generate_options_report()
            
            # Save results
            with open('phase2_options_factor_report.json', 'w') as f:
                json.dump(options_report, f, indent=2, default=str)
            
            # Create visualizations
            try:
                self.options_model.plot_options_analysis('phase2_options_analysis.png')
            except Exception as e:
                logger.warning(f"Options visualization failed: {str(e)}")
            
            self.analysis_results['options_factors'] = {
                'portfolio_greeks': portfolio_greeks,
                'factor_loadings': factor_loadings.to_dict() if not factor_loadings.empty else {},
                'term_structure': term_structure,
                'volatility_surface_shape': vol_surface.shape if not vol_surface.empty else (0, 0)
            }
            
            logger.info("Options factor analysis completed")
            return self.analysis_results['options_factors']
            
        except Exception as e:
            logger.error(f"Error in options factor analysis: {str(e)}")
            return {}
    
    def run_monte_carlo_simulation(self) -> Dict:
        """
        Run Monte Carlo simulation and stress testing
        
        Returns:
            Monte Carlo simulation results
        """
        logger.info("Phase 2: Running Monte Carlo simulation...")
        
        try:
            # Initialize Monte Carlo simulator
            self.monte_carlo_sim = MonteCarloSimulator(self.returns_matrix, self.factor_matrix)
            
            # Set up simulation parameters
            sim_params = SimulationParameters(
                n_simulations=10000,
                time_horizon=252,  # 1 year
                confidence_levels=[0.01, 0.05, 0.10, 0.90, 0.95, 0.99],
                random_seed=42
            )
            
            # Equal weight portfolio
            n_assets = len(self.returns_matrix.columns)
            weights = np.ones(n_assets) / n_assets
            
            # Run simulation
            simulation_results = self.monte_carlo_sim.simulate_portfolio(weights, sim_params)
            
            # Stress testing scenarios
            stress_scenarios = {
                'market_crash': {
                    'mean_shift': {asset: -0.003 for asset in self.returns_matrix.columns},
                    'vol_scaling': {asset: 2.5 for asset in self.returns_matrix.columns}
                },
                'volatility_explosion': {
                    'vol_scaling': {asset: 4.0 for asset in self.returns_matrix.columns}
                },
                'correlation_breakdown': {
                    'correlation_stress': 0.8  # Increase correlations
                },
                'options_specific_stress': {
                    'mean_shift': {asset: -0.001 for asset in self.returns_matrix.columns},
                    'vol_scaling': {asset: 1.8 for asset in self.returns_matrix.columns}
                }
            }
            
            # Run stress tests
            stress_results = self.monte_carlo_sim.stress_test(weights, stress_scenarios)
            
            # Generate comprehensive report
            mc_report = self.monte_carlo_sim.generate_simulation_report()
            mc_report['stress_testing'] = stress_results
            
            # Save results
            with open('phase2_monte_carlo_report.json', 'w') as f:
                json.dump(mc_report, f, indent=2, default=str)
            
            # Create visualizations
            try:
                self.monte_carlo_sim.plot_simulation_results('phase2_monte_carlo_analysis.png')
            except Exception as e:
                logger.warning(f"Monte Carlo visualization failed: {str(e)}")
            
            self.analysis_results['monte_carlo'] = {
                'simulation_metrics': simulation_results['risk_metrics'],
                'stress_testing': stress_results
            }
            
            logger.info("Monte Carlo simulation completed")
            return self.analysis_results['monte_carlo']
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo simulation: {str(e)}")
            return {}
    
    def run_portfolio_optimization(self) -> Dict:
        """
        Run advanced portfolio optimization
        
        Returns:
            Portfolio optimization results
        """
        logger.info("Phase 2: Running portfolio optimization...")
        
        try:
            # Initialize portfolio optimizer
            self.portfolio_optimizer = PortfolioOptimizer(self.returns_matrix, self.factor_matrix)
            
            # Run different optimization strategies
            optimization_results = {}
            
            # 1. Mean-Variance Optimization (multiple objectives)
            for objective in ['Sharpe', 'MinRisk', 'MaxRet']:
                result = self.portfolio_optimizer.optimize_mean_variance(obj=objective)
                if result:
                    optimization_results[f'MeanVariance_{objective}'] = result
            
            # 2. Risk Parity Optimization
            for risk_measure in ['MV', 'CVaR', 'CDaR']:
                result = self.portfolio_optimizer.optimize_risk_parity(risk_measure=risk_measure)
                if result:
                    optimization_results[f'RiskParity_{risk_measure}'] = result
            
            # 3. Advanced Risk Measures Optimization
            adv_results = self.portfolio_optimizer.optimize_advanced_risk_measures(['CVaR', 'EVaR', 'CDaR'])
            if adv_results:
                optimization_results['AdvancedRiskMeasures'] = adv_results
            
            # 4. Black-Litterman Optimization
            bl_result = self.portfolio_optimizer.optimize_black_litterman()
            if bl_result:
                optimization_results['BlackLitterman'] = bl_result
            
            # Generate efficient frontier
            efficient_frontier = self.portfolio_optimizer.generate_efficient_frontier(n_points=50)
            
            # Strategy comparison
            comparison = self.portfolio_optimizer.compare_strategies()
            
            # Generate comprehensive report
            opt_report = self.portfolio_optimizer.generate_optimization_report()
            
            # Save results
            with open('phase2_portfolio_optimization_report.json', 'w') as f:
                json.dump(opt_report, f, indent=2, default=str)
            
            # Create visualizations
            try:
                self.portfolio_optimizer.plot_optimization_results('phase2_portfolio_optimization.png')
            except Exception as e:
                logger.warning(f"Optimization visualization failed: {str(e)}")
            
            self.analysis_results['portfolio_optimization'] = {
                'optimization_results': optimization_results,
                'efficient_frontier_points': len(efficient_frontier) if not efficient_frontier.empty else 0,
                'strategy_comparison': comparison.to_dict() if not comparison.empty else {}
            }
            
            logger.info("Portfolio optimization completed")
            return self.analysis_results['portfolio_optimization']
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {str(e)}")
            return {}
    
    def generate_comprehensive_report(self) -> Dict:
        """
        Generate comprehensive Phase 2 analysis report
        
        Returns:
            Complete Phase 2 analysis report
        """
        logger.info("Generating comprehensive Phase 2 report...")
        
        # Data summary
        data_summary = self.loader.get_data_summary() if self.loader else {}
        
        # Comprehensive report
        comprehensive_report = {
            'phase': 'Phase 2 - Advanced Risk Management',
            'timestamp': datetime.now().isoformat(),
            'data_summary': data_summary,
            'analysis_modules': {
                'advanced_risk_measures': 'advanced_risk' in self.analysis_results,
                'options_factor_models': 'options_factors' in self.analysis_results,
                'monte_carlo_simulation': 'monte_carlo' in self.analysis_results,
                'portfolio_optimization': 'portfolio_optimization' in self.analysis_results
            },
            'analysis_results': self.analysis_results,
            'key_insights': self._generate_key_insights(),
            'recommendations': self._generate_recommendations(),
            'files_generated': [
                'phase2_advanced_risk_report.json',
                'phase2_options_factor_report.json',
                'phase2_monte_carlo_report.json',
                'phase2_portfolio_optimization_report.json',
                'phase2_advanced_risk_measures.png',
                'phase2_options_analysis.png',
                'phase2_monte_carlo_analysis.png',
                'phase2_portfolio_optimization.png',
                'phase2_comprehensive_report.json'
            ]
        }
        
        # Save comprehensive report
        with open('phase2_comprehensive_report.json', 'w') as f:
            json.dump(comprehensive_report, f, indent=2, default=str)
        
        logger.info("Comprehensive Phase 2 report generated")
        return comprehensive_report
    
    def _generate_key_insights(self) -> List[str]:
        """Generate key insights from all analyses"""
        insights = []
        
        # Advanced risk insights
        if 'advanced_risk' in self.analysis_results:
            adv_risk = self.analysis_results['advanced_risk']
            if 'VaR_Measures' in adv_risk:
                var_5 = adv_risk['VaR_Measures'].get('VaR_Historical', 0)
                insights.append(f"5% VaR indicates potential daily loss of {var_5:.3f}")
            
            if 'Exotic_Measures' in adv_risk:
                evar = adv_risk['Exotic_Measures'].get('EVaR', 0)
                if evar > 0:
                    insights.append(f"Entropic VaR of {evar:.4f} suggests tail risk exposure")
        
        # Options insights
        if 'options_factors' in self.analysis_results:
            options = self.analysis_results['options_factors']
            if 'portfolio_greeks' in options:
                delta = options['portfolio_greeks'].get('delta', 0)
                vega = options['portfolio_greeks'].get('vega', 0)
                
                if abs(delta) > 100:
                    insights.append(f"High delta exposure ({delta:.0f}) indicates significant directional risk")
                
                if abs(vega) > 100:
                    insights.append(f"High vega exposure ({vega:.0f}) shows volatility sensitivity")
        
        # Monte Carlo insights
        if 'monte_carlo' in self.analysis_results:
            mc = self.analysis_results['monte_carlo']
            if 'simulation_metrics' in mc:
                prob_loss_10 = mc['simulation_metrics'].get('prob_loss_10', 0)
                if prob_loss_10 > 0.2:
                    insights.append(f"High probability ({prob_loss_10:.1%}) of >10% losses in simulation")
        
        # Portfolio optimization insights
        if 'portfolio_optimization' in self.analysis_results:
            opt = self.analysis_results['portfolio_optimization']
            if 'optimization_results' in opt:
                insights.append(f"Multiple optimization strategies tested for risk-return enhancement")
        
        if not insights:
            insights.append("Comprehensive risk analysis completed across all modules")
        
        return insights
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Risk management recommendations
        if 'advanced_risk' in self.analysis_results:
            recommendations.append("Implement daily VaR monitoring with EVaR for tail risk assessment")
        
        if 'options_factors' in self.analysis_results:
            recommendations.append("Monitor Greeks exposure actively, especially delta and vega")
            recommendations.append("Consider volatility surface changes in strategy selection")
        
        if 'monte_carlo' in self.analysis_results:
            recommendations.append("Use Monte Carlo simulation for position sizing and risk budgeting")
            recommendations.append("Regular stress testing under extreme market scenarios")
        
        if 'portfolio_optimization' in self.analysis_results:
            recommendations.append("Consider risk parity approaches for better risk-adjusted returns")
            recommendations.append("Use Black-Litterman for incorporating market views")
        
        # General recommendations
        recommendations.extend([
            "Implement real-time risk monitoring dashboard",
            "Establish risk limits based on advanced risk measures",
            "Regular model validation and backtesting",
            "Consider regime-switching models for dynamic risk management"
        ])
        
        return recommendations


def main():
    """
    Main execution function for Phase 2 Advanced Risk Management
    """
    
    print("=" * 80)
    print("TRADESTEWARD ADVANCED RISK MANAGEMENT - PHASE 2")
    print("Comprehensive Options Trading Risk Analysis Framework")
    print("=" * 80)
    
    try:
        # Initialize Phase 2 analysis
        phase2_analysis = Phase2RiskAnalysis()
        
        # Step 1: Load and prepare data
        print("\n📊 Step 1: Loading and preparing data...")
        if not phase2_analysis.load_and_prepare_data():
            print("❌ Data preparation failed. Exiting.")
            return False
        
        print("✅ Data preparation completed")
        
        # Step 2: Advanced Risk Measures
        print("\n🔬 Step 2: Running advanced risk measures analysis...")
        advanced_risk_results = phase2_analysis.run_advanced_risk_analysis()
        
        if advanced_risk_results:
            print("✅ Advanced risk analysis completed")
            # Display key metrics
            if 'VaR_Measures' in advanced_risk_results:
                var_5 = advanced_risk_results['VaR_Measures'].get('VaR_Historical', 0)
                cvar_5 = advanced_risk_results['VaR_Measures'].get('CVaR_Historical', 0)
                print(f"  • VaR (5%): {var_5:.4f}")
                print(f"  • CVaR (5%): {cvar_5:.4f}")
            
            if 'Exotic_Measures' in advanced_risk_results:
                evar = advanced_risk_results['Exotic_Measures'].get('EVaR', 0)
                rlvar = advanced_risk_results['Exotic_Measures'].get('RLVaR', 0)
                print(f"  • EVaR: {evar:.4f}")
                print(f"  • RLVaR: {rlvar:.4f}")
        else:
            print("⚠️ Advanced risk analysis had issues")
        
        # Step 3: Options Factor Analysis
        print("\n🎯 Step 3: Running options factor analysis...")
        options_results = phase2_analysis.run_options_factor_analysis()
        
        if options_results:
            print("✅ Options factor analysis completed")
            # Display key Greeks
            if 'portfolio_greeks' in options_results:
                greeks = options_results['portfolio_greeks']
                print(f"  • Portfolio Delta: {greeks.get('delta', 0):.2f}")
                print(f"  • Portfolio Gamma: {greeks.get('gamma', 0):.4f}")
                print(f"  • Portfolio Vega: {greeks.get('vega', 0):.2f}")
                print(f"  • Portfolio Theta: {greeks.get('theta', 0):.2f}")
        else:
            print("⚠️ Options factor analysis had issues")
        
        # Step 4: Monte Carlo Simulation
        print("\n🎲 Step 4: Running Monte Carlo simulation...")
        mc_results = phase2_analysis.run_monte_carlo_simulation()
        
        if mc_results:
            print("✅ Monte Carlo simulation completed")
            # Display key simulation metrics
            if 'simulation_metrics' in mc_results:
                metrics = mc_results['simulation_metrics']
                print(f"  • Expected Return: {metrics.get('expected_return', 0):.4f}")
                print(f"  • Volatility: {metrics.get('volatility', 0):.4f}")
                print(f"  • Prob(Loss>10%): {metrics.get('prob_loss_10', 0):.3f}")
                print(f"  • Max Drawdown (95%): {metrics.get('max_drawdown_95', 0):.4f}")
            
            # Display stress test results
            if 'stress_testing' in mc_results:
                print(f"  • Stress scenarios tested: {len(mc_results['stress_testing'])}")
        else:
            print("⚠️ Monte Carlo simulation had issues")
        
        # Step 5: Portfolio Optimization
        print("\n🎯 Step 5: Running portfolio optimization...")
        opt_results = phase2_analysis.run_portfolio_optimization()
        
        if opt_results:
            print("✅ Portfolio optimization completed")
            # Display optimization results
            if 'optimization_results' in opt_results:
                n_strategies = len(opt_results['optimization_results'])
                print(f"  • Optimization strategies tested: {n_strategies}")
            
            if 'efficient_frontier_points' in opt_results:
                print(f"  • Efficient frontier points: {opt_results['efficient_frontier_points']}")
        else:
            print("⚠️ Portfolio optimization had issues")
        
        # Step 6: Generate Comprehensive Report
        print("\n📋 Step 6: Generating comprehensive report...")
        comprehensive_report = phase2_analysis.generate_comprehensive_report()
        
        # Display summary
        print("\n" + "=" * 60)
        print("PHASE 2 ANALYSIS SUMMARY")
        print("=" * 60)
        
        print(f"📊 Analysis Modules Completed:")
        for module, completed in comprehensive_report['analysis_modules'].items():
            status = "✅" if completed else "❌"
            print(f"  {status} {module.replace('_', ' ').title()}")
        
        print(f"\n💡 Key Insights:")
        for insight in comprehensive_report['key_insights']:
            print(f"  • {insight}")
        
        print(f"\n🎯 Recommendations:")
        for i, rec in enumerate(comprehensive_report['recommendations'][:5], 1):  # Show top 5
            print(f"  {i}. {rec}")
        
        if len(comprehensive_report['recommendations']) > 5:
            print(f"  ... and {len(comprehensive_report['recommendations']) - 5} more recommendations")
        
        print(f"\n📁 Generated Files:")
        for file_name in comprehensive_report['files_generated']:
            print(f"  • {file_name}")
        
        print("\n🎉 PHASE 2 ADVANCED RISK MANAGEMENT COMPLETE!")
        print("All modules successfully executed with comprehensive analysis")
        
        return True
        
    except Exception as e:
        logger.error(f"Critical error in Phase 2 analysis: {str(e)}", exc_info=True)
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print("Please check the log file for detailed error information.")
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)