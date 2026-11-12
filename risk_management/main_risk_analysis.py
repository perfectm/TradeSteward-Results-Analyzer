"""
Main Risk Analysis Script - Phase 1 Implementation
Section 6 Priority: Risk Attribution and Decomposition for TradeSteward Options Trading
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import TradeStewardDataLoader
from risk_attribution import FactorRiskModel
import json
import logging
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('risk_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    """
    Main execution function for TradeSteward Risk Attribution Analysis
    Implements Phase 1 with Section 6 (Risk Attribution and Decomposition) priority
    """
    
    print("=" * 80)
    print("TRADESTEWARD RISK ATTRIBUTION ANALYSIS - PHASE 1")
    print("Section 6 Priority: Risk Attribution and Decomposition")
    print("=" * 80)
    
    try:
        # Step 1: Initialize Data Loader
        logger.info("Step 1: Initializing TradeSteward Data Loader...")
        loader = TradeStewardDataLoader(db_path="../trades.db")
        
        # Step 2: Load and Process Data
        logger.info("Step 2: Loading trades from database...")
        trades_df = loader.load_user_trades()
        
        if trades_df.empty:
            logger.error("No trades found in database. Please ensure the database contains trade data.")
            print("\n❌ ERROR: No trades found in database.")
            print("Please run the main application and upload CSV data first.")
            return
        
        print(f"✅ Loaded {len(trades_df)} trades from database")
        
        # Step 3: Parse JSON Data
        logger.info("Step 3: Parsing raw trade data...")
        enhanced_df = loader.parse_raw_data_json(trades_df)
        print(f"✅ Enhanced {len(enhanced_df)} trades with complete option leg data")
        
        # Step 4: Clean and Validate Data
        logger.info("Step 4: Cleaning and validating data...")
        clean_df = loader.clean_and_validate_data(enhanced_df)
        print(f"✅ Data cleaning complete: {len(clean_df)} valid trades")
        
        # Step 5: Create Time Series
        logger.info("Step 5: Creating daily P&L time series...")
        daily_series = loader.create_daily_returns_series(clean_df)
        print(f"✅ Created daily series with {len(daily_series)} days")
        
        # Step 6: Create Strategy Portfolios
        logger.info("Step 6: Creating strategy-based portfolios...")
        strategy_portfolios = loader.create_strategy_portfolios(clean_df)
        print(f"✅ Created {len(strategy_portfolios)} strategy portfolios")
        
        if not strategy_portfolios:
            logger.error("No strategy portfolios created")
            return
        
        # Step 7: Export for Riskfolio Analysis
        logger.info("Step 7: Preparing data for risk attribution analysis...")
        returns_matrix, factor_matrix = loader.export_for_riskfolio()
        print(f"✅ Returns matrix: {returns_matrix.shape}, Factor matrix: {factor_matrix.shape}")
        
        # Save processed data
        returns_matrix.to_csv('returns_matrix.csv')
        factor_matrix.to_csv('factor_matrix.csv')
        logger.info("Processed data saved to CSV files")
        
        # Step 8: Risk Attribution Analysis (Section 6 Priority)
        print("\n" + "=" * 60)
        print("SECTION 6: RISK ATTRIBUTION AND DECOMPOSITION ANALYSIS")
        print("=" * 60)
        
        logger.info("Step 8: Initializing Factor Risk Model...")
        risk_model = FactorRiskModel(returns_matrix, factor_matrix)
        
        # Step 8.1: Estimate Factor Loadings
        logger.info("Step 8.1: Estimating factor loadings using OLS regression...")
        factor_loadings = risk_model.estimate_factor_loadings(method='ols')
        
        print("\n📊 FACTOR LOADINGS MATRIX:")
        print(factor_loadings.round(4))
        
        if hasattr(risk_model, 'factor_r_squared'):
            avg_r_squared = np.mean(list(risk_model.factor_r_squared.values()))
            print(f"\n📈 Average R-squared: {avg_r_squared:.3f}")
            print("Individual R-squared by Strategy:")
            for strategy, r2 in risk_model.factor_r_squared.items():
                print(f"  • {strategy}: {r2:.3f}")
        
        # Step 8.2: Calculate Portfolio Factor Exposures
        logger.info("Step 8.2: Calculating portfolio factor exposures...")
        
        # Create portfolio weights based on strategy performance
        total_pnl_by_strategy = {}
        for strategy, data in strategy_portfolios.items():
            total_pnl = data['net_pnl'].sum()
            total_pnl_by_strategy[strategy] = total_pnl
        
        # Use equal weights as baseline (can be optimized later)
        n_strategies = len(strategy_portfolios)
        portfolio_weights = {strategy: 1.0/n_strategies for strategy in strategy_portfolios.keys()}
        
        factor_exposures = risk_model.calculate_factor_exposures(portfolio_weights)
        
        print("\n🎯 PORTFOLIO FACTOR EXPOSURES:")
        print(factor_exposures.round(4))
        
        # Step 8.3: Risk Decomposition
        logger.info("Step 8.3: Decomposing portfolio risk...")
        risk_decomposition = risk_model.decompose_portfolio_risk(portfolio_weights)
        
        print("\n⚡ RISK DECOMPOSITION RESULTS:")
        print(f"  Total Portfolio Risk (Annual): {risk_decomposition['total_risk_volatility']:.2%}")
        print(f"  Factor Risk Component: {risk_decomposition['factor_risk_percentage']:.1f}%")
        print(f"  Idiosyncratic Risk Component: {risk_decomposition['idiosyncratic_risk_percentage']:.1f}%")
        
        print("\n🔍 FACTOR RISK CONTRIBUTIONS:")
        for factor, contrib in risk_decomposition['factor_contributions'].items():
            print(f"  • {factor}: {contrib['percentage_of_risk']:.2f}% of total risk")
        
        # Step 8.4: Risk Budgeting
        logger.info("Step 8.4: Calculating risk budgets...")
        risk_budgets = risk_model.calculate_risk_budgets(portfolio_weights)
        
        print("\n💰 RISK BUDGET ANALYSIS:")
        print(risk_budgets[['Asset', 'Weight', 'Risk_Contribution_Percentage', 'Risk_Budget_Ratio']].round(4))
        
        # Step 8.5: Generate Comprehensive Report
        logger.info("Step 8.5: Generating comprehensive risk attribution report...")
        risk_report = risk_model.generate_risk_report()
        
        # Save report to JSON
        with open('risk_attribution_report.json', 'w') as f:
            json.dump(risk_report, f, indent=2, default=str)
        
        # Step 9: Visualizations
        logger.info("Step 9: Creating risk attribution visualizations...")
        try:
            risk_model.plot_risk_attribution('risk_attribution_analysis.png')
            print("✅ Risk attribution visualizations created")
        except Exception as e:
            logger.warning(f"Visualization creation failed: {str(e)}")
        
        # Step 10: Summary Statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        
        summary = loader.get_data_summary()
        
        print(f"📈 Trading Performance:")
        print(f"  • Total Trades: {summary['total_trades']:,}")
        print(f"  • Total P&L: ${summary['total_pnl']:,.2f}")
        print(f"  • Total Commissions: ${summary['total_commission']:,.2f}")
        print(f"  • Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        
        print(f"\n🎯 Strategy Breakdown:")
        for strategy in summary['strategies']:
            if strategy in strategy_portfolios:
                strategy_pnl = strategy_portfolios[strategy]['net_pnl'].sum()
                strategy_trades = strategy_portfolios[strategy]['trade_count'].sum()
                print(f"  • {strategy}: ${strategy_pnl:.2f} ({strategy_trades} trades)")
        
        print(f"\n📊 Risk Metrics:")
        print(f"  • Portfolio Risk (Annual): {risk_decomposition['total_risk_volatility']:.2%}")
        print(f"  • Factor Risk Contribution: {risk_decomposition['factor_risk_percentage']:.1f}%")
        print(f"  • Number of Factors: {len(factor_loadings.columns)}")
        print(f"  • Average Factor Model R²: {avg_r_squared:.3f}" if 'avg_r_squared' in locals() else "")
        
        print(f"\n🌊 Market Environment:")
        if 'vix_range' in summary:
            print(f"  • VIX Range: {summary['vix_range']['min']:.1f} - {summary['vix_range']['max']:.1f}")
            print(f"  • Average VIX: {summary['vix_range']['mean']:.1f}")
        
        # Step 11: Key Insights and Recommendations
        print("\n" + "=" * 60)
        print("KEY INSIGHTS AND RECOMMENDATIONS")
        print("=" * 60)
        
        # Risk concentration analysis
        max_risk_contrib = risk_budgets['Risk_Contribution_Percentage'].max()
        max_risk_strategy = risk_budgets.loc[
            risk_budgets['Risk_Contribution_Percentage'].idxmax(), 'Asset'
        ]
        
        print(f"🔴 Risk Concentration:")
        print(f"  • Highest risk contributor: {max_risk_strategy} ({max_risk_contrib:.1f}%)")
        
        if max_risk_contrib > 50:
            print(f"  ⚠️  WARNING: High concentration risk detected!")
            print(f"     Consider reducing exposure to {max_risk_strategy}")
        
        # Factor exposure analysis
        max_exposure = factor_exposures.abs().max(axis=1).iloc[0]
        max_exposure_factor = factor_exposures.abs().idxmax(axis=1).iloc[0]
        
        print(f"\n🎯 Factor Exposure:")
        print(f"  • Largest factor exposure: {max_exposure_factor} ({max_exposure:.3f})")
        
        # Risk efficiency analysis
        avg_risk_budget_ratio = risk_budgets['Risk_Budget_Ratio'].mean()
        print(f"\n⚖️  Risk Efficiency:")
        print(f"  • Average risk budget ratio: {avg_risk_budget_ratio:.2f}")
        if avg_risk_budget_ratio > 1.2:
            print(f"  ⚠️  Some strategies may be taking excessive risk relative to allocation")
        elif avg_risk_budget_ratio < 0.8:
            print(f"  ℹ️  Portfolio appears to be risk-efficient")
        
        print(f"\n✅ PHASE 1 RISK ATTRIBUTION ANALYSIS COMPLETE!")
        print(f"📁 Results saved to:")
        print(f"  • returns_matrix.csv")
        print(f"  • factor_matrix.csv") 
        print(f"  • risk_attribution_report.json")
        print(f"  • risk_attribution_analysis.png")
        print(f"  • risk_analysis.log")
        
    except Exception as e:
        logger.error(f"Critical error in risk analysis: {str(e)}", exc_info=True)
        print(f"\n❌ CRITICAL ERROR: {str(e)}")
        print("Please check the log file for detailed error information.")
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Risk attribution analysis completed successfully!")
    else:
        print("\n💥 Risk attribution analysis failed. Check logs for details.")
        sys.exit(1)