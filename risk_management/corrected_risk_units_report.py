"""
Corrected Risk Units Sample Report Generator
Creates risk analysis reports using actual TradeSteward strategy data with proper strategy identification
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from cvar_risk_calculator import CVaRRiskCalculator, TradeRiskData, CVaRMethod
from risk_units_converter import RiskUnitsConverter
from strategy_risk_profiler import StrategyRiskProfiler

def analyze_tradesteward_with_risk_units():
    """Generate comprehensive risk units analysis using actual TradeSteward data"""
    
    # File path to TradeSteward CSV
    csv_file = "/Users/closet/production/TradeSteward-Results-Analyzer/tradeSteward-performanceLogs-1753563010.csv"
    
    print("🔍 TRADESTEWARD RISK UNITS ANALYSIS")
    print("=" * 60)
    
    # Load and analyze the data
    print("Loading TradeSteward data...")
    df = pd.read_csv(csv_file)
    
    # Clean and prepare data
    print(f"Loaded {len(df)} trades from CSV")
    
    # Remove empty rows and clean data
    df = df.dropna(subset=['OpenOrderNumber', 'Strategy', 'OpenDate'])
    df['OpenDate'] = pd.to_datetime(df['OpenDate'], errors='coerce')
    df['FinalTradeClosedDate'] = pd.to_datetime(df['FinalTradeClosedDate'], errors='coerce')
    
    # Use actual strategy information
    df['Strategy_Clean'] = df['Strategy'].str.replace(' Separate', '').str.strip()
    
    # Clean numeric columns
    numeric_cols = ['TotalNetProfitLoss', 'TotalAverageOpen', 'VIXOpenQuote', 'UnderlyingOpenQuote']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Calculate days to expiration
    df['Opt1-Expiration'] = pd.to_datetime(df['Opt1-Expiration'], errors='coerce')
    df['DaysToExpiration'] = (df['Opt1-Expiration'] - df['OpenDate']).dt.days
    df['DaysToExpiration'] = df['DaysToExpiration'].fillna(30).clip(lower=1)
    
    # Calculate implied volatility
    df['ImpliedVolatility'] = df['VIXOpenQuote'] / 100
    df['ImpliedVolatility'] = df['ImpliedVolatility'].clip(lower=0.1, upper=1.0)
    
    # Estimate maximum loss (for options strategies, often the premium paid)
    df['MaxLossEstimate'] = -(df['TotalAverageOpen'] * 100)
    
    # Calculate approximate Greeks
    greek_cols = ['Opt1-Delta', 'Opt2-Delta', 'Opt3-Delta', 'Opt4-Delta']
    for col in greek_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    df['DeltaSum'] = df[greek_cols].sum(axis=1)
    df['GammaEstimate'] = abs(df['DeltaSum']) * 0.1
    df['ThetaEstimate'] = -df['TotalAverageOpen'] * 0.05
    df['VegaEstimate'] = df['TotalAverageOpen'] * df['ImpliedVolatility'] * 0.1
    
    print(f"Cleaned data: {len(df)} valid trades")
    print(f"Unique strategies: {df['Strategy_Clean'].unique()}")
    print(f"Date range: {df['OpenDate'].min()} to {df['OpenDate'].max()}")
    
    # Initialize risk calculation components
    cvar_calculator = CVaRRiskCalculator()
    risk_converter = RiskUnitsConverter()
    
    # Process each trade and calculate risk units
    print("\n📊 Calculating Risk Units for all trades...")
    risk_results = []
    
    for idx, row in df.iterrows():
        try:
            # Create trade data object
            trade_data = TradeRiskData(
                trade_id=str(row['OpenOrderNumber']),
                strategy=row['Strategy_Clean'],
                entry_date=row['OpenDate'],
                exit_date=row['FinalTradeClosedDate'],
                underlying_symbol=row['Underlying'],
                trade_pnl=row['TotalNetProfitLoss'],
                max_loss=row['MaxLossEstimate'],
                days_to_expiration=int(row['DaysToExpiration']),
                implied_volatility=row['ImpliedVolatility'],
                delta=row['DeltaSum'],
                gamma=row['GammaEstimate'],
                theta=row['ThetaEstimate'],
                vega=row['VegaEstimate'],
                premium_paid=row['TotalAverageOpen'] * 100,
                premium_received=0.0
            )
            
            # Calculate risk units
            conversion_result = risk_converter.convert_trade_to_risk_units(trade_data)
            
            # Store results
            risk_results.append({
                'trade_id': trade_data.trade_id,
                'date': trade_data.entry_date,
                'strategy': trade_data.strategy,
                'underlying': trade_data.underlying_symbol,
                'risk_units': conversion_result.risk_units,
                'cvar_amount': conversion_result.original_cvar,
                'pnl': trade_data.trade_pnl,
                'max_loss': trade_data.max_loss,
                'days_to_expiration': trade_data.days_to_expiration,
                'implied_volatility': trade_data.implied_volatility,
                'risk_adjusted_return': trade_data.trade_pnl / conversion_result.risk_units if conversion_result.risk_units > 0 else 0
            })
            
        except Exception as e:
            print(f"Warning: Error processing trade {row['OpenOrderNumber']}: {e}")
            continue
    
    # Convert to DataFrame for analysis
    risk_df = pd.DataFrame(risk_results)
    risk_df['date'] = pd.to_datetime(risk_df['date'])
    
    print(f"✅ Successfully calculated risk units for {len(risk_df)} trades")
    
    # Generate comprehensive analysis
    print("\n📈 RISK UNITS ANALYSIS RESULTS")
    print("=" * 60)
    
    # Overall statistics
    total_trades = len(risk_df)
    total_risk_units = risk_df['risk_units'].sum()
    total_pnl = risk_df['pnl'].sum()
    total_cvar = risk_df['cvar_amount'].sum()
    
    print(f"📊 EXECUTIVE SUMMARY")
    print("-" * 40)
    print(f"Total Trades Analyzed:      {total_trades:,}")
    print(f"Total Risk Units:           {total_risk_units:.1f}")
    print(f"Total P&L:                  ${total_pnl:,.2f}")
    print(f"Total CVaR Exposure:        ${total_cvar:,.2f}")
    print(f"Average Risk per Trade:     {total_risk_units/total_trades:.2f} units")
    print(f"Risk Efficiency:            ${total_pnl/total_risk_units:.2f} per risk unit" if total_risk_units > 0 else "Risk Efficiency: N/A")
    print(f"Average CVaR per Trade:     ${total_cvar/total_trades:.2f}")
    
    # Strategy breakdown
    print(f"\n📋 STRATEGY ANALYSIS")
    print("-" * 40)
    
    strategy_analysis = risk_df.groupby('strategy').agg({
        'risk_units': ['count', 'sum', 'mean', 'std'],
        'pnl': ['sum', 'mean'],
        'cvar_amount': 'sum',
        'risk_adjusted_return': 'mean'
    }).round(2)
    
    # Flatten column names
    strategy_analysis.columns = [f"{col[0]}_{col[1]}" for col in strategy_analysis.columns]
    strategy_analysis = strategy_analysis.reset_index()
    
    # Calculate additional metrics
    strategy_analysis['win_rate'] = risk_df.groupby('strategy')['pnl'].apply(lambda x: (x > 0).mean()).values
    strategy_analysis['risk_efficiency'] = strategy_analysis['pnl_sum'] / strategy_analysis['risk_units_sum']
    strategy_analysis['risk_concentration'] = strategy_analysis['risk_units_sum'] / total_risk_units
    
    # Display strategy analysis
    print(f"{'Strategy':<20} {'Trades':<8} {'Risk Units':<12} {'Avg Risk':<10} {'Total P&L':<12} {'Win Rate':<10} {'Risk Eff':<12}")
    print("-" * 95)
    
    for _, row in strategy_analysis.iterrows():
        print(f"{row['strategy']:<20} "
              f"{row['risk_units_count']:<8.0f} "
              f"{row['risk_units_sum']:<12.1f} "
              f"{row['risk_units_mean']:<10.2f} "
              f"${row['pnl_sum']:<11,.0f} "
              f"{row['win_rate']:<10.1%} "
              f"${row['risk_efficiency']:<11.2f}")
    
    # Daily and Weekly Analysis
    print(f"\n📅 TEMPORAL RISK ANALYSIS")
    print("-" * 40)
    
    # Daily risk analysis
    daily_risk = risk_df.groupby([risk_df['date'].dt.date, 'strategy']).agg({
        'risk_units': 'sum',
        'pnl': 'sum'
    }).reset_index()
    
    daily_summary = daily_risk.groupby('date').agg({
        'risk_units': ['sum', 'mean'],
        'pnl': 'sum'
    }).round(2)
    
    avg_daily_risk = daily_summary['risk_units']['sum'].mean()
    avg_daily_pnl = daily_summary['pnl']['sum'].mean()
    
    print(f"Average Daily Risk Units:   {avg_daily_risk:.1f}")
    print(f"Average Daily P&L:          ${avg_daily_pnl:,.2f}")
    print(f"Daily Risk Efficiency:      ${avg_daily_pnl/avg_daily_risk:.2f} per risk unit" if avg_daily_risk > 0 else "Daily Risk Efficiency: N/A")
    
    # Weekly risk analysis
    risk_df['week'] = risk_df['date'].dt.to_period('W')
    weekly_risk = risk_df.groupby(['week', 'strategy']).agg({
        'risk_units': 'sum',
        'pnl': 'sum'
    }).reset_index()
    
    weekly_summary = weekly_risk.groupby('week').agg({
        'risk_units': ['sum', 'mean'],
        'pnl': 'sum'
    }).round(2)
    
    avg_weekly_risk = weekly_summary['risk_units']['sum'].mean()
    avg_weekly_pnl = weekly_summary['pnl']['sum'].mean()
    
    print(f"Average Weekly Risk Units:  {avg_weekly_risk:.1f}")
    print(f"Average Weekly P&L:         ${avg_weekly_pnl:,.2f}")
    print(f"Weekly Risk Efficiency:     ${avg_weekly_pnl/avg_weekly_risk:.2f} per risk unit" if avg_weekly_risk > 0 else "Weekly Risk Efficiency: N/A")
    
    # Risk concentration analysis
    print(f"\n🎯 RISK CONCENTRATION ANALYSIS")
    print("-" * 40)
    
    # Calculate Herfindahl-Hirschman Index
    strategy_concentrations = strategy_analysis['risk_concentration']
    hhi = (strategy_concentrations ** 2).sum()
    diversification_score = 1 - hhi
    
    print(f"Number of Strategies:       {len(strategy_analysis)}")
    print(f"Largest Strategy Share:     {strategy_concentrations.max():.1%}")
    print(f"Top 2 Strategies Share:     {strategy_concentrations.nlargest(2).sum():.1%}")
    print(f"Herfindahl Index:          {hhi:.3f}")
    print(f"Diversification Score:      {diversification_score:.3f}")
    print(f"Effective # Strategies:     {1/hhi:.1f}" if hhi > 0 else "Effective # Strategies: N/A")
    
    if hhi > 0.25:
        concentration_level = "High"
    elif hhi > 0.15:
        concentration_level = "Medium"
    else:
        concentration_level = "Low"
    
    print(f"Concentration Risk Level:   {concentration_level}")
    
    # Performance insights
    print(f"\n💡 PERFORMANCE INSIGHTS")
    print("-" * 40)
    
    best_strategy = strategy_analysis.loc[strategy_analysis['risk_efficiency'].idxmax()]
    worst_strategy = strategy_analysis.loc[strategy_analysis['risk_efficiency'].idxmin()]
    
    print(f"Most Efficient Strategy:    {best_strategy['strategy']} (${best_strategy['risk_efficiency']:.2f}/unit)")
    print(f"Least Efficient Strategy:   {worst_strategy['strategy']} (${worst_strategy['risk_efficiency']:.2f}/unit)")
    
    highest_risk_strategy = strategy_analysis.loc[strategy_analysis['risk_units_mean'].idxmax()]
    lowest_risk_strategy = strategy_analysis.loc[strategy_analysis['risk_units_mean'].idxmin()]
    
    print(f"Highest Risk per Trade:     {highest_risk_strategy['strategy']} ({highest_risk_strategy['risk_units_mean']:.2f} units)")
    print(f"Lowest Risk per Trade:      {lowest_risk_strategy['strategy']} ({lowest_risk_strategy['risk_units_mean']:.2f} units)")
    
    # Recent performance (last 30 days)
    recent_cutoff = risk_df['date'].max() - timedelta(days=30)
    recent_trades = risk_df[risk_df['date'] >= recent_cutoff]
    
    if len(recent_trades) > 0:
        print(f"\n📊 RECENT PERFORMANCE (Last 30 Days)")
        print("-" * 40)
        print(f"Recent Trades:              {len(recent_trades)}")
        print(f"Recent Risk Units:          {recent_trades['risk_units'].sum():.1f}")
        print(f"Recent P&L:                 ${recent_trades['pnl'].sum():,.2f}")
        print(f"Recent Risk Efficiency:     ${recent_trades['pnl'].sum()/recent_trades['risk_units'].sum():.2f}/unit" if recent_trades['risk_units'].sum() > 0 else "Recent Risk Efficiency: N/A")
    
    # Create visualization
    create_risk_units_visualization(risk_df, strategy_analysis)
    
    # Export data
    print(f"\n💾 EXPORTING DATA")
    print("-" * 40)
    
    # Export risk units data
    risk_df.to_csv('tradesteward_risk_units_analysis.csv', index=False)
    print("✅ Risk units data exported to: tradesteward_risk_units_analysis.csv")
    
    # Export strategy summary
    strategy_analysis.to_csv('tradesteward_strategy_analysis.csv', index=False)
    print("✅ Strategy analysis exported to: tradesteward_strategy_analysis.csv")
    
    # Export daily summary
    daily_summary.to_csv('tradesteward_daily_risk_summary.csv')
    print("✅ Daily summary exported to: tradesteward_daily_risk_summary.csv")
    
    print(f"\n🎉 Analysis Complete!")
    print("=" * 60)

def create_risk_units_visualization(risk_df, strategy_analysis):
    """Create comprehensive risk units visualizations"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a comprehensive dashboard
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Risk Units by Strategy (Pie Chart)
    ax1 = plt.subplot(3, 3, 1)
    strategy_risk = strategy_analysis.set_index('strategy')['risk_units_sum']
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategy_risk)))
    wedges, texts, autotexts = ax1.pie(strategy_risk.values, labels=strategy_risk.index, 
                                      autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Risk Units Distribution by Strategy', fontsize=14, fontweight='bold')
    
    # 2. Risk Units Timeline
    ax2 = plt.subplot(3, 3, 2)
    daily_risk = risk_df.groupby(risk_df['date'].dt.date)['risk_units'].sum()
    ax2.plot(daily_risk.index, daily_risk.values, marker='o', linewidth=2, markersize=4)
    ax2.set_title('Daily Risk Units Over Time', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Risk Units')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 3. Risk vs P&L Scatter
    ax3 = plt.subplot(3, 3, 3)
    scatter = ax3.scatter(risk_df['risk_units'], risk_df['pnl'], 
                         c=risk_df['strategy'].astype('category').cat.codes, 
                         alpha=0.6, s=50)
    ax3.set_xlabel('Risk Units')
    ax3.set_ylabel('P&L ($)')
    ax3.set_title('Risk Units vs P&L', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # 4. Strategy Risk Efficiency
    ax4 = plt.subplot(3, 3, 4)
    strategy_names = strategy_analysis['strategy']
    risk_efficiency = strategy_analysis['risk_efficiency']
    bars = ax4.bar(range(len(strategy_names)), risk_efficiency)
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('Risk Efficiency ($/Risk Unit)')
    ax4.set_title('Risk Efficiency by Strategy', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(strategy_names)))
    ax4.set_xticklabels(strategy_names, rotation=45)
    
    # Color bars based on efficiency
    for i, bar in enumerate(bars):
        if risk_efficiency.iloc[i] >= 0:
            bar.set_color('green')
            bar.set_alpha(0.7)
        else:
            bar.set_color('red')
            bar.set_alpha(0.7)
    
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 5. Risk Units Distribution (Histogram)
    ax5 = plt.subplot(3, 3, 5)
    ax5.hist(risk_df['risk_units'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    ax5.set_xlabel('Risk Units')
    ax5.set_ylabel('Frequency')
    ax5.set_title('Risk Units Distribution', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Cumulative P&L with Risk
    ax6 = plt.subplot(3, 3, 6)
    risk_df_sorted = risk_df.sort_values('date')
    cumulative_pnl = risk_df_sorted['pnl'].cumsum()
    cumulative_risk = risk_df_sorted['risk_units'].cumsum()
    
    ax6_twin = ax6.twinx()
    line1 = ax6.plot(risk_df_sorted['date'], cumulative_pnl, 'g-', linewidth=2, label='Cumulative P&L')
    line2 = ax6_twin.plot(risk_df_sorted['date'], cumulative_risk, 'b-', linewidth=2, label='Cumulative Risk Units')
    
    ax6.set_xlabel('Date')
    ax6.set_ylabel('Cumulative P&L ($)', color='g')
    ax6_twin.set_ylabel('Cumulative Risk Units', color='b')
    ax6.set_title('Cumulative P&L vs Risk Over Time', fontsize=14, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # 7. Average Risk by Strategy (Bar Chart)
    ax7 = plt.subplot(3, 3, 7)
    avg_risk = strategy_analysis['risk_units_mean']
    bars = ax7.bar(range(len(strategy_names)), avg_risk, color='orange', alpha=0.7)
    ax7.set_xlabel('Strategy')
    ax7.set_ylabel('Average Risk Units per Trade')
    ax7.set_title('Average Risk Units by Strategy', fontsize=14, fontweight='bold')
    ax7.set_xticks(range(len(strategy_names)))
    ax7.set_xticklabels(strategy_names, rotation=45)
    ax7.grid(True, alpha=0.3)
    
    # 8. Win Rate vs Risk Efficiency
    ax8 = plt.subplot(3, 3, 8)
    win_rate = strategy_analysis['win_rate']
    ax8.scatter(win_rate, risk_efficiency, s=100, alpha=0.7, c=strategy_analysis['risk_units_sum'], cmap='viridis')
    ax8.set_xlabel('Win Rate')
    ax8.set_ylabel('Risk Efficiency ($/Risk Unit)')
    ax8.set_title('Win Rate vs Risk Efficiency', fontsize=14, fontweight='bold')
    ax8.grid(True, alpha=0.3)
    
    # Add strategy labels
    for i, strategy in enumerate(strategy_names):
        ax8.annotate(strategy, (win_rate.iloc[i], risk_efficiency.iloc[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 9. Weekly Risk Aggregation
    ax9 = plt.subplot(3, 3, 9)
    risk_df['week'] = risk_df['date'].dt.to_period('W')
    weekly_risk = risk_df.groupby('week')['risk_units'].sum()
    weekly_pnl = risk_df.groupby('week')['pnl'].sum()
    
    bars = ax9.bar(range(len(weekly_risk)), weekly_risk.values, alpha=0.7)
    ax9.set_xlabel('Week')
    ax9.set_ylabel('Weekly Risk Units')
    ax9.set_title('Weekly Risk Units', fontsize=14, fontweight='bold')
    ax9.grid(True, alpha=0.3)
    
    # Color bars based on P&L
    for i, bar in enumerate(bars):
        if weekly_pnl.iloc[i] >= 0:
            bar.set_color('green')
            bar.set_alpha(0.7)
        else:
            bar.set_color('red')
            bar.set_alpha(0.7)
    
    plt.tight_layout(pad=3.0)
    plt.savefig('tradesteward_risk_units_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ Comprehensive dashboard saved as: tradesteward_risk_units_dashboard.png")

if __name__ == "__main__":
    analyze_tradesteward_with_risk_units()