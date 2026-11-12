"""
Risk Units Sample Report Generator
Creates comprehensive risk analysis reports using the TradeSteward data with Risk Units framework
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
from risk_units_database import RiskUnitsDatabaseManager

@dataclass
class RiskUnitsReportData:
    total_trades: int
    total_risk_units: float
    avg_daily_risk_units: float
    avg_weekly_risk_units: float
    strategy_breakdown: Dict[str, Dict]
    daily_summary: pd.DataFrame
    weekly_summary: pd.DataFrame
    risk_efficiency_metrics: Dict[str, float]
    concentration_analysis: Dict[str, float]

class TradeStewardRiskUnitsAnalyzer:
    """
    Analyzes TradeSteward CSV data using Risk Units framework
    """
    
    def __init__(self, csv_file_path: str):
        self.csv_file_path = csv_file_path
        self.cvar_calculator = CVaRRiskCalculator()
        self.risk_converter = RiskUnitsConverter()
        self.strategy_profiler = StrategyRiskProfiler()
        
        # Load and process TradeSteward data
        self.trades_df = self._load_and_clean_data()
        self.risk_units_data = []
        
    def _load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean TradeSteward CSV data"""
        try:
            # Read CSV with proper handling
            df = pd.read_csv(self.csv_file_path, encoding='utf-8')
            
            print(f"Loaded {len(df)} raw trades from CSV")
            
            # Remove empty rows
            df = df.dropna(subset=['OpenOrderNumber', 'Strategy', 'OpenDate'])
            
            # Clean and standardize data
            df['OpenDate'] = pd.to_datetime(df['OpenDate'], errors='coerce')
            df['FinalTradeClosedDate'] = pd.to_datetime(df['FinalTradeClosedDate'], errors='coerce')
            
            # Calculate days to expiration (approximate)
            df['Opt1-Expiration'] = pd.to_datetime(df['Opt1-Expiration'], errors='coerce')
            df['DaysToExpiration'] = (df['Opt1-Expiration'] - df['OpenDate']).dt.days
            df['DaysToExpiration'] = df['DaysToExpiration'].fillna(30).clip(lower=1)  # Default to 30 days, minimum 1
            
            # Clean strategy names - use the actual strategy column
            df['Strategy_Clean'] = df['Strategy'].astype(str).str.replace(' Separate', '').str.lower()
            df['Strategy_Clean'] = df['Strategy_Clean'].str.replace('ratio condor', 'ratio_condor')
            
            # For this analysis, we'll group by the actual strategy type, not the date-based naming
            # Let's extract the core strategy from the 'Strategy' column
            df['Core_Strategy'] = df['Strategy'].str.replace(' Separate', '').str.strip()
            
            print(f"Unique strategies found: {df['Core_Strategy'].unique()}")
            
            # Use Core_Strategy for analysis
            df['Strategy_Clean'] = df['Core_Strategy'].str.lower().str.replace(' ', '_')
            
            # Calculate implied volatility proxy (using VIX)
            df['VIXOpenQuote'] = pd.to_numeric(df['VIXOpenQuote'], errors='coerce').fillna(20.0)
            df['ImpliedVolatility'] = df['VIXOpenQuote'] / 100
            
            # Calculate maximum loss estimate based on actual data
            df['TotalAverageOpen'] = pd.to_numeric(df['TotalAverageOpen'], errors='coerce').fillna(10.0)
            df['TotalNetProfitLoss'] = pd.to_numeric(df['TotalNetProfitLoss'], errors='coerce').fillna(0.0)
            
            # For options strategies, max loss is often the premium paid
            df['MaxLossEstimate'] = -(df['TotalAverageOpen'] * 100)  # Convert to dollars
            
            # Calculate Greeks (simplified estimates)
            greek_cols = ['Opt1-Delta', 'Opt2-Delta', 'Opt3-Delta', 'Opt4-Delta']
            for col in greek_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            
            df['DeltaEstimate'] = df[greek_cols].sum(axis=1)
            df['GammaEstimate'] = abs(df['DeltaEstimate']) * 0.1  # Simplified gamma estimate
            df['ThetaEstimate'] = -df['TotalAverageOpen'] * 0.05  # Simplified theta estimate
            df['VegaEstimate'] = df['TotalAverageOpen'] * df['ImpliedVolatility'] * 0.1
            
            # Ensure we have valid underlying symbols
            df['Underlying'] = df['Underlying'].fillna('SPX')
            
            print(f"Cleaned data: {len(df)} valid trades")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def calculate_risk_units_for_all_trades(self) -> List[Dict]:
        """Calculate risk units for all trades in the dataset"""
        risk_units_results = []
        
        for idx, trade_row in self.trades_df.iterrows():
            try:
                # Create TradeRiskData object
                trade_data = TradeRiskData(
                    trade_id=str(trade_row['OpenOrderNumber']),
                    strategy=trade_row['Strategy_Clean'],
                    entry_date=trade_row['OpenDate'],
                    exit_date=trade_row['FinalTradeClosedDate'],
                    underlying_symbol=trade_row['Underlying'],
                    trade_pnl=trade_row['TotalNetProfitLoss'],
                    max_loss=trade_row['MaxLossEstimate'],
                    days_to_expiration=max(1, trade_row['DaysToExpiration']),
                    implied_volatility=max(0.1, trade_row['ImpliedVolatility']),
                    delta=trade_row['DeltaEstimate'],
                    gamma=trade_row['GammaEstimate'],
                    theta=trade_row['ThetaEstimate'],
                    vega=trade_row['VegaEstimate'],
                    premium_paid=trade_row['TotalAverageOpen'] * 100,
                    premium_received=0.0  # Simplified
                )
                
                # Calculate risk units
                conversion_result = self.risk_converter.convert_trade_to_risk_units(trade_data)
                
                # Store results
                risk_result = {
                    'trade_id': trade_data.trade_id,
                    'date': trade_data.entry_date,
                    'strategy': trade_data.strategy,
                    'underlying': trade_data.underlying_symbol,
                    'risk_units': conversion_result.risk_units,
                    'cvar_amount': conversion_result.original_cvar,
                    'pnl': trade_data.trade_pnl,
                    'risk_adjusted_return': trade_data.trade_pnl / conversion_result.risk_units if conversion_result.risk_units > 0 else 0,
                    'days_to_expiration': trade_data.days_to_expiration,
                    'implied_volatility': trade_data.implied_volatility,
                    'max_loss': trade_data.max_loss
                }
                
                risk_units_results.append(risk_result)
                
            except Exception as e:
                print(f"Error processing trade {trade_row['OpenOrderNumber']}: {e}")
                continue
        
        self.risk_units_data = risk_units_results
        return risk_units_results
    
    def generate_daily_summary(self) -> pd.DataFrame:
        """Generate daily risk units summary"""
        if not self.risk_units_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.risk_units_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Group by date and strategy
        daily_summary = df.groupby(['date', 'strategy']).agg({
            'risk_units': ['sum', 'mean', 'count', 'std'],
            'pnl': 'sum',
            'risk_adjusted_return': 'mean',
            'cvar_amount': 'sum'
        }).round(2)
        
        # Flatten column names
        daily_summary.columns = [f"{col[0]}_{col[1]}" for col in daily_summary.columns]
        daily_summary = daily_summary.reset_index()
        
        # Calculate risk efficiency
        daily_summary['risk_efficiency'] = (
            daily_summary['pnl_sum'] / daily_summary['risk_units_sum']
        ).replace([np.inf, -np.inf], 0)
        
        return daily_summary
    
    def generate_weekly_summary(self) -> pd.DataFrame:
        """Generate weekly risk units summary"""
        if not self.risk_units_data:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.risk_units_data)
        df['date'] = pd.to_datetime(df['date'])
        df['week'] = df['date'].dt.to_period('W')
        
        # Group by week and strategy
        weekly_summary = df.groupby(['week', 'strategy']).agg({
            'risk_units': ['sum', 'mean', 'count', 'std'],
            'pnl': 'sum',
            'risk_adjusted_return': 'mean',
            'cvar_amount': 'sum'
        }).round(2)
        
        # Flatten column names
        weekly_summary.columns = [f"{col[0]}_{col[1]}" for col in weekly_summary.columns]
        weekly_summary = weekly_summary.reset_index()
        
        # Calculate risk efficiency
        weekly_summary['risk_efficiency'] = (
            weekly_summary['pnl_sum'] / weekly_summary['risk_units_sum']
        ).replace([np.inf, -np.inf], 0)
        
        return weekly_summary
    
    def generate_strategy_breakdown(self) -> Dict[str, Dict]:
        """Generate comprehensive strategy breakdown"""
        if not self.risk_units_data:
            return {}
        
        df = pd.DataFrame(self.risk_units_data)
        strategy_breakdown = {}
        
        for strategy in df['strategy'].unique():
            strategy_data = df[df['strategy'] == strategy]
            
            breakdown = {
                'total_trades': len(strategy_data),
                'total_risk_units': strategy_data['risk_units'].sum(),
                'avg_risk_units': strategy_data['risk_units'].mean(),
                'std_risk_units': strategy_data['risk_units'].std(),
                'min_risk_units': strategy_data['risk_units'].min(),
                'max_risk_units': strategy_data['risk_units'].max(),
                'total_pnl': strategy_data['pnl'].sum(),
                'avg_pnl': strategy_data['pnl'].mean(),
                'win_rate': (strategy_data['pnl'] > 0).mean(),
                'risk_efficiency': strategy_data['pnl'].sum() / strategy_data['risk_units'].sum() if strategy_data['risk_units'].sum() > 0 else 0,
                'avg_risk_adjusted_return': strategy_data['risk_adjusted_return'].mean(),
                'total_cvar': strategy_data['cvar_amount'].sum(),
                'concentration': strategy_data['risk_units'].sum() / df['risk_units'].sum()
            }
            
            strategy_breakdown[strategy] = breakdown
        
        return strategy_breakdown
    
    def generate_concentration_analysis(self) -> Dict[str, float]:
        """Analyze risk concentration"""
        if not self.risk_units_data:
            return {}
        
        df = pd.DataFrame(self.risk_units_data)
        total_risk_units = df['risk_units'].sum()
        
        # Strategy concentration
        strategy_concentrations = df.groupby('strategy')['risk_units'].sum() / total_risk_units
        
        # Herfindahl-Hirschman Index
        hhi = (strategy_concentrations ** 2).sum()
        
        # Concentration analysis
        concentration_analysis = {
            'herfindahl_index': hhi,
            'max_strategy_concentration': strategy_concentrations.max(),
            'number_of_strategies': len(strategy_concentrations),
            'concentration_risk_level': 'High' if hhi > 0.25 else 'Medium' if hhi > 0.15 else 'Low',
            'diversification_score': 1 - hhi,
            'top_strategy_share': strategy_concentrations.max(),
            'top_2_strategies_share': strategy_concentrations.nlargest(2).sum(),
            'effective_number_strategies': 1 / hhi if hhi > 0 else 1
        }
        
        return concentration_analysis
    
    def create_visualizations(self, output_dir: str = "risk_units_charts/"):
        """Create risk units visualizations"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.risk_units_data:
            print("No risk units data available for visualization")
            return
        
        df = pd.DataFrame(self.risk_units_data)
        df['date'] = pd.to_datetime(df['date'])
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Daily Risk Units by Strategy
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Daily risk units timeline
        daily_risk = df.groupby(['date', 'strategy'])['risk_units'].sum().unstack(fill_value=0)
        daily_risk.plot(kind='area', stacked=True, ax=axes[0,0], alpha=0.7)
        axes[0,0].set_title('Daily Risk Units by Strategy (Stacked)', fontsize=14, fontweight='bold')
        axes[0,0].set_xlabel('Date')
        axes[0,0].set_ylabel('Risk Units')
        axes[0,0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Strategy risk distribution
        strategy_risk = df.groupby('strategy')['risk_units'].sum()
        axes[0,1].pie(strategy_risk.values, labels=strategy_risk.index, autopct='%1.1f%%', startangle=90)
        axes[0,1].set_title('Risk Units Distribution by Strategy', fontsize=14, fontweight='bold')
        
        # Risk vs P&L scatter
        axes[1,0].scatter(df['risk_units'], df['pnl'], alpha=0.6, c=df['strategy'].astype('category').cat.codes)
        axes[1,0].set_xlabel('Risk Units')
        axes[1,0].set_ylabel('P&L ($)')
        axes[1,0].set_title('Risk Units vs P&L', fontsize=14, fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # Risk efficiency by strategy
        strategy_efficiency = df.groupby('strategy').apply(
            lambda x: x['pnl'].sum() / x['risk_units'].sum() if x['risk_units'].sum() > 0 else 0
        )
        bars = axes[1,1].bar(range(len(strategy_efficiency)), strategy_efficiency.values)
        axes[1,1].set_xlabel('Strategy')
        axes[1,1].set_ylabel('Risk Efficiency ($/Risk Unit)')
        axes[1,1].set_title('Risk Efficiency by Strategy', fontsize=14, fontweight='bold')
        axes[1,1].set_xticks(range(len(strategy_efficiency)))
        axes[1,1].set_xticklabels(strategy_efficiency.index, rotation=45)
        
        # Color bars based on positive/negative efficiency
        for i, bar in enumerate(bars):
            if strategy_efficiency.iloc[i] >= 0:
                bar.set_color('green')
                bar.set_alpha(0.7)
            else:
                bar.set_color('red')
                bar.set_alpha(0.7)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/risk_units_overview.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Weekly Risk Patterns
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # Weekly risk units
        df['week'] = df['date'].dt.to_period('W')
        weekly_risk = df.groupby(['week', 'strategy'])['risk_units'].sum().unstack(fill_value=0)
        weekly_risk.plot(kind='bar', stacked=True, ax=axes[0], alpha=0.8)
        axes[0].set_title('Weekly Risk Units by Strategy', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Week')
        axes[0].set_ylabel('Risk Units')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Weekly P&L vs Risk
        weekly_pnl = df.groupby('week')['pnl'].sum()
        weekly_risk_total = df.groupby('week')['risk_units'].sum()
        
        ax2 = axes[1].twinx()
        line1 = axes[1].plot(range(len(weekly_pnl)), weekly_pnl.values, 'g-', marker='o', linewidth=2, label='Weekly P&L')
        line2 = ax2.plot(range(len(weekly_risk_total)), weekly_risk_total.values, 'b-', marker='s', linewidth=2, label='Weekly Risk Units')
        
        axes[1].set_xlabel('Week')
        axes[1].set_ylabel('P&L ($)', color='g')
        ax2.set_ylabel('Risk Units', color='b')
        axes[1].set_title('Weekly P&L vs Risk Units', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Combine legends
        lines1, labels1 = axes[1].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/weekly_risk_patterns.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {output_dir}")
    
    def generate_comprehensive_report(self) -> RiskUnitsReportData:
        """Generate comprehensive risk units report"""
        print("Calculating risk units for all trades...")
        self.calculate_risk_units_for_all_trades()
        
        print("Generating daily summary...")
        daily_summary = self.generate_daily_summary()
        
        print("Generating weekly summary...")
        weekly_summary = self.generate_weekly_summary()
        
        print("Analyzing strategy breakdown...")
        strategy_breakdown = self.generate_strategy_breakdown()
        
        print("Analyzing concentration...")
        concentration_analysis = self.generate_concentration_analysis()
        
        # Calculate overall metrics
        if not self.risk_units_data:
            # Return empty report if no data
            return RiskUnitsReportData(
                total_trades=0,
                total_risk_units=0.0,
                avg_daily_risk_units=0.0,
                avg_weekly_risk_units=0.0,
                strategy_breakdown={},
                daily_summary=pd.DataFrame(),
                weekly_summary=pd.DataFrame(),
                risk_efficiency_metrics={},
                concentration_analysis={}
            )
        
        df = pd.DataFrame(self.risk_units_data)
        
        total_trades = len(df)
        total_risk_units = df['risk_units'].sum()
        
        # Calculate averages
        avg_daily_risk_units = daily_summary.groupby('date')['risk_units_sum'].sum().mean() if not daily_summary.empty else 0
        avg_weekly_risk_units = weekly_summary.groupby('week')['risk_units_sum'].sum().mean() if not weekly_summary.empty else 0
        
        # Risk efficiency metrics
        risk_efficiency_metrics = {
            'overall_risk_efficiency': df['pnl'].sum() / df['risk_units'].sum() if df['risk_units'].sum() > 0 else 0,
            'avg_risk_per_trade': df['risk_units'].mean(),
            'risk_adjusted_sharpe': df['risk_adjusted_return'].mean() / df['risk_adjusted_return'].std() if df['risk_adjusted_return'].std() > 0 else 0,
            'total_cvar': df['cvar_amount'].sum(),
            'avg_cvar_per_trade': df['cvar_amount'].mean()
        }
        
        return RiskUnitsReportData(
            total_trades=total_trades,
            total_risk_units=total_risk_units,
            avg_daily_risk_units=avg_daily_risk_units,
            avg_weekly_risk_units=avg_weekly_risk_units,
            strategy_breakdown=strategy_breakdown,
            daily_summary=daily_summary,
            weekly_summary=weekly_summary,
            risk_efficiency_metrics=risk_efficiency_metrics,
            concentration_analysis=concentration_analysis
        )

def format_currency(amount: float) -> str:
    """Format currency with proper signs and commas"""
    return f"${amount:,.2f}" if amount >= 0 else f"-${abs(amount):,.2f}"

def format_percentage(value: float) -> str:
    """Format percentage with proper sign"""
    return f"{value:.1%}" if value >= 0 else f"{value:.1%}"

def print_risk_units_report(report_data: RiskUnitsReportData):
    """Print comprehensive risk units report"""
    print("\n" + "="*80)
    print("TRADESTWARD RISK UNITS ANALYSIS REPORT")
    print("="*80)
    print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Risk Unit Standard: $1,000 CVaR (95% confidence)")
    
    # Executive Summary
    print(f"\n📊 EXECUTIVE SUMMARY")
    print("-" * 40)
    print(f"Total Trades Analyzed:      {report_data.total_trades:,}")
    print(f"Total Risk Units:           {report_data.total_risk_units:.1f}")
    print(f"Average Daily Risk Units:   {report_data.avg_daily_risk_units:.1f}")
    print(f"Average Weekly Risk Units:  {report_data.avg_weekly_risk_units:.1f}")
    print(f"Overall Risk Efficiency:    {format_currency(report_data.risk_efficiency_metrics['overall_risk_efficiency'])}/unit")
    print(f"Average Risk per Trade:     {report_data.risk_efficiency_metrics['avg_risk_per_trade']:.2f} units")
    
    # Strategy Breakdown
    print(f"\n📈 STRATEGY RISK ANALYSIS")
    print("-" * 40)
    
    # Sort strategies by total risk units
    sorted_strategies = sorted(
        report_data.strategy_breakdown.items(), 
        key=lambda x: x[1]['total_risk_units'], 
        reverse=True
    )
    
    print(f"{'Strategy':<15} {'Trades':<8} {'Risk Units':<12} {'Avg Risk':<10} {'Win Rate':<10} {'Risk Eff':<12} {'Conc %':<8}")
    print("-" * 85)
    
    for strategy, data in sorted_strategies:
        print(f"{strategy:<15} "
              f"{data['total_trades']:<8} "
              f"{data['total_risk_units']:<12.1f} "
              f"{data['avg_risk_units']:<10.2f} "
              f"{data['win_rate']:<10.1%} "
              f"{format_currency(data['risk_efficiency']):<12} "
              f"{data['concentration']:<8.1%}")
    
    # Risk Concentration Analysis
    print(f"\n🎯 RISK CONCENTRATION ANALYSIS")
    print("-" * 40)
    conc = report_data.concentration_analysis
    print(f"Diversification Score:      {conc['diversification_score']:.3f}")
    print(f"Concentration Risk Level:   {conc['concentration_risk_level']}")
    print(f"Top Strategy Share:         {format_percentage(conc['top_strategy_share'])}")
    print(f"Top 2 Strategies Share:     {format_percentage(conc['top_2_strategies_share'])}")
    print(f"Effective # of Strategies:  {conc['effective_number_strategies']:.1f}")
    print(f"Herfindahl Index:          {conc['herfindahl_index']:.3f}")
    
    # Daily Risk Summary (last 10 days)
    if not report_data.daily_summary.empty:
        print(f"\n📅 RECENT DAILY RISK SUMMARY (Last 10 Days)")
        print("-" * 40)
        
        recent_daily = report_data.daily_summary.tail(10)
        print(f"{'Date':<12} {'Strategy':<15} {'Risk Units':<12} {'Trades':<8} {'P&L':<12} {'Risk Eff':<12}")
        print("-" * 75)
        
        for _, row in recent_daily.iterrows():
            print(f"{row['date'].strftime('%Y-%m-%d'):<12} "
                  f"{row['strategy']:<15} "
                  f"{row['risk_units_sum']:<12.1f} "
                  f"{row['risk_units_count']:<8.0f} "
                  f"{format_currency(row['pnl_sum']):<12} "
                  f"{format_currency(row['risk_efficiency']):<12}")
    
    # Weekly Risk Summary
    if not report_data.weekly_summary.empty:
        print(f"\n📊 WEEKLY RISK SUMMARY")
        print("-" * 40)
        
        print(f"{'Week':<12} {'Strategy':<15} {'Risk Units':<12} {'Trades':<8} {'P&L':<12} {'Risk Eff':<12}")
        print("-" * 75)
        
        for _, row in report_data.weekly_summary.iterrows():
            print(f"{str(row['week']):<12} "
                  f"{row['strategy']:<15} "
                  f"{row['risk_units_sum']:<12.1f} "
                  f"{row['risk_units_count']:<8.0f} "
                  f"{format_currency(row['pnl_sum']):<12} "
                  f"{format_currency(row['risk_efficiency']):<12}")
    
    # Risk Efficiency Metrics
    print(f"\n⚡ RISK EFFICIENCY METRICS")
    print("-" * 40)
    metrics = report_data.risk_efficiency_metrics
    print(f"Total CVaR Exposure:        {format_currency(metrics['total_cvar'])}")
    print(f"Average CVaR per Trade:     {format_currency(metrics['avg_cvar_per_trade'])}")
    print(f"Risk-Adjusted Sharpe:       {metrics['risk_adjusted_sharpe']:.3f}")
    print(f"Overall Risk Efficiency:    {format_currency(metrics['overall_risk_efficiency'])}/unit")
    
    # Risk Management Insights
    print(f"\n💡 RISK MANAGEMENT INSIGHTS")
    print("-" * 40)
    
    # Find highest risk strategy
    highest_risk_strategy = max(sorted_strategies, key=lambda x: x[1]['avg_risk_units'])
    lowest_risk_strategy = min(sorted_strategies, key=lambda x: x[1]['avg_risk_units'])
    
    print(f"• Highest Risk Strategy: {highest_risk_strategy[0]} ({highest_risk_strategy[1]['avg_risk_units']:.2f} units/trade)")
    print(f"• Lowest Risk Strategy: {lowest_risk_strategy[0]} ({lowest_risk_strategy[1]['avg_risk_units']:.2f} units/trade)")
    
    # Risk concentration warning
    if conc['concentration_risk_level'] == 'High':
        print(f"• ⚠️  HIGH CONCENTRATION RISK: {format_percentage(conc['top_strategy_share'])} in {sorted_strategies[0][0]}")
    elif conc['concentration_risk_level'] == 'Medium':
        print(f"• ⚠️  MEDIUM CONCENTRATION RISK: Consider diversification")
    else:
        print(f"• ✅ GOOD DIVERSIFICATION: Risk well distributed across strategies")
    
    # Efficiency insights
    best_efficiency = max(sorted_strategies, key=lambda x: x[1]['risk_efficiency'])
    worst_efficiency = min(sorted_strategies, key=lambda x: x[1]['risk_efficiency'])
    
    print(f"• Best Risk Efficiency: {best_efficiency[0]} ({format_currency(best_efficiency[1]['risk_efficiency'])}/unit)")
    print(f"• Worst Risk Efficiency: {worst_efficiency[0]} ({format_currency(worst_efficiency[1]['risk_efficiency'])}/unit)")
    
    print("\n" + "="*80)
    print("END OF RISK UNITS ANALYSIS REPORT")
    print("="*80)

def main():
    """Generate sample risk units report using TradeSteward data"""
    
    # File path to TradeSteward CSV
    csv_file = "/Users/closet/production/TradeSteward-Results-Analyzer/tradeSteward-performanceLogs-1753563010.csv"
    
    print("Initializing TradeSteward Risk Units Analyzer...")
    analyzer = TradeStewardRiskUnitsAnalyzer(csv_file)
    
    print(f"Loaded {len(analyzer.trades_df)} trades from TradeSteward data")
    
    # Generate comprehensive report
    print("\nGenerating comprehensive Risk Units analysis...")
    report_data = analyzer.generate_comprehensive_report()
    
    # Print the report
    print_risk_units_report(report_data)
    
    # Create visualizations
    print("\nCreating visualizations...")
    analyzer.create_visualizations()
    
    # Export summary data
    print("\nExporting summary data...")
    if not report_data.daily_summary.empty:
        report_data.daily_summary.to_csv("risk_units_daily_summary.csv", index=False)
        print("Daily summary exported to: risk_units_daily_summary.csv")
    
    if not report_data.weekly_summary.empty:
        report_data.weekly_summary.to_csv("risk_units_weekly_summary.csv", index=False)
        print("Weekly summary exported to: risk_units_weekly_summary.csv")
    
    print("\n✅ Risk Units Analysis Complete!")
    print("📊 Charts saved to: risk_units_charts/")
    print("📄 Data exported to CSV files")

if __name__ == "__main__":
    main()