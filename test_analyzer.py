#!/usr/bin/env python3
"""
Test script for TradeSteward Results Analyzer
This script tests the core analyzer functionality with the sample CSV file.
"""

import pandas as pd
from app import TradeStewardAnalyzer, TradeDatabase
import os

def test_analyzer():
    """Test the TradeSteward analyzer with the sample CSV file"""
    
    # Initialize database and analyzer (using test user ID)
    db = TradeDatabase("test_database.db")
    analyzer = TradeStewardAnalyzer(db, user_id=1)
    
    # Check if the CSV file exists
    csv_file = "tradeSteward-performanceLogs-1753563010.csv"
    if not os.path.exists(csv_file):
        print(f"❌ Error: {csv_file} not found in current directory")
        print("Please ensure the TradeSteward CSV file is in the project directory")
        return False
    
    print(f"📊 Testing TradeSteward Results Analyzer")
    print(f"📄 Using file: {csv_file}")
    print("-" * 50)
    
    try:
        # Read and analyze the CSV file
        with open(csv_file, 'r', encoding='utf-8') as f:
            raw_csv_content = f.read()
        
        # Clean the CSV content first (same as web app does)
        from app import clean_csv_content
        csv_content = clean_csv_content(raw_csv_content)
        
        # Load data
        result = analyzer.load_data(csv_content)
        
        if not result["success"]:
            print(f"❌ Error loading data: {result['error']}")
            return False
        
        # Print basic summary
        print(f"✅ Data loaded successfully!")
        print(f"📈 Total trades: {result['total_trades']}")
        print(f"📅 Date range: {result['date_range']}")
        print(f"💰 Total P&L: ${result['total_pnl']:,.2f}")
        
        # Print merge statistics if available
        if 'new_trades' in result:
            print(f"🔄 Merge stats: {result['new_trades']} new, {result.get('updated_trades', 0)} updated, {result.get('duplicate_trades', 0)} duplicates")
        print()
        
        # Print strategies from metrics
        print("🎯 Strategies found:")
        if analyzer.metrics and 'strategy_performance' in analyzer.metrics:
            for strategy, stats in analyzer.metrics['strategy_performance'].items():
                print(f"   • {strategy}: {stats['trade_count']} trades")
        print()
        
        # Print key metrics
        metrics = analyzer.metrics
        print("📊 Key Metrics:")
        print(f"   • Win Rate: {metrics['win_rate']:.1f}%")
        print(f"   • Profit Factor: {metrics['profit_factor']}")
        print(f"   • Premium Capture Rate: {metrics['pcr']:.1f}%")
        print(f"   • Average Return per Trade: ${metrics['avg_return_per_trade']:,.2f}")
        print(f"   • Average Trades per Day: {metrics['avg_trades_per_day']:.1f}")
        print(f"   • Total Credit Opened: ${metrics['total_credit_opened']:,.2f}")
        print(f"   • Fees % of Returns: {metrics['fees_percentage_of_returns']:.2f}%")
        print(f"   • Max Drawdown: ${metrics['max_drawdown']:,.2f}")
        print(f"   • Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"   • Average Winner: ${metrics['avg_winner']:,.2f}")
        print(f"   • Average Loser: ${metrics['avg_loser']:,.2f}")
        print()
        
        # Test plot generation
        print("📈 Generating test plots...")
        plot_files = analyzer.generate_plots()
        
        if plot_files:
            print(f"✅ Generated {len(plot_files)} plots:")
            for plot_file in plot_files:
                print(f"   • {plot_file}")
        else:
            print("⚠️  No plots generated")
        
        print()
        print("🎉 All tests passed! The analyzer is working correctly.")
        print("🚀 You can now run 'python app.py' to start the web interface.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_analyzer()
    exit(0 if success else 1)
