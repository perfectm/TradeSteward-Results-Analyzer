# 🎯 TradeSteward Results Analyzer

A comprehensive analysis tool specifically designed for TradeSteward options trading performance logs. This application provides advanced analytics, risk metrics, and visualizations for complex multi-leg options strategies.

## 📊 Features

### **Strategy Performance Analysis**
- **Multi-Strategy Support**: Ratio Condors, Call/Put Side strategies, and more
- **P&L Breakdown**: Individual strategy performance with detailed metrics
- **Trade Count Analysis**: Volume and frequency analysis by strategy type

### **Advanced Risk Metrics**
- **Win Rate & Profit Factor**: Trading efficiency measurements
- **Maximum Drawdown**: Peak-to-trough decline analysis
- **Sharpe Ratio**: Risk-adjusted return calculations
- **Volatility Analysis**: Daily and annualized volatility metrics

### **Market Environment Analysis**
- **VIX Correlation**: Market fear index impact on trading performance
- **Market Condition Mapping**: Performance across different volatility regimes
- **Underlying Price Movement**: SPX price correlation with trade outcomes

### **Time-Based Analytics**
- **Daily P&L Tracking**: Cumulative performance over time
- **Monthly Performance**: Seasonal and periodic analysis
- **Trading Day Statistics**: Active trading patterns and frequencies

### **Commission & Fee Analysis**
- **Gross vs Net P&L**: Impact of trading costs on performance
- **Fee Breakdown**: Regulatory fees, broker commissions, exchange fees
- **Cost Efficiency**: Fee analysis across different strategies

## 🚀 Quick Start

### Installation

1. **Clone or navigate to the project directory**:
```bash
cd TradeSteward-Results-Analyzer
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
python app.py
```

4. **Open your browser** and navigate to `http://localhost:8000`

### Usage

1. **Upload your TradeSteward CSV file** using the drag-and-drop interface
2. **View comprehensive metrics** including win rate, profit factor, and drawdown
3. **Analyze strategy performance** with detailed breakdowns by strategy type
4. **Examine risk charts** including P&L distribution and VIX correlation
5. **Review time-based analysis** with daily and monthly performance tracking

## 📈 Supported Data Format

The analyzer is specifically designed for TradeSteward performance logs with the following key columns:

### **Core Trade Information**
- `OpenOrderNumber` - Unique trade identifier
- `BotName` - Trading bot/strategy name
- `Strategy` - Strategy type (e.g., "Ratio Condor Separate")
- `OpenDate` & `OpenTime` - Trade entry timestamp
- `FinalTradeClosedDate` & `FinalTradeClosedTime` - Trade exit timestamp

### **Market Data**
- `Underlying` - SPX (underlying symbol)
- `UnderlyingOpenQuote` & `UnderlyingCloseQuote` - SPX prices
- `VIXOpenQuote` & `VIXCloseQuote` - VIX volatility readings

### **Multi-Leg Options Details**
- `Opt1-Symbol` through `Opt4-Symbol` - Option symbols for up to 4 legs
- `Opt1-Strike` through `Opt4-Strike` - Strike prices
- `Opt1-PutCall` through `Opt4-PutCall` - Put/Call designation
- `Opt1-Qty` through `Opt4-Qty` - Position quantities
- `Opt1-Delta` through `Opt4-Delta` - Greeks information

### **Performance Metrics**
- `TotalNetProfitLoss` - Final P&L after all fees
- `TotalGrossProfitLoss` - P&L before fees
- `TotalProfitLossPercent` - Percentage return
- `RegulatoryFees_Open/Close` - Fee breakdowns
- `BrokerCommissions_Open/Close` - Commission costs

## 🎨 Dashboard Features

### **Interactive Metrics Dashboard**
- Real-time calculation of key trading metrics
- Color-coded performance indicators (green/red)
- Professional metric cards with clear labeling

### **Comprehensive Visualizations**
1. **Cumulative P&L Chart**: Track portfolio growth over time
2. **Daily P&L Bar Chart**: Daily performance with profit/loss indicators
3. **Strategy Performance Analysis**: P&L and trade count by strategy
4. **Risk Analysis Quartet**:
   - P&L Distribution histogram
   - VIX vs Trade P&L scatter plot
   - Monthly performance bar chart
   - Drawdown timeline

### **Strategy Performance Table**
- Sortable table with strategy breakdown
- Total P&L, average P&L, trade count, and average percentage
- Color-coded positive/negative performance indicators

## 🔧 Technical Architecture

### **Backend (FastAPI)**
- **TradeStewardAnalyzer Class**: Core analysis engine
- **Data Processing**: Pandas-based CSV parsing and cleaning
- **Metrics Calculation**: Comprehensive financial metrics computation
- **Visualization Engine**: Matplotlib/Seaborn chart generation

### **Frontend (HTML/CSS/JavaScript)**
- **Responsive Design**: Mobile-friendly interface
- **Drag & Drop Upload**: Intuitive file upload experience
- **Real-time Loading**: Progress indicators and error handling
- **Interactive Dashboard**: Dynamic metric and chart display

### **Key Components**
- `app.py` - Main FastAPI application with analysis engine
- `templates/index.html` - Frontend interface with interactive features
- `requirements.txt` - Python dependencies
- Auto-generated `uploads/plots/` - Chart storage directory

## 📊 Calculated Metrics

### **Performance Metrics**
- **Total P&L**: Sum of all trade net profits/losses
- **Win Rate**: Percentage of profitable trades
- **Average Winner/Loser**: Mean profit/loss of winning/losing trades
- **Profit Factor**: Ratio of gross profits to gross losses

### **Risk Metrics**
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Annualized standard deviation of daily returns
- **Sharpe Ratio**: Risk-adjusted return (daily returns * 252 / volatility)

### **Time-Based Metrics**
- **Trading Days**: Number of active trading days
- **Average Daily P&L**: Mean daily profit/loss
- **Monthly Performance**: Aggregated monthly P&L analysis

### **Strategy-Specific Metrics**
- **Strategy P&L**: Total and average P&L by strategy type
- **Trade Count**: Number of trades per strategy
- **Strategy Percentage**: Average percentage return by strategy

## 🎯 Perfect For

- **Options Traders** using TradeSteward for automated trading
- **Portfolio Managers** analyzing complex multi-leg strategies
- **Risk Managers** requiring detailed drawdown and volatility analysis
- **Quantitative Analysts** studying strategy performance patterns
- **Trading System Developers** optimizing algorithmic strategies

## 📝 Example Analysis Output

When you upload a TradeSteward CSV file, you'll receive:

1. **8 Key Metric Cards**: Total P&L, Trade Count, Win Rate, Profit Factor, Max Drawdown, Sharpe Ratio, Average Winner/Loser
2. **Strategy Performance Table**: Detailed breakdown by strategy type
3. **3 Comprehensive Chart Sets**: 
   - Cumulative & Daily P&L performance
   - Strategy analysis with trade counts
   - Risk analysis with VIX correlation and drawdown

## 🛠️ Requirements

- Python 3.8+
- FastAPI for web framework
- Pandas for data processing
- Matplotlib/Seaborn for visualizations
- Modern web browser for interface

## 📄 License

This project is designed specifically for TradeSteward CSV format analysis. The tool processes complex options trading data with multi-leg strategies and provides institutional-grade analytics for trading performance evaluation.

---

**Built for traders who demand sophisticated analysis of their TradeSteward options trading results.** 🚀📈
