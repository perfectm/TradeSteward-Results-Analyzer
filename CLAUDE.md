# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
python app.py
```
This starts the FastAPI web server on http://localhost:8000 using uvicorn.

### Installing Dependencies
```bash
pip install -r requirements.txt
```

### Running Tests
```bash
python test_analyzer.py
```
This runs the test script which analyzes a sample TradeSteward CSV file.

## Architecture Overview

This is a Python FastAPI web application for analyzing TradeSteward options trading performance logs. The architecture consists of:

### Core Components
- **`app.py`**: Main FastAPI application containing the `TradeStewardAnalyzer` class
- **`TradeStewardAnalyzer`**: Core analysis engine that processes CSV data and generates metrics/visualizations
- **`templates/index.html`**: Frontend dashboard with drag-and-drop file upload
- **`static/`**: CSS and JavaScript files for the web interface
- **`uploads/plots/`**: Auto-generated directory for chart storage

### Data Processing Pipeline
1. **CSV Cleaning**: `clean_csv_content()` function handles malformed TradeSteward CSV exports
2. **Data Analysis**: Pandas-based processing of multi-leg options trading data
3. **Metrics Calculation**: Comprehensive financial metrics (Sharpe ratio, drawdown, win rate, etc.)
4. **Visualization**: Matplotlib/Seaborn chart generation with three main chart sets:
   - Cumulative and daily P&L performance
   - Strategy analysis with trade counts
   - Risk analysis including VIX correlation and drawdown timeline

### Key Data Structure
The analyzer expects TradeSteward CSV files with specific columns:
- Multi-leg options data: `Opt1-Symbol` through `Opt4-Symbol` (up to 4 legs)
- Performance metrics: `TotalNetProfitLoss`, `TotalGrossProfitLoss`
- Market data: `VIXOpenQuote`, `VIXCloseQuote`, `UnderlyingOpenQuote`
- Strategy information: `Strategy`, `BotName`
- Timestamp data: `OpenDate`, `OpenTime`, `FinalTradeClosedDate`

### API Endpoints
- **GET `/`**: Serves the main dashboard interface
- **POST `/analyze`**: Accepts CSV upload and returns analysis results
- **GET `/plots/{filename}`**: Serves generated chart images

### Testing
The `test_analyzer.py` script tests the analyzer with the included sample CSV file `tradeSteward-performanceLogs-1753563010.csv`.