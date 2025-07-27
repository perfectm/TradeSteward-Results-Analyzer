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

```bash
python test_forgot_password.py
```
This tests the forgot password functionality for unauthenticated users.

## Architecture Overview

This is a Python FastAPI web application for analyzing TradeSteward options trading performance logs. The architecture consists of:

### Core Components
- **`app.py`**: Main FastAPI application containing the `TradeStewardAnalyzer` class and authentication system
- **`TradeStewardAnalyzer`**: Core analysis engine that processes CSV data and generates metrics/visualizations
- **`templates/index.html`**: Frontend dashboard with authentication, drag-and-drop file upload, and daily performance analysis
- **`static/`**: CSS and JavaScript files for the web interface
- **`uploads/plots/`**: Auto-generated directory for chart storage
- **SQLite Database**: User authentication and trade data storage with account-based privacy protection

### Data Processing Pipeline
1. **Authentication**: JWT-based user authentication with bcrypt password hashing
2. **CSV Upload**: User-specific file uploads with data validation and deduplication
3. **CSV Cleaning**: `clean_csv_content()` function handles malformed TradeSteward CSV exports
4. **Data Analysis**: Pandas-based processing of multi-leg options trading data with account privacy protection
5. **Metrics Calculation**: Comprehensive financial metrics (Sharpe ratio, UPI, drawdown, win rate, etc.)
6. **Visualization**: Matplotlib/Seaborn chart generation with four main chart sets:
   - Cumulative and daily P&L performance
   - Strategy analysis with trade counts  
   - Risk analysis including VIX correlation and drawdown timeline
   - Commission analysis with account-based filtering
7. **Daily Analysis**: Interactive daily performance breakdown with persistent view protection

### Key Data Structure
The analyzer expects TradeSteward CSV files with specific columns:
- Multi-leg options data: `Opt1-Symbol` through `Opt4-Symbol` (up to 4 legs)
- Performance metrics: `TotalNetProfitLoss`, `TotalGrossProfitLoss`
- Market data: `VIXOpenQuote`, `VIXCloseQuote`, `UnderlyingOpenQuote`
- Strategy information: `Strategy`, `BotName`
- Timestamp data: `OpenDate`, `OpenTime`, `FinalTradeClosedDate`

### API Endpoints

#### Authentication Endpoints
- **POST `/register`**: User registration with username, email, and password
- **POST `/login`**: User authentication with JWT token generation
- **POST `/logout`**: User logout (frontend-only token clearing)
- **POST `/forgot-password`**: Initiate password reset for unauthenticated users
- **POST `/update-forgotten-password`**: Update password without current authentication
- **POST `/reset-password`**: Reset password for authenticated users (requires current password)

#### Analysis Endpoints
- **GET `/`**: Serves the main dashboard interface
- **POST `/upload`**: Accepts CSV upload and stores trade data (authenticated)
- **GET `/database-analysis`**: Retrieves analysis from stored trade data (authenticated)
- **GET `/daily-analysis`**: Provides daily performance breakdown (authenticated)
- **GET `/plots/{filename}`**: Serves generated chart images

### Testing
The `test_analyzer.py` script tests the analyzer with the included sample CSV file `tradeSteward-performanceLogs-1753563010.csv`.