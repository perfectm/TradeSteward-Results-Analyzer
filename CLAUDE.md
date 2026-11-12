# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Running the Application
```bash
python app.py
```
This starts the FastAPI web server on http://localhost:8003 using uvicorn.

### Running Risk Management System
```bash
cd risk_management
python basic_risk_analysis.py      # Phase 1: Basic risk analysis
python advanced_risk_measures.py   # Phase 2: Advanced risk measures  
python real_time_monitoring.py     # Phase 3: Real-time monitoring
python deep_learning_models.py     # Phase 4: Deep learning models
python api_gateway.py              # Phase 4: API gateway
python mobile_dashboard.py         # Phase 4: Mobile dashboard
```

### Installing Dependencies
```bash
pip install -r requirements.txt

# Additional dependencies for risk management system:
pip install torch tensorflow scikit-learn
pip install plotly dash streamlit
pip install redis celery
pip install numba asyncio websockets
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

This is a comprehensive Python FastAPI application for analyzing TradeSteward options trading performance with an advanced enterprise-grade risk management system. The architecture consists of:

### Core Components
- **`app.py`**: Main FastAPI application containing the `TradeStewardAnalyzer` class and authentication system
- **`TradeStewardAnalyzer`**: Core analysis engine that processes CSV data and generates metrics/visualizations
- **`templates/index.html`**: Frontend dashboard with authentication, drag-and-drop file upload, and daily performance analysis
- **`static/`**: CSS and JavaScript files for the web interface
- **`uploads/plots/`**: Auto-generated directory for chart storage
- **SQLite Database**: User authentication and trade data storage with account-based privacy protection

### Risk Management System (`risk_management/`)
Comprehensive 4-phase enterprise risk management framework:

#### Phase 1: Basic Risk Analysis
- **`basic_risk_analysis.py`**: Foundation risk metrics (VaR, CVaR, Sharpe ratio, maximum drawdown)
- Portfolio-level risk assessment with Monte Carlo simulations
- Stress testing and scenario analysis
- Historical volatility and correlation analysis

#### Phase 2: Advanced Risk Measures  
- **`advanced_risk_measures.py`**: Sophisticated risk models (GARCH, copulas, extreme value theory)
- Expected shortfall calculations and tail risk analysis
- Dynamic hedging strategies and Greeks calculations
- Multi-factor risk attribution

#### Phase 3: Real-time Monitoring
- **`real_time_monitoring.py`**: Live risk monitoring with WebSocket feeds
- Automated alert systems with customizable thresholds
- Real-time P&L tracking and position monitoring
- Integration with market data providers

#### Phase 4: Enterprise Features
- **`deep_learning_models.py`**: AI-enhanced risk prediction with PyTorch/TensorFlow
  - LSTM volatility surface modeling
  - Transformer-based risk prediction
  - Black-Scholes neural network enhancement
- **`alternative_data_integration.py`**: News sentiment, options flow, social media analysis
- **`high_frequency_risk_monitor.py`**: Microsecond-level risk monitoring with Numba optimization
- **`cloud_deployment.py`**: Kubernetes auto-scaling deployment architecture
- **`multi_user_auth.py`**: Enterprise RBAC with 8 user roles and 20+ permissions
- **`api_gateway.py`**: External system integration (Bloomberg, Refinitiv, IBKR)
- **`advanced_reporting.py`**: Custom report builder with PDF/Excel/HTML export
- **`mobile_dashboard.py`**: Responsive mobile interface with real-time updates

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

#### Core Application Endpoints
- **GET `/`**: Serves the main dashboard interface
- **POST `/register`**: User registration with username, email, and password
- **POST `/login`**: User authentication with JWT token generation
- **POST `/logout`**: User logout (frontend-only token clearing)
- **POST `/forgot-password`**: Initiate password reset for unauthenticated users
- **POST `/update-forgotten-password`**: Update password without current authentication
- **POST `/reset-password`**: Reset password for authenticated users (requires current password)
- **POST `/upload`**: Accepts CSV upload and stores trade data (authenticated)
- **GET `/database-analysis`**: Retrieves analysis from stored trade data (authenticated)
- **GET `/daily-analysis`**: Provides daily performance breakdown (authenticated)
- **GET `/plots/{filename}`**: Serves generated chart images

#### Risk Management API Endpoints (via API Gateway)
- **GET `/api/v1/risk/portfolio/{portfolio_id}`**: Portfolio risk metrics
- **GET `/api/v1/risk/positions`**: Position-level risk analysis
- **GET `/api/v1/risk/var/{confidence_level}`**: Value at Risk calculations
- **GET `/api/v1/analytics/performance/{period}`**: Performance analytics
- **GET `/api/v1/analytics/correlations`**: Correlation matrix data
- **POST `/api/v1/webhooks`**: Create webhook endpoints
- **GET `/api/metrics/{user_id}`**: Mobile dashboard metrics
- **GET `/api/alerts/{user_id}`**: Risk alerts for users
- **WebSocket `/ws/{user_id}`**: Real-time updates for mobile dashboard

### Key Features

#### Enterprise Risk Management
- **Comprehensive Risk Metrics**: VaR, CVaR, Expected Shortfall, Greeks, stress testing
- **Advanced Modeling**: GARCH volatility models, copula analysis, extreme value theory
- **Real-time Monitoring**: Live risk feeds with customizable alerts and thresholds
- **AI-Enhanced Analysis**: Deep learning models for volatility prediction and risk forecasting
- **Alternative Data**: News sentiment, options flow, social media integration
- **High-Frequency Monitoring**: Microsecond-level risk calculation with Numba optimization
- **Enterprise Security**: Multi-user RBAC system with 8 roles and granular permissions
- **Mobile Access**: Responsive mobile dashboard with real-time WebSocket updates
- **Advanced Reporting**: Custom report builder with multiple export formats
- **Cloud Deployment**: Kubernetes auto-scaling architecture with Terraform IaC

#### External System Integration
- **Bloomberg Terminal API**: Market data and options pricing
- **Refinitiv Eikon**: Real-time financial data feeds
- **Interactive Brokers**: Account data and position monitoring
- **TD Ameritrade**: Quote data and market information
- **Alpha Vantage**: Historical market data
- **Quandl**: Economic and financial datasets

#### Technology Stack
- **Backend**: FastAPI, Python, AsyncIO, WebSockets
- **Database**: SQLite (development), PostgreSQL (production)
- **ML/AI**: PyTorch, TensorFlow, scikit-learn, Numba
- **Visualization**: Matplotlib, Seaborn, Plotly, Chart.js
- **Frontend**: HTML5, Bootstrap, JavaScript, WebSocket clients
- **Deployment**: Docker, Kubernetes, AWS, Terraform
- **Caching/Queuing**: Redis, Celery
- **Security**: JWT, bcrypt, RBAC, audit logging

### Testing
- **`test_analyzer.py`**: Tests core analyzer with sample TradeSteward CSV file
- **`test_forgot_password.py`**: Tests authentication and password reset functionality
- **Risk Management Tests**: Each module includes comprehensive testing capabilities
- **Integration Tests**: API gateway and external system connectivity testing