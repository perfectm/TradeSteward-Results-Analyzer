"""
Real-Time Risk Monitoring Dashboard - Phase 3 Implementation
Interactive web-based dashboard for live options trading risk monitoring
"""

import dash
from dash import dcc, html, dash_table, Input, Output, State, callback
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import threading
import time
import logging
from typing import Dict, List, Tuple, Optional
import warnings

# Import our risk management modules
from data_loader import TradeStewardDataLoader
from advanced_risk_measures import AdvancedRiskCalculator
from options_factor_models import OptionsFactorModel
from monte_carlo_simulation import MonteCarloSimulator, SimulationParameters
from portfolio_optimization import PortfolioOptimizer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class RealTimeRiskDashboard:
    """
    Real-time risk monitoring dashboard for options trading portfolios
    """
    
    def __init__(self, db_path: str = "../trades.db", update_interval: int = 30):
        """
        Initialize real-time dashboard
        
        Args:
            db_path: Path to TradeSteward database
            update_interval: Update interval in seconds
        """
        self.db_path = db_path
        self.update_interval = update_interval
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])
        
        # Data storage
        self.current_data = {}
        self.risk_metrics = {}
        self.alerts = []
        self.last_update = None
        
        # Risk limits (configurable)
        self.risk_limits = {
            'var_5_limit': 0.02,           # 2% daily VaR limit
            'max_drawdown_limit': 0.05,    # 5% maximum drawdown limit
            'delta_limit': 1000,           # Delta exposure limit
            'vega_limit': 2000,            # Vega exposure limit
            'concentration_limit': 0.4     # Maximum single strategy weight
        }
        
        # Initialize layout and callbacks
        self._setup_layout()
        self._setup_callbacks()
        
        # Start background data update thread
        self.data_thread = threading.Thread(target=self._update_data_loop, daemon=True)
        self.data_thread.start()
        
        logger.info("Real-time risk dashboard initialized")
    
    def _setup_layout(self):
        """Setup the dashboard layout"""
        
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1("TradeSteward Real-Time Risk Monitor", 
                       style={'textAlign': 'center', 'color': '#2c3e50', 'marginBottom': 30}),
                html.Div([
                    html.Div([
                        html.H4("Last Update:", style={'margin': 0}),
                        html.P(id='last-update-time', style={'margin': 0, 'fontSize': 14})
                    ], className='three columns'),
                    html.Div([
                        html.H4("Status:", style={'margin': 0}),
                        html.P(id='system-status', style={'margin': 0, 'fontSize': 14})
                    ], className='three columns'),
                    html.Div([
                        html.Button('Refresh Data', id='refresh-button', 
                                   style={'backgroundColor': '#3498db', 'color': 'white', 'border': 'none', 'padding': '10px 20px'})
                    ], className='three columns'),
                    html.Div([
                        html.Button('Export Report', id='export-button',
                                   style={'backgroundColor': '#27ae60', 'color': 'white', 'border': 'none', 'padding': '10px 20px'})
                    ], className='three columns')
                ], className='row', style={'marginBottom': 30})
            ]),
            
            # Alert Section
            html.Div(id='alerts-section', style={'marginBottom': 20}),
            
            # Key Metrics Cards
            html.Div([
                html.Div([
                    html.H4("Portfolio VaR (5%)", style={'textAlign': 'center', 'margin': 0}),
                    html.H2(id='var-metric', style={'textAlign': 'center', 'color': '#e74c3c'})
                ], className='three columns', style={'backgroundColor': '#ecf0f1', 'padding': 20, 'margin': 5}),
                
                html.Div([
                    html.H4("Total P&L", style={'textAlign': 'center', 'margin': 0}),
                    html.H2(id='pnl-metric', style={'textAlign': 'center'})
                ], className='three columns', style={'backgroundColor': '#ecf0f1', 'padding': 20, 'margin': 5}),
                
                html.Div([
                    html.H4("Portfolio Delta", style={'textAlign': 'center', 'margin': 0}),
                    html.H2(id='delta-metric', style={'textAlign': 'center', 'color': '#f39c12'})
                ], className='three columns', style={'backgroundColor': '#ecf0f1', 'padding': 20, 'margin': 5}),
                
                html.Div([
                    html.H4("Portfolio Vega", style={'textAlign': 'center', 'margin': 0}),
                    html.H2(id='vega-metric', style={'textAlign': 'center', 'color': '#9b59b6'})
                ], className='three columns', style={'backgroundColor': '#ecf0f1', 'padding': 20, 'margin': 5})
            ], className='row', style={'marginBottom': 30}),
            
            # Charts Section
            html.Div([
                # Risk Measures Chart
                html.Div([
                    dcc.Graph(id='risk-measures-chart')
                ], className='six columns'),
                
                # Portfolio Composition
                html.Div([
                    dcc.Graph(id='portfolio-composition-chart')
                ], className='six columns')
            ], className='row', style={'marginBottom': 30}),
            
            html.Div([
                # Greeks Evolution
                html.Div([
                    dcc.Graph(id='greeks-evolution-chart')
                ], className='six columns'),
                
                # P&L Attribution
                html.Div([
                    dcc.Graph(id='pnl-attribution-chart')
                ], className='six columns')
            ], className='row', style={'marginBottom': 30}),
            
            # Detailed Tables
            html.Div([
                html.H3("Strategy Risk Breakdown", style={'marginBottom': 20}),
                dash_table.DataTable(
                    id='strategy-table',
                    columns=[
                        {"name": "Strategy", "id": "strategy"},
                        {"name": "P&L", "id": "pnl", "type": "numeric", "format": {"specifier": ",.2f"}},
                        {"name": "VaR", "id": "var", "type": "numeric", "format": {"specifier": ".4f"}},
                        {"name": "Delta", "id": "delta", "type": "numeric", "format": {"specifier": ",.0f"}},
                        {"name": "Vega", "id": "vega", "type": "numeric", "format": {"specifier": ",.0f"}},
                        {"name": "Trades", "id": "trades", "type": "numeric"},
                        {"name": "Weight", "id": "weight", "type": "numeric", "format": {"specifier": ".2%"}}
                    ],
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {
                            'if': {'row_index': 'odd'},
                            'backgroundColor': 'rgb(248, 248, 248)'
                        }
                    ],
                    style_header={
                        'backgroundColor': 'rgb(230, 230, 230)',
                        'fontWeight': 'bold'
                    }
                )
            ], style={'marginBottom': 30}),
            
            # Risk Limits Monitor
            html.Div([
                html.H3("Risk Limits Monitor", style={'marginBottom': 20}),
                dash_table.DataTable(
                    id='limits-table',
                    columns=[
                        {"name": "Risk Metric", "id": "metric"},
                        {"name": "Current", "id": "current"},
                        {"name": "Limit", "id": "limit"},
                        {"name": "Utilization", "id": "utilization", "type": "numeric", "format": {"specifier": ".1%"}},
                        {"name": "Status", "id": "status"}
                    ],
                    style_cell={'textAlign': 'center'},
                    style_data_conditional=[
                        {
                            'if': {'filter_query': '{status} = BREACH'},
                            'backgroundColor': '#ffebee',
                            'color': 'black',
                        },
                        {
                            'if': {'filter_query': '{status} = WARNING'},
                            'backgroundColor': '#fff3e0',
                            'color': 'black',
                        }
                    ]
                )
            ]),
            
            # Auto-refresh interval component
            dcc.Interval(
                id='interval-component',
                interval=self.update_interval * 1000,  # in milliseconds
                n_intervals=0
            ),
            
            # Hidden div to store data
            html.Div(id='data-store', style={'display': 'none'})
            
        ], style={'margin': '20px'})
    
    def _setup_callbacks(self):
        """Setup dashboard callbacks"""
        
        @self.app.callback(
            [Output('data-store', 'children'),
             Output('last-update-time', 'children'),
             Output('system-status', 'children')],
            [Input('interval-component', 'n_intervals'),
             Input('refresh-button', 'n_clicks')]
        )
        def update_data_store(n_intervals, refresh_clicks):
            """Update the data store"""
            try:
                self._load_current_data()
                
                status = "🟢 ACTIVE" if self.current_data else "🔴 NO DATA"
                last_update = self.last_update.strftime("%Y-%m-%d %H:%M:%S") if self.last_update else "Never"
                
                return json.dumps(self.current_data), last_update, status
                
            except Exception as e:
                logger.error(f"Error updating data: {str(e)}")
                return json.dumps({}), "Error", "🔴 ERROR"
        
        @self.app.callback(
            [Output('var-metric', 'children'),
             Output('pnl-metric', 'children'),
             Output('pnl-metric', 'style'),
             Output('delta-metric', 'children'),
             Output('vega-metric', 'children')],
            [Input('data-store', 'children')]
        )
        def update_key_metrics(data_json):
            """Update key metric cards"""
            try:
                if not data_json:
                    return "N/A", "N/A", {}, "N/A", "N/A"
                
                data = json.loads(data_json)
                
                # VaR
                var_value = data.get('risk_metrics', {}).get('VaR_5', 0)
                var_display = f"{var_value:.3f}"
                
                # P&L
                total_pnl = data.get('total_pnl', 0)
                pnl_display = f"${total_pnl:,.0f}"
                pnl_color = '#27ae60' if total_pnl >= 0 else '#e74c3c'
                pnl_style = {'textAlign': 'center', 'color': pnl_color}
                
                # Greeks
                delta = data.get('portfolio_greeks', {}).get('delta', 0)
                vega = data.get('portfolio_greeks', {}).get('vega', 0)
                delta_display = f"{delta:.0f}"
                vega_display = f"{vega:.0f}"
                
                return var_display, pnl_display, pnl_style, delta_display, vega_display
                
            except Exception as e:
                logger.error(f"Error updating metrics: {str(e)}")
                return "Error", "Error", {}, "Error", "Error"
        
        @self.app.callback(
            Output('alerts-section', 'children'),
            [Input('data-store', 'children')]
        )
        def update_alerts(data_json):
            """Update alerts section"""
            try:
                if not data_json:
                    return html.Div()
                
                # Check for risk limit breaches
                alerts = self._check_risk_limits(json.loads(data_json))
                
                if not alerts:
                    return html.Div([
                        html.Div("✅ All risk limits within acceptable ranges", 
                                style={'backgroundColor': '#d4edda', 'color': '#155724', 
                                      'padding': 10, 'borderRadius': 5, 'marginBottom': 10})
                    ])
                
                alert_divs = []
                for alert in alerts:
                    color = '#f8d7da' if alert['severity'] == 'BREACH' else '#fff3cd'
                    text_color = '#721c24' if alert['severity'] == 'BREACH' else '#856404'
                    
                    alert_divs.append(
                        html.Div(f"⚠️ {alert['message']}", 
                                style={'backgroundColor': color, 'color': text_color,
                                      'padding': 10, 'borderRadius': 5, 'marginBottom': 5})
                    )
                
                return html.Div(alert_divs)
                
            except Exception as e:
                logger.error(f"Error updating alerts: {str(e)}")
                return html.Div()
        
        @self.app.callback(
            Output('risk-measures-chart', 'figure'),
            [Input('data-store', 'children')]
        )
        def update_risk_chart(data_json):
            """Update risk measures chart"""
            try:
                if not data_json:
                    return go.Figure()
                
                data = json.loads(data_json)
                risk_metrics = data.get('risk_metrics', {})
                
                # Create risk measures comparison
                measures = ['VaR_1', 'VaR_5', 'VaR_10', 'CVaR_5', 'Max_Drawdown']
                values = [risk_metrics.get(measure, 0) for measure in measures]
                
                fig = go.Figure(data=[
                    go.Bar(x=[m.replace('_', ' ') for m in measures], y=values,
                           marker_color=['#e74c3c', '#f39c12', '#f1c40f', '#e67e22', '#c0392b'])
                ])
                
                fig.update_layout(
                    title='Risk Measures Overview',
                    yaxis_title='Risk Value',
                    template='plotly_white'
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error creating risk chart: {str(e)}")
                return go.Figure()
        
        @self.app.callback(
            Output('portfolio-composition-chart', 'figure'),
            [Input('data-store', 'children')]
        )
        def update_composition_chart(data_json):
            """Update portfolio composition chart"""
            try:
                if not data_json:
                    return go.Figure()
                
                data = json.loads(data_json)
                strategy_data = data.get('strategy_breakdown', {})
                
                if not strategy_data:
                    return go.Figure()
                
                strategies = list(strategy_data.keys())
                pnl_values = [strategy_data[s].get('total_pnl', 0) for s in strategies]
                
                fig = go.Figure(data=[
                    go.Pie(labels=strategies, values=[abs(p) for p in pnl_values],
                           hole=0.3)
                ])
                
                fig.update_layout(
                    title='Portfolio Composition by Strategy P&L',
                    template='plotly_white'
                )
                
                return fig
                
            except Exception as e:
                logger.error(f"Error creating composition chart: {str(e)}")
                return go.Figure()
        
        @self.app.callback(
            Output('strategy-table', 'data'),
            [Input('data-store', 'children')]
        )
        def update_strategy_table(data_json):
            """Update strategy breakdown table"""
            try:
                if not data_json:
                    return []
                
                data = json.loads(data_json)
                strategy_data = data.get('strategy_breakdown', {})
                total_pnl = sum(s.get('total_pnl', 0) for s in strategy_data.values())
                
                table_data = []
                for strategy, info in strategy_data.items():
                    pnl = info.get('total_pnl', 0)
                    weight = pnl / total_pnl if total_pnl != 0 else 0
                    
                    table_data.append({
                        'strategy': strategy,
                        'pnl': pnl,
                        'var': info.get('var_5', 0),
                        'delta': info.get('delta', 0),
                        'vega': info.get('vega', 0),
                        'trades': info.get('trade_count', 0),
                        'weight': weight
                    })
                
                return table_data
                
            except Exception as e:
                logger.error(f"Error updating strategy table: {str(e)}")
                return []
        
        @self.app.callback(
            Output('limits-table', 'data'),
            [Input('data-store', 'children')]
        )
        def update_limits_table(data_json):
            """Update risk limits table"""
            try:
                if not data_json:
                    return []
                
                data = json.loads(data_json)
                
                # Calculate current values and limit utilization
                current_var = data.get('risk_metrics', {}).get('VaR_5', 0)
                current_dd = data.get('risk_metrics', {}).get('Max_Drawdown', 0)
                current_delta = abs(data.get('portfolio_greeks', {}).get('delta', 0))
                current_vega = abs(data.get('portfolio_greeks', {}).get('vega', 0))
                
                # Max concentration
                strategy_data = data.get('strategy_breakdown', {})
                total_pnl = sum(abs(s.get('total_pnl', 0)) for s in strategy_data.values())
                max_concentration = max(abs(s.get('total_pnl', 0)) / total_pnl 
                                      for s in strategy_data.values()) if total_pnl > 0 else 0
                
                limits_data = [
                    {
                        'metric': 'VaR (5%)',
                        'current': f"{current_var:.4f}",
                        'limit': f"{self.risk_limits['var_5_limit']:.4f}",
                        'utilization': current_var / self.risk_limits['var_5_limit'],
                        'status': self._get_limit_status(current_var, self.risk_limits['var_5_limit'])
                    },
                    {
                        'metric': 'Max Drawdown',
                        'current': f"{current_dd:.4f}",
                        'limit': f"{self.risk_limits['max_drawdown_limit']:.4f}",
                        'utilization': current_dd / self.risk_limits['max_drawdown_limit'],
                        'status': self._get_limit_status(current_dd, self.risk_limits['max_drawdown_limit'])
                    },
                    {
                        'metric': 'Delta Exposure',
                        'current': f"{current_delta:.0f}",
                        'limit': f"{self.risk_limits['delta_limit']:.0f}",
                        'utilization': current_delta / self.risk_limits['delta_limit'],
                        'status': self._get_limit_status(current_delta, self.risk_limits['delta_limit'])
                    },
                    {
                        'metric': 'Vega Exposure',
                        'current': f"{current_vega:.0f}",
                        'limit': f"{self.risk_limits['vega_limit']:.0f}",
                        'utilization': current_vega / self.risk_limits['vega_limit'],
                        'status': self._get_limit_status(current_vega, self.risk_limits['vega_limit'])
                    },
                    {
                        'metric': 'Concentration',
                        'current': f"{max_concentration:.2%}",
                        'limit': f"{self.risk_limits['concentration_limit']:.2%}",
                        'utilization': max_concentration / self.risk_limits['concentration_limit'],
                        'status': self._get_limit_status(max_concentration, self.risk_limits['concentration_limit'])
                    }
                ]
                
                return limits_data
                
            except Exception as e:
                logger.error(f"Error updating limits table: {str(e)}")
                return []
    
    def _get_limit_status(self, current: float, limit: float) -> str:
        """Get status based on limit utilization"""
        utilization = current / limit if limit > 0 else 0
        
        if utilization >= 1.0:
            return "BREACH"
        elif utilization >= 0.8:
            return "WARNING"
        else:
            return "OK"
    
    def _load_current_data(self):
        """Load current risk data"""
        try:
            # Load data using existing modules
            loader = TradeStewardDataLoader(self.db_path)
            trades_df = loader.load_user_trades()
            
            if trades_df.empty:
                logger.warning("No trades data available")
                return
            
            # Process data
            enhanced_df = loader.parse_raw_data_json(trades_df)
            clean_df = loader.clean_and_validate_data(enhanced_df)
            daily_series = loader.create_daily_returns_series(clean_df)
            strategy_portfolios = loader.create_strategy_portfolios(clean_df)
            returns_matrix, factor_matrix = loader.export_for_riskfolio()
            
            # Calculate risk metrics
            risk_calc = AdvancedRiskCalculator(returns_matrix)
            n_assets = len(returns_matrix.columns)
            weights = np.ones(n_assets) / n_assets
            risk_analysis = risk_calc.comprehensive_risk_analysis(weights)
            
            # Calculate Greeks
            options_model = OptionsFactorModel(clean_df)
            portfolio_greeks = options_model.calculate_portfolio_greeks()
            
            # Get data summary
            data_summary = loader.get_data_summary()
            
            # Store current data
            self.current_data = {
                'risk_metrics': risk_analysis.get('VaR_Measures', {}),
                'portfolio_greeks': portfolio_greeks,
                'strategy_breakdown': self._format_strategy_data(strategy_portfolios, portfolio_greeks),
                'total_pnl': data_summary.get('total_pnl', 0),
                'total_trades': data_summary.get('total_trades', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Update drawdown metrics
            if 'Drawdown_Measures' in risk_analysis:
                self.current_data['risk_metrics']['Max_Drawdown'] = risk_analysis['Drawdown_Measures'].get('Maximum_Drawdown', 0)
            
            self.last_update = datetime.now()
            logger.info("Data updated successfully")
            
        except Exception as e:
            logger.error(f"Error loading current data: {str(e)}")
    
    def _format_strategy_data(self, strategy_portfolios: Dict, portfolio_greeks: Dict) -> Dict:
        """Format strategy data for dashboard"""
        strategy_data = {}
        
        # Distribute Greeks across strategies (simplified)
        n_strategies = len(strategy_portfolios)
        delta_per_strategy = portfolio_greeks.get('delta', 0) / n_strategies if n_strategies > 0 else 0
        vega_per_strategy = portfolio_greeks.get('vega', 0) / n_strategies if n_strategies > 0 else 0
        
        for strategy, data in strategy_portfolios.items():
            total_pnl = data['net_pnl'].sum()
            trade_count = data['trade_count'].sum()
            
            # Simplified VaR calculation (proportional to P&L)
            strategy_var = abs(total_pnl) * 0.001  # Simplified estimation
            
            strategy_data[strategy] = {
                'total_pnl': total_pnl,
                'trade_count': int(trade_count),
                'var_5': strategy_var,
                'delta': delta_per_strategy,
                'vega': vega_per_strategy
            }
        
        return strategy_data
    
    def _check_risk_limits(self, data: Dict) -> List[Dict]:
        """Check for risk limit breaches"""
        alerts = []
        
        try:
            # VaR limit check
            current_var = data.get('risk_metrics', {}).get('VaR_5', 0)
            if current_var > self.risk_limits['var_5_limit']:
                alerts.append({
                    'severity': 'BREACH',
                    'message': f"VaR breach: {current_var:.4f} > {self.risk_limits['var_5_limit']:.4f}"
                })
            elif current_var > self.risk_limits['var_5_limit'] * 0.8:
                alerts.append({
                    'severity': 'WARNING',
                    'message': f"VaR approaching limit: {current_var:.4f}"
                })
            
            # Delta limit check
            current_delta = abs(data.get('portfolio_greeks', {}).get('delta', 0))
            if current_delta > self.risk_limits['delta_limit']:
                alerts.append({
                    'severity': 'BREACH',
                    'message': f"Delta exposure breach: {current_delta:.0f} > {self.risk_limits['delta_limit']:.0f}"
                })
            
            # Vega limit check
            current_vega = abs(data.get('portfolio_greeks', {}).get('vega', 0))
            if current_vega > self.risk_limits['vega_limit']:
                alerts.append({
                    'severity': 'BREACH',
                    'message': f"Vega exposure breach: {current_vega:.0f} > {self.risk_limits['vega_limit']:.0f}"
                })
            
        except Exception as e:
            logger.error(f"Error checking risk limits: {str(e)}")
        
        return alerts
    
    def _update_data_loop(self):
        """Background thread for data updates"""
        while True:
            try:
                time.sleep(self.update_interval)
                # Data will be updated by dashboard callbacks
                logger.debug("Background update cycle")
            except Exception as e:
                logger.error(f"Error in background update: {str(e)}")
    
    def run(self, host='127.0.0.1', port=8050, debug=False):
        """Run the dashboard"""
        logger.info(f"Starting real-time risk dashboard on {host}:{port}")
        
        # Initial data load
        self._load_current_data()
        
        # Run the app
        self.app.run_server(host=host, port=port, debug=debug)


def main():
    """Example usage of Real-Time Risk Dashboard"""
    
    # Install required packages if not available
    try:
        import dash
        import plotly
    except ImportError:
        print("Installing required packages for dashboard...")
        import subprocess
        subprocess.check_call(["pip", "install", "dash", "plotly", "dash-bootstrap-components"])
        import dash
        import plotly
    
    print("Real-Time Risk Monitoring Dashboard - Phase 3")
    print("=" * 60)
    
    try:
        # Initialize dashboard
        dashboard = RealTimeRiskDashboard(
            db_path="../trades.db",
            update_interval=30  # Update every 30 seconds
        )
        
        print("✅ Dashboard initialized successfully")
        print(f"🌐 Starting web server...")
        print(f"📊 Dashboard will be available at: http://127.0.0.1:8050")
        print(f"🔄 Auto-refresh interval: 30 seconds")
        print(f"⚠️  Risk limits configured:")
        print(f"   • VaR (5%): {dashboard.risk_limits['var_5_limit']:.2%}")
        print(f"   • Max Drawdown: {dashboard.risk_limits['max_drawdown_limit']:.2%}")
        print(f"   • Delta Limit: {dashboard.risk_limits['delta_limit']:,}")
        print(f"   • Vega Limit: {dashboard.risk_limits['vega_limit']:,}")
        
        print("\n🚀 Starting dashboard server...")
        print("Press Ctrl+C to stop the server")
        
        # Run dashboard
        dashboard.run(host='127.0.0.1', port=8050, debug=False)
        
    except Exception as e:
        logger.error(f"Error running dashboard: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()