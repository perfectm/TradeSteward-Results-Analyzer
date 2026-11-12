"""
Automated Alert and Notification System - Phase 3 Implementation
Comprehensive risk monitoring alerts with multiple notification channels
"""

import smtplib
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from email.mime.base import MimeBase
from email import encoders
import threading
import time
import requests
import sqlite3
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# Import our risk management modules
from data_loader import TradeStewardDataLoader
from advanced_risk_measures import AdvancedRiskCalculator
from options_factor_models import OptionsFactorModel
from ml_risk_prediction import MLRiskPredictor

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"

class AlertType(Enum):
    """Types of risk alerts"""
    VAR_BREACH = "VAR_BREACH"
    GREEKS_LIMIT = "GREEKS_LIMIT"
    DRAWDOWN_ALERT = "DRAWDOWN_ALERT"
    CONCENTRATION_RISK = "CONCENTRATION_RISK"
    VOLATILITY_SPIKE = "VOLATILITY_SPIKE"
    CORRELATION_BREAKDOWN = "CORRELATION_BREAKDOWN"
    ML_PREDICTION = "ML_PREDICTION"
    SYSTEM_ERROR = "SYSTEM_ERROR"
    MARKET_REGIME_CHANGE = "MARKET_REGIME_CHANGE"

@dataclass
class RiskAlert:
    """Risk alert data structure"""
    alert_id: str
    timestamp: datetime
    alert_type: AlertType
    severity: AlertSeverity
    title: str
    message: str
    current_value: float
    threshold_value: float
    metric_name: str
    strategy: Optional[str] = None
    recommendations: List[str] = None
    data: Dict[str, Any] = None
    
    def to_dict(self):
        """Convert alert to dictionary"""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp.isoformat(),
            'alert_type': self.alert_type.value,
            'severity': self.severity.value,
            'title': self.title,
            'message': self.message,
            'current_value': self.current_value,
            'threshold_value': self.threshold_value,
            'metric_name': self.metric_name,
            'strategy': self.strategy,
            'recommendations': self.recommendations or [],
            'data': self.data or {}
        }

@dataclass
class NotificationConfig:
    """Notification configuration"""
    email_enabled: bool = True
    email_smtp_server: str = "smtp.gmail.com"
    email_port: int = 587
    email_username: str = ""
    email_password: str = ""
    email_recipients: List[str] = None
    
    slack_enabled: bool = False
    slack_webhook_url: str = ""
    slack_channel: str = "#risk-alerts"
    
    sms_enabled: bool = False
    sms_api_key: str = ""
    sms_recipients: List[str] = None
    
    webhook_enabled: bool = False
    webhook_url: str = ""
    
    file_logging: bool = True
    log_file: str = "risk_alerts.log"

class RiskLimitsConfig:
    """Risk limits configuration"""
    
    def __init__(self):
        self.limits = {
            # VaR limits
            'var_1_daily': 0.01,      # 1% daily VaR limit
            'var_5_daily': 0.02,      # 2% daily VaR limit
            'var_10_daily': 0.035,    # 3.5% daily VaR limit
            
            # CVaR limits
            'cvar_5_daily': 0.025,    # 2.5% daily CVaR limit
            
            # Greeks limits
            'delta_limit': 1000,       # Delta exposure limit
            'gamma_limit': 50,         # Gamma exposure limit
            'vega_limit': 2000,        # Vega exposure limit
            'theta_limit': 500,        # Theta exposure limit
            
            # Drawdown limits
            'max_drawdown': 0.05,      # 5% maximum drawdown
            'current_drawdown': 0.03,  # 3% current drawdown alert
            
            # Concentration limits
            'single_strategy_weight': 0.4,  # 40% max single strategy
            'top_3_strategies_weight': 0.7, # 70% max top 3 strategies
            
            # Volatility limits
            'portfolio_volatility': 0.02,   # 2% daily volatility limit
            'volatility_spike_factor': 2.0,  # Alert if vol > 2x average
            
            # Correlation limits
            'max_correlation': 0.8,          # Maximum strategy correlation
            'correlation_spike': 0.3,        # Alert if correlation jumps 30%
            
            # ML prediction limits
            'ml_risk_score': 0.7,           # ML risk score threshold
            'prediction_confidence': 0.6    # Minimum prediction confidence
        }
    
    def get_limit(self, metric: str) -> float:
        """Get limit for specific metric"""
        return self.limits.get(metric, 0.0)
    
    def update_limit(self, metric: str, value: float):
        """Update limit for specific metric"""
        self.limits[metric] = value

class AlertDatabase:
    """Database for storing and managing alerts"""
    
    def __init__(self, db_path: str = "risk_alerts.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize alerts database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS risk_alerts (
                alert_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                alert_type TEXT NOT NULL,
                severity TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                current_value REAL NOT NULL,
                threshold_value REAL NOT NULL,
                metric_name TEXT NOT NULL,
                strategy TEXT,
                acknowledged INTEGER DEFAULT 0,
                resolved INTEGER DEFAULT 0,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                data TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp ON risk_alerts(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_severity ON risk_alerts(severity)
        ''')
        
        conn.commit()
        conn.close()
    
    def store_alert(self, alert: RiskAlert):
        """Store alert in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO risk_alerts 
            (alert_id, timestamp, alert_type, severity, title, message, 
             current_value, threshold_value, metric_name, strategy, data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            alert.alert_id,
            alert.timestamp.isoformat(),
            alert.alert_type.value,
            alert.severity.value,
            alert.title,
            alert.message,
            alert.current_value,
            alert.threshold_value,
            alert.metric_name,
            alert.strategy,
            json.dumps(alert.data) if alert.data else None
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        conn = sqlite3.connect(self.db_path)
        
        since_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        query = '''
            SELECT * FROM risk_alerts 
            WHERE timestamp >= ? 
            ORDER BY timestamp DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=[since_time])
        conn.close()
        
        return df.to_dict('records')
    
    def acknowledge_alert(self, alert_id: str):
        """Mark alert as acknowledged"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE risk_alerts 
            SET acknowledged = 1 
            WHERE alert_id = ?
        ''', (alert_id,))
        
        conn.commit()
        conn.close()
    
    def get_alert_statistics(self) -> Dict:
        """Get alert statistics"""
        conn = sqlite3.connect(self.db_path)
        
        # Total alerts by severity
        severity_query = '''
            SELECT severity, COUNT(*) as count 
            FROM risk_alerts 
            GROUP BY severity
        '''
        severity_df = pd.read_sql_query(severity_query, conn)
        
        # Alerts by type
        type_query = '''
            SELECT alert_type, COUNT(*) as count 
            FROM risk_alerts 
            GROUP BY alert_type
        '''
        type_df = pd.read_sql_query(type_query, conn)
        
        conn.close()
        
        return {
            'by_severity': severity_df.to_dict('records'),
            'by_type': type_df.to_dict('records'),
            'total_alerts': len(severity_df)
        }

class NotificationManager:
    """Manages different notification channels"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.NotificationManager')
    
    def send_alert(self, alert: RiskAlert) -> bool:
        """Send alert through all enabled channels"""
        success = True
        
        try:
            # Email notification
            if self.config.email_enabled and self.config.email_recipients:
                success &= self._send_email(alert)
            
            # Slack notification
            if self.config.slack_enabled and self.config.slack_webhook_url:
                success &= self._send_slack(alert)
            
            # SMS notification
            if self.config.sms_enabled and self.config.sms_recipients:
                success &= self._send_sms(alert)
            
            # Webhook notification
            if self.config.webhook_enabled and self.config.webhook_url:
                success &= self._send_webhook(alert)
            
            # File logging
            if self.config.file_logging:
                self._log_to_file(alert)
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending notifications: {str(e)}")
            return False
    
    def _send_email(self, alert: RiskAlert) -> bool:
        """Send email notification"""
        try:
            msg = MimeMultipart()
            msg['From'] = self.config.email_username
            msg['To'] = ', '.join(self.config.email_recipients)
            msg['Subject'] = f"[{alert.severity.value}] TradeSteward Risk Alert: {alert.title}"
            
            # Create HTML email body
            html_body = self._create_email_body(alert)
            msg.attach(MimeText(html_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.config.email_smtp_server, self.config.email_port)
            server.starttls()
            server.login(self.config.email_username, self.config.email_password)
            
            text = msg.as_string()
            server.sendmail(self.config.email_username, self.config.email_recipients, text)
            server.quit()
            
            self.logger.info(f"Email alert sent successfully: {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")
            return False
    
    def _send_slack(self, alert: RiskAlert) -> bool:
        """Send Slack notification"""
        try:
            # Create Slack message
            color = {
                AlertSeverity.INFO: "good",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.EMERGENCY: "danger"
            }.get(alert.severity, "warning")
            
            slack_message = {
                "channel": self.config.slack_channel,
                "username": "TradeSteward Risk Monitor",
                "icon_emoji": ":warning:",
                "attachments": [{
                    "color": color,
                    "title": f"{alert.severity.value}: {alert.title}",
                    "text": alert.message,
                    "fields": [
                        {"title": "Metric", "value": alert.metric_name, "short": True},
                        {"title": "Current Value", "value": f"{alert.current_value:.4f}", "short": True},
                        {"title": "Threshold", "value": f"{alert.threshold_value:.4f}", "short": True},
                        {"title": "Timestamp", "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "short": True}
                    ],
                    "footer": "TradeSteward Risk Management",
                    "ts": int(alert.timestamp.timestamp())
                }]
            }
            
            # Send to Slack
            response = requests.post(self.config.slack_webhook_url, json=slack_message)
            response.raise_for_status()
            
            self.logger.info(f"Slack alert sent successfully: {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send Slack alert: {str(e)}")
            return False
    
    def _send_sms(self, alert: RiskAlert) -> bool:
        """Send SMS notification (placeholder for SMS service integration)"""
        try:
            # This would integrate with SMS services like Twilio, AWS SNS, etc.
            # For now, just log the SMS that would be sent
            
            sms_text = f"TradeSteward Alert [{alert.severity.value}]: {alert.title}. {alert.metric_name}: {alert.current_value:.4f} (Limit: {alert.threshold_value:.4f})"
            
            self.logger.info(f"SMS alert (simulated): {sms_text}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send SMS alert: {str(e)}")
            return False
    
    def _send_webhook(self, alert: RiskAlert) -> bool:
        """Send webhook notification"""
        try:
            payload = alert.to_dict()
            response = requests.post(self.config.webhook_url, json=payload, timeout=10)
            response.raise_for_status()
            
            self.logger.info(f"Webhook alert sent successfully: {alert.alert_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {str(e)}")
            return False
    
    def _log_to_file(self, alert: RiskAlert):
        """Log alert to file"""
        try:
            with open(self.config.log_file, 'a') as f:
                log_entry = {
                    'timestamp': alert.timestamp.isoformat(),
                    'severity': alert.severity.value,
                    'type': alert.alert_type.value,
                    'title': alert.title,
                    'message': alert.message,
                    'metric': alert.metric_name,
                    'current_value': alert.current_value,
                    'threshold': alert.threshold_value
                }
                f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            self.logger.error(f"Failed to log alert to file: {str(e)}")
    
    def _create_email_body(self, alert: RiskAlert) -> str:
        """Create HTML email body"""
        color = {
            AlertSeverity.INFO: "#17a2b8",
            AlertSeverity.WARNING: "#ffc107",
            AlertSeverity.CRITICAL: "#dc3545",
            AlertSeverity.EMERGENCY: "#dc3545"
        }.get(alert.severity, "#ffc107")
        
        html = f"""
        <html>
        <head></head>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background-color: {color}; color: white; padding: 20px; text-align: center;">
                    <h1>{alert.severity.value} ALERT</h1>
                    <h2>{alert.title}</h2>
                </div>
                
                <div style="padding: 20px; background-color: #f8f9fa;">
                    <h3>Alert Details</h3>
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 8px; font-weight: bold;">Timestamp:</td>
                            <td style="padding: 8px;">{alert.timestamp.strftime("%Y-%m-%d %H:%M:%S")}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 8px; font-weight: bold;">Alert Type:</td>
                            <td style="padding: 8px;">{alert.alert_type.value}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 8px; font-weight: bold;">Metric:</td>
                            <td style="padding: 8px;">{alert.metric_name}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 8px; font-weight: bold;">Current Value:</td>
                            <td style="padding: 8px;">{alert.current_value:.4f}</td>
                        </tr>
                        <tr style="border-bottom: 1px solid #dee2e6;">
                            <td style="padding: 8px; font-weight: bold;">Threshold:</td>
                            <td style="padding: 8px;">{alert.threshold_value:.4f}</td>
                        </tr>
                        {f'<tr style="border-bottom: 1px solid #dee2e6;"><td style="padding: 8px; font-weight: bold;">Strategy:</td><td style="padding: 8px;">{alert.strategy}</td></tr>' if alert.strategy else ''}
                    </table>
                </div>
                
                <div style="padding: 20px;">
                    <h3>Description</h3>
                    <p>{alert.message}</p>
                    
                    {f'<h3>Recommendations</h3><ul>{"".join([f"<li>{rec}</li>" for rec in alert.recommendations])}</ul>' if alert.recommendations else ''}
                </div>
                
                <div style="background-color: #e9ecef; padding: 10px; text-align: center; font-size: 12px;">
                    TradeSteward Risk Management System - Alert ID: {alert.alert_id}
                </div>
            </div>
        </body>
        </html>
        """
        
        return html

class AutomatedAlertSystem:
    """Main automated alert and notification system"""
    
    def __init__(self, 
                 db_path: str = "../trades.db",
                 notification_config: Optional[NotificationConfig] = None,
                 risk_limits: Optional[RiskLimitsConfig] = None,
                 check_interval: int = 60):
        """
        Initialize automated alert system
        
        Args:
            db_path: Path to TradeSteward database
            notification_config: Notification configuration
            risk_limits: Risk limits configuration
            check_interval: Check interval in seconds
        """
        self.db_path = db_path
        self.notification_config = notification_config or NotificationConfig()
        self.risk_limits = risk_limits or RiskLimitsConfig()
        self.check_interval = check_interval
        
        # Initialize components
        self.alert_db = AlertDatabase()
        self.notification_manager = NotificationManager(self.notification_config)
        
        # Risk monitoring components
        self.data_loader = None
        self.risk_calculator = None
        self.options_model = None
        self.ml_predictor = None
        
        # Running state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Alert suppression (prevent spam)
        self.alert_cooldown = {}  # Track recent alerts to prevent spam
        self.cooldown_period = 300  # 5 minutes cooldown
        
        logger.info("Automated alert system initialized")
    
    def start_monitoring(self):
        """Start continuous risk monitoring"""
        if self.monitoring_active:
            logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Risk monitoring started with {self.check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous risk monitoring"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Risk monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                self._perform_risk_checks()
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(self.check_interval)
    
    def _perform_risk_checks(self):
        """Perform comprehensive risk checks"""
        try:
            # Load current data
            self._initialize_risk_components()
            
            # Load and process trades data
            trades_df = self.data_loader.load_user_trades()
            if trades_df.empty:
                return
            
            enhanced_df = self.data_loader.parse_raw_data_json(trades_df)
            clean_df = self.data_loader.clean_and_validate_data(enhanced_df)
            returns_matrix, factor_matrix = self.data_loader.export_for_riskfolio()
            
            # Perform various risk checks
            self._check_var_limits(returns_matrix)
            self._check_greeks_limits(clean_df)
            self._check_drawdown_limits(returns_matrix)
            self._check_concentration_limits(clean_df)
            self._check_volatility_spikes(returns_matrix)
            self._check_ml_predictions(returns_matrix, factor_matrix)
            
            logger.debug("Risk checks completed successfully")
            
        except Exception as e:
            logger.error(f"Error performing risk checks: {str(e)}")
            self._create_system_error_alert(str(e))
    
    def _initialize_risk_components(self):
        """Initialize risk calculation components if needed"""
        if not self.data_loader:
            self.data_loader = TradeStewardDataLoader(self.db_path)
    
    def _check_var_limits(self, returns_matrix: pd.DataFrame):
        """Check VaR limit breaches"""
        try:
            # Initialize risk calculator
            risk_calc = AdvancedRiskCalculator(returns_matrix, alpha=0.05)
            
            # Equal weight portfolio
            n_assets = len(returns_matrix.columns)
            weights = np.ones(n_assets) / n_assets
            
            # Calculate VaR measures
            risk_analysis = risk_calc.comprehensive_risk_analysis(weights)
            var_measures = risk_analysis.get('VaR_Measures', {})
            
            # Check different VaR levels
            var_checks = [
                ('VaR_1', 'var_1_daily', 'Daily VaR (1%)'),
                ('VaR_5', 'var_5_daily', 'Daily VaR (5%)'),
                ('VaR_10', 'var_10_daily', 'Daily VaR (10%)'),
                ('CVaR_5', 'cvar_5_daily', 'Daily CVaR (5%)')
            ]
            
            for var_key, limit_key, metric_name in var_checks:
                var_value = var_measures.get(var_key, 0)
                limit = self.risk_limits.get_limit(limit_key)
                
                if var_value > limit:
                    self._create_var_alert(var_value, limit, metric_name, var_key)
            
        except Exception as e:
            logger.error(f"Error checking VaR limits: {str(e)}")
    
    def _check_greeks_limits(self, clean_df: pd.DataFrame):
        """Check Greeks exposure limits"""
        try:
            # Initialize options model
            options_model = OptionsFactorModel(clean_df)
            portfolio_greeks = options_model.calculate_portfolio_greeks()
            
            # Check Greeks limits
            greeks_checks = [
                ('delta', 'delta_limit', 'Portfolio Delta'),
                ('gamma', 'gamma_limit', 'Portfolio Gamma'),
                ('vega', 'vega_limit', 'Portfolio Vega'),
                ('theta', 'theta_limit', 'Portfolio Theta')
            ]
            
            for greek, limit_key, metric_name in greeks_checks:
                current_value = abs(portfolio_greeks.get(greek, 0))
                limit = self.risk_limits.get_limit(limit_key)
                
                if current_value > limit:
                    self._create_greeks_alert(current_value, limit, metric_name, greek)
            
        except Exception as e:
            logger.error(f"Error checking Greeks limits: {str(e)}")
    
    def _check_drawdown_limits(self, returns_matrix: pd.DataFrame):
        """Check drawdown limits"""
        try:
            # Calculate portfolio returns
            n_assets = len(returns_matrix.columns)
            weights = np.ones(n_assets) / n_assets
            portfolio_returns = (returns_matrix * weights).sum(axis=1)
            
            # Calculate cumulative returns and drawdown
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            current_drawdown = abs(drawdown.iloc[-1])
            max_drawdown = abs(drawdown.min())
            
            # Check current drawdown
            current_limit = self.risk_limits.get_limit('current_drawdown')
            if current_drawdown > current_limit:
                self._create_drawdown_alert(current_drawdown, current_limit, 'Current Drawdown')
            
            # Check maximum drawdown
            max_limit = self.risk_limits.get_limit('max_drawdown')
            if max_drawdown > max_limit:
                self._create_drawdown_alert(max_drawdown, max_limit, 'Maximum Drawdown')
            
        except Exception as e:
            logger.error(f"Error checking drawdown limits: {str(e)}")
    
    def _check_concentration_limits(self, clean_df: pd.DataFrame):
        """Check concentration risk limits"""
        try:
            # Calculate strategy weights by P&L
            strategy_pnl = clean_df.groupby('strategy')['net_pnl'].sum()
            total_pnl = abs(strategy_pnl).sum()
            
            if total_pnl > 0:
                strategy_weights = abs(strategy_pnl) / total_pnl
                
                # Check single strategy concentration
                max_weight = strategy_weights.max()
                single_limit = self.risk_limits.get_limit('single_strategy_weight')
                
                if max_weight > single_limit:
                    max_strategy = strategy_weights.idxmax()
                    self._create_concentration_alert(
                        max_weight, single_limit, 
                        f'Single Strategy Concentration ({max_strategy})', 
                        max_strategy
                    )
                
                # Check top 3 strategies concentration
                top_3_weight = strategy_weights.nlargest(3).sum()
                top_3_limit = self.risk_limits.get_limit('top_3_strategies_weight')
                
                if top_3_weight > top_3_limit:
                    self._create_concentration_alert(
                        top_3_weight, top_3_limit, 
                        'Top 3 Strategies Concentration'
                    )
            
        except Exception as e:
            logger.error(f"Error checking concentration limits: {str(e)}")
    
    def _check_volatility_spikes(self, returns_matrix: pd.DataFrame):
        """Check for volatility spikes"""
        try:
            # Calculate portfolio volatility
            n_assets = len(returns_matrix.columns)
            weights = np.ones(n_assets) / n_assets
            portfolio_returns = (returns_matrix * weights).sum(axis=1)
            
            # Rolling volatility (20-day window)
            rolling_vol = portfolio_returns.rolling(20).std()
            current_vol = rolling_vol.iloc[-1]
            avg_vol = rolling_vol.mean()
            
            # Check absolute volatility limit
            vol_limit = self.risk_limits.get_limit('portfolio_volatility')
            if current_vol > vol_limit:
                self._create_volatility_alert(current_vol, vol_limit, 'Portfolio Volatility')
            
            # Check volatility spike
            spike_factor = self.risk_limits.get_limit('volatility_spike_factor')
            if current_vol > avg_vol * spike_factor:
                self._create_volatility_spike_alert(current_vol, avg_vol * spike_factor, 'Volatility Spike')
            
        except Exception as e:
            logger.error(f"Error checking volatility spikes: {str(e)}")
    
    def _check_ml_predictions(self, returns_matrix: pd.DataFrame, factor_matrix: pd.DataFrame):
        """Check ML risk predictions"""
        try:
            # Initialize ML predictor if needed
            if not self.ml_predictor:
                self.ml_predictor = MLRiskPredictor(returns_matrix, factor_matrix)
            
            # Get risk predictions
            predictions = self.ml_predictor.predict_portfolio_risk()
            
            # Check risk score threshold
            risk_score = predictions.get('risk_score', 0)
            score_limit = self.risk_limits.get_limit('ml_risk_score')
            
            if risk_score > score_limit:
                confidence = predictions.get('confidence', 0)
                min_confidence = self.risk_limits.get_limit('prediction_confidence')
                
                if confidence >= min_confidence:
                    self._create_ml_prediction_alert(risk_score, score_limit, 'ML Risk Score', predictions)
            
        except Exception as e:
            logger.error(f"Error checking ML predictions: {str(e)}")
    
    def _create_var_alert(self, current_value: float, limit: float, metric_name: str, var_type: str):
        """Create VaR breach alert"""
        alert_key = f"var_{var_type}"
        
        if self._should_suppress_alert(alert_key):
            return
        
        severity = AlertSeverity.CRITICAL if current_value > limit * 1.5 else AlertSeverity.WARNING
        
        alert = RiskAlert(
            alert_id=f"{alert_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.VAR_BREACH,
            severity=severity,
            title=f"{metric_name} Limit Breach",
            message=f"{metric_name} has exceeded the configured limit. Current: {current_value:.4f}, Limit: {limit:.4f}",
            current_value=current_value,
            threshold_value=limit,
            metric_name=metric_name,
            recommendations=[
                "Review current portfolio exposures",
                "Consider reducing position sizes",
                "Implement hedging strategies",
                "Review risk limit calibration"
            ]
        )
        
        self._send_alert(alert)
    
    def _create_greeks_alert(self, current_value: float, limit: float, metric_name: str, greek: str):
        """Create Greeks limit alert"""
        alert_key = f"greeks_{greek}"
        
        if self._should_suppress_alert(alert_key):
            return
        
        severity = AlertSeverity.WARNING if current_value > limit else AlertSeverity.INFO
        
        alert = RiskAlert(
            alert_id=f"{alert_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.GREEKS_LIMIT,
            severity=severity,
            title=f"{metric_name} Exposure Alert",
            message=f"{metric_name} exposure has exceeded limits. Current: {current_value:.2f}, Limit: {limit:.2f}",
            current_value=current_value,
            threshold_value=limit,
            metric_name=metric_name,
            recommendations=[
                f"Review {greek} exposure across strategies",
                "Consider offsetting positions",
                "Monitor underlying price movements",
                "Adjust position sizing"
            ]
        )
        
        self._send_alert(alert)
    
    def _create_drawdown_alert(self, current_value: float, limit: float, metric_name: str):
        """Create drawdown alert"""
        alert_key = f"drawdown_{metric_name.lower().replace(' ', '_')}"
        
        if self._should_suppress_alert(alert_key):
            return
        
        severity = AlertSeverity.CRITICAL if current_value > limit * 1.2 else AlertSeverity.WARNING
        
        alert = RiskAlert(
            alert_id=f"{alert_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.DRAWDOWN_ALERT,
            severity=severity,
            title=f"{metric_name} Alert",
            message=f"{metric_name} has exceeded acceptable levels. Current: {current_value:.2%}, Limit: {limit:.2%}",
            current_value=current_value,
            threshold_value=limit,
            metric_name=metric_name,
            recommendations=[
                "Review recent trading performance",
                "Consider reducing portfolio risk",
                "Implement stop-loss strategies",
                "Evaluate strategy effectiveness"
            ]
        )
        
        self._send_alert(alert)
    
    def _create_concentration_alert(self, current_value: float, limit: float, metric_name: str, strategy: str = None):
        """Create concentration risk alert"""
        alert_key = f"concentration_{strategy or 'portfolio'}"
        
        if self._should_suppress_alert(alert_key):
            return
        
        alert = RiskAlert(
            alert_id=f"{alert_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.CONCENTRATION_RISK,
            severity=AlertSeverity.WARNING,
            title=f"{metric_name} Risk Alert",
            message=f"Portfolio concentration risk detected. Current: {current_value:.2%}, Limit: {limit:.2%}",
            current_value=current_value,
            threshold_value=limit,
            metric_name=metric_name,
            strategy=strategy,
            recommendations=[
                "Diversify across more strategies",
                "Reduce oversized positions",
                "Review strategy allocation",
                "Consider correlation effects"
            ]
        )
        
        self._send_alert(alert)
    
    def _create_volatility_alert(self, current_value: float, limit: float, metric_name: str):
        """Create volatility alert"""
        alert_key = "portfolio_volatility"
        
        if self._should_suppress_alert(alert_key):
            return
        
        alert = RiskAlert(
            alert_id=f"{alert_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.VOLATILITY_SPIKE,
            severity=AlertSeverity.WARNING,
            title=f"{metric_name} Alert",
            message=f"Portfolio volatility is elevated. Current: {current_value:.4f}, Limit: {limit:.4f}",
            current_value=current_value,
            threshold_value=limit,
            metric_name=metric_name,
            recommendations=[
                "Monitor market conditions",
                "Consider reducing position sizes",
                "Review correlation patterns",
                "Evaluate hedging options"
            ]
        )
        
        self._send_alert(alert)
    
    def _create_volatility_spike_alert(self, current_value: float, threshold: float, metric_name: str):
        """Create volatility spike alert"""
        alert_key = "volatility_spike"
        
        if self._should_suppress_alert(alert_key):
            return
        
        alert = RiskAlert(
            alert_id=f"{alert_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.VOLATILITY_SPIKE,
            severity=AlertSeverity.WARNING,
            title=f"{metric_name} Detected",
            message=f"Volatility has spiked significantly. Current: {current_value:.4f}, Threshold: {threshold:.4f}",
            current_value=current_value,
            threshold_value=threshold,
            metric_name=metric_name,
            recommendations=[
                "Review recent market events",
                "Consider temporary risk reduction",
                "Monitor for regime changes",
                "Evaluate strategy performance"
            ]
        )
        
        self._send_alert(alert)
    
    def _create_ml_prediction_alert(self, risk_score: float, limit: float, metric_name: str, predictions: Dict):
        """Create ML prediction alert"""
        alert_key = "ml_risk_prediction"
        
        if self._should_suppress_alert(alert_key):
            return
        
        severity = AlertSeverity.CRITICAL if risk_score > 0.8 else AlertSeverity.WARNING
        
        alert = RiskAlert(
            alert_id=f"{alert_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.ML_PREDICTION,
            severity=severity,
            title=f"{metric_name} Alert",
            message=f"ML model predicts elevated risk. Risk Score: {risk_score:.3f}, Limit: {limit:.3f}",
            current_value=risk_score,
            threshold_value=limit,
            metric_name=metric_name,
            data=predictions,
            recommendations=[
                "Review ML model predictions in detail",
                "Consider preemptive risk reduction",
                "Monitor predicted risk factors",
                "Validate model predictions"
            ]
        )
        
        self._send_alert(alert)
    
    def _create_system_error_alert(self, error_message: str):
        """Create system error alert"""
        alert_key = "system_error"
        
        if self._should_suppress_alert(alert_key):
            return
        
        alert = RiskAlert(
            alert_id=f"{alert_key}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            alert_type=AlertType.SYSTEM_ERROR,
            severity=AlertSeverity.CRITICAL,
            title="Risk Monitoring System Error",
            message=f"Risk monitoring system encountered an error: {error_message}",
            current_value=1.0,
            threshold_value=0.0,
            metric_name="System Health",
            recommendations=[
                "Check system logs for details",
                "Verify data connectivity",
                "Restart monitoring if needed",
                "Contact system administrator"
            ]
        )
        
        self._send_alert(alert)
    
    def _should_suppress_alert(self, alert_key: str) -> bool:
        """Check if alert should be suppressed due to cooldown"""
        now = datetime.now()
        
        if alert_key in self.alert_cooldown:
            last_alert = self.alert_cooldown[alert_key]
            if (now - last_alert).total_seconds() < self.cooldown_period:
                return True
        
        self.alert_cooldown[alert_key] = now
        return False
    
    def _send_alert(self, alert: RiskAlert):
        """Send alert through all channels and store in database"""
        try:
            # Store in database
            self.alert_db.store_alert(alert)
            
            # Send notifications
            success = self.notification_manager.send_alert(alert)
            
            if success:
                logger.info(f"Alert sent successfully: {alert.alert_id}")
            else:
                logger.warning(f"Failed to send some notifications for alert: {alert.alert_id}")
                
        except Exception as e:
            logger.error(f"Error sending alert: {str(e)}")
    
    def get_alert_statistics(self) -> Dict:
        """Get alert system statistics"""
        return self.alert_db.get_alert_statistics()
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        return self.alert_db.get_recent_alerts(hours)
    
    def acknowledge_alert(self, alert_id: str):
        """Acknowledge an alert"""
        self.alert_db.acknowledge_alert(alert_id)


def main():
    """Example usage of Automated Alert System"""
    
    print("Automated Alert and Notification System - Phase 3")
    print("=" * 60)
    
    try:
        # Configure notifications (example configuration)
        notification_config = NotificationConfig(
            email_enabled=False,  # Set to True and configure SMTP settings for email alerts
            email_recipients=["risk-manager@company.com"],
            slack_enabled=False,  # Set to True and configure webhook for Slack alerts
            file_logging=True,
            log_file="risk_alerts.log"
        )
        
        # Configure risk limits
        risk_limits = RiskLimitsConfig()
        
        # Initialize alert system
        alert_system = AutomatedAlertSystem(
            db_path="../trades.db",
            notification_config=notification_config,
            risk_limits=risk_limits,
            check_interval=60  # Check every minute
        )
        
        print("✅ Alert system initialized successfully")
        print(f"📧 Email alerts: {'Enabled' if notification_config.email_enabled else 'Disabled'}")
        print(f"💬 Slack alerts: {'Enabled' if notification_config.slack_enabled else 'Disabled'}")
        print(f"📝 File logging: {'Enabled' if notification_config.file_logging else 'Disabled'}")
        print(f"⏰ Check interval: {alert_system.check_interval} seconds")
        
        # Display configured risk limits
        print(f"\n⚠️  Configured Risk Limits:")
        limits = risk_limits.limits
        print(f"   • VaR (5%): {limits['var_5_daily']:.2%}")
        print(f"   • Delta Limit: {limits['delta_limit']:,}")
        print(f"   • Vega Limit: {limits['vega_limit']:,}")
        print(f"   • Max Drawdown: {limits['max_drawdown']:.2%}")
        print(f"   • Single Strategy: {limits['single_strategy_weight']:.2%}")
        
        # Test with manual risk check
        print(f"\n🔍 Performing initial risk assessment...")
        alert_system._perform_risk_checks()
        
        # Get recent alerts
        recent_alerts = alert_system.get_recent_alerts(24)
        print(f"📊 Recent alerts (24h): {len(recent_alerts)}")
        
        # Display alert statistics
        stats = alert_system.get_alert_statistics()
        print(f"📈 Alert statistics: {stats}")
        
        # Start continuous monitoring
        print(f"\n🚀 Starting continuous risk monitoring...")
        print(f"Press Ctrl+C to stop monitoring")
        
        alert_system.start_monitoring()
        
        # Keep the program running
        try:
            while True:
                time.sleep(10)
                # Display monitoring status periodically
                if alert_system.monitoring_active:
                    recent = alert_system.get_recent_alerts(1)  # Last hour
                    if recent:
                        print(f"⚠️  New alerts in last hour: {len(recent)}")
                
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping risk monitoring...")
            alert_system.stop_monitoring()
            print(f"✅ Risk monitoring stopped")
        
    except Exception as e:
        logger.error(f"Error running alert system: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()