"""
Risk Units Database Schema - Phase 1 Implementation
Database models and management for Risk Units framework
"""

import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import pandas as pd
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Boolean, Text, JSON, ForeignKey, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from sqlalchemy.dialects.sqlite import REAL
from sqlalchemy.sql import func

from cvar_risk_calculator import CVaRMethod, RiskUnitBasis
from risk_units_converter import RiskUnitStandard, RiskScalingMethod

Base = declarative_base()

class StrategyRiskProfiles(Base):
    """Strategy risk profile characteristics"""
    __tablename__ = "strategy_risk_profiles"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    strategy_id = Column(String(50), unique=True, nullable=False)
    strategy_name = Column(String(100), nullable=False)
    category = Column(String(20))  # directional, neutral, volatility, income, hedge
    risk_profile = Column(String(20))  # conservative, moderate, aggressive, speculative
    
    # Risk Unit Metrics
    avg_risk_units = Column(Float, default=0.0)
    std_dev_risk_units = Column(Float, default=0.0)
    min_risk_units = Column(Float, default=0.0)
    max_risk_units = Column(Float, default=0.0)
    median_risk_units = Column(Float, default=0.0)
    
    # Risk Efficiency Metrics
    risk_efficiency_ratio = Column(Float, default=0.0)  # Return per risk unit
    consistency_score = Column(Float, default=0.0)  # 1 - coefficient of variation
    capital_efficiency = Column(Float, default=0.0)  # Risk units per dollar at risk
    
    # Strategy Characteristics
    delta_neutrality = Column(Float, default=0.5)
    gamma_sensitivity = Column(Float, default=0.5)
    theta_dependency = Column(Float, default=0.5)
    vega_sensitivity = Column(Float, default=0.5)
    max_loss_predictability = Column(Float, default=0.5)
    
    # Performance Expectations
    expected_win_rate = Column(Float, default=0.5)
    expected_profit_loss_ratio = Column(Float, default=1.0)
    
    # Metadata
    sample_size = Column(Integer, default=0)
    last_calculated = Column(DateTime, default=datetime.utcnow)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    trades = relationship("TradeRiskUnits", back_populates="strategy_profile")

class TradeRiskUnits(Base):
    """Individual trade risk units data"""
    __tablename__ = "trade_risk_units"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(50), unique=True, nullable=False)
    original_trade_id = Column(String(50))  # Reference to original trade system
    
    # Strategy Information
    strategy_id = Column(String(50), ForeignKey('strategy_risk_profiles.strategy_id'))
    strategy_name = Column(String(100))
    underlying_symbol = Column(String(20))
    
    # Trade Dates
    entry_date = Column(DateTime)
    exit_date = Column(DateTime)
    trade_date = Column(DateTime, default=datetime.utcnow)  # Risk calculation date
    
    # Risk Units Calculation
    risk_units = Column(Float, nullable=False)
    original_cvar = Column(Float)  # Original CVaR amount
    standardized_cvar = Column(Float)  # Standardized CVaR
    var_amount = Column(Float)  # VaR amount
    unit_size = Column(Float, default=1000.0)  # Risk unit size ($)
    confidence_level = Column(Float, default=0.95)
    
    # CVaR Calculation Details
    cvar_method = Column(String(30))  # Method used for CVaR calculation
    scaling_method = Column(String(30))  # Scaling method applied
    scaling_factor = Column(Float, default=1.0)
    
    # Trade Characteristics
    days_to_expiration = Column(Integer)
    implied_volatility = Column(Float)
    max_loss = Column(Float)
    premium_paid = Column(Float)
    premium_received = Column(Float)
    
    # Greeks
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    
    # Performance Data
    realized_pnl = Column(Float)  # Actual P&L when trade closed
    risk_adjusted_return = Column(Float)  # P&L per risk unit
    max_risk_during_trade = Column(Float)  # Maximum risk units during trade life
    
    # Market Conditions at Trade Time
    market_volatility_regime = Column(String(20))  # low, normal, high, extreme
    vix_level = Column(Float)
    
    # Metadata
    calculation_metadata = Column(Text)  # JSON string with additional data
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    strategy_profile = relationship("StrategyRiskProfiles", back_populates="trades")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_trade_date', 'trade_date'),
        Index('idx_strategy_date', 'strategy_id', 'trade_date'),
        Index('idx_symbol_date', 'underlying_symbol', 'trade_date'),
        Index('idx_entry_date', 'entry_date'),
    )

class DailyRiskSummary(Base):
    """Daily aggregated risk metrics by strategy"""
    __tablename__ = "daily_risk_summary"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    strategy_id = Column(String(50), ForeignKey('strategy_risk_profiles.strategy_id'))
    strategy_name = Column(String(100))
    
    # Daily Risk Metrics
    total_risk_units = Column(Float, default=0.0)
    active_trades_count = Column(Integer, default=0)
    new_trades_count = Column(Integer, default=0)
    closed_trades_count = Column(Integer, default=0)
    avg_risk_per_trade = Column(Float, default=0.0)
    max_risk_per_trade = Column(Float, default=0.0)
    
    # Portfolio Risk Metrics
    portfolio_risk_concentration = Column(Float, default=0.0)  # % of total portfolio risk
    risk_capacity_utilization = Column(Float, default=0.0)  # % of max allowed risk
    
    # Performance Metrics
    daily_pnl = Column(Float, default=0.0)
    daily_risk_adjusted_return = Column(Float, default=0.0)
    cumulative_risk_adjusted_return = Column(Float, default=0.0)
    
    # Risk Efficiency
    risk_efficiency_daily = Column(Float, default=0.0)  # Daily P&L per risk unit
    risk_velocity = Column(Float, default=0.0)  # Rate of risk deployment/reduction
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Unique constraint on date + strategy
    __table_args__ = (
        Index('idx_date_strategy', 'date', 'strategy_id', unique=True),
        Index('idx_date_only', 'date'),
    )

class WeeklyRiskAnalysis(Base):
    """Weekly risk pattern analysis"""
    __tablename__ = "weekly_risk_analysis"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    week_start_date = Column(DateTime, nullable=False)
    week_end_date = Column(DateTime, nullable=False)
    strategy_id = Column(String(50), ForeignKey('strategy_risk_profiles.strategy_id'))
    strategy_name = Column(String(100))
    
    # Weekly Aggregated Metrics
    total_weekly_risk_units = Column(Float, default=0.0)
    avg_daily_risk_units = Column(Float, default=0.0)
    peak_daily_risk_units = Column(Float, default=0.0)
    total_trades = Column(Integer, default=0)
    
    # Weekly Patterns
    monday_risk_units = Column(Float, default=0.0)
    tuesday_risk_units = Column(Float, default=0.0)
    wednesday_risk_units = Column(Float, default=0.0)
    thursday_risk_units = Column(Float, default=0.0)
    friday_risk_units = Column(Float, default=0.0)
    
    # Risk Deployment Patterns
    risk_deployment_trend = Column(String(20))  # increasing, decreasing, stable, volatile
    risk_concentration_score = Column(Float, default=0.0)
    risk_diversification_score = Column(Float, default=0.0)
    
    # Performance
    weekly_pnl = Column(Float, default=0.0)
    weekly_risk_adjusted_return = Column(Float, default=0.0)
    weekly_sharpe_ratio = Column(Float, default=0.0)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_week_strategy', 'week_start_date', 'strategy_id'),
        Index('idx_week_start', 'week_start_date'),
    )

class RiskUnitsConfiguration(Base):
    """Configuration settings for risk units calculations"""
    __tablename__ = "risk_units_configuration"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    config_name = Column(String(100), unique=True, nullable=False)
    
    # Risk Unit Standards
    unit_standard = Column(String(30), default='standard_1k')  # standard_1k, institutional_10k, retail_100
    unit_size = Column(Float, default=1000.0)
    currency = Column(String(10), default='USD')
    
    # Calculation Parameters
    risk_basis = Column(String(30), default='cvar_95')  # cvar_95, cvar_99, var_95, var_99
    confidence_level = Column(Float, default=0.95)
    default_cvar_method = Column(String(30), default='historical_simulation')
    scaling_method = Column(String(30), default='linear')
    
    # Risk Limits
    max_risk_units_per_trade = Column(Float, default=10.0)
    max_total_daily_risk_units = Column(Float, default=100.0)
    max_strategy_concentration = Column(Float, default=0.4)  # 40% max in any strategy
    risk_utilization_warning_threshold = Column(Float, default=0.8)  # 80% warning level
    
    # Active Configuration
    is_active = Column(Boolean, default=True)
    is_default = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class RiskAlertsLog(Base):
    """Log of risk-related alerts and breaches"""
    __tablename__ = "risk_alerts_log"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(50), unique=True, nullable=False)
    alert_type = Column(String(50), nullable=False)  # risk_limit_breach, concentration_warning, etc.
    severity = Column(String(20), default='medium')  # low, medium, high, critical
    
    # Alert Details
    strategy_id = Column(String(50))
    trade_id = Column(String(50))
    alert_message = Column(Text)
    current_value = Column(Float)
    threshold_value = Column(Float)
    
    # Alert Status
    is_acknowledged = Column(Boolean, default=False)
    acknowledged_by = Column(String(100))
    acknowledged_at = Column(DateTime)
    resolution_notes = Column(Text)
    
    # Metadata
    additional_data = Column(Text)  # JSON string with extra context
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_alert_type_date', 'alert_type', 'created_at'),
        Index('idx_severity_date', 'severity', 'created_at'),
    )

class RiskUnitsDatabaseManager:
    """
    Database manager for Risk Units framework
    """
    
    def __init__(self, database_url: str = "sqlite:///risk_units.db"):
        self.database_url = database_url
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        
        # Create all tables
        self.create_tables()
        
        # Initialize default configuration
        self.initialize_default_config()
        
        logging.info(f"Risk Units Database Manager initialized with {database_url}")
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
        logging.info("Risk units database tables created")
    
    def get_session(self) -> Session:
        """Get database session"""
        return self.SessionLocal()
    
    def initialize_default_config(self):
        """Initialize default risk units configuration"""
        session = self.get_session()
        try:
            # Check if default config exists
            existing_config = session.query(RiskUnitsConfiguration).filter_by(is_default=True).first()
            
            if not existing_config:
                default_config = RiskUnitsConfiguration(
                    config_name="default_config",
                    unit_standard="standard_1k",
                    unit_size=1000.0,
                    currency="USD",
                    risk_basis="cvar_95",
                    confidence_level=0.95,
                    default_cvar_method="historical_simulation",
                    scaling_method="linear",
                    max_risk_units_per_trade=10.0,
                    max_total_daily_risk_units=100.0,
                    max_strategy_concentration=0.4,
                    risk_utilization_warning_threshold=0.8,
                    is_active=True,
                    is_default=True
                )
                
                session.add(default_config)
                session.commit()
                logging.info("Default risk units configuration created")
                
        except Exception as e:
            session.rollback()
            logging.error(f"Error initializing default config: {e}")
        finally:
            session.close()
    
    def insert_trade_risk_units(self, trade_risk_data: Dict[str, Any]) -> bool:
        """Insert trade risk units data"""
        session = self.get_session()
        try:
            trade_record = TradeRiskUnits(**trade_risk_data)
            session.add(trade_record)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Error inserting trade risk units: {e}")
            return False
        finally:
            session.close()
    
    def update_strategy_risk_profile(self, strategy_data: Dict[str, Any]) -> bool:
        """Update or insert strategy risk profile"""
        session = self.get_session()
        try:
            strategy_id = strategy_data['strategy_id']
            
            # Check if profile exists
            existing_profile = session.query(StrategyRiskProfiles).filter_by(strategy_id=strategy_id).first()
            
            if existing_profile:
                # Update existing profile
                for key, value in strategy_data.items():
                    if hasattr(existing_profile, key):
                        setattr(existing_profile, key, value)
                existing_profile.updated_at = datetime.utcnow()
            else:
                # Create new profile
                new_profile = StrategyRiskProfiles(**strategy_data)
                session.add(new_profile)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Error updating strategy risk profile: {e}")
            return False
        finally:
            session.close()
    
    def get_strategy_risk_profile(self, strategy_id: str) -> Optional[Dict[str, Any]]:
        """Get strategy risk profile by ID"""
        session = self.get_session()
        try:
            profile = session.query(StrategyRiskProfiles).filter_by(strategy_id=strategy_id).first()
            if profile:
                return {
                    'strategy_id': profile.strategy_id,
                    'strategy_name': profile.strategy_name,
                    'avg_risk_units': profile.avg_risk_units,
                    'std_dev_risk_units': profile.std_dev_risk_units,
                    'risk_efficiency_ratio': profile.risk_efficiency_ratio,
                    'consistency_score': profile.consistency_score,
                    'sample_size': profile.sample_size,
                    'last_calculated': profile.last_calculated
                }
            return None
        except Exception as e:
            logging.error(f"Error getting strategy risk profile: {e}")
            return None
        finally:
            session.close()
    
    def calculate_daily_risk_summary(self, date: datetime, strategy_id: str = None) -> bool:
        """Calculate and store daily risk summary"""
        session = self.get_session()
        try:
            # Get all strategies or specific strategy
            if strategy_id:
                strategies = [strategy_id]
            else:
                strategies = session.query(StrategyRiskProfiles.strategy_id).distinct().all()
                strategies = [s[0] for s in strategies]
            
            for strat_id in strategies:
                # Get trades for the date and strategy
                trades = session.query(TradeRiskUnits).filter(
                    func.date(TradeRiskUnits.trade_date) == date.date(),
                    TradeRiskUnits.strategy_id == strat_id
                ).all()
                
                if not trades:
                    continue
                
                # Calculate metrics
                total_risk_units = sum(trade.risk_units for trade in trades)
                active_trades = len(trades)
                avg_risk_per_trade = total_risk_units / active_trades if active_trades > 0 else 0
                max_risk_per_trade = max(trade.risk_units for trade in trades) if trades else 0
                
                # Daily P&L (if available)
                daily_pnl = sum(trade.realized_pnl or 0 for trade in trades)
                daily_risk_adjusted_return = daily_pnl / total_risk_units if total_risk_units > 0 else 0
                
                # Check if summary already exists
                existing_summary = session.query(DailyRiskSummary).filter(
                    func.date(DailyRiskSummary.date) == date.date(),
                    DailyRiskSummary.strategy_id == strat_id
                ).first()
                
                summary_data = {
                    'date': date,
                    'strategy_id': strat_id,
                    'strategy_name': trades[0].strategy_name,
                    'total_risk_units': total_risk_units,
                    'active_trades_count': active_trades,
                    'avg_risk_per_trade': avg_risk_per_trade,
                    'max_risk_per_trade': max_risk_per_trade,
                    'daily_pnl': daily_pnl,
                    'daily_risk_adjusted_return': daily_risk_adjusted_return
                }
                
                if existing_summary:
                    # Update existing
                    for key, value in summary_data.items():
                        if hasattr(existing_summary, key):
                            setattr(existing_summary, key, value)
                else:
                    # Create new
                    new_summary = DailyRiskSummary(**summary_data)
                    session.add(new_summary)
            
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Error calculating daily risk summary: {e}")
            return False
        finally:
            session.close()
    
    def get_risk_units_summary(self, 
                             start_date: datetime, 
                             end_date: datetime,
                             strategy_id: str = None) -> Dict[str, Any]:
        """Get risk units summary for date range"""
        session = self.get_session()
        try:
            # Base query
            query = session.query(TradeRiskUnits).filter(
                TradeRiskUnits.trade_date >= start_date,
                TradeRiskUnits.trade_date <= end_date
            )
            
            if strategy_id:
                query = query.filter(TradeRiskUnits.strategy_id == strategy_id)
            
            trades = query.all()
            
            if not trades:
                return {"error": "No trades found for the specified period"}
            
            # Calculate summary statistics
            risk_units = [trade.risk_units for trade in trades]
            
            summary = {
                "period": {
                    "start_date": start_date.isoformat(),
                    "end_date": end_date.isoformat(),
                    "strategy_id": strategy_id
                },
                "total_trades": len(trades),
                "total_risk_units": sum(risk_units),
                "average_risk_units": sum(risk_units) / len(risk_units),
                "min_risk_units": min(risk_units),
                "max_risk_units": max(risk_units),
                "std_risk_units": pd.Series(risk_units).std(),
                "median_risk_units": pd.Series(risk_units).median()
            }
            
            # Strategy breakdown
            strategy_breakdown = {}
            for trade in trades:
                strategy = trade.strategy_name
                if strategy not in strategy_breakdown:
                    strategy_breakdown[strategy] = {
                        'count': 0,
                        'total_risk_units': 0.0
                    }
                strategy_breakdown[strategy]['count'] += 1
                strategy_breakdown[strategy]['total_risk_units'] += trade.risk_units
            
            # Calculate averages
            for strategy_data in strategy_breakdown.values():
                strategy_data['avg_risk_units'] = (
                    strategy_data['total_risk_units'] / strategy_data['count']
                )
            
            summary['strategy_breakdown'] = strategy_breakdown
            
            return summary
            
        except Exception as e:
            logging.error(f"Error getting risk units summary: {e}")
            return {"error": str(e)}
        finally:
            session.close()
    
    def create_risk_alert(self, alert_data: Dict[str, Any]) -> bool:
        """Create a risk alert"""
        session = self.get_session()
        try:
            alert = RiskAlertsLog(**alert_data)
            session.add(alert)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            logging.error(f"Error creating risk alert: {e}")
            return False
        finally:
            session.close()
    
    def get_active_risk_alerts(self, severity: str = None) -> List[Dict[str, Any]]:
        """Get active (unacknowledged) risk alerts"""
        session = self.get_session()
        try:
            query = session.query(RiskAlertsLog).filter(
                RiskAlertsLog.is_acknowledged == False
            )
            
            if severity:
                query = query.filter(RiskAlertsLog.severity == severity)
            
            alerts = query.order_by(RiskAlertsLog.created_at.desc()).all()
            
            return [
                {
                    'alert_id': alert.alert_id,
                    'alert_type': alert.alert_type,
                    'severity': alert.severity,
                    'alert_message': alert.alert_message,
                    'current_value': alert.current_value,
                    'threshold_value': alert.threshold_value,
                    'created_at': alert.created_at.isoformat()
                }
                for alert in alerts
            ]
            
        except Exception as e:
            logging.error(f"Error getting risk alerts: {e}")
            return []
        finally:
            session.close()
    
    def export_risk_data(self, output_file: str, table_name: str, start_date: datetime = None, end_date: datetime = None):
        """Export risk data to CSV"""
        session = self.get_session()
        try:
            if table_name == "trade_risk_units":
                query = session.query(TradeRiskUnits)
                if start_date:
                    query = query.filter(TradeRiskUnits.trade_date >= start_date)
                if end_date:
                    query = query.filter(TradeRiskUnits.trade_date <= end_date)
                
                df = pd.read_sql(query.statement, session.bind)
                
            elif table_name == "daily_risk_summary":
                query = session.query(DailyRiskSummary)
                if start_date:
                    query = query.filter(DailyRiskSummary.date >= start_date)
                if end_date:
                    query = query.filter(DailyRiskSummary.date <= end_date)
                
                df = pd.read_sql(query.statement, session.bind)
                
            else:
                raise ValueError(f"Unknown table name: {table_name}")
            
            df.to_csv(output_file, index=False)
            logging.info(f"Risk data exported to {output_file}")
            
        except Exception as e:
            logging.error(f"Error exporting risk data: {e}")
        finally:
            session.close()

# Example usage and testing
def main():
    # Initialize database manager
    db_manager = RiskUnitsDatabaseManager("sqlite:///test_risk_units.db")
    
    print("Risk Units Database Schema - Phase 1 Implementation")
    print("=" * 55)
    
    # Test inserting sample strategy profile
    strategy_data = {
        'strategy_id': 'iron_condor',
        'strategy_name': 'Iron Condor',
        'category': 'neutral',
        'risk_profile': 'moderate',
        'avg_risk_units': 1.8,
        'std_dev_risk_units': 0.4,
        'min_risk_units': 1.0,
        'max_risk_units': 3.2,
        'median_risk_units': 1.7,
        'risk_efficiency_ratio': 0.15,
        'consistency_score': 0.75,
        'delta_neutrality': 0.95,
        'gamma_sensitivity': 0.4,
        'vega_sensitivity': 0.5,
        'expected_win_rate': 0.65,
        'sample_size': 25
    }
    
    success = db_manager.update_strategy_risk_profile(strategy_data)
    print(f"Strategy profile insert: {'✓' if success else '✗'}")
    
    # Test inserting sample trade
    trade_data = {
        'trade_id': 'TRADE_001',
        'original_trade_id': 'TradeSteward_001',
        'strategy_id': 'iron_condor',
        'strategy_name': 'Iron Condor',
        'underlying_symbol': 'SPY',
        'entry_date': datetime(2024, 1, 15),
        'trade_date': datetime(2024, 1, 15),
        'risk_units': 1.75,
        'original_cvar': 1750.0,
        'standardized_cvar': 1750.0,
        'var_amount': 1600.0,
        'unit_size': 1000.0,
        'confidence_level': 0.95,
        'cvar_method': 'historical_simulation',
        'scaling_method': 'linear',
        'scaling_factor': 1.0,
        'days_to_expiration': 45,
        'implied_volatility': 0.18,
        'max_loss': -2500.0,
        'premium_paid': 2500.0,
        'premium_received': 500.0,
        'delta': -0.05,
        'gamma': 0.02,
        'theta': -15.0,
        'vega': 25.0,
        'market_volatility_regime': 'normal',
        'calculation_metadata': '{"market_conditions": {"vix": 18.5}}'
    }
    
    success = db_manager.insert_trade_risk_units(trade_data)
    print(f"Trade risk units insert: {'✓' if success else '✗'}")
    
    # Test calculating daily summary
    success = db_manager.calculate_daily_risk_summary(datetime(2024, 1, 15))
    print(f"Daily risk summary calculation: {'✓' if success else '✗'}")
    
    # Test getting strategy profile
    profile = db_manager.get_strategy_risk_profile('iron_condor')
    if profile:
        print(f"Strategy profile retrieved: ✓")
        print(f"  Average Risk Units: {profile['avg_risk_units']}")
        print(f"  Sample Size: {profile['sample_size']}")
    else:
        print("Strategy profile retrieved: ✗")
    
    # Test getting risk summary
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    summary = db_manager.get_risk_units_summary(start_date, end_date)
    
    if 'error' not in summary:
        print(f"Risk units summary retrieved: ✓")
        print(f"  Total Trades: {summary['total_trades']}")
        print(f"  Total Risk Units: {summary['total_risk_units']:.2f}")
        print(f"  Average Risk Units: {summary['average_risk_units']:.2f}")
    else:
        print("Risk units summary retrieved: ✗")
    
    # Test creating risk alert
    alert_data = {
        'alert_id': 'ALERT_001',
        'alert_type': 'risk_limit_breach',
        'severity': 'high',
        'strategy_id': 'iron_condor',
        'alert_message': 'Strategy risk units exceeded daily limit',
        'current_value': 12.5,
        'threshold_value': 10.0,
        'additional_data': '{"breach_percentage": 25.0}'
    }
    
    success = db_manager.create_risk_alert(alert_data)
    print(f"Risk alert creation: {'✓' if success else '✗'}")
    
    # Test getting active alerts
    alerts = db_manager.get_active_risk_alerts()
    print(f"Active alerts retrieved: {len(alerts)} alerts")
    
    if alerts:
        for alert in alerts:
            print(f"  Alert: {alert['alert_type']} - {alert['severity']}")
    
    print(f"\nDatabase tables created successfully!")
    print(f"Schema includes:")
    print(f"  - Strategy Risk Profiles")
    print(f"  - Trade Risk Units")
    print(f"  - Daily Risk Summary")
    print(f"  - Weekly Risk Analysis")
    print(f"  - Risk Units Configuration")
    print(f"  - Risk Alerts Log")

if __name__ == "__main__":
    main()