"""
Regulatory Reporting and Compliance Module - Phase 3 Implementation
Comprehensive regulatory compliance framework for options trading risk management
"""

import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from datetime import datetime, timedelta, date
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import xml.etree.ElementTree as ET
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import os
from pathlib import Path

# Import our risk management modules
from data_loader import TradeStewardDataLoader
from advanced_risk_measures import AdvancedRiskCalculator
from options_factor_models import OptionsFactorModel
from portfolio_optimization import PortfolioOptimizer

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class RegulatoryFramework(Enum):
    """Supported regulatory frameworks"""
    BASEL_III = "basel_iii"
    FRTB = "frtb"  # Fundamental Review of the Trading Book
    CCAR = "ccar"  # Comprehensive Capital Analysis and Review
    EMIR = "emir"  # European Market Infrastructure Regulation
    DODD_FRANK = "dodd_frank"
    MIFID_II = "mifid_ii"
    SOLVENCY_II = "solvency_ii"
    SEC_RULE_15C3_1 = "sec_rule_15c3_1"  # Net Capital Rule

class ReportType(Enum):
    """Types of regulatory reports"""
    CAPITAL_ADEQUACY = "capital_adequacy"
    RISK_METRICS = "risk_metrics"
    STRESS_TEST = "stress_test"
    LIQUIDITY_COVERAGE = "liquidity_coverage"
    LARGE_EXPOSURES = "large_exposures"
    TRADING_BOOK = "trading_book"
    COUNTERPARTY_RISK = "counterparty_risk"
    OPERATIONAL_RISK = "operational_risk"
    MARKET_RISK = "market_risk"
    POSITION_LIMITS = "position_limits"

class ReportingFrequency(Enum):
    """Reporting frequencies"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    ANNUALLY = "annually"
    AD_HOC = "ad_hoc"

@dataclass
class RegulatoryLimit:
    """Regulatory limit definition"""
    limit_id: str
    framework: RegulatoryFramework
    metric_name: str
    limit_value: float
    warning_threshold: float
    hard_limit: bool
    description: str
    calculation_method: str
    regulatory_reference: str

@dataclass
class ComplianceCheck:
    """Compliance check result"""
    check_id: str
    timestamp: datetime
    framework: RegulatoryFramework
    limit_id: str
    current_value: float
    limit_value: float
    warning_threshold: float
    status: str  # COMPLIANT, WARNING, BREACH
    breach_amount: Optional[float]
    recommendations: List[str]
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RegulatoryReport:
    """Regulatory report structure"""
    report_id: str
    framework: RegulatoryFramework
    report_type: ReportType
    reporting_date: date
    period_start: date
    period_end: date
    entity_name: str
    data: Dict[str, Any]
    calculations: Dict[str, Any]
    compliance_status: Dict[str, str]
    certifications: List[str]
    attachments: List[str]
    created_at: datetime
    version: str = "1.0"

class BaselIIICalculator:
    """Basel III capital adequacy calculations"""
    
    def __init__(self, returns_matrix: pd.DataFrame, factor_matrix: pd.DataFrame):
        self.returns_matrix = returns_matrix
        self.factor_matrix = factor_matrix
        self.logger = logging.getLogger(__name__ + '.BaselIIICalculator')
    
    def calculate_capital_ratios(self, total_capital: float, tier1_capital: float, 
                                common_equity_tier1: float, rwa: float) -> Dict[str, float]:
        """Calculate Basel III capital ratios"""
        
        ratios = {}
        
        # Common Equity Tier 1 (CET1) ratio
        ratios['cet1_ratio'] = common_equity_tier1 / rwa if rwa > 0 else 0
        
        # Tier 1 capital ratio
        ratios['tier1_ratio'] = tier1_capital / rwa if rwa > 0 else 0
        
        # Total capital ratio
        ratios['total_capital_ratio'] = total_capital / rwa if rwa > 0 else 0
        
        return ratios
    
    def calculate_risk_weighted_assets(self, portfolio_weights: np.ndarray) -> Dict[str, float]:
        """Calculate Risk Weighted Assets (RWA) for trading book"""
        
        try:
            # Market Risk RWA using Internal Models Approach
            risk_calc = AdvancedRiskCalculator(self.returns_matrix, alpha=0.01)
            risk_analysis = risk_calc.comprehensive_risk_analysis(portfolio_weights)
            
            # VaR-based capital charge
            var_1_day = risk_analysis.get('VaR_Measures', {}).get('VaR_Historical', 0)
            
            # Stressed VaR (typically 3x normal VaR)
            stressed_var = var_1_day * 3.0
            
            # Market Risk Capital Charge (max of VaR and Stressed VaR, with multiplier)
            multiplier = 3.0  # Regulatory multiplier
            market_risk_charge = max(var_1_day, stressed_var) * multiplier
            
            # Convert to RWA (capital charge / 8%)
            market_rwa = market_risk_charge / 0.08
            
            # Specific Risk (for options positions)
            specific_risk_charge = self._calculate_specific_risk_charge(portfolio_weights)
            specific_rwa = specific_risk_charge / 0.08
            
            # Total RWA
            total_rwa = market_rwa + specific_rwa
            
            return {
                'market_rwa': market_rwa,
                'specific_rwa': specific_rwa,
                'total_rwa': total_rwa,
                'market_risk_charge': market_risk_charge,
                'specific_risk_charge': specific_risk_charge,
                'var_1_day': var_1_day,
                'stressed_var': stressed_var
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating RWA: {str(e)}")
            return {'total_rwa': 0, 'market_rwa': 0, 'specific_rwa': 0}
    
    def _calculate_specific_risk_charge(self, portfolio_weights: np.ndarray) -> float:
        """Calculate specific risk charge for options positions"""
        
        # Simplified specific risk calculation
        # In practice, this would be more sophisticated based on underlying exposures
        
        portfolio_value = np.sum(np.abs(portfolio_weights))
        
        # Apply specific risk charges based on position types
        # Equity options: 8% for general market risk + specific risk
        equity_specific_risk = portfolio_value * 0.04  # 4% specific risk charge
        
        return equity_specific_risk
    
    def calculate_leverage_ratio(self, tier1_capital: float, total_exposure: float) -> float:
        """Calculate Basel III leverage ratio"""
        return tier1_capital / total_exposure if total_exposure > 0 else 0

class FRTBCalculator:
    """Fundamental Review of the Trading Book calculations"""
    
    def __init__(self, returns_matrix: pd.DataFrame, factor_matrix: pd.DataFrame):
        self.returns_matrix = returns_matrix
        self.factor_matrix = factor_matrix
        self.logger = logging.getLogger(__name__ + '.FRTBCalculator')
    
    def calculate_sensitivities_based_method(self, portfolio_weights: np.ndarray) -> Dict[str, float]:
        """Calculate FRTB Sensitivities-Based Method capital charge"""
        
        try:
            # Delta risk charge
            delta_charge = self._calculate_delta_risk_charge(portfolio_weights)
            
            # Vega risk charge
            vega_charge = self._calculate_vega_risk_charge(portfolio_weights)
            
            # Curvature risk charge
            curvature_charge = self._calculate_curvature_risk_charge(portfolio_weights)
            
            # Total sensitivities-based capital charge
            total_charge = delta_charge + vega_charge + curvature_charge
            
            return {
                'delta_charge': delta_charge,
                'vega_charge': vega_charge,
                'curvature_charge': curvature_charge,
                'total_sb_charge': total_charge
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating FRTB SBM: {str(e)}")
            return {'total_sb_charge': 0}
    
    def _calculate_delta_risk_charge(self, portfolio_weights: np.ndarray) -> float:
        """Calculate delta risk charge"""
        
        # Simplified delta risk calculation
        # Would need actual Greeks for precise calculation
        
        portfolio_returns = (self.returns_matrix * portfolio_weights).sum(axis=1)
        portfolio_volatility = portfolio_returns.std()
        
        # Delta risk factor: based on equity risk
        equity_risk_weight = 0.55  # FRTB equity risk weight
        delta_charge = portfolio_volatility * equity_risk_weight * np.sqrt(250)  # Annualized
        
        return delta_charge
    
    def _calculate_vega_risk_charge(self, portfolio_weights: np.ndarray) -> float:
        """Calculate vega risk charge"""
        
        # Simplified vega risk calculation
        # Vega risk based on implied volatility changes
        
        portfolio_value = np.sum(np.abs(portfolio_weights))
        vega_risk_weight = 0.55  # FRTB vega risk weight
        
        # Assume 1% vega exposure relative to portfolio value
        vega_exposure = portfolio_value * 0.01
        vega_charge = vega_exposure * vega_risk_weight
        
        return vega_charge
    
    def _calculate_curvature_risk_charge(self, portfolio_weights: np.ndarray) -> float:
        """Calculate curvature (gamma) risk charge"""
        
        # Simplified curvature risk calculation
        # Based on gamma exposure and price volatility
        
        portfolio_value = np.sum(np.abs(portfolio_weights))
        curvature_risk_weight = 0.28  # FRTB curvature risk weight
        
        # Assume 0.5% gamma exposure relative to portfolio value
        gamma_exposure = portfolio_value * 0.005
        curvature_charge = gamma_exposure * curvature_risk_weight
        
        return curvature_charge
    
    def calculate_default_risk_charge(self, portfolio_weights: np.ndarray) -> float:
        """Calculate FRTB default risk charge"""
        
        # Simplified default risk calculation
        # Based on concentration and credit quality
        
        portfolio_value = np.sum(np.abs(portfolio_weights))
        
        # Default risk weights by rating (simplified)
        default_risk_weights = {
            'AAA': 0.005,
            'AA': 0.01,
            'A': 0.02,
            'BBB': 0.05,
            'BB': 0.15,
            'B': 0.30,
            'CCC': 0.50
        }
        
        # Assume average rating of A for simplification
        avg_risk_weight = 0.02
        default_charge = portfolio_value * avg_risk_weight
        
        return default_charge

class CCARStressTester:
    """CCAR stress testing framework"""
    
    def __init__(self, returns_matrix: pd.DataFrame, factor_matrix: pd.DataFrame):
        self.returns_matrix = returns_matrix
        self.factor_matrix = factor_matrix
        self.logger = logging.getLogger(__name__ + '.CCARStressTester')
    
    def run_supervisory_scenarios(self, portfolio_weights: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Run CCAR supervisory stress scenarios"""
        
        scenarios = {
            'baseline': self._baseline_scenario(portfolio_weights),
            'adverse': self._adverse_scenario(portfolio_weights), 
            'severely_adverse': self._severely_adverse_scenario(portfolio_weights)
        }
        
        return scenarios
    
    def _baseline_scenario(self, portfolio_weights: np.ndarray) -> Dict[str, float]:
        """CCAR baseline scenario"""
        
        # Baseline: Normal market conditions
        portfolio_returns = (self.returns_matrix * portfolio_weights).sum(axis=1)
        
        # 9-quarter projection
        quarters = 9
        quarterly_returns = []
        
        for q in range(quarters):
            # Simulate normal growth
            mean_return = portfolio_returns.mean() * 63  # Quarterly (63 trading days)
            vol = portfolio_returns.std() * np.sqrt(63)
            
            quarter_return = np.random.normal(mean_return, vol)
            quarterly_returns.append(quarter_return)
        
        return {
            'total_return': sum(quarterly_returns),
            'max_loss': min(quarterly_returns),
            'average_return': np.mean(quarterly_returns),
            'volatility': np.std(quarterly_returns)
        }
    
    def _adverse_scenario(self, portfolio_weights: np.ndarray) -> Dict[str, float]:
        """CCAR adverse scenario"""
        
        # Adverse: Moderate recession
        portfolio_returns = (self.returns_matrix * portfolio_weights).sum(axis=1)
        
        # Apply stress factors
        stress_mean = portfolio_returns.mean() * 0.7  # 30% reduction in returns
        stress_vol = portfolio_returns.std() * 1.5    # 50% increase in volatility
        
        quarters = 9
        quarterly_returns = []
        
        for q in range(quarters):
            quarter_return = np.random.normal(stress_mean * 63, stress_vol * np.sqrt(63))
            quarterly_returns.append(quarter_return)
        
        return {
            'total_return': sum(quarterly_returns),
            'max_loss': min(quarterly_returns),
            'average_return': np.mean(quarterly_returns),
            'volatility': np.std(quarterly_returns)
        }
    
    def _severely_adverse_scenario(self, portfolio_weights: np.ndarray) -> Dict[str, float]:
        """CCAR severely adverse scenario"""
        
        # Severely Adverse: Severe recession with financial stress
        portfolio_returns = (self.returns_matrix * portfolio_weights).sum(axis=1)
        
        # Apply severe stress factors
        stress_mean = portfolio_returns.mean() * 0.4  # 60% reduction in returns
        stress_vol = portfolio_returns.std() * 2.5    # 150% increase in volatility
        
        quarters = 9
        quarterly_returns = []
        
        # Add correlation breakdown and market liquidity stress
        for q in range(quarters):
            base_return = np.random.normal(stress_mean * 63, stress_vol * np.sqrt(63))
            
            # Add tail events
            if np.random.random() < 0.1:  # 10% chance of extreme event each quarter
                tail_shock = np.random.normal(-0.2, 0.1)  # Large negative shock
                base_return += tail_shock
            
            quarterly_returns.append(base_return)
        
        return {
            'total_return': sum(quarterly_returns),
            'max_loss': min(quarterly_returns),
            'average_return': np.mean(quarterly_returns),
            'volatility': np.std(quarterly_returns)
        }

class RegulatoryLimitsManager:
    """Manager for regulatory limits and compliance checking"""
    
    def __init__(self):
        self.limits = {}
        self._initialize_default_limits()
        self.logger = logging.getLogger(__name__ + '.RegulatoryLimitsManager')
    
    def _initialize_default_limits(self):
        """Initialize default regulatory limits"""
        
        # Basel III limits
        basel_limits = [
            RegulatoryLimit(
                limit_id="basel_cet1_ratio",
                framework=RegulatoryFramework.BASEL_III,
                metric_name="CET1 Ratio",
                limit_value=0.045,  # 4.5% minimum
                warning_threshold=0.05,  # 5% warning
                hard_limit=True,
                description="Common Equity Tier 1 Capital Ratio",
                calculation_method="CET1 Capital / Risk Weighted Assets",
                regulatory_reference="Basel III Framework"
            ),
            RegulatoryLimit(
                limit_id="basel_tier1_ratio",
                framework=RegulatoryFramework.BASEL_III,
                metric_name="Tier 1 Ratio", 
                limit_value=0.06,  # 6% minimum
                warning_threshold=0.065,
                hard_limit=True,
                description="Tier 1 Capital Ratio",
                calculation_method="Tier 1 Capital / Risk Weighted Assets",
                regulatory_reference="Basel III Framework"
            ),
            RegulatoryLimit(
                limit_id="basel_total_ratio",
                framework=RegulatoryFramework.BASEL_III,
                metric_name="Total Capital Ratio",
                limit_value=0.08,  # 8% minimum
                warning_threshold=0.085,
                hard_limit=True,
                description="Total Capital Ratio",
                calculation_method="Total Capital / Risk Weighted Assets",
                regulatory_reference="Basel III Framework"
            ),
            RegulatoryLimit(
                limit_id="basel_leverage_ratio",
                framework=RegulatoryFramework.BASEL_III,
                metric_name="Leverage Ratio",
                limit_value=0.03,  # 3% minimum
                warning_threshold=0.035,
                hard_limit=True,
                description="Basel III Leverage Ratio",
                calculation_method="Tier 1 Capital / Total Exposure",
                regulatory_reference="Basel III Framework"
            )
        ]
        
        # FRTB limits
        frtb_limits = [
            RegulatoryLimit(
                limit_id="frtb_market_risk_charge",
                framework=RegulatoryFramework.FRTB,
                metric_name="Market Risk Charge",
                limit_value=0.02,  # 2% of portfolio value
                warning_threshold=0.015,
                hard_limit=False,
                description="FRTB Market Risk Capital Charge",
                calculation_method="Sensitivities-Based Method",
                regulatory_reference="FRTB Framework"
            )
        ]
        
        # Position limits
        position_limits = [
            RegulatoryLimit(
                limit_id="single_name_concentration",
                framework=RegulatoryFramework.SEC_RULE_15C3_1,
                metric_name="Single Name Concentration",
                limit_value=0.25,  # 25% max single position
                warning_threshold=0.20,
                hard_limit=True,
                description="Maximum single name concentration",
                calculation_method="Single Position Value / Total Portfolio Value",
                regulatory_reference="SEC Rule 15c3-1"
            ),
            RegulatoryLimit(
                limit_id="sector_concentration",
                framework=RegulatoryFramework.SEC_RULE_15C3_1,
                metric_name="Sector Concentration",
                limit_value=0.40,  # 40% max sector concentration
                warning_threshold=0.35,
                hard_limit=True,
                description="Maximum sector concentration",
                calculation_method="Sector Position Value / Total Portfolio Value",
                regulatory_reference="SEC Rule 15c3-1"
            )
        ]
        
        # Store all limits
        all_limits = basel_limits + frtb_limits + position_limits
        
        for limit in all_limits:
            self.limits[limit.limit_id] = limit
    
    def add_limit(self, limit: RegulatoryLimit):
        """Add a regulatory limit"""
        self.limits[limit.limit_id] = limit
    
    def get_limits_by_framework(self, framework: RegulatoryFramework) -> List[RegulatoryLimit]:
        """Get all limits for a specific framework"""
        return [limit for limit in self.limits.values() if limit.framework == framework]
    
    def check_compliance(self, limit_id: str, current_value: float) -> ComplianceCheck:
        """Check compliance against a specific limit"""
        
        if limit_id not in self.limits:
            raise ValueError(f"Unknown limit ID: {limit_id}")
        
        limit = self.limits[limit_id]
        
        # Determine compliance status
        if current_value >= limit.limit_value:
            status = "COMPLIANT"
            breach_amount = None
        elif current_value >= limit.warning_threshold:
            status = "WARNING"
            breach_amount = None
        else:
            status = "BREACH"
            breach_amount = limit.limit_value - current_value
        
        # Generate recommendations based on status
        recommendations = self._generate_recommendations(limit, status, current_value)
        
        return ComplianceCheck(
            check_id=f"{limit_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            framework=limit.framework,
            limit_id=limit_id,
            current_value=current_value,
            limit_value=limit.limit_value,
            warning_threshold=limit.warning_threshold,
            status=status,
            breach_amount=breach_amount,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, limit: RegulatoryLimit, status: str, current_value: float) -> List[str]:
        """Generate recommendations based on compliance status"""
        
        recommendations = []
        
        if status == "BREACH":
            recommendations.extend([
                f"URGENT: {limit.metric_name} is below regulatory minimum",
                "Immediate capital raising or risk reduction required",
                "Notify regulators and senior management",
                "Implement emergency capital plan"
            ])
        elif status == "WARNING":
            recommendations.extend([
                f"WARNING: {limit.metric_name} approaching regulatory minimum",
                "Consider proactive capital management",
                "Review risk exposures and concentration",
                "Monitor closely for further deterioration"
            ])
        
        # Metric-specific recommendations
        if "ratio" in limit.metric_name.lower():
            recommendations.append("Consider capital retention or new capital issuance")
        
        if "concentration" in limit.metric_name.lower():
            recommendations.append("Diversify portfolio exposures")
            recommendations.append("Reduce position sizes in concentrated areas")
        
        return recommendations

class RegulatoryReportGenerator:
    """Generate regulatory reports in required formats"""
    
    def __init__(self, 
                 db_path: str = "../trades.db",
                 output_dir: str = "regulatory_reports"):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize calculators
        self.data_loader = TradeStewardDataLoader(db_path)
        self.limits_manager = RegulatoryLimitsManager()
        
        self.logger = logging.getLogger(__name__ + '.RegulatoryReportGenerator')
    
    def generate_basel_iii_report(self, reporting_date: date, entity_name: str = "TradeSteward Portfolio") -> RegulatoryReport:
        """Generate Basel III capital adequacy report"""
        
        try:
            # Load and process data
            trades_df = self.data_loader.load_user_trades()
            enhanced_df = self.data_loader.parse_raw_data_json(trades_df)
            clean_df = self.data_loader.clean_and_validate_data(enhanced_df)
            returns_matrix, factor_matrix = self.data_loader.export_for_riskfolio()
            
            # Portfolio weights (equal weight for analysis)
            n_assets = len(returns_matrix.columns)
            portfolio_weights = np.ones(n_assets) / n_assets
            
            # Initialize Basel III calculator
            basel_calc = BaselIIICalculator(returns_matrix, factor_matrix)
            
            # Example capital figures (would come from actual balance sheet)
            total_capital = 1000000  # $1M
            tier1_capital = 800000   # $800K
            cet1_capital = 600000    # $600K
            
            # Calculate RWA
            rwa_calc = basel_calc.calculate_risk_weighted_assets(portfolio_weights)
            total_rwa = rwa_calc['total_rwa']
            
            # Calculate capital ratios
            capital_ratios = basel_calc.calculate_capital_ratios(
                total_capital, tier1_capital, cet1_capital, total_rwa
            )
            
            # Calculate leverage ratio
            total_exposure = 1200000  # $1.2M
            leverage_ratio = basel_calc.calculate_leverage_ratio(tier1_capital, total_exposure)
            
            # Check compliance
            compliance_checks = {}
            
            for ratio_name, ratio_value in capital_ratios.items():
                limit_id = f"basel_{ratio_name}"
                if limit_id in self.limits_manager.limits:
                    check = self.limits_manager.check_compliance(limit_id, ratio_value)
                    compliance_checks[ratio_name] = check.status
            
            # Check leverage ratio
            leverage_check = self.limits_manager.check_compliance("basel_leverage_ratio", leverage_ratio)
            compliance_checks['leverage_ratio'] = leverage_check.status
            
            # Compile report data
            report_data = {
                'capital_figures': {
                    'total_capital': total_capital,
                    'tier1_capital': tier1_capital,
                    'cet1_capital': cet1_capital
                },
                'risk_weighted_assets': rwa_calc,
                'capital_ratios': capital_ratios,
                'leverage_ratio': leverage_ratio,
                'total_exposure': total_exposure
            }
            
            calculations = {
                'portfolio_weights': portfolio_weights.tolist(),
                'returns_matrix_shape': returns_matrix.shape,
                'calculation_date': datetime.now().isoformat()
            }
            
            # Create report
            report = RegulatoryReport(
                report_id=f"BASEL_III_{reporting_date.strftime('%Y%m%d')}",
                framework=RegulatoryFramework.BASEL_III,
                report_type=ReportType.CAPITAL_ADEQUACY,
                reporting_date=reporting_date,
                period_start=reporting_date - timedelta(days=90),  # Quarterly
                period_end=reporting_date,
                entity_name=entity_name,
                data=report_data,
                calculations=calculations,
                compliance_status=compliance_checks,
                certifications=["Risk Management Team", "Chief Risk Officer"],
                attachments=[],
                created_at=datetime.now()
            )
            
            # Save report
            self._save_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating Basel III report: {str(e)}")
            raise
    
    def generate_frtb_report(self, reporting_date: date, entity_name: str = "TradeSteward Portfolio") -> RegulatoryReport:
        """Generate FRTB market risk report"""
        
        try:
            # Load and process data
            trades_df = self.data_loader.load_user_trades()
            enhanced_df = self.data_loader.parse_raw_data_json(trades_df)
            clean_df = self.data_loader.clean_and_validate_data(enhanced_df)
            returns_matrix, factor_matrix = self.data_loader.export_for_riskfolio()
            
            # Portfolio weights
            n_assets = len(returns_matrix.columns)
            portfolio_weights = np.ones(n_assets) / n_assets
            
            # Initialize FRTB calculator
            frtb_calc = FRTBCalculator(returns_matrix, factor_matrix)
            
            # Calculate sensitivities-based method charges
            sbm_charges = frtb_calc.calculate_sensitivities_based_method(portfolio_weights)
            
            # Calculate default risk charge
            default_charge = frtb_calc.calculate_default_risk_charge(portfolio_weights)
            
            # Total FRTB capital charge
            total_frtb_charge = sbm_charges['total_sb_charge'] + default_charge
            
            # Check compliance
            portfolio_value = np.sum(np.abs(portfolio_weights))
            charge_ratio = total_frtb_charge / portfolio_value if portfolio_value > 0 else 0
            
            compliance_check = self.limits_manager.check_compliance("frtb_market_risk_charge", charge_ratio)
            
            # Compile report data
            report_data = {
                'sensitivities_based_method': sbm_charges,
                'default_risk_charge': default_charge,
                'total_capital_charge': total_frtb_charge,
                'portfolio_value': portfolio_value,
                'charge_ratio': charge_ratio
            }
            
            calculations = {
                'portfolio_weights': portfolio_weights.tolist(),
                'calculation_method': 'Sensitivities-Based Method',
                'calculation_date': datetime.now().isoformat()
            }
            
            compliance_status = {
                'market_risk_charge': compliance_check.status
            }
            
            # Create report
            report = RegulatoryReport(
                report_id=f"FRTB_{reporting_date.strftime('%Y%m%d')}",
                framework=RegulatoryFramework.FRTB,
                report_type=ReportType.MARKET_RISK,
                reporting_date=reporting_date,
                period_start=reporting_date - timedelta(days=30),  # Monthly
                period_end=reporting_date,
                entity_name=entity_name,
                data=report_data,
                calculations=calculations,
                compliance_status=compliance_status,
                certifications=["Market Risk Team", "Head of Trading"],
                attachments=[],
                created_at=datetime.now()
            )
            
            # Save report
            self._save_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating FRTB report: {str(e)}")
            raise
    
    def generate_ccar_stress_test_report(self, reporting_date: date, entity_name: str = "TradeSteward Portfolio") -> RegulatoryReport:
        """Generate CCAR stress test report"""
        
        try:
            # Load and process data
            trades_df = self.data_loader.load_user_trades()
            enhanced_df = self.data_loader.parse_raw_data_json(trades_df)
            clean_df = self.data_loader.clean_and_validate_data(enhanced_df)
            returns_matrix, factor_matrix = self.data_loader.export_for_riskfolio()
            
            # Portfolio weights
            n_assets = len(returns_matrix.columns)
            portfolio_weights = np.ones(n_assets) / n_assets
            
            # Initialize CCAR stress tester
            ccar_tester = CCARStressTester(returns_matrix, factor_matrix)
            
            # Run supervisory scenarios
            stress_scenarios = ccar_tester.run_supervisory_scenarios(portfolio_weights)
            
            # Calculate capital impact
            baseline_capital = 1000000  # Starting capital
            
            scenario_impacts = {}
            for scenario_name, results in stress_scenarios.items():
                max_loss = abs(results['max_loss'])
                remaining_capital = baseline_capital - max_loss
                capital_ratio_impact = remaining_capital / baseline_capital
                
                scenario_impacts[scenario_name] = {
                    'max_loss': max_loss,
                    'remaining_capital': remaining_capital,
                    'capital_ratio_impact': capital_ratio_impact,
                    'pass_threshold': capital_ratio_impact >= 0.045  # 4.5% minimum CET1
                }
            
            # Overall stress test assessment
            all_scenarios_pass = all(impact['pass_threshold'] for impact in scenario_impacts.values())
            
            # Compile report data
            report_data = {
                'stress_scenarios': stress_scenarios,
                'scenario_impacts': scenario_impacts,
                'baseline_capital': baseline_capital,
                'overall_assessment': 'PASS' if all_scenarios_pass else 'FAIL'
            }
            
            calculations = {
                'portfolio_weights': portfolio_weights.tolist(),
                'projection_quarters': 9,
                'calculation_date': datetime.now().isoformat()
            }
            
            compliance_status = {
                'stress_test_result': 'COMPLIANT' if all_scenarios_pass else 'BREACH'
            }
            
            # Create report
            report = RegulatoryReport(
                report_id=f"CCAR_{reporting_date.strftime('%Y%m%d')}",
                framework=RegulatoryFramework.CCAR,
                report_type=ReportType.STRESS_TEST,
                reporting_date=reporting_date,
                period_start=reporting_date,
                period_end=reporting_date + timedelta(days=630),  # 9 quarters forward
                entity_name=entity_name,
                data=report_data,
                calculations=calculations,
                compliance_status=compliance_status,
                certifications=["Chief Risk Officer", "Chief Executive Officer"],
                attachments=[],
                created_at=datetime.now()
            )
            
            # Save report
            self._save_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating CCAR stress test report: {str(e)}")
            raise
    
    def _save_report(self, report: RegulatoryReport):
        """Save regulatory report to file"""
        
        # Create framework-specific directory
        framework_dir = self.output_dir / report.framework.value
        framework_dir.mkdir(exist_ok=True)
        
        # Generate filename
        filename = f"{report.report_id}.json"
        filepath = framework_dir / filename
        
        # Convert report to dictionary
        report_dict = asdict(report)
        
        # Convert enums to strings for JSON serialization
        report_dict['framework'] = report.framework.value
        report_dict['report_type'] = report.report_type.value
        report_dict['reporting_date'] = report.reporting_date.isoformat()
        report_dict['period_start'] = report.period_start.isoformat()
        report_dict['period_end'] = report.period_end.isoformat()
        report_dict['created_at'] = report.created_at.isoformat()
        
        # Save to JSON file
        with open(filepath, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        self.logger.info(f"Regulatory report saved: {filepath}")
        
        # Also save as Excel for easy viewing
        self._save_report_excel(report, framework_dir)
    
    def _save_report_excel(self, report: RegulatoryReport, output_dir: Path):
        """Save regulatory report as Excel file"""
        
        try:
            excel_filename = f"{report.report_id}.xlsx"
            excel_filepath = output_dir / excel_filename
            
            with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = {
                    'Report ID': [report.report_id],
                    'Framework': [report.framework.value],
                    'Report Type': [report.report_type.value],
                    'Reporting Date': [report.reporting_date],
                    'Entity': [report.entity_name],
                    'Created': [report.created_at]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Data sheets (flatten nested dictionaries)
                self._write_nested_dict_to_excel(report.data, writer, 'Data')
                self._write_nested_dict_to_excel(report.compliance_status, writer, 'Compliance')
                
                # Recommendations sheet if applicable
                if hasattr(report, 'recommendations') and report.recommendations:
                    rec_df = pd.DataFrame({'Recommendations': report.recommendations})
                    rec_df.to_excel(writer, sheet_name='Recommendations', index=False)
            
            self.logger.info(f"Excel report saved: {excel_filepath}")
            
        except Exception as e:
            self.logger.warning(f"Could not save Excel report: {str(e)}")
    
    def _write_nested_dict_to_excel(self, data_dict: Dict, writer, sheet_name: str):
        """Write nested dictionary to Excel sheet"""
        
        flattened_data = []
        
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        if isinstance(data_dict, dict):
            flat_dict = flatten_dict(data_dict)
            df = pd.DataFrame(list(flat_dict.items()), columns=['Metric', 'Value'])
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    def generate_compliance_summary(self, frameworks: List[RegulatoryFramework] = None) -> pd.DataFrame:
        """Generate compliance summary across frameworks"""
        
        if frameworks is None:
            frameworks = [RegulatoryFramework.BASEL_III, RegulatoryFramework.FRTB]
        
        summary_data = []
        
        for framework in frameworks:
            limits = self.limits_manager.get_limits_by_framework(framework)
            
            for limit in limits:
                # For demonstration, use mock current values
                # In practice, these would be calculated from current portfolio state
                if "ratio" in limit.metric_name.lower():
                    current_value = np.random.uniform(limit.warning_threshold, limit.limit_value * 1.2)
                else:
                    current_value = np.random.uniform(0, limit.limit_value * 0.8)
                
                check = self.limits_manager.check_compliance(limit.limit_id, current_value)
                
                summary_data.append({
                    'Framework': framework.value,
                    'Metric': limit.metric_name,
                    'Current Value': f"{current_value:.4f}",
                    'Limit': f"{limit.limit_value:.4f}",
                    'Warning Threshold': f"{limit.warning_threshold:.4f}",
                    'Status': check.status,
                    'Breach Amount': f"{check.breach_amount:.4f}" if check.breach_amount else "N/A"
                })
        
        return pd.DataFrame(summary_data)


def main():
    """Example usage of Regulatory Compliance Module"""
    
    print("Regulatory Reporting and Compliance Module - Phase 3")
    print("=" * 60)
    
    try:
        # Initialize regulatory report generator
        report_generator = RegulatoryReportGenerator()
        
        print("✅ Regulatory compliance module initialized")
        print(f"📋 Output directory: {report_generator.output_dir}")
        
        # Set reporting date
        reporting_date = date.today()
        entity_name = "TradeSteward Options Portfolio"
        
        print(f"📅 Generating reports for: {reporting_date}")
        print(f"🏢 Entity: {entity_name}")
        
        # Generate Basel III report
        print(f"\n📊 Generating Basel III Capital Adequacy Report...")
        basel_report = report_generator.generate_basel_iii_report(reporting_date, entity_name)
        
        print(f"✅ Basel III report generated: {basel_report.report_id}")
        print(f"   • CET1 Ratio: {basel_report.data['capital_ratios']['cet1_ratio']:.3f}")
        print(f"   • Tier 1 Ratio: {basel_report.data['capital_ratios']['tier1_ratio']:.3f}")
        print(f"   • Total Capital Ratio: {basel_report.data['capital_ratios']['total_capital_ratio']:.3f}")
        print(f"   • Leverage Ratio: {basel_report.data['leverage_ratio']:.3f}")
        
        # Generate FRTB report
        print(f"\n📈 Generating FRTB Market Risk Report...")
        frtb_report = report_generator.generate_frtb_report(reporting_date, entity_name)
        
        print(f"✅ FRTB report generated: {frtb_report.report_id}")
        print(f"   • Total Capital Charge: ${frtb_report.data['total_capital_charge']:,.0f}")
        print(f"   • Delta Charge: ${frtb_report.data['sensitivities_based_method']['delta_charge']:,.0f}")
        print(f"   • Vega Charge: ${frtb_report.data['sensitivities_based_method']['vega_charge']:,.0f}")
        print(f"   • Curvature Charge: ${frtb_report.data['sensitivities_based_method']['curvature_charge']:,.0f}")
        
        # Generate CCAR stress test report
        print(f"\n🧪 Generating CCAR Stress Test Report...")
        ccar_report = report_generator.generate_ccar_stress_test_report(reporting_date, entity_name)
        
        print(f"✅ CCAR stress test report generated: {ccar_report.report_id}")
        print(f"   • Overall Assessment: {ccar_report.data['overall_assessment']}")
        
        for scenario, impact in ccar_report.data['scenario_impacts'].items():
            print(f"   • {scenario.title()}: Max Loss ${impact['max_loss']:,.0f}, Pass: {impact['pass_threshold']}")
        
        # Generate compliance summary
        print(f"\n📋 Generating Compliance Summary...")
        compliance_summary = report_generator.generate_compliance_summary()
        
        print(f"✅ Compliance summary generated")
        print(f"\n{compliance_summary.to_string(index=False)}")
        
        # Count compliance status
        status_counts = compliance_summary['Status'].value_counts()
        print(f"\n📊 Compliance Status Summary:")
        for status, count in status_counts.items():
            print(f"   • {status}: {count}")
        
        print(f"\n📁 Generated Files:")
        for framework_dir in report_generator.output_dir.iterdir():
            if framework_dir.is_dir():
                files = list(framework_dir.glob("*"))
                print(f"   • {framework_dir.name}: {len(files)} files")
                for file in files:
                    print(f"     - {file.name}")
        
        print(f"\n🎉 Regulatory compliance reporting demonstration complete!")
        print(f"All reports saved to: {report_generator.output_dir}")
        
    except Exception as e:
        logger.error(f"Error running regulatory compliance module: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()