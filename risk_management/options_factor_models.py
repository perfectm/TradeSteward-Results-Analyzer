"""
Options-Specific Factor Models - Phase 2 Implementation
Implements Greeks, volatility surface, and options-specific risk factors
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import griddata, RBFInterpolator
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from datetime import datetime, timedelta
import json

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class OptionsFactorModel:
    """
    Options-specific factor model for risk attribution
    Handles Greeks exposure, volatility surface, and term structure
    """
    
    def __init__(self, trades_data: pd.DataFrame):
        """
        Initialize options factor model
        
        Args:
            trades_data: Enhanced trades data with option leg information
        """
        self.trades_data = trades_data
        self.option_legs_data = None
        self.greeks_exposure = None
        self.volatility_surface = None
        self.term_structure = None
        
        # Extract option legs data
        self._extract_option_legs()
    
    def _extract_option_legs(self):
        """Extract and process option legs data from trades"""
        logger.info("Extracting option legs data...")
        
        option_legs = []
        
        for idx, trade in self.trades_data.iterrows():
            if 'option_legs' in trade and trade['option_legs']:
                for leg in trade['option_legs']:
                    if leg.get('symbol') and pd.notna(leg.get('symbol')):
                        leg_data = {
                            'trade_id': trade.get('id', idx),
                            'open_date': trade.get('open_date'),
                            'close_date': trade.get('final_trade_closed_date'),
                            'underlying': trade.get('underlying'),
                            'strategy': trade.get('strategy'),
                            'leg_number': leg.get('leg_number'),
                            'symbol': leg.get('symbol'),
                            'option_type': leg.get('put_call'),
                            'strike': self._safe_float(leg.get('strike')),
                            'expiration': leg.get('expiration'),
                            'quantity': self._safe_float(leg.get('quantity')),
                            'pnl': self._safe_float(leg.get('pnl')),
                            'open_price': self._safe_float(leg.get('open_price')),
                            'close_price': self._safe_float(leg.get('close_price')),
                            'underlying_price': trade.get('underlying_open_quote'),
                            'vix': trade.get('vix_open_quote')
                        }
                        
                        # Calculate derived fields
                        leg_data.update(self._calculate_option_metrics(leg_data))
                        option_legs.append(leg_data)
        
        self.option_legs_data = pd.DataFrame(option_legs)
        logger.info(f"Extracted {len(self.option_legs_data)} option legs from {len(self.trades_data)} trades")
    
    def _safe_float(self, value):
        """Safely convert value to float"""
        try:
            return float(value) if pd.notna(value) and value != '' else 0.0
        except (ValueError, TypeError):
            return 0.0
    
    def _calculate_option_metrics(self, leg_data: Dict) -> Dict:
        """Calculate option-specific metrics for each leg"""
        metrics = {}
        
        try:
            # Time to expiration (approximate)
            if leg_data.get('expiration') and leg_data.get('open_date'):
                try:
                    exp_date = pd.to_datetime(leg_data['expiration'])
                    open_date = pd.to_datetime(leg_data['open_date'])
                    tte = (exp_date - open_date).days / 365.0
                    metrics['time_to_expiration'] = max(tte, 0.01)  # Minimum 1 day
                except:
                    metrics['time_to_expiration'] = 0.25  # Default to 3 months
            else:
                metrics['time_to_expiration'] = 0.25
            
            # Moneyness
            strike = leg_data.get('strike', 0)
            underlying_price = leg_data.get('underlying_price', 0)
            
            if strike > 0 and underlying_price > 0:
                metrics['moneyness'] = underlying_price / strike
                metrics['log_moneyness'] = np.log(underlying_price / strike)
            else:
                metrics['moneyness'] = 1.0
                metrics['log_moneyness'] = 0.0
            
            # Option value and implied metrics
            open_price = leg_data.get('open_price', 0)
            close_price = leg_data.get('close_price', 0)
            
            if open_price > 0:
                metrics['premium_ratio'] = open_price / underlying_price if underlying_price > 0 else 0
                metrics['intrinsic_value'] = self._calculate_intrinsic_value(
                    leg_data.get('option_type'), underlying_price, strike
                )
                metrics['time_value'] = max(0, open_price - metrics['intrinsic_value'])
            else:
                metrics['premium_ratio'] = 0
                metrics['intrinsic_value'] = 0
                metrics['time_value'] = 0
            
            # Greeks approximations (simplified Black-Scholes)
            metrics.update(self._approximate_greeks(leg_data, metrics))
            
        except Exception as e:
            logger.warning(f"Error calculating option metrics: {str(e)}")
            # Set default values
            metrics.update({
                'time_to_expiration': 0.25,
                'moneyness': 1.0,
                'log_moneyness': 0.0,
                'premium_ratio': 0,
                'intrinsic_value': 0,
                'time_value': 0,
                'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0
            })
        
        return metrics
    
    def _calculate_intrinsic_value(self, option_type: str, spot: float, strike: float) -> float:
        """Calculate intrinsic value of option"""
        if not option_type or spot <= 0 or strike <= 0:
            return 0.0
        
        if option_type.upper() in ['C', 'CALL']:
            return max(0, spot - strike)
        elif option_type.upper() in ['P', 'PUT']:
            return max(0, strike - spot)
        else:
            return 0.0
    
    def _approximate_greeks(self, leg_data: Dict, metrics: Dict) -> Dict:
        """Approximate Greeks using simplified Black-Scholes"""
        greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        try:
            S = leg_data.get('underlying_price', 100)  # Spot price
            K = leg_data.get('strike', 100)  # Strike price
            T = metrics.get('time_to_expiration', 0.25)  # Time to expiration
            r = 0.05  # Risk-free rate (assumption)
            
            # Estimate implied volatility from VIX
            vix = leg_data.get('vix', 20)
            sigma = (vix / 100) if vix > 0 else 0.20  # Convert VIX to decimal
            
            option_type = leg_data.get('option_type', '').upper()
            
            if S > 0 and K > 0 and T > 0 and sigma > 0:
                # Black-Scholes calculations
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                
                # Calculate Greeks
                if option_type in ['C', 'CALL']:
                    greeks['delta'] = stats.norm.cdf(d1)
                    greeks['rho'] = K * T * np.exp(-r*T) * stats.norm.cdf(d2) / 100
                elif option_type in ['P', 'PUT']:
                    greeks['delta'] = stats.norm.cdf(d1) - 1
                    greeks['rho'] = -K * T * np.exp(-r*T) * stats.norm.cdf(-d2) / 100
                
                # Greeks common to both calls and puts
                greeks['gamma'] = stats.norm.pdf(d1) / (S * sigma * np.sqrt(T))
                greeks['theta'] = (-S * stats.norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                                 - r * K * np.exp(-r*T) * stats.norm.cdf(d2)) / 365
                greeks['vega'] = S * stats.norm.pdf(d1) * np.sqrt(T) / 100
                
                # Scale by quantity
                quantity = leg_data.get('quantity', 1)
                for greek in greeks:
                    greeks[greek] *= quantity
        
        except Exception as e:
            logger.warning(f"Error calculating Greeks: {str(e)}")
        
        return greeks
    
    def calculate_portfolio_greeks(self, date: str = None) -> Dict[str, float]:
        """
        Calculate portfolio-level Greeks exposure
        
        Args:
            date: Specific date for calculation, None for all positions
            
        Returns:
            Dictionary of portfolio Greeks
        """
        logger.info("Calculating portfolio Greeks exposure...")
        
        if self.option_legs_data is None or len(self.option_legs_data) == 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        # Filter by date if specified
        if date:
            date_filtered = self.option_legs_data[
                self.option_legs_data['open_date'] <= date
            ]
        else:
            date_filtered = self.option_legs_data
        
        # Aggregate Greeks
        portfolio_greeks = {
            'delta': date_filtered['delta'].sum(),
            'gamma': date_filtered['gamma'].sum(),
            'theta': date_filtered['theta'].sum(),
            'vega': date_filtered['vega'].sum(),
            'rho': date_filtered['rho'].sum()
        }
        
        # Calculate normalized Greeks (per $100k)
        total_notional = date_filtered['underlying_price'].sum() * 100  # Assuming 100 shares per contract
        if total_notional > 0:
            portfolio_greeks['delta_normalized'] = portfolio_greeks['delta'] / (total_notional / 100000)
            portfolio_greeks['gamma_normalized'] = portfolio_greeks['gamma'] / (total_notional / 100000)
            portfolio_greeks['vega_normalized'] = portfolio_greeks['vega'] / (total_notional / 100000)
        
        logger.info(f"Portfolio Greeks: Delta={portfolio_greeks['delta']:.2f}, "
                   f"Gamma={portfolio_greeks['gamma']:.4f}, Vega={portfolio_greeks['vega']:.2f}")
        
        self.greeks_exposure = portfolio_greeks
        return portfolio_greeks
    
    def create_volatility_surface(self) -> pd.DataFrame:
        """
        Create implied volatility surface from option data
        
        Returns:
            DataFrame representing the volatility surface
        """
        logger.info("Creating volatility surface...")
        
        if self.option_legs_data is None or len(self.option_legs_data) == 0:
            logger.warning("No option data available for volatility surface")
            return pd.DataFrame()
        
        # Estimate implied volatility for each option
        vol_data = []
        
        for idx, option in self.option_legs_data.iterrows():
            if (option['open_price'] > 0 and option['strike'] > 0 and 
                option['underlying_price'] > 0 and option['time_to_expiration'] > 0):
                
                # Use VIX as proxy for implied volatility (simplified)
                # In production, would use Newton-Raphson to solve for IV
                implied_vol = option['vix'] / 100 if option['vix'] > 0 else 0.20
                
                vol_data.append({
                    'moneyness': option['moneyness'],
                    'time_to_expiration': option['time_to_expiration'],
                    'implied_volatility': implied_vol,
                    'option_type': option['option_type'],
                    'volume': abs(option['quantity']) if option['quantity'] else 1
                })
        
        if not vol_data:
            logger.warning("No valid option data for volatility surface")
            return pd.DataFrame()
        
        vol_df = pd.DataFrame(vol_data)
        
        # Create volatility surface grid
        moneyness_grid = np.linspace(0.8, 1.2, 20)
        tte_grid = np.linspace(0.02, 1.0, 15)  # 1 week to 1 year
        
        # Interpolate volatility surface
        points = vol_df[['moneyness', 'time_to_expiration']].values
        values = vol_df['implied_volatility'].values
        
        try:
            # Create meshgrid for interpolation
            M, T = np.meshgrid(moneyness_grid, tte_grid)
            
            # Use RBF interpolation for smooth surface
            rbf = RBFInterpolator(points, values, kernel='thin_plate_spline')
            vol_surface = rbf(np.column_stack([M.ravel(), T.ravel()]))
            vol_surface = vol_surface.reshape(M.shape)
            
            # Create surface DataFrame
            surface_df = pd.DataFrame(vol_surface, index=tte_grid, columns=moneyness_grid)
            
            self.volatility_surface = surface_df
            logger.info(f"Volatility surface created: {surface_df.shape}")
            
            return surface_df
            
        except Exception as e:
            logger.warning(f"Error creating volatility surface: {str(e)}")
            return pd.DataFrame()
    
    def calculate_factor_loadings(self) -> pd.DataFrame:
        """
        Calculate options-specific factor loadings
        
        Returns:
            DataFrame with factor loadings
        """
        logger.info("Calculating options-specific factor loadings...")
        
        if self.option_legs_data is None:
            return pd.DataFrame()
        
        # Group by strategy and calculate factor exposures
        strategy_factors = []
        
        for strategy in self.option_legs_data['strategy'].unique():
            if pd.isna(strategy):
                continue
                
            strategy_data = self.option_legs_data[self.option_legs_data['strategy'] == strategy]
            
            # Calculate factor exposures
            factors = {
                'strategy': strategy,
                # Greeks factors
                'delta_exposure': strategy_data['delta'].mean(),
                'gamma_exposure': strategy_data['gamma'].mean(),
                'theta_exposure': strategy_data['theta'].mean(),
                'vega_exposure': strategy_data['vega'].mean(),
                'rho_exposure': strategy_data['rho'].mean(),
                
                # Structural factors
                'avg_moneyness': strategy_data['moneyness'].mean(),
                'avg_time_to_expiration': strategy_data['time_to_expiration'].mean(),
                'avg_premium_ratio': strategy_data['premium_ratio'].mean(),
                
                # Volatility factors
                'volatility_exposure': strategy_data['vix'].mean() / 100,
                'volatility_sensitivity': strategy_data['vega'].std(),
                
                # Concentration factors
                'strike_concentration': strategy_data['strike'].std() / strategy_data['strike'].mean() if strategy_data['strike'].mean() > 0 else 0,
                'expiration_concentration': strategy_data['time_to_expiration'].std(),
                
                # Performance factors
                'option_count': len(strategy_data),
                'avg_pnl_per_option': strategy_data['pnl'].mean(),
                'pnl_volatility': strategy_data['pnl'].std()
            }
            
            strategy_factors.append(factors)
        
        factor_df = pd.DataFrame(strategy_factors).set_index('strategy')
        
        logger.info(f"Options factor loadings calculated for {len(factor_df)} strategies")
        return factor_df
    
    def analyze_term_structure_risk(self) -> Dict[str, float]:
        """
        Analyze term structure risk exposure
        
        Returns:
            Dictionary of term structure metrics
        """
        logger.info("Analyzing term structure risk...")
        
        if self.option_legs_data is None:
            return {}
        
        # Group by time to expiration buckets
        tte_buckets = {
            'short_term': (0, 0.083),    # 0-1 month
            'medium_term': (0.083, 0.25), # 1-3 months
            'long_term': (0.25, 1.0)      # 3-12 months
        }
        
        term_analysis = {}
        
        for bucket_name, (min_tte, max_tte) in tte_buckets.items():
            bucket_data = self.option_legs_data[
                (self.option_legs_data['time_to_expiration'] >= min_tte) &
                (self.option_legs_data['time_to_expiration'] < max_tte)
            ]
            
            if len(bucket_data) > 0:
                term_analysis[bucket_name] = {
                    'position_count': len(bucket_data),
                    'total_delta': bucket_data['delta'].sum(),
                    'total_gamma': bucket_data['gamma'].sum(),
                    'total_theta': bucket_data['theta'].sum(),
                    'total_vega': bucket_data['vega'].sum(),
                    'avg_moneyness': bucket_data['moneyness'].mean(),
                    'pnl_contribution': bucket_data['pnl'].sum(),
                    'risk_weight': len(bucket_data) / len(self.option_legs_data)
                }
            else:
                term_analysis[bucket_name] = {
                    'position_count': 0, 'total_delta': 0, 'total_gamma': 0,
                    'total_theta': 0, 'total_vega': 0, 'avg_moneyness': 1.0,
                    'pnl_contribution': 0, 'risk_weight': 0
                }
        
        # Calculate term structure risk metrics
        term_structure_risk = {
            'short_term_weight': term_analysis['short_term']['risk_weight'],
            'medium_term_weight': term_analysis['medium_term']['risk_weight'],
            'long_term_weight': term_analysis['long_term']['risk_weight'],
            'term_structure_skew': (
                term_analysis['short_term']['risk_weight'] - 
                term_analysis['long_term']['risk_weight']
            ),
            'theta_concentration': sum([
                abs(bucket['total_theta']) for bucket in term_analysis.values()
            ]),
            'vega_term_spread': (
                term_analysis['long_term']['total_vega'] - 
                term_analysis['short_term']['total_vega']
            )
        }
        
        self.term_structure = term_structure_risk
        logger.info("Term structure analysis completed")
        
        return term_structure_risk
    
    def plot_options_analysis(self, save_path: str = None) -> None:
        """
        Create comprehensive options analysis visualizations
        
        Args:
            save_path: Path to save the plot
        """
        if self.option_legs_data is None:
            logger.error("No option data available for plotting")
            return
        
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Greeks Distribution
        ax1 = plt.subplot(3, 3, 1)
        greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        greek_values = [self.option_legs_data[greek].sum() for greek in greeks]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        bars = ax1.bar(greeks, greek_values, color=colors, alpha=0.7)
        ax1.set_title('Portfolio Greeks Exposure', fontweight='bold')
        ax1.set_ylabel('Greek Value')
        
        # Add value labels on bars
        for bar, value in zip(bars, greek_values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # 2. Moneyness Distribution
        ax2 = plt.subplot(3, 3, 2)
        ax2.hist(self.option_legs_data['moneyness'], bins=30, alpha=0.7, color='steelblue')
        ax2.axvline(x=1.0, color='red', linestyle='--', label='ATM')
        ax2.set_title('Moneyness Distribution', fontweight='bold')
        ax2.set_xlabel('Moneyness (S/K)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 3. Time to Expiration Distribution
        ax3 = plt.subplot(3, 3, 3)
        ax3.hist(self.option_legs_data['time_to_expiration'] * 365, bins=20, alpha=0.7, color='green')
        ax3.set_title('Time to Expiration Distribution', fontweight='bold')
        ax3.set_xlabel('Days to Expiration')
        ax3.set_ylabel('Frequency')
        
        # 4. P&L by Option Type
        ax4 = plt.subplot(3, 3, 4)
        pnl_by_type = self.option_legs_data.groupby('option_type')['pnl'].sum()
        if len(pnl_by_type) > 0:
            ax4.bar(pnl_by_type.index, pnl_by_type.values, alpha=0.7)
            ax4.set_title('P&L by Option Type', fontweight='bold')
            ax4.set_ylabel('Total P&L ($)')
        
        # 5. Volatility Surface (if available)
        if self.volatility_surface is not None and not self.volatility_surface.empty:
            ax5 = plt.subplot(3, 3, 5, projection='3d')
            
            M, T = np.meshgrid(self.volatility_surface.columns, self.volatility_surface.index)
            surf = ax5.plot_surface(M, T, self.volatility_surface.values, 
                                   cmap='viridis', alpha=0.8)
            
            ax5.set_title('Implied Volatility Surface', fontweight='bold')
            ax5.set_xlabel('Moneyness')
            ax5.set_ylabel('Time to Expiration')
            ax5.set_zlabel('Implied Volatility')
        
        # 6. Delta vs Gamma Scatter
        ax6 = plt.subplot(3, 3, 6)
        scatter = ax6.scatter(self.option_legs_data['delta'], self.option_legs_data['gamma'],
                             c=self.option_legs_data['pnl'], cmap='RdYlGn', alpha=0.6)
        ax6.set_title('Delta vs Gamma (colored by P&L)', fontweight='bold')
        ax6.set_xlabel('Delta')
        ax6.set_ylabel('Gamma')
        plt.colorbar(scatter, ax=ax6, label='P&L ($)')
        
        # 7. Strategy Greeks Comparison
        ax7 = plt.subplot(3, 3, 7)
        strategy_greeks = self.option_legs_data.groupby('strategy')[['delta', 'gamma', 'vega']].sum()
        
        if len(strategy_greeks) > 0:
            strategy_greeks.plot(kind='bar', ax=ax7, alpha=0.7)
            ax7.set_title('Greeks by Strategy', fontweight='bold')
            ax7.set_ylabel('Greek Value')
            ax7.tick_params(axis='x', rotation=45)
            ax7.legend()
        
        # 8. Term Structure Analysis
        if self.term_structure:
            ax8 = plt.subplot(3, 3, 8)
            term_weights = [
                self.term_structure['short_term_weight'],
                self.term_structure['medium_term_weight'],
                self.term_structure['long_term_weight']
            ]
            ax8.pie(term_weights, labels=['Short-term', 'Medium-term', 'Long-term'], 
                   autopct='%1.1f%%', startangle=90)
            ax8.set_title('Term Structure Distribution', fontweight='bold')
        
        # 9. Vega vs VIX Relationship
        ax9 = plt.subplot(3, 3, 9)
        ax9.scatter(self.option_legs_data['vix'], self.option_legs_data['vega'], alpha=0.6)
        ax9.set_title('Vega vs VIX Relationship', fontweight='bold')
        ax9.set_xlabel('VIX Level')
        ax9.set_ylabel('Vega')
        
        # Add trend line
        if len(self.option_legs_data) > 1:
            z = np.polyfit(self.option_legs_data['vix'].dropna(), 
                          self.option_legs_data['vega'].dropna(), 1)
            p = np.poly1d(z)
            x_trend = np.linspace(self.option_legs_data['vix'].min(), 
                                self.option_legs_data['vix'].max(), 100)
            ax9.plot(x_trend, p(x_trend), "r--", alpha=0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Options analysis plot saved to {save_path}")
        
        plt.show()
    
    def generate_options_report(self) -> Dict:
        """
        Generate comprehensive options factor model report
        
        Returns:
            Detailed options analysis report
        """
        logger.info("Generating options factor model report...")
        
        # Calculate all metrics
        portfolio_greeks = self.calculate_portfolio_greeks()
        factor_loadings = self.calculate_factor_loadings()
        term_structure = self.analyze_term_structure_risk()
        volatility_surface = self.create_volatility_surface()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_option_legs': len(self.option_legs_data) if self.option_legs_data is not None else 0,
                'unique_strategies': len(self.option_legs_data['strategy'].unique()) if self.option_legs_data is not None else 0,
                'analysis_period': {
                    'start': self.option_legs_data['open_date'].min().strftime('%Y-%m-%d') if self.option_legs_data is not None and len(self.option_legs_data) > 0 else 'N/A',
                    'end': self.option_legs_data['close_date'].max().strftime('%Y-%m-%d') if self.option_legs_data is not None and len(self.option_legs_data) > 0 else 'N/A'
                }
            },
            'portfolio_greeks': portfolio_greeks,
            'factor_loadings': factor_loadings.to_dict() if not factor_loadings.empty else {},
            'term_structure_risk': term_structure,
            'volatility_surface_shape': volatility_surface.shape if not volatility_surface.empty else (0, 0),
            'risk_insights': self._generate_risk_insights(portfolio_greeks, term_structure)
        }
        
        return report
    
    def _generate_risk_insights(self, greeks: Dict, term_structure: Dict) -> List[str]:
        """Generate risk insights based on analysis"""
        insights = []
        
        # Greeks analysis
        if abs(greeks.get('delta', 0)) > 100:
            insights.append(f"High delta exposure ({greeks['delta']:.0f}) indicates significant directional risk")
        
        if abs(greeks.get('gamma', 0)) > 10:
            insights.append(f"High gamma exposure ({greeks['gamma']:.2f}) suggests large delta sensitivity to price moves")
        
        if greeks.get('theta', 0) < -50:
            insights.append(f"High negative theta ({greeks['theta']:.0f}) indicates significant time decay risk")
        
        if abs(greeks.get('vega', 0)) > 100:
            insights.append(f"High vega exposure ({greeks['vega']:.0f}) shows vulnerability to volatility changes")
        
        # Term structure analysis
        if term_structure:
            if term_structure.get('short_term_weight', 0) > 0.6:
                insights.append("Portfolio heavily concentrated in short-term options (high theta risk)")
            
            if abs(term_structure.get('term_structure_skew', 0)) > 0.3:
                insights.append("Significant term structure skew detected")
        
        if not insights:
            insights.append("Portfolio shows balanced options risk profile")
        
        return insights


def main():
    """Example usage of Options Factor Model"""
    
    try:
        # Load enhanced trades data from Phase 1
        import sys
        sys.path.append('..')
        from data_loader import TradeStewardDataLoader
        
        print("Options Factor Model Analysis - Phase 2")
        print("=" * 60)
        
        # Load data
        loader = TradeStewardDataLoader(db_path="../trades.db")
        trades_df = loader.load_user_trades()
        
        if trades_df.empty:
            print("❌ No trades data available")
            return
        
        enhanced_df = loader.parse_raw_data_json(trades_df)
        clean_df = loader.clean_and_validate_data(enhanced_df)
        
        # Initialize options factor model
        options_model = OptionsFactorModel(clean_df)
        
        # Calculate portfolio Greeks
        portfolio_greeks = options_model.calculate_portfolio_greeks()
        print("\n📊 Portfolio Greeks:")
        for greek, value in portfolio_greeks.items():
            print(f"  • {greek.title()}: {value:.2f}")
        
        # Factor loadings
        factor_loadings = options_model.calculate_factor_loadings()
        print(f"\n🎯 Factor Loadings calculated for {len(factor_loadings)} strategies")
        
        # Term structure analysis
        term_structure = options_model.analyze_term_structure_risk()
        print(f"\n⏰ Term Structure Analysis:")
        for metric, value in term_structure.items():
            print(f"  • {metric.replace('_', ' ').title()}: {value:.3f}")
        
        # Create volatility surface
        vol_surface = options_model.create_volatility_surface()
        print(f"\n🌊 Volatility Surface: {vol_surface.shape}")
        
        # Generate visualizations
        options_model.plot_options_analysis('options_factor_analysis.png')
        
        # Generate comprehensive report
        report = options_model.generate_options_report()
        
        # Save report
        with open('options_factor_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display insights
        print(f"\n💡 Risk Insights:")
        for insight in report['risk_insights']:
            print(f"  • {insight}")
        
        print("\n✅ Options factor analysis complete!")
        print("📁 Results saved to:")
        print("  • options_factor_analysis.png")
        print("  • options_factor_report.json")
        
    except Exception as e:
        logger.error(f"Error in options factor analysis: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()