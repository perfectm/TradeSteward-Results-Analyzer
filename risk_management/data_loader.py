"""
TradeSteward Data Loader for Risk Analysis
Handles database integration and data preprocessing for risk attribution analysis
"""

import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradeStewardDataLoader:
    """
    Data loader for TradeSteward trading data with focus on risk attribution analysis.
    Prioritizes Section 6: Risk Attribution and Decomposition
    """
    
    def __init__(self, db_path: str = "../trades.db"):
        """
        Initialize the data loader
        
        Args:
            db_path: Path to the TradeSteward SQLite database
        """
        self.db_path = db_path
        self.raw_data = None
        self.processed_data = None
        self.daily_returns = None
        self.strategy_portfolios = {}
        
    def connect_database(self) -> sqlite3.Connection:
        """Create database connection with row factory"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def load_user_trades(self, user_id: Optional[int] = None) -> pd.DataFrame:
        """
        Load trades from database for a specific user or all users
        
        Args:
            user_id: Specific user ID to load, None for all users
            
        Returns:
            DataFrame with trade data
        """
        try:
            with self.connect_database() as conn:
                if user_id:
                    query = """
                    SELECT * FROM trades 
                    WHERE user_id = ? 
                    ORDER BY open_date, open_time
                    """
                    df = pd.read_sql_query(query, conn, params=(user_id,))
                else:
                    query = """
                    SELECT * FROM trades 
                    ORDER BY user_id, open_date, open_time
                    """
                    df = pd.read_sql_query(query, conn)
                
                logger.info(f"Loaded {len(df)} trades from database")
                self.raw_data = df
                return df
                
        except Exception as e:
            logger.error(f"Error loading trades: {str(e)}")
            return pd.DataFrame()
    
    def parse_raw_data_json(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse the raw_data JSON column to extract complete trade information
        
        Args:
            df: DataFrame with raw_data column
            
        Returns:
            Enhanced DataFrame with parsed JSON data
        """
        logger.info("Parsing raw_data JSON for complete trade information...")
        
        enhanced_data = []
        
        for idx, row in df.iterrows():
            try:
                # Parse the JSON data
                if pd.notna(row['raw_data']):
                    raw_json = json.loads(row['raw_data'])
                    
                    # Create enhanced record
                    enhanced_record = dict(row)
                    
                    # Add parsed JSON fields
                    for key, value in raw_json.items():
                        if key not in enhanced_record or pd.isna(enhanced_record[key]):
                            enhanced_record[key] = value
                    
                    # Extract multi-leg option data
                    option_legs = []
                    for i in range(1, 5):  # Up to 4 legs
                        opt_symbol = raw_json.get(f'Opt{i}-Symbol')
                        if opt_symbol and pd.notna(opt_symbol):
                            leg_data = {
                                'leg_number': i,
                                'symbol': opt_symbol,
                                'put_call': raw_json.get(f'Opt{i}-PutCall'),
                                'strike': raw_json.get(f'Opt{i}-Strike'),
                                'expiration': raw_json.get(f'Opt{i}-Expiration'),
                                'quantity': raw_json.get(f'Opt{i}-Quantity'),
                                'pnl': raw_json.get(f'Opt{i}-PL'),
                                'open_price': raw_json.get(f'Opt{i}-OpenPrice'),
                                'close_price': raw_json.get(f'Opt{i}-ClosePrice')
                            }
                            option_legs.append(leg_data)
                    
                    enhanced_record['option_legs'] = option_legs
                    enhanced_record['num_legs'] = len(option_legs)
                    
                    enhanced_data.append(enhanced_record)
                else:
                    enhanced_data.append(dict(row))
                    
            except Exception as e:
                logger.warning(f"Error parsing JSON for row {idx}: {str(e)}")
                enhanced_data.append(dict(row))
        
        enhanced_df = pd.DataFrame(enhanced_data)
        logger.info(f"Enhanced {len(enhanced_df)} trades with parsed JSON data")
        
        return enhanced_df
    
    def clean_and_validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate trading data for risk analysis
        
        Args:
            df: Raw trading data
            
        Returns:
            Cleaned and validated DataFrame
        """
        logger.info("Cleaning and validating data...")
        
        # Create a copy for processing
        clean_df = df.copy()
        
        # Convert date columns
        date_columns = ['open_date', 'final_trade_closed_date']
        for col in date_columns:
            if col in clean_df.columns:
                clean_df[col] = pd.to_datetime(clean_df[col], errors='coerce')
        
        # Clean numeric columns
        numeric_columns = [
            'total_net_profit_loss', 'total_gross_profit_loss', 'total_profit_loss_percent',
            'underlying_open_quote', 'underlying_close_quote', 'vix_open_quote', 'vix_close_quote'
        ]
        
        for col in numeric_columns:
            if col in clean_df.columns:
                clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce')
        
        # Remove invalid trades
        initial_count = len(clean_df)
        
        # Filter for valid P&L and dates
        clean_df = clean_df.dropna(subset=['total_net_profit_loss', 'open_date', 'final_trade_closed_date'])
        
        # Remove extreme outliers (beyond 5 standard deviations)
        if 'total_net_profit_loss' in clean_df.columns:
            pnl_mean = clean_df['total_net_profit_loss'].mean()
            pnl_std = clean_df['total_net_profit_loss'].std()
            outlier_threshold = 5 * pnl_std
            clean_df = clean_df[
                abs(clean_df['total_net_profit_loss'] - pnl_mean) <= outlier_threshold
            ]
        
        # Ensure logical date ordering
        clean_df = clean_df[clean_df['final_trade_closed_date'] >= clean_df['open_date']]
        
        final_count = len(clean_df)
        removed_count = initial_count - final_count
        
        logger.info(f"Data cleaning complete: {removed_count} invalid trades removed, {final_count} trades remaining")
        
        # Add calculated fields for risk analysis
        clean_df['trade_duration_days'] = (clean_df['final_trade_closed_date'] - clean_df['open_date']).dt.days
        clean_df['commission'] = clean_df['total_gross_profit_loss'] - clean_df['total_net_profit_loss']
        clean_df['commission'].fillna(0, inplace=True)
        
        # Add market regime classification based on VIX
        clean_df['vix_regime'] = pd.cut(
            clean_df['vix_open_quote'],
            bins=[0, 20, 30, 40, 100],
            labels=['Low_Vol', 'Normal_Vol', 'High_Vol', 'Crisis_Vol'],
            include_lowest=True
        )
        
        self.processed_data = clean_df
        return clean_df
    
    def create_daily_returns_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert trade-level data to daily P&L time series for risk analysis
        
        Args:
            df: Processed trade data
            
        Returns:
            Daily returns DataFrame
        """
        logger.info("Creating daily P&L time series...")
        
        # Group by final trade closed date to get daily P&L
        daily_pnl = df.groupby(df['final_trade_closed_date'].dt.date).agg({
            'total_net_profit_loss': 'sum',
            'total_gross_profit_loss': 'sum',
            'commission': 'sum',
            'id': 'count'  # Trade count per day
        }).reset_index()
        
        daily_pnl.columns = ['date', 'net_pnl', 'gross_pnl', 'commission', 'trade_count']
        daily_pnl['date'] = pd.to_datetime(daily_pnl['date'])
        
        # Create complete date range
        date_range = pd.date_range(
            start=daily_pnl['date'].min(),
            end=daily_pnl['date'].max(),
            freq='D'
        )
        
        # Reindex to include all dates, fill missing with 0
        daily_series = daily_pnl.set_index('date').reindex(date_range, fill_value=0)
        daily_series.index.name = 'date'
        daily_series = daily_series.reset_index()
        
        # Calculate cumulative P&L
        daily_series['cumulative_pnl'] = daily_series['net_pnl'].cumsum()
        
        # Calculate returns (percentage change from cumulative capital)
        # Use a base capital assumption for return calculation
        base_capital = 100000  # $100k base
        daily_series['capital'] = base_capital + daily_series['cumulative_pnl']
        daily_series['daily_return'] = daily_series['net_pnl'] / daily_series['capital'].shift(1)
        daily_series['daily_return'].fillna(0, inplace=True)
        
        # Add rolling metrics
        daily_series['rolling_vol_30d'] = daily_series['daily_return'].rolling(30).std() * np.sqrt(252)
        daily_series['rolling_sharpe_30d'] = (
            daily_series['daily_return'].rolling(30).mean() * 252 / 
            (daily_series['daily_return'].rolling(30).std() * np.sqrt(252))
        )
        
        logger.info(f"Created daily series with {len(daily_series)} days")
        
        self.daily_returns = daily_series
        return daily_series
    
    def create_strategy_portfolios(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Create strategy-based portfolio time series for risk attribution
        
        Args:
            df: Processed trade data
            
        Returns:
            Dictionary of strategy portfolios
        """
        logger.info("Creating strategy-based portfolios...")
        
        strategies = df['strategy'].unique()
        strategy_portfolios = {}
        
        for strategy in strategies:
            if pd.isna(strategy):
                continue
                
            strategy_trades = df[df['strategy'] == strategy]
            
            # Create daily P&L series for this strategy
            strategy_daily = strategy_trades.groupby(
                strategy_trades['final_trade_closed_date'].dt.date
            ).agg({
                'total_net_profit_loss': 'sum',
                'total_gross_profit_loss': 'sum',
                'commission': 'sum',
                'id': 'count'
            }).reset_index()
            
            strategy_daily.columns = ['date', 'net_pnl', 'gross_pnl', 'commission', 'trade_count']
            strategy_daily['date'] = pd.to_datetime(strategy_daily['date'])
            
            # Fill complete date range
            if len(strategy_daily) > 0:
                full_date_range = pd.date_range(
                    start=self.daily_returns['date'].min(),
                    end=self.daily_returns['date'].max(),
                    freq='D'
                )
                
                strategy_series = strategy_daily.set_index('date').reindex(
                    full_date_range, fill_value=0
                ).reset_index()
                strategy_series.columns = ['date', 'net_pnl', 'gross_pnl', 'commission', 'trade_count']
                
                # Calculate strategy-specific metrics
                strategy_series['cumulative_pnl'] = strategy_series['net_pnl'].cumsum()
                
                # Calculate returns for this strategy
                base_capital = 50000  # Assume equal capital allocation
                strategy_series['capital'] = base_capital + strategy_series['cumulative_pnl']
                strategy_series['daily_return'] = (
                    strategy_series['net_pnl'] / strategy_series['capital'].shift(1)
                )
                strategy_series['daily_return'].fillna(0, inplace=True)
                
                strategy_portfolios[strategy] = strategy_series
                
        logger.info(f"Created {len(strategy_portfolios)} strategy portfolios")
        
        self.strategy_portfolios = strategy_portfolios
        return strategy_portfolios
    
    def get_factor_data(self, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        Get market factor data for risk attribution (VIX, SPY, etc.)
        This is a placeholder - in production, you'd fetch from financial data provider
        
        Args:
            start_date: Start date for factor data
            end_date: End date for factor data
            
        Returns:
            DataFrame with factor returns
        """
        logger.info("Creating placeholder factor data...")
        
        if self.daily_returns is None:
            logger.warning("No daily returns data available")
            return pd.DataFrame()
        
        # Create date range from daily returns
        dates = self.daily_returns['date']
        
        # Placeholder factor data (in production, use yfinance or similar)
        np.random.seed(42)  # For reproducible results
        
        factor_data = pd.DataFrame({
            'date': dates,
            'market_factor': np.random.normal(0.0005, 0.015, len(dates)),  # SPY-like returns
            'volatility_factor': np.random.normal(-0.0001, 0.05, len(dates)),  # VIX-like returns
            'interest_rate_factor': np.random.normal(0.0001, 0.002, len(dates)),  # Interest rate changes
            'momentum_factor': np.random.normal(0.0002, 0.01, len(dates)),  # Momentum factor
            'value_factor': np.random.normal(0.0001, 0.008, len(dates))  # Value factor
        })
        
        logger.info("Created placeholder factor data (replace with real data in production)")
        return factor_data
    
    def export_for_riskfolio(self, strategy_name: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Export data in format suitable for riskfolio-lib analysis
        
        Args:
            strategy_name: Specific strategy to export, None for all
            
        Returns:
            Tuple of (returns_matrix, factor_matrix)
        """
        logger.info("Preparing data for riskfolio-lib analysis...")
        
        if strategy_name and strategy_name in self.strategy_portfolios:
            # Export single strategy
            strategy_data = self.strategy_portfolios[strategy_name]
            returns_df = pd.DataFrame({
                'date': strategy_data['date'],
                strategy_name: strategy_data['daily_return']
            })
        else:
            # Export all strategies as multi-asset portfolio
            returns_data = {'date': self.daily_returns['date']}
            
            for strategy, data in self.strategy_portfolios.items():
                returns_data[strategy] = data['daily_return']
            
            returns_df = pd.DataFrame(returns_data)
        
        # Set date as index for riskfolio-lib
        returns_matrix = returns_df.set_index('date')
        
        # Get factor data
        factor_data = self.get_factor_data()
        factor_matrix = factor_data.set_index('date')
        
        logger.info(f"Exported returns matrix: {returns_matrix.shape}, factor matrix: {factor_matrix.shape}")
        
        return returns_matrix, factor_matrix
    
    def get_data_summary(self) -> Dict:
        """
        Get summary statistics of loaded data
        
        Returns:
            Dictionary with data summary
        """
        if self.processed_data is None:
            return {"error": "No processed data available"}
        
        summary = {
            "total_trades": len(self.processed_data),
            "date_range": {
                "start": self.processed_data['open_date'].min().strftime('%Y-%m-%d'),
                "end": self.processed_data['final_trade_closed_date'].max().strftime('%Y-%m-%d')
            },
            "strategies": list(self.processed_data['strategy'].unique()),
            "total_pnl": float(self.processed_data['total_net_profit_loss'].sum()),
            "total_commission": float(self.processed_data['commission'].sum()),
            "unique_underlyings": list(self.processed_data['underlying'].unique()),
            "vix_range": {
                "min": float(self.processed_data['vix_open_quote'].min()),
                "max": float(self.processed_data['vix_open_quote'].max()),
                "mean": float(self.processed_data['vix_open_quote'].mean())
            }
        }
        
        if self.daily_returns is not None:
            summary["daily_series_length"] = len(self.daily_returns)
            summary["trading_days"] = int(self.daily_returns[self.daily_returns['trade_count'] > 0]['trade_count'].count())
        
        if self.strategy_portfolios:
            summary["strategy_portfolios"] = len(self.strategy_portfolios)
        
        return summary


def main():
    """Example usage of the TradeSteward Data Loader"""
    
    # Initialize data loader
    loader = TradeStewardDataLoader()
    
    # Load trades (all users for now)
    trades_df = loader.load_user_trades()
    
    if trades_df.empty:
        logger.error("No trades loaded from database")
        return
    
    # Parse JSON data
    enhanced_df = loader.parse_raw_data_json(trades_df)
    
    # Clean and validate
    clean_df = loader.clean_and_validate_data(enhanced_df)
    
    # Create time series
    daily_series = loader.create_daily_returns_series(clean_df)
    
    # Create strategy portfolios
    strategy_portfolios = loader.create_strategy_portfolios(clean_df)
    
    # Export for riskfolio analysis
    returns_matrix, factor_matrix = loader.export_for_riskfolio()
    
    # Print summary
    summary = loader.get_data_summary()
    print("\nData Loading Summary:")
    print("=" * 50)
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nReturns matrix shape: {returns_matrix.shape}")
    print(f"Factor matrix shape: {factor_matrix.shape}")
    
    # Save processed data
    returns_matrix.to_csv('risk_management/returns_matrix.csv')
    factor_matrix.to_csv('risk_management/factor_matrix.csv')
    
    logger.info("Data loading and preprocessing complete!")


if __name__ == "__main__":
    main()