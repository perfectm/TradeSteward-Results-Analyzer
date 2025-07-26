from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import io
import base64
import os
from typing import List, Optional, Dict, Tuple
import json
from pathlib import Path
import logging
import traceback
import csv
import sqlite3
from contextlib import contextmanager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_csv_content(raw_csv_content: str) -> str:
    """Clean CSV content to handle malformed rows and formatting issues"""
    try:
        # Remove Windows carriage returns
        content = raw_csv_content.replace('\r\n', '\n').replace('\r', '\n')
        
        # Split into lines
        lines = content.strip().split('\n')
        if not lines:
            return raw_csv_content
        
        # Get header and expected column count
        header = lines[0]
        expected_cols = len(header.split(','))
        
        cleaned_lines = [header]
        
        # Process each data line
        for line in lines[1:]:
            if not line.strip():
                continue
                
            # Remove trailing comma if present (common TradeSteward CSV issue)
            if line.endswith(','):
                line = line[:-1]
            
            # Count commas to estimate columns
            comma_count = line.count(',')
            
            # If line has too many commas (embedded commas in quoted fields)
            if comma_count > expected_cols - 1:
                # Try to parse with csv module to handle quoted fields properly
                try:
                    reader = csv.reader([line])
                    row = next(reader)
                    
                    # If we got too many fields, truncate to expected count
                    if len(row) > expected_cols:
                        row = row[:expected_cols]
                    
                    # Reconstruct the line
                    cleaned_line = ','.join(str(field) for field in row)
                    cleaned_lines.append(cleaned_line)
                    
                except:
                    # If csv parsing fails, try simple truncation
                    parts = line.split(',')
                    if len(parts) > expected_cols:
                        parts = parts[:expected_cols]
                    cleaned_lines.append(','.join(parts))
            else:
                # Line seems fine, keep as is
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
        
    except Exception as e:
        logger.warning(f"CSV cleaning failed: {e}. Using original content.")
        return raw_csv_content

class TradeDatabase:
    """Database manager for persistent trade storage"""
    
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        self.init_database()
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def init_database(self):
        """Initialize the database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    open_order_number TEXT UNIQUE NOT NULL,
                    account_num TEXT,
                    bot_name TEXT,
                    strategy TEXT,
                    open_date TEXT,
                    open_time TEXT,
                    underlying TEXT,
                    underlying_open_quote REAL,
                    vix_open_quote REAL,
                    final_trade_closed_date TEXT,
                    final_trade_closed_time TEXT,
                    underlying_close_quote REAL,
                    vix_close_quote REAL,
                    total_net_profit_loss REAL,
                    total_gross_profit_loss REAL,
                    total_profit_loss_percent REAL,
                    raw_data TEXT,  -- Store complete row as JSON for full analysis
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create index on open_order_number for fast duplicate detection
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_open_order_number 
                ON trades(open_order_number)
            ''')
            
            # Create index on dates for time-based queries
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_dates 
                ON trades(open_date, final_trade_closed_date)
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def insert_trades(self, df: pd.DataFrame) -> Tuple[int, int, int]:
        """
        Insert trades from DataFrame, avoiding duplicates
        Returns: (inserted_count, updated_count, duplicate_count)
        """
        inserted_count = 0
        updated_count = 0
        duplicate_count = 0
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            for _, row in df.iterrows():
                open_order_number = str(row.get('OpenOrderNumber', ''))
                
                # Check if trade already exists
                cursor.execute(
                    'SELECT id, updated_at FROM trades WHERE open_order_number = ?',
                    (open_order_number,)
                )
                existing = cursor.fetchone()
                
                # Prepare trade data
                trade_data = {
                    'open_order_number': open_order_number,
                    'account_num': str(row.get('AccountNum', '')),
                    'bot_name': str(row.get('BotName', '')),
                    'strategy': str(row.get('Strategy', '')),
                    'open_date': str(row.get('OpenDate', '')),
                    'open_time': str(row.get('OpenTime', '')),
                    'underlying': str(row.get('Underlying', '')),
                    'underlying_open_quote': float(row.get('UnderlyingOpenQuote', 0)) if pd.notna(row.get('UnderlyingOpenQuote')) else None,
                    'vix_open_quote': float(row.get('VIXOpenQuote', 0)) if pd.notna(row.get('VIXOpenQuote')) else None,
                    'final_trade_closed_date': str(row.get('FinalTradeClosedDate', '')),
                    'final_trade_closed_time': str(row.get('FinalTradeClosedTime', '')),
                    'underlying_close_quote': float(row.get('UnderlyingCloseQuote', 0)) if pd.notna(row.get('UnderlyingCloseQuote')) else None,
                    'vix_close_quote': float(row.get('VIXCloseQuote', 0)) if pd.notna(row.get('VIXCloseQuote')) else None,
                    'total_net_profit_loss': float(row.get('TotalNetProfitLoss', 0)) if pd.notna(row.get('TotalNetProfitLoss')) else None,
                    'total_gross_profit_loss': float(row.get('TotalGrossProfitLoss', 0)) if pd.notna(row.get('TotalGrossProfitLoss')) else None,
                    'total_profit_loss_percent': float(row.get('TotalProfitLossPercent', 0)) if pd.notna(row.get('TotalProfitLossPercent')) else None,
                    'raw_data': json.dumps(row.to_dict(), default=str)  # Store complete row for full analysis
                }
                
                if existing:
                    # Update existing record
                    cursor.execute('''
                        UPDATE trades SET
                            account_num = ?, bot_name = ?, strategy = ?, open_date = ?, open_time = ?,
                            underlying = ?, underlying_open_quote = ?, vix_open_quote = ?,
                            final_trade_closed_date = ?, final_trade_closed_time = ?,
                            underlying_close_quote = ?, vix_close_quote = ?,
                            total_net_profit_loss = ?, total_gross_profit_loss = ?, total_profit_loss_percent = ?,
                            raw_data = ?, updated_at = CURRENT_TIMESTAMP
                        WHERE open_order_number = ?
                    ''', (
                        trade_data['account_num'], trade_data['bot_name'], trade_data['strategy'],
                        trade_data['open_date'], trade_data['open_time'], trade_data['underlying'],
                        trade_data['underlying_open_quote'], trade_data['vix_open_quote'],
                        trade_data['final_trade_closed_date'], trade_data['final_trade_closed_time'],
                        trade_data['underlying_close_quote'], trade_data['vix_close_quote'],
                        trade_data['total_net_profit_loss'], trade_data['total_gross_profit_loss'], trade_data['total_profit_loss_percent'],
                        trade_data['raw_data'], open_order_number
                    ))
                    updated_count += 1
                else:
                    # Insert new record
                    try:
                        cursor.execute('''
                            INSERT INTO trades (
                                open_order_number, account_num, bot_name, strategy, open_date, open_time,
                                underlying, underlying_open_quote, vix_open_quote,
                                final_trade_closed_date, final_trade_closed_time,
                                underlying_close_quote, vix_close_quote,
                                total_net_profit_loss, total_gross_profit_loss, total_profit_loss_percent,
                                raw_data
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            trade_data['open_order_number'], trade_data['account_num'], trade_data['bot_name'], trade_data['strategy'],
                            trade_data['open_date'], trade_data['open_time'], trade_data['underlying'],
                            trade_data['underlying_open_quote'], trade_data['vix_open_quote'],
                            trade_data['final_trade_closed_date'], trade_data['final_trade_closed_time'],
                            trade_data['underlying_close_quote'], trade_data['vix_close_quote'],
                            trade_data['total_net_profit_loss'], trade_data['total_gross_profit_loss'], trade_data['total_profit_loss_percent'],
                            trade_data['raw_data']
                        ))
                        inserted_count += 1
                    except sqlite3.IntegrityError:
                        # Handle race condition where record was inserted between check and insert
                        duplicate_count += 1
            
            conn.commit()
        
        logger.info(f"Database operation completed: {inserted_count} inserted, {updated_count} updated, {duplicate_count} duplicates")
        return inserted_count, updated_count, duplicate_count
    
    def get_all_trades(self) -> pd.DataFrame:
        """Retrieve all trades as a pandas DataFrame"""
        with self.get_connection() as conn:
            # Get the raw data and reconstruct the DataFrame
            cursor = conn.cursor()
            cursor.execute('SELECT raw_data FROM trades ORDER BY open_date, open_time')
            
            rows = []
            for row in cursor.fetchall():
                trade_data = json.loads(row['raw_data'])
                rows.append(trade_data)
            
            if not rows:
                return pd.DataFrame()
            
            df = pd.DataFrame(rows)
            
            # Convert data types to match original processing
            if 'OpenDate' in df.columns:
                df['OpenDate'] = pd.to_datetime(df['OpenDate'], errors='coerce')
            if 'FinalTradeClosedDate' in df.columns:
                df['FinalTradeClosedDate'] = pd.to_datetime(df['FinalTradeClosedDate'], errors='coerce')
            
            # Convert numeric columns
            numeric_cols = ['TotalNetProfitLoss', 'TotalGrossProfitLoss', 'TotalProfitLossPercent', 
                          'UnderlyingOpenQuote', 'UnderlyingCloseQuote', 'VIXOpenQuote', 'VIXCloseQuote']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            return df
    
    def get_trade_count(self) -> int:
        """Get total number of trades in database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM trades')
            return cursor.fetchone()[0]

app = FastAPI(title="TradeSteward Results Analyzer", 
              description="Advanced analysis tool for TradeSteward options trading results")

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads/plots", exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

class TradeStewardAnalyzer:
    """Core analyzer for TradeSteward trading data"""
    
    def __init__(self, db_path: str = "trades.db"):
        self.db = TradeDatabase(db_path)
        self.data = None
        self.daily_pnl = None
        self.metrics = {}
        
        # Load existing data from database on initialization
        self._load_from_database()
        
    def _load_from_database(self):
        """Load all existing trades from database"""
        try:
            self.data = self.db.get_all_trades()
            if not self.data.empty:
                logger.info(f"Loaded {len(self.data)} existing trades from database")
                self._calculate_daily_pnl()
                self._calculate_metrics()
            else:
                logger.info("No existing trades found in database")
        except Exception as e:
            logger.error(f"Error loading from database: {e}")
        
    def load_data(self, csv_content: str) -> dict:
        """Load and parse TradeSteward CSV data, merging with existing database"""
        try:
            logger.info("Starting to load CSV data")
            
            # Read CSV directly - no line number prefixes in actual file
            new_df = pd.read_csv(io.StringIO(csv_content))
            logger.info(f"CSV loaded successfully. Shape: {new_df.shape}")
            logger.info(f"Original columns: {list(new_df.columns)[:10]}...")  # Log first 10 columns
            
            # Clean column names by stripping whitespace
            new_df.columns = new_df.columns.str.strip()
            logger.info(f"Cleaned columns: {list(new_df.columns)[:10]}...")  # Log cleaned columns
            
            # Check for required columns
            required_cols = ['OpenDate', 'FinalTradeClosedDate', 'TotalNetProfitLoss', 'Strategy', 'OpenOrderNumber']
            missing_cols = [col for col in required_cols if col not in new_df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            logger.info("Parsing dates...")
            # Parse dates with proper handling (flexible format for dates and datetimes)
            new_df['OpenDate'] = pd.to_datetime(new_df['OpenDate'], errors='coerce')
            new_df['FinalTradeClosedDate'] = pd.to_datetime(new_df['FinalTradeClosedDate'], errors='coerce')
            
            # Log date parsing results
            open_date_nas = new_df['OpenDate'].isna().sum()
            close_date_nas = new_df['FinalTradeClosedDate'].isna().sum()
            logger.info(f"Date parsing complete. OpenDate NAs: {open_date_nas}, FinalTradeClosedDate NAs: {close_date_nas}")
            
            # Filter out rows with invalid dates
            before_filter = len(new_df)
            new_df = new_df.dropna(subset=['OpenDate', 'FinalTradeClosedDate'])
            after_filter = len(new_df)
            logger.info(f"Filtered out {before_filter - after_filter} rows with invalid dates. Remaining: {after_filter}")
            
            if len(new_df) == 0:
                raise ValueError("No valid trades remain after filtering invalid dates")
            
            logger.info("Converting numeric columns...")
            # Clean and prepare key columns
            numeric_cols = ['TotalNetProfitLoss', 'TotalGrossProfitLoss', 'TotalProfitLossPercent', 
                          'UnderlyingOpenQuote', 'UnderlyingCloseQuote', 'VIXOpenQuote', 'VIXCloseQuote']
            
            for col in numeric_cols:
                if col in new_df.columns:
                    new_df[col] = pd.to_numeric(new_df[col], errors='coerce')
                    nas = new_df[col].isna().sum()
                    logger.info(f"Column {col}: {nas} NAs after conversion")
            
            # Filter out invalid trades
            before_filter = len(new_df)
            new_df = new_df.dropna(subset=['TotalNetProfitLoss', 'OpenDate'])
            after_filter = len(new_df)
            logger.info(f"Filtered out {before_filter - after_filter} rows with invalid P&L or dates. Final count: {after_filter}")
            
            if len(new_df) == 0:
                raise ValueError("No valid trades remain after filtering")
            
            # Store new data in database and get merge statistics
            logger.info("Storing new data in database...")
            inserted_count, updated_count, duplicate_count = self.db.insert_trades(new_df)
            
            # Reload all data from database (includes both old and new data)
            logger.info("Reloading all data from database...")
            self._load_from_database()
            
            logger.info("Data processing complete")
            
            return {
                "success": True,
                "total_trades": len(self.data) if self.data is not None else 0,
                "new_trades": inserted_count,
                "updated_trades": updated_count,
                "duplicate_trades": duplicate_count,
                "date_range": f"{self.data['OpenDate'].min().strftime('%Y-%m-%d')} to {self.data['FinalTradeClosedDate'].max().strftime('%Y-%m-%d')}" if self.data is not None and not self.data.empty else "N/A",
                "total_pnl": float(self.data['TotalNetProfitLoss'].sum()) if self.data is not None and not self.data.empty else 0.0
            }
            
        except Exception as e:
            logger.error(f"Error in load_data: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}
    
    def _calculate_daily_pnl(self):
        """Calculate daily P&L from individual trades"""
        if self.data is None:
            return
        
        # Use FinalTradeClosedDate for P&L aggregation
        daily_data = self.data.groupby(self.data['FinalTradeClosedDate'].dt.date).agg({
            'TotalNetProfitLoss': 'sum',
            'TotalGrossProfitLoss': 'sum',
            'OpenOrderNumber': 'count'
        }).reset_index()
        
        daily_data.columns = ['Date', 'NetPnL', 'GrossPnL', 'TradeCount']
        daily_data['Date'] = pd.to_datetime(daily_data['Date'])
        daily_data['CumulativePnL'] = daily_data['NetPnL'].cumsum()
        
        self.daily_pnl = daily_data
    
    def _calculate_metrics(self):
        """Calculate comprehensive trading metrics"""
        if self.data is None or self.daily_pnl is None:
            return
        
        trades = self.data
        daily = self.daily_pnl
        
        # Basic Performance Metrics
        total_trades = len(trades)
        total_pnl = trades['TotalNetProfitLoss'].sum()
        total_gross_pnl = trades['TotalGrossProfitLoss'].sum()
        total_fees = total_pnl - total_gross_pnl if not pd.isna(total_gross_pnl) else 0
        
        # Win Rate & Trade Analysis
        winning_trades = trades[trades['TotalNetProfitLoss'] > 0]
        losing_trades = trades[trades['TotalNetProfitLoss'] < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_winner = winning_trades['TotalNetProfitLoss'].mean() if len(winning_trades) > 0 else 0
        avg_loser = losing_trades['TotalNetProfitLoss'].mean() if len(losing_trades) > 0 else 0
        
        # Risk Metrics
        daily_returns = daily['NetPnL'].values
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = np.mean(daily_returns) * 252 / volatility if volatility != 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Drawdown Analysis
        cumulative = daily['CumulativePnL'].values
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - rolling_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Time-based Analysis
        trading_days = len(daily)
        if trading_days > 0:
            start_date = daily['Date'].min()
            end_date = daily['Date'].max()
            total_days = (end_date - start_date).days + 1
            avg_daily_pnl = total_pnl / trading_days
        else:
            total_days = 0
            avg_daily_pnl = 0
        
        # Strategy Breakdown
        strategy_performance = trades.groupby('Strategy').agg({
            'TotalNetProfitLoss': ['sum', 'mean', 'count'],
            'TotalProfitLossPercent': 'mean'
        }).round(2)
        
        # Convert strategy performance to JSON-serializable format
        strategy_dict = {}
        for strategy in strategy_performance.index:
            strategy_dict[strategy] = {
                'total_pnl': float(strategy_performance.loc[strategy, ('TotalNetProfitLoss', 'sum')]),
                'avg_pnl': float(strategy_performance.loc[strategy, ('TotalNetProfitLoss', 'mean')]),
                'trade_count': int(strategy_performance.loc[strategy, ('TotalNetProfitLoss', 'count')]),
                'avg_percentage': float(strategy_performance.loc[strategy, ('TotalProfitLossPercent', 'mean')])
            }
        
        # VIX Analysis
        vix_stats = {
            'avg_vix_open': float(trades['VIXOpenQuote'].mean()),
            'avg_vix_close': float(trades['VIXCloseQuote'].mean()),
            'vix_range': [float(trades['VIXOpenQuote'].min()), float(trades['VIXOpenQuote'].max())]
        }
        
        self.metrics = {
            'total_trades': int(total_trades),
            'total_pnl': float(round(total_pnl, 2)),
            'total_gross_pnl': float(round(total_gross_pnl, 2)),
            'total_fees': float(round(total_fees, 2)),
            'win_rate': float(round(win_rate, 2)),
            'avg_winner': float(round(avg_winner, 2)),
            'avg_loser': float(round(avg_loser, 2)),
            'profit_factor': float(round(abs(avg_winner * len(winning_trades) / (avg_loser * len(losing_trades))), 2)) if len(losing_trades) > 0 and avg_loser != 0 else float('inf'),
            'max_drawdown': float(round(max_drawdown, 2)),
            'volatility': float(round(volatility, 2)),
            'sharpe_ratio': float(round(sharpe_ratio, 2)),
            'trading_days': int(trading_days),
            'total_days': int(total_days),
            'avg_daily_pnl': float(round(avg_daily_pnl, 2)),
            'strategy_performance': strategy_dict,
            'vix_stats': vix_stats,
            'date_range': {
                'start': start_date.strftime('%Y-%m-%d') if trading_days > 0 else 'N/A',
                'end': end_date.strftime('%Y-%m-%d') if trading_days > 0 else 'N/A'
            }
        }
    
    def generate_plots(self) -> List[str]:
        """Generate comprehensive analysis plots"""
        if self.data is None or self.daily_pnl is None:
            return []
        
        plot_files = []
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Cumulative P&L Chart
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Cumulative P&L
        ax1.plot(self.daily_pnl['Date'], self.daily_pnl['CumulativePnL'], 
                linewidth=2, color='#2E86AB', marker='o', markersize=3)
        ax1.fill_between(self.daily_pnl['Date'], self.daily_pnl['CumulativePnL'], 
                        alpha=0.3, color='#2E86AB')
        ax1.set_title('Cumulative P&L Over Time', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # Daily P&L
        colors = ['green' if x > 0 else 'red' for x in self.daily_pnl['NetPnL']]
        ax2.bar(self.daily_pnl['Date'], self.daily_pnl['NetPnL'], 
               color=colors, alpha=0.7)
        ax2.set_title('Daily P&L', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Daily P&L ($)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        
        plt.tight_layout()
        filename = f"plots/cumulative_pnl_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f"uploads/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(filename)
        
        # 2. Strategy Performance Analysis
        strategy_pnl = self.data.groupby('Strategy')['TotalNetProfitLoss'].agg(['sum', 'count']).reset_index()
        strategy_pnl.columns = ['Strategy', 'TotalPnL', 'TradeCount']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Strategy P&L
        bars1 = ax1.bar(range(len(strategy_pnl)), strategy_pnl['TotalPnL'], 
                       color=['green' if x > 0 else 'red' for x in strategy_pnl['TotalPnL']])
        ax1.set_title('P&L by Strategy', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Total P&L ($)', fontsize=12)
        ax1.set_xticks(range(len(strategy_pnl)))
        ax1.set_xticklabels(strategy_pnl['Strategy'], rotation=45, ha='right')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, strategy_pnl['TotalPnL']):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + (50 if height > 0 else -150),
                    f'${value:.0f}', ha='center', va='bottom' if height > 0 else 'top')
        
        # Trade Count by Strategy
        ax2.bar(range(len(strategy_pnl)), strategy_pnl['TradeCount'], color='steelblue', alpha=0.7)
        ax2.set_title('Trade Count by Strategy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Number of Trades', fontsize=12)
        ax2.set_xticks(range(len(strategy_pnl)))
        ax2.set_xticklabels(strategy_pnl['Strategy'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"plots/strategy_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f"uploads/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(filename)
        
        # 3. Risk Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # P&L Distribution
        ax1.hist(self.data['TotalNetProfitLoss'], bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.set_title('P&L Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('P&L ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        
        # VIX vs P&L (colored by strategy)
        strategies = self.data['Strategy'].unique()
        colors = plt.cm.Set1(np.linspace(0, 1, len(strategies)))
        
        for i, strategy in enumerate(strategies):
            strategy_data = self.data[self.data['Strategy'] == strategy]
            ax2.scatter(strategy_data['VIXOpenQuote'], strategy_data['TotalNetProfitLoss'], 
                       alpha=0.7, color=colors[i], label=strategy, s=30)
        
        ax2.set_title('VIX vs Trade P&L (by Strategy)', fontsize=14, fontweight='bold')
        ax2.set_xlabel('VIX at Open', fontsize=12)
        ax2.set_ylabel('Trade P&L ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Monthly Performance
        self.data['YearMonth'] = self.data['OpenDate'].dt.to_period('M')
        monthly_pnl = self.data.groupby('YearMonth')['TotalNetProfitLoss'].sum()
        
        ax3.bar(range(len(monthly_pnl)), monthly_pnl.values, 
               color=['green' if x > 0 else 'red' for x in monthly_pnl.values])
        ax3.set_title('Monthly P&L', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Monthly P&L ($)', fontsize=12)
        ax3.set_xticks(range(len(monthly_pnl)))
        ax3.set_xticklabels([str(m) for m in monthly_pnl.index], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Drawdown Chart
        cumulative = self.daily_pnl['CumulativePnL'].values
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - rolling_max
        
        ax4.fill_between(self.daily_pnl['Date'], drawdown, alpha=0.7, color='red')
        ax4.set_title('Drawdown Over Time', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Drawdown ($)', fontsize=12)
        ax4.set_xlabel('Date', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = f"plots/risk_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f"uploads/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(filename)
        
        return plot_files

# Global analyzer instance
analyzer = TradeStewardAnalyzer()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main upload page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Process uploaded TradeSteward CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    
    try:
        # Read file content
        content = await file.read()
        raw_csv_content = content.decode('utf-8')
        
        # Clean the CSV content using the same logic as previous cleaning steps
        csv_content = clean_csv_content(raw_csv_content)
        
        # Analyze data
        result = analyzer.load_data(csv_content)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=f"Error processing file: {result['error']}")
        
        # Generate plots
        plot_files = analyzer.generate_plots()
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "summary": result,
            "metrics": analyzer.metrics,
            "plots": plot_files,
            # Add merge statistics at top level for frontend
            "total_trades": result.get("total_trades", 0),
            "new_trades": result.get("new_trades", 0),
            "updated_trades": result.get("updated_trades", 0),
            "duplicate_trades": result.get("duplicate_trades", 0),
            "date_range": result.get("date_range", "N/A"),
            "total_pnl": result.get("total_pnl", 0.0)
        }
        
        return response
        
    except Exception as e:
        logger.error("Error processing file", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    """Get current analysis metrics"""
    if analyzer.data is None:
        raise HTTPException(status_code=404, detail="No data loaded. Please upload a file first.")
    
    return analyzer.metrics

@app.get("/database-analysis")
async def get_database_analysis():
    """Get complete analysis of all data in the database"""
    try:
        # Check if there's data in the database
        total_trades = analyzer.db.get_trade_count()
        if total_trades == 0:
            raise HTTPException(status_code=404, detail="No trading data found in database. Please upload a CSV file first.")
        
        # Reload data from database (in case there were any changes)
        analyzer._load_from_database()
        
        if analyzer.data is None or analyzer.data.empty:
            raise HTTPException(status_code=404, detail="No valid trading data found in database.")
        
        # Generate plots
        plot_files = analyzer.generate_plots()
        
        # Prepare response similar to upload but without merge statistics
        response = {
            "success": True,
            "source": "database",
            "metrics": analyzer.metrics,
            "plots": plot_files,
            "total_trades": len(analyzer.data),
            "date_range": f"{analyzer.data['OpenDate'].min().strftime('%Y-%m-%d')} to {analyzer.data['FinalTradeClosedDate'].max().strftime('%Y-%m-%d')}",
            "total_pnl": float(analyzer.data['TotalNetProfitLoss'].sum())
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting database analysis", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving database analysis: {str(e)}")

@app.get("/analyze")
async def analyze_fixed_csv():
    """Automatically load and analyze the fixed CSV file"""
    fixed_csv_path = "tradeSteward-performanceLogs-1753563010_fixed.csv"
    
    # Check if the fixed CSV file exists
    if not os.path.exists(fixed_csv_path):
        raise HTTPException(status_code=404, detail=f"Fixed CSV file not found: {fixed_csv_path}")
    
    try:
        # Read the fixed CSV file
        with open(fixed_csv_path, 'r', encoding='utf-8') as file:
            csv_content = file.read()
        
        # Analyze data
        result = analyzer.load_data(csv_content)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=f"Error processing file: {result['error']}")
        
        # Generate plots
        plot_files = analyzer.generate_plots()
        
        # Prepare response
        response = {
            "success": True,
            "filename": fixed_csv_path,
            "summary": result,
            "metrics": analyzer.metrics,
            "plots": plot_files
        }
        
        return response
        
    except Exception as e:
        logger.error("Error processing fixed CSV file", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/data/summary")
async def get_data_summary():
    """Get summary of loaded data"""
    if analyzer.data is None:
        raise HTTPException(status_code=404, detail="No data loaded")
    
    return {
        "total_trades": len(analyzer.data),
        "strategies": analyzer.data['Strategy'].value_counts().to_dict(),
        "date_range": {
            "start": analyzer.data['OpenDate'].min().strftime('%Y-%m-%d'),
            "end": analyzer.data['OpenDate'].max().strftime('%Y-%m-%d')
        },
        "columns": list(analyzer.data.columns)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)