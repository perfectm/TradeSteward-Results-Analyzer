from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Depends, status, Form
from fastapi.responses import HTMLResponse, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Optional
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
from passlib.context import CryptContext
from jose import JWTError, jwt
import hashlib
import secrets

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Authentication configuration
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer(auto_error=False)

# User models
class User:
    def __init__(self, user_id: int, username: str, email: str, hashed_password: str, created_at: str):
        self.user_id = user_id
        self.username = username
        self.email = email
        self.hashed_password = hashed_password
        self.created_at = created_at

class UserCreate:
    def __init__(self, username: str, email: str, password: str):
        self.username = username
        self.email = email
        self.password = password

class UserLogin:
    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password

class StrategyFilterRequest(BaseModel):
    strategies: List[str]

class AccountFilterRequest(BaseModel):
    account_num: Optional[str] = None

class DateRangeRequest(BaseModel):
    start_date: str
    end_date: str

class PasswordResetRequest(BaseModel):
    username: str
    current_password: str
    new_password: str

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
    """Database manager for persistent trade storage with user authentication"""
    
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
            
            # Create users table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    email TEXT UNIQUE NOT NULL,
                    hashed_password TEXT NOT NULL,
                    initial_capital REAL DEFAULT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                )
            ''')
            
            # Add initial_capital column to existing users table if it doesn't exist
            try:
                cursor.execute('ALTER TABLE users ADD COLUMN initial_capital REAL DEFAULT NULL')
                logger.info("Added initial_capital column to users table")
            except Exception:
                # Column already exists
                pass
            
            # Create trades table (now with user_id foreign key)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    open_order_number TEXT NOT NULL,
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
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id),
                    UNIQUE(user_id, open_order_number)  -- Unique per user
                )
            ''')
            
            # Create indexes
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_users_username ON users(username)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_users_email ON users(email)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_user_order 
                ON trades(user_id, open_order_number)
            ''')
            
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_trades_dates 
                ON trades(user_id, open_date, final_trade_closed_date)
            ''')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    # User management methods
    def create_user(self, username: str, email: str, hashed_password: str) -> Optional[int]:
        """Create a new user and return user_id"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO users (username, email, hashed_password)
                    VALUES (?, ?, ?)
                ''', (username, email, hashed_password))
                user_id = cursor.lastrowid
                conn.commit()
                logger.info(f"Created new user: {username} (ID: {user_id})")
                return user_id
        except sqlite3.IntegrityError as e:
            logger.error(f"User creation failed: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, username, email, hashed_password, created_at
                FROM users WHERE username = ? AND is_active = TRUE
            ''', (username,))
            row = cursor.fetchone()
            if row:
                return User(
                    user_id=row['user_id'],
                    username=row['username'], 
                    email=row['email'],
                    hashed_password=row['hashed_password'],
                    created_at=row['created_at']
                )
            return None
    
    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by user_id"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT user_id, username, email, hashed_password, created_at
                FROM users WHERE user_id = ? AND is_active = TRUE
            ''', (user_id,))
            row = cursor.fetchone()
            if row:
                return User(
                    user_id=row['user_id'],
                    username=row['username'],
                    email=row['email'], 
                    hashed_password=row['hashed_password'],
                    created_at=row['created_at']
                )
            return None
    
    def update_user_password(self, user_id: int, new_hashed_password: str) -> bool:
        """Update user password"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users 
                    SET hashed_password = ?
                    WHERE user_id = ? AND is_active = TRUE
                ''', (new_hashed_password, user_id))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated password for user ID: {user_id}")
                    return True
                else:
                    logger.warning(f"No user found with ID: {user_id}")
                    return False
        except Exception as e:
            logger.error(f"Error updating password for user {user_id}: {e}")
            return False
    
    def set_initial_capital(self, user_id: int, initial_capital: float) -> bool:
        """Set user's initial capital for return calculations"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE users 
                    SET initial_capital = ?
                    WHERE user_id = ? AND is_active = TRUE
                ''', (initial_capital, user_id))
                conn.commit()
                
                if cursor.rowcount > 0:
                    logger.info(f"Updated initial capital to ${initial_capital:.2f} for user ID: {user_id}")
                    return True
                else:
                    logger.warning(f"No user found with ID: {user_id}")
                    return False
        except Exception as e:
            logger.error(f"Error updating initial capital for user {user_id}: {e}")
            return False
    
    def get_initial_capital(self, user_id: int) -> Optional[float]:
        """Get user's initial capital setting"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT initial_capital 
                    FROM users 
                    WHERE user_id = ? AND is_active = TRUE
                ''', (user_id,))
                
                result = cursor.fetchone()
                if result:
                    return result[0]  # May be None if not set
                return None
        except Exception as e:
            logger.error(f"Error getting initial capital for user {user_id}: {e}")
            return None
    
    def insert_trades(self, df: pd.DataFrame, user_id: int) -> Tuple[int, int, int]:
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
                
                # Check if trade already exists for this user
                cursor.execute(
                    'SELECT id, updated_at FROM trades WHERE user_id = ? AND open_order_number = ?',
                    (user_id, open_order_number)
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
                        WHERE user_id = ? AND open_order_number = ?
                    ''', (
                        trade_data['account_num'], trade_data['bot_name'], trade_data['strategy'],
                        trade_data['open_date'], trade_data['open_time'], trade_data['underlying'],
                        trade_data['underlying_open_quote'], trade_data['vix_open_quote'],
                        trade_data['final_trade_closed_date'], trade_data['final_trade_closed_time'],
                        trade_data['underlying_close_quote'], trade_data['vix_close_quote'],
                        trade_data['total_net_profit_loss'], trade_data['total_gross_profit_loss'], trade_data['total_profit_loss_percent'],
                        trade_data['raw_data'], user_id, open_order_number
                    ))
                    updated_count += 1
                else:
                    # Insert new record
                    try:
                        cursor.execute('''
                            INSERT INTO trades (
                                user_id, open_order_number, account_num, bot_name, strategy, open_date, open_time,
                                underlying, underlying_open_quote, vix_open_quote,
                                final_trade_closed_date, final_trade_closed_time,
                                underlying_close_quote, vix_close_quote,
                                total_net_profit_loss, total_gross_profit_loss, total_profit_loss_percent,
                                raw_data
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            user_id, trade_data['open_order_number'], trade_data['account_num'], trade_data['bot_name'], trade_data['strategy'],
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
    
    def get_all_trades(self, user_id: int) -> pd.DataFrame:
        """Retrieve all trades for a specific user as a pandas DataFrame"""
        with self.get_connection() as conn:
            # Get the raw data and reconstruct the DataFrame
            cursor = conn.cursor()
            cursor.execute('SELECT raw_data FROM trades WHERE user_id = ? ORDER BY open_date, open_time', (user_id,))
            
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
    
    def get_trade_count(self, user_id: int) -> int:
        """Get total number of trades for a specific user"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM trades WHERE user_id = ?', (user_id,))
            return cursor.fetchone()[0]
    
    def get_user_accounts(self, user_id: int) -> List[dict]:
        """Get all accounts for a specific user with trade counts and obfuscated display names"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # First check if account_num data exists at all for this user
            cursor.execute('''
                SELECT account_num, COUNT(*) as trade_count
                FROM trades 
                WHERE user_id = ?
                GROUP BY account_num
                ORDER BY account_num
            ''', (user_id,))
            
            all_accounts = cursor.fetchall()
            logger.info(f"All account data for user {user_id}: {[(row['account_num'], row['trade_count']) for row in all_accounts]}")
            
            # Now get only non-null accounts
            cursor.execute('''
                SELECT account_num, COUNT(*) as trade_count
                FROM trades 
                WHERE user_id = ? AND account_num IS NOT NULL AND account_num != ''
                GROUP BY account_num
                ORDER BY account_num
            ''', (user_id,))
            
            accounts = []
            for row in cursor.fetchall():
                account_num = row['account_num']
                accounts.append({
                    'account_num': account_num,  # Real account number for backend filtering
                    'account_display': obfuscate_account_number(account_num),  # Obfuscated for display
                    'trade_count': row['trade_count']
                })
            
            logger.info(f"Valid accounts for user {user_id}: {accounts}")
            return accounts
    
    def delete_all_user_trades(self, user_id: int) -> int:
        """Delete all trades for a specific user and return count of deleted trades"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # First get the count of trades to be deleted
            cursor.execute('SELECT COUNT(*) FROM trades WHERE user_id = ?', (user_id,))
            trade_count = cursor.fetchone()[0]
            
            # Delete all trades for this user
            cursor.execute('DELETE FROM trades WHERE user_id = ?', (user_id,))
            conn.commit()
            
            logger.info(f"Deleted {trade_count} trades for user {user_id}")
            return trade_count

# Utility functions
def obfuscate_account_number(account_num: str) -> str:
    """Obfuscate account number for privacy (show first 2 and last 2 digits)"""
    if not account_num or len(account_num) < 4:
        return "****"
    
    # Show first 2 and last 2 characters, mask the middle
    visible_start = account_num[:2]
    visible_end = account_num[-2:]
    masked_middle = "*" * (len(account_num) - 4)
    
    return f"{visible_start}{masked_middle}{visible_end}"

# Authentication helper functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """Verify a JWT token and return the payload"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[User]:
    """Get the current authenticated user"""
    if not credentials:
        return None
    
    payload = verify_token(credentials.credentials)
    if not payload:
        return None
    
    user_id = payload.get("sub")
    if not user_id:
        return None
    
    try:
        user_id = int(user_id)
        user = db.get_user_by_id(user_id)
        return user
    except (ValueError, TypeError):
        return None

async def require_user(current_user: Optional[User] = Depends(get_current_user)) -> User:
    """Require authentication and return the current user"""
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user

app = FastAPI(title="TradeSteward Results Analyzer", 
              description="Advanced analysis tool for TradeSteward options trading results with user authentication")

# Create directories
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads/plots", exist_ok=True)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
templates = Jinja2Templates(directory="templates")

# Global database instance
db = TradeDatabase()

class TradeStewardAnalyzer:
    """Core analyzer for TradeSteward trading data with user support"""
    
    def __init__(self, db: TradeDatabase, user_id: Optional[int] = None):
        self.db = db
        self.user_id = user_id
        self.data = None
        self.daily_pnl = None
        self.metrics = {}
        
        # Load existing data from database if user is specified
        if self.user_id:
            self._load_from_database()
        
    def _load_from_database(self):
        """Load all existing trades from database for current user"""
        try:
            if not self.user_id:
                logger.warning("No user_id specified, cannot load trades")
                return
                
            self.data = self.db.get_all_trades(self.user_id)
            if not self.data.empty:
                logger.info(f"Loaded {len(self.data)} existing trades from database for user {self.user_id}")
                self._calculate_daily_pnl()
                
                # Get user's initial capital setting
                user_initial_capital = self.db.get_initial_capital(self.user_id)
                self._calculate_metrics(user_initial_capital)
            else:
                logger.info(f"No existing trades found in database for user {self.user_id}")
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
            if not self.user_id:
                raise ValueError("User ID is required to store trades")
                
            logger.info("Storing new data in database...")
            inserted_count, updated_count, duplicate_count = self.db.insert_trades(new_df, self.user_id)
            
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
    
    def _calculate_metrics(self, user_initial_capital: Optional[float] = None):
        """Calculate comprehensive trading metrics"""
        if self.data is None or self.daily_pnl is None:
            return
        
        trades = self.data
        daily = self.daily_pnl
        
        # Basic Performance Metrics
        total_trades = len(trades)
        total_pnl = trades['TotalNetProfitLoss'].sum()
        total_gross_pnl = trades['TotalGrossProfitLoss'].sum()
        
        # Commission Analysis (Gross P&L - Net P&L = Commissions)
        total_commissions = total_gross_pnl - total_pnl if not pd.isna(total_gross_pnl) else 0
        avg_commission_per_trade = total_commissions / total_trades if total_trades > 0 else 0
        commission_percentage = (total_commissions / abs(total_gross_pnl)) * 100 if abs(total_gross_pnl) > 0 else 0
        
        # Win Rate & Trade Analysis
        winning_trades = trades[trades['TotalNetProfitLoss'] > 0]
        losing_trades = trades[trades['TotalNetProfitLoss'] < 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_winner = winning_trades['TotalNetProfitLoss'].mean() if len(winning_trades) > 0 else 0
        avg_loser = losing_trades['TotalNetProfitLoss'].mean() if len(losing_trades) > 0 else 0
        
        # Risk Metrics
        daily_returns = daily['NetPnL'].values
        # Risk-free rate of 4.3% (0.043)
        risk_free_rate = 0.043
        
        if len(daily_returns) > 1:
            volatility = np.std(daily_returns) * np.sqrt(252)  # Annualized
            annualized_return = np.mean(daily_returns) * 252
            sharpe_ratio = (annualized_return - risk_free_rate) / volatility if volatility != 0 else 0
        else:
            volatility = 0
            annualized_return = 0
            sharpe_ratio = 0
        
        # Drawdown Analysis
        cumulative = daily['CumulativePnL'].values
        rolling_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - rolling_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0
        
        # Advanced Risk Metrics
        # MAR Ratio (Mean Annual Return / Maximum Drawdown)
        if len(daily_returns) > 1 and max_drawdown < 0:
            annual_return = np.mean(daily_returns) * 252
            mar_ratio = abs(annual_return / max_drawdown)
        else:
            mar_ratio = 0
        
        # Sortino Ratio (risk-adjusted return using downside deviation)
        if len(daily_returns) > 1:
            negative_returns = daily_returns[daily_returns < 0]
            if len(negative_returns) > 0:
                downside_deviation = np.std(negative_returns) * np.sqrt(252)
                sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation if downside_deviation != 0 else 0
            else:
                sortino_ratio = float('inf') if annualized_return > risk_free_rate else 0
        else:
            sortino_ratio = 0
        
        # UPI (Ulcer Performance Index) - risk-adjusted performance using drawdown
        if len(cumulative) > 0 and len(daily_returns) > 1:
            drawdown_percentages = []
            for i in range(len(cumulative)):
                if rolling_max[i] != 0:
                    # Calculate drawdown as percentage
                    dd_pct = abs(drawdown[i] / rolling_max[i]) * 100
                    drawdown_percentages.append(dd_pct ** 2)
                else:
                    drawdown_percentages.append(0)
            
            # Calculate Ulcer Index (as percentage)
            ulcer_index = np.sqrt(np.mean(drawdown_percentages)) if len(drawdown_percentages) > 0 else 0
            
            # Convert annualized return to percentage for UPI calculation
            # Use user-provided initial capital if available, otherwise use proxy
            if user_initial_capital and user_initial_capital > 0:
                # Use user-provided initial capital
                initial_capital_proxy = user_initial_capital
                annualized_return_pct = (annualized_return / initial_capital_proxy) * 100
                capital_source = "user-provided"
            elif len(cumulative) > 0 and rolling_max[0] != 0:
                # Use the first peak as a proxy for initial capital
                initial_capital_proxy = rolling_max[0]
                annualized_return_pct = (annualized_return / initial_capital_proxy) * 100
                capital_source = "first peak proxy"
            else:
                # Fallback: assume $10,000 initial capital
                initial_capital_proxy = 10000
                annualized_return_pct = (annualized_return / 10000) * 100
                capital_source = "$10,000 fallback"
            
            risk_free_rate_pct = risk_free_rate * 100  # Convert to percentage
            
            # UPI = (Return% - Risk-Free Rate%) / Ulcer Index%
            if ulcer_index == 0:
                if annualized_return_pct > risk_free_rate_pct:
                    upi = 999.99  # Very high UPI for no-drawdown profitable strategy
                else:
                    upi = 0
            else:
                upi = (annualized_return_pct - risk_free_rate_pct) / ulcer_index
            
            # Debug logging
            logger.info(f"UPI Calculation - Annualized Return: ${annualized_return:.2f} ({annualized_return_pct:.2f}%), Risk-Free Rate: {risk_free_rate_pct:.1f}%, Ulcer Index: {ulcer_index:.4f}%, UPI: {upi:.4f}")
            logger.info(f"Initial capital: ${initial_capital_proxy:.2f} ({capital_source})")
        else:
            ulcer_index = 0
            upi = 0
            logger.info(f"UPI not calculated - Cumulative length: {len(cumulative)}, Daily returns length: {len(daily_returns)}")
        
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
        
        # Strategy Breakdown with Commission Analysis
        strategy_performance = trades.groupby('Strategy').agg({
            'TotalNetProfitLoss': ['sum', 'mean', 'count'],
            'TotalGrossProfitLoss': ['sum', 'mean'],
            'TotalProfitLossPercent': 'mean'
        }).round(2)
        
        # Convert strategy performance to JSON-serializable format
        strategy_dict = {}
        for strategy in strategy_performance.index:
            net_sum = float(strategy_performance.loc[strategy, ('TotalNetProfitLoss', 'sum')])
            gross_sum = float(strategy_performance.loc[strategy, ('TotalGrossProfitLoss', 'sum')])
            trade_count = int(strategy_performance.loc[strategy, ('TotalNetProfitLoss', 'count')])
            commissions = gross_sum - net_sum
            
            strategy_dict[strategy] = {
                'total_pnl': net_sum,
                'total_gross_pnl': gross_sum,
                'total_commissions': commissions,
                'avg_pnl': float(strategy_performance.loc[strategy, ('TotalNetProfitLoss', 'mean')]),
                'avg_commission': commissions / trade_count if trade_count > 0 else 0,
                'trade_count': trade_count,
                'avg_percentage': float(strategy_performance.loc[strategy, ('TotalProfitLossPercent', 'mean')])
            }
        
        # Monthly Commission Analysis
        trades['YearMonth'] = trades['OpenDate'].dt.to_period('M')
        monthly_commissions = trades.groupby('YearMonth').agg({
            'TotalNetProfitLoss': 'sum',
            'TotalGrossProfitLoss': 'sum'
        })
        monthly_commissions['Commissions'] = monthly_commissions['TotalGrossProfitLoss'] - monthly_commissions['TotalNetProfitLoss']
        monthly_commission_dict = {
            str(period): float(commission) 
            for period, commission in monthly_commissions['Commissions'].items()
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
            'total_commissions': float(round(total_commissions, 2)),
            'avg_commission_per_trade': float(round(avg_commission_per_trade, 2)),
            'commission_percentage': float(round(commission_percentage, 2)),
            'win_rate': float(round(win_rate, 2)),
            'avg_winner': float(round(avg_winner, 2)),
            'avg_loser': float(round(avg_loser, 2)),
            'profit_factor': float(round(abs(avg_winner * len(winning_trades) / (avg_loser * len(losing_trades))), 2)) if len(losing_trades) > 0 and avg_loser != 0 else float('inf'),
            'max_drawdown': float(round(max_drawdown, 2)),
            'volatility': float(round(volatility, 2)),
            'sharpe_ratio': float(round(sharpe_ratio, 2)),
            'mar_ratio': float(round(mar_ratio, 2)),
            'sortino_ratio': float(round(sortino_ratio, 2)),
            'risk_free_rate': float(risk_free_rate),
            'ulcer_index': float(round(ulcer_index, 2)),
            'upi': float(round(upi, 2)),
            'trading_days': int(trading_days),
            'total_days': int(total_days),
            'avg_daily_pnl': float(round(avg_daily_pnl, 2)),
            'strategy_performance': strategy_dict,
            'monthly_commissions': monthly_commission_dict,
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
        
        # 4. Commission Analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Commission per trade over time
        self.data['Commission'] = self.data['TotalGrossProfitLoss'] - self.data['TotalNetProfitLoss']
        commission_by_date = self.data.groupby(self.data['FinalTradeClosedDate'].dt.date)['Commission'].sum().reset_index()
        commission_by_date['Date'] = pd.to_datetime(commission_by_date['FinalTradeClosedDate'])
        
        ax1.plot(commission_by_date['Date'], commission_by_date['Commission'], 
                linewidth=2, color='red', marker='o', markersize=3)
        ax1.fill_between(commission_by_date['Date'], commission_by_date['Commission'], 
                        alpha=0.3, color='red')
        ax1.set_title('Daily Commission Costs', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Daily Commissions ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Commission by strategy
        strategy_commissions = self.data.groupby('Strategy')['Commission'].agg(['sum', 'mean']).reset_index()
        strategy_commissions.columns = ['Strategy', 'TotalCommission', 'AvgCommission']
        
        bars = ax2.bar(range(len(strategy_commissions)), strategy_commissions['TotalCommission'], 
                      color='red', alpha=0.7)
        ax2.set_title('Total Commissions by Strategy', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Total Commissions ($)', fontsize=12)
        ax2.set_xticks(range(len(strategy_commissions)))
        ax2.set_xticklabels(strategy_commissions['Strategy'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, strategy_commissions['TotalCommission']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${value:.0f}', ha='center', va='bottom', fontsize=10)
        
        # Monthly commission trend
        monthly_commissions = self.data.groupby('YearMonth')['Commission'].sum()
        ax3.bar(range(len(monthly_commissions)), monthly_commissions.values, 
               color='darkred', alpha=0.7)
        ax3.set_title('Monthly Commission Costs', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Monthly Commissions ($)', fontsize=12)
        ax3.set_xticks(range(len(monthly_commissions)))
        ax3.set_xticklabels([str(m) for m in monthly_commissions.index], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Commission vs P&L scatter
        ax4.scatter(self.data['TotalNetProfitLoss'], self.data['Commission'], 
                   alpha=0.6, color='purple', s=20)
        ax4.set_title('Commission vs Trade P&L', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Trade P&L ($)', fontsize=12)
        ax4.set_ylabel('Commission ($)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add trend line
        if len(self.data) > 1:
            z = np.polyfit(self.data['TotalNetProfitLoss'], self.data['Commission'], 1)
            p = np.poly1d(z)
            ax4.plot(sorted(self.data['TotalNetProfitLoss']), 
                    p(sorted(self.data['TotalNetProfitLoss'])), 
                    "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        filename = f"plots/commission_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f"uploads/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(filename)
        
        return plot_files
    
    def generate_interactive_chart_data(self) -> dict:
        """Generate data for interactive charts using Plotly.js"""
        if self.data is None or self.daily_pnl is None:
            return {}
        
        chart_data = {}
        
        try:
            # 1. Cumulative P&L Chart Data
            daily_pnl_clean = self.daily_pnl.fillna(0)  # Replace NaN with 0
            chart_data['cumulative_pnl'] = {
                'x': daily_pnl_clean['Date'].dt.strftime('%Y-%m-%d').tolist(),
                'y': daily_pnl_clean['CumulativePnL'].tolist(),
                'daily_pnl': daily_pnl_clean['NetPnL'].tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Cumulative P&L',
                'hovertemplate': '<b>Date:</b> %{x}<br><b>Cumulative P&L:</b> $%{y:,.2f}<br><b>Daily P&L:</b> $%{customdata:,.2f}<extra></extra>',
                'customdata': daily_pnl_clean['NetPnL'].tolist()
            }
            
            # 2. Strategy Performance Data
            strategy_pnl = self.data.groupby('Strategy')['TotalNetProfitLoss'].agg(['sum', 'count']).reset_index()
            strategy_pnl.columns = ['Strategy', 'TotalPnL', 'TradeCount']
            strategy_pnl = strategy_pnl.fillna(0).sort_values('TotalPnL', ascending=False)
            
            chart_data['strategy_performance'] = {
                'x': strategy_pnl['Strategy'].tolist(),
                'y': strategy_pnl['TotalPnL'].tolist(),
                'trade_counts': strategy_pnl['TradeCount'].tolist(),
                'type': 'bar',
                'name': 'Strategy P&L',
                'hovertemplate': '<b>Strategy:</b> %{x}<br><b>Total P&L:</b> $%{y:,.2f}<br><b>Trade Count:</b> %{customdata}<extra></extra>',
                'customdata': strategy_pnl['TradeCount'].tolist()
            }
            
            # 3. P&L Distribution (Histogram)  
            pnl_values = self.data['TotalNetProfitLoss'].fillna(0).tolist()
            chart_data['pnl_distribution'] = {
                'x': pnl_values,
                'type': 'histogram',
                'nbinsx': 30,
                'name': 'P&L Distribution',
                'hovertemplate': '<b>P&L Range:</b> $%{x:,.2f}<br><b>Count:</b> %{y}<extra></extra>'
            }
            
            # 4. VIX vs P&L Scatter Plot
            if 'VIXOpenQuote' in self.data.columns and 'VIXCloseQuote' in self.data.columns:
                # Clean VIX data
                data_clean = self.data.copy()
                data_clean['VIXOpenQuote'] = data_clean['VIXOpenQuote'].fillna(0)
                data_clean['VIXCloseQuote'] = data_clean['VIXCloseQuote'].fillna(0)
                data_clean['VIXAvg'] = (data_clean['VIXOpenQuote'] + data_clean['VIXCloseQuote']) / 2
                data_clean['TotalNetProfitLoss'] = data_clean['TotalNetProfitLoss'].fillna(0)
                
                # Group by strategy for color coding
                strategies = data_clean['Strategy'].unique()
                vix_scatter_data = []
                
                for i, strategy in enumerate(strategies):
                    strategy_data = data_clean[data_clean['Strategy'] == strategy]
                    vix_scatter_data.append({
                        'x': strategy_data['VIXAvg'].fillna(0).tolist(),
                        'y': strategy_data['TotalNetProfitLoss'].fillna(0).tolist(),
                        'dates': strategy_data['OpenDate'].dt.strftime('%Y-%m-%d').tolist(),
                        'type': 'scatter',
                        'mode': 'markers',
                        'name': strategy,
                        'hovertemplate': f'<b>Strategy:</b> {strategy}<br><b>VIX:</b> %{{x:.2f}}<br><b>P&L:</b> $%{{y:,.2f}}<br><b>Date:</b> %{{customdata}}<extra></extra>',
                        'customdata': strategy_data['OpenDate'].dt.strftime('%Y-%m-%d').tolist()
                    })
                
                chart_data['vix_scatter'] = vix_scatter_data
            
            # 5. Commission Analysis
            commission_data = self.data.copy()
            commission_data['TotalGrossProfitLoss'] = commission_data['TotalGrossProfitLoss'].fillna(0)
            commission_data['TotalNetProfitLoss'] = commission_data['TotalNetProfitLoss'].fillna(0)
            commission_data['Commission'] = commission_data['TotalGrossProfitLoss'] - commission_data['TotalNetProfitLoss']
            commission_by_date = commission_data.groupby(commission_data['FinalTradeClosedDate'].dt.date)['Commission'].sum().reset_index()
            commission_by_date.columns = ['Date', 'Commission']
            commission_by_date = commission_by_date.fillna(0)
            
            chart_data['commission_analysis'] = {
                'x': commission_by_date['Date'].astype(str).tolist(),
                'y': commission_by_date['Commission'].tolist(),
                'type': 'scatter',
                'mode': 'lines+markers',
                'name': 'Daily Commission',
                'hovertemplate': '<b>Date:</b> %{x}<br><b>Commission:</b> $%{y:,.2f}<extra></extra>'
            }
            
        except Exception as e:
            logger.error(f"Error generating interactive chart data: {e}", exc_info=True)
            return {}
        
        return chart_data
    
    def generate_commission_chart(self, account_filter: Optional[str] = None) -> str:
        """Generate commission analysis chart for specific account or all accounts"""
        if self.data is None:
            raise ValueError("No data available for chart generation")
        
        # Filter data by account if specified
        data_to_use = self.data.copy()
        if account_filter:
            # Check the actual column name in the data
            account_col = 'AccountNum' if 'AccountNum' in data_to_use.columns else 'account_num'
            data_to_use = data_to_use[data_to_use[account_col] == account_filter]
            if data_to_use.empty:
                raise ValueError(f"No data found for account {account_filter}")
        
        # Calculate commission for the filtered data
        data_to_use['Commission'] = data_to_use['TotalGrossProfitLoss'] - data_to_use['TotalNetProfitLoss']
        
        # 4-panel commission analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Commission per trade over time
        commission_by_date = data_to_use.groupby(data_to_use['FinalTradeClosedDate'].dt.date)['Commission'].sum().reset_index()
        commission_by_date['Date'] = pd.to_datetime(commission_by_date['FinalTradeClosedDate'])
        
        ax1.plot(commission_by_date['Date'], commission_by_date['Commission'], 
                linewidth=2, color='red', marker='o', markersize=3)
        ax1.fill_between(commission_by_date['Date'], commission_by_date['Commission'], 
                        alpha=0.3, color='red')
        
        title_suffix = f" - Account {obfuscate_account_number(account_filter)}" if account_filter else " - All Accounts"
        ax1.set_title(f'Daily Commission Costs{title_suffix}', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Daily Commissions ($)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Commission by strategy
        strategy_commissions = data_to_use.groupby('Strategy')['Commission'].agg(['sum', 'mean']).reset_index()
        strategy_commissions.columns = ['Strategy', 'TotalCommission', 'AvgCommission']
        
        bars = ax2.bar(range(len(strategy_commissions)), strategy_commissions['TotalCommission'], 
                      color='red', alpha=0.7)
        ax2.set_title(f'Total Commissions by Strategy{title_suffix}', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Total Commissions ($)', fontsize=12)
        ax2.set_xticks(range(len(strategy_commissions)))
        ax2.set_xticklabels(strategy_commissions['Strategy'], rotation=45, ha='right')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, strategy_commissions['TotalCommission']):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                    f'${value:.0f}', ha='center', va='bottom', fontsize=10)
        
        # Monthly commission trend
        data_to_use['YearMonth'] = data_to_use['OpenDate'].dt.to_period('M')
        monthly_commissions = data_to_use.groupby('YearMonth')['Commission'].sum()
        ax3.bar(range(len(monthly_commissions)), monthly_commissions.values, 
               color='darkred', alpha=0.7)
        ax3.set_title(f'Monthly Commission Costs{title_suffix}', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Monthly Commissions ($)', fontsize=12)
        ax3.set_xticks(range(len(monthly_commissions)))
        ax3.set_xticklabels([str(m) for m in monthly_commissions.index], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Commission vs P&L scatter
        ax4.scatter(data_to_use['TotalNetProfitLoss'], data_to_use['Commission'], 
                   alpha=0.6, color='purple', s=20)
        ax4.set_title(f'Commission vs Trade P&L{title_suffix}', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Trade P&L ($)', fontsize=12)
        ax4.set_ylabel('Commission ($)', fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # Add trend line
        if len(data_to_use) > 1:
            z = np.polyfit(data_to_use['TotalNetProfitLoss'], data_to_use['Commission'], 1)
            p = np.poly1d(z)
            ax4.plot(sorted(data_to_use['TotalNetProfitLoss']), 
                    p(sorted(data_to_use['TotalNetProfitLoss'])), 
                    "r--", alpha=0.8, linewidth=2)
        
        plt.tight_layout()
        
        # Generate filename
        account_suffix = f"_account_{account_filter}" if account_filter else "_all_accounts"
        filename = f"plots/commission_analysis{account_suffix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(f"uploads/{filename}", dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
    
    def calculate_daily_metrics(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> dict:
        """Calculate comprehensive daily performance metrics"""
        if self.data is None or self.daily_pnl is None:
            return {}
        
        # Filter daily P&L by date range if specified
        daily_data = self.daily_pnl.copy()
        if start_date:
            daily_data = daily_data[daily_data['Date'] >= pd.to_datetime(start_date)]
        if end_date:
            daily_data = daily_data[daily_data['Date'] <= pd.to_datetime(end_date)]
        
        if daily_data.empty:
            return {}
        
        # Calculate daily metrics
        total_trading_days = len(daily_data)
        avg_daily_pnl = daily_data['NetPnL'].mean()
        best_day = daily_data['NetPnL'].max()
        worst_day = daily_data['NetPnL'].min()
        positive_days = (daily_data['NetPnL'] > 0).sum()
        positive_days_pct = (positive_days / total_trading_days * 100) if total_trading_days > 0 else 0
        
        # Calculate average trades per day
        trades_data = self.data.copy()
        if start_date:
            trades_data = trades_data[trades_data['OpenDate'] >= pd.to_datetime(start_date)]
        if end_date:
            trades_data = trades_data[trades_data['OpenDate'] <= pd.to_datetime(end_date)]
        
        total_trades = len(trades_data)
        avg_trades_per_day = total_trades / total_trading_days if total_trading_days > 0 else 0
        
        return {
            'total_trading_days': total_trading_days,
            'avg_daily_pnl': avg_daily_pnl,
            'best_day': best_day,
            'worst_day': worst_day,
            'positive_days_pct': round(positive_days_pct, 1),
            'avg_trades_per_day': avg_trades_per_day
        }
    
    def get_daily_performance_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[dict]:
        """Get day-by-day performance data for table display"""
        if self.data is None or self.daily_pnl is None:
            return []
        
        # Filter data by date range if specified
        daily_data = self.daily_pnl.copy()
        trades_data = self.data.copy()
        
        if start_date:
            daily_data = daily_data[daily_data['Date'] >= pd.to_datetime(start_date)]
            trades_data = trades_data[trades_data['OpenDate'] >= pd.to_datetime(start_date)]
        if end_date:
            daily_data = daily_data[daily_data['Date'] <= pd.to_datetime(end_date)]
            trades_data = trades_data[trades_data['OpenDate'] <= pd.to_datetime(end_date)]
        
        if daily_data.empty:
            return []
        
        # Group trades by date to get detailed daily information
        trades_by_date = trades_data.groupby(trades_data['OpenDate'].dt.date)
        
        daily_performance = []
        for _, row in daily_data.iterrows():
            date = row['Date'].date()
            date_str = date.strftime('%Y-%m-%d')
            
            # Get trades for this date
            if date in trades_by_date.groups:
                day_trades = trades_by_date.get_group(date)
                trade_count = len(day_trades)
                winning_trades = (day_trades['TotalNetProfitLoss'] > 0).sum()
                win_rate = round((winning_trades / trade_count * 100) if trade_count > 0 else 0, 1)
                strategies = day_trades['Strategy'].unique().tolist()
            else:
                trade_count = 0
                win_rate = 0
                strategies = []
            
            daily_performance.append({
                'date': date_str,
                'trade_count': trade_count,
                'daily_pnl': round(row['NetPnL'], 2),
                'cumulative_pnl': round(row['CumulativePnL'], 2),
                'win_rate': win_rate,
                'strategies': strategies
            })
        
        # Sort by date descending (most recent first)
        daily_performance.sort(key=lambda x: x['date'], reverse=True)
        
        return daily_performance

# Authentication endpoints
@app.post("/register")
async def register(username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    """Register a new user"""
    if len(password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long")
    
    hashed_password = get_password_hash(password)
    user_id = db.create_user(username, email, hashed_password)
    
    if not user_id:
        raise HTTPException(status_code=400, detail="Username or email already exists")
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user_id)}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user_id,
        "username": username
    }

@app.post("/login")
async def login(username: str = Form(...), password: str = Form(...)):
    """Login user"""
    user = db.get_user_by_username(username)
    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": str(user.user_id)}, expires_delta=access_token_expires
    )
    
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user_id": user.user_id,
        "username": user.username
    }

@app.get("/me")
async def get_current_user_info(current_user: User = Depends(require_user)):
    """Get current user information"""
    return {
        "user_id": current_user.user_id,
        "username": current_user.username,
        "email": current_user.email,
        "created_at": current_user.created_at
    }

@app.post("/reset-password")
async def reset_password(
    username: str = Form(...),
    current_password: str = Form(...), 
    new_password: str = Form(...),
    current_user: User = Depends(require_user)
):
    """Reset user password"""
    try:
        # Verify the username matches the authenticated user
        if username != current_user.username:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You can only reset your own password"
            )
        
        # Verify current password
        if not verify_password(current_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )
        
        # Validate new password
        if len(new_password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 6 characters long"
            )
        
        # Check if new password is different from current password
        if verify_password(new_password, current_user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be different from current password"
            )
        
        # Hash new password and update in database
        new_hashed_password = get_password_hash(new_password)
        success = db.update_user_password(current_user.user_id, new_hashed_password)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )
        
        logger.info(f"Password reset successful for user: {username}")
        
        return {
            "success": True,
            "message": "Password reset successfully",
            "username": username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during password reset for {username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during password reset"
        )

@app.post("/forgot-password")
async def forgot_password(username: str = Form(...)):
    """Initiate forgot password process for unauthenticated users"""
    try:
        # Check if user exists
        user = db.get_user_by_username(username)
        if not user:
            # For security, don't reveal if username exists or not
            return {
                "success": True,
                "message": "If the username exists, password reset has been initiated"
            }
        
        logger.info(f"Forgot password initiated for user: {username}")
        
        # In a real application, you would:
        # 1. Generate a secure reset token
        # 2. Store it in the database with expiration
        # 3. Send it via email
        # For this demo, we'll allow immediate password reset
        
        return {
            "success": True,
            "message": "Password reset initiated",
            "username": username,
            "allow_reset": True  # This would normally be sent via email
        }
        
    except Exception as e:
        logger.error(f"Error during forgot password for {username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during password reset initiation"
        )

@app.post("/update-forgotten-password")
async def update_forgotten_password(
    username: str = Form(...),
    new_password: str = Form(...)
):
    """Update password for users who forgot their password (no authentication required)"""
    try:
        # Validate new password
        if len(new_password) < 6:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Password must be at least 6 characters long"
            )
        
        # Check if user exists
        user = db.get_user_by_username(username)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Hash new password and update in database
        new_hashed_password = get_password_hash(new_password)
        success = db.update_user_password(user.user_id, new_hashed_password)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update password"
            )
        
        logger.info(f"Forgotten password reset successful for user: {username}")
        
        return {
            "success": True,
            "message": "Password updated successfully",
            "username": username
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during forgotten password update for {username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during password update"
        )

@app.post("/set-initial-capital")
async def set_initial_capital(
    initial_capital: float = Form(...),
    current_user: User = Depends(require_user)
):
    """Set user's initial capital for return calculations"""
    try:
        # Validate initial capital
        if initial_capital < 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Initial capital must be at least $100"
            )
        
        if initial_capital > 1000000000:  # 1 billion limit
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Initial capital cannot exceed $1 billion"
            )
        
        # Update in database
        success = db.set_initial_capital(current_user.user_id, initial_capital)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update initial capital"
            )
        
        logger.info(f"Initial capital set to ${initial_capital:.2f} for user: {current_user.username}")
        
        return {
            "success": True,
            "message": f"Initial capital updated to ${initial_capital:,.2f}",
            "initial_capital": initial_capital
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error setting initial capital for {current_user.username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while updating initial capital"
        )

@app.get("/get-initial-capital")
async def get_initial_capital(current_user: User = Depends(require_user)):
    """Get user's initial capital setting"""
    try:
        initial_capital = db.get_initial_capital(current_user.user_id)
        
        return {
            "success": True,
            "initial_capital": initial_capital
        }
        
    except Exception as e:
        logger.error(f"Error getting initial capital for {current_user.username}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while retrieving initial capital"
        )

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Main upload page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: User = Depends(require_user)):
    """Process uploaded TradeSteward CSV file"""
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Please upload a CSV file")
    
    try:
        # Read file content
        content = await file.read()
        raw_csv_content = content.decode('utf-8')
        
        # Clean the CSV content using the same logic as previous cleaning steps
        csv_content = clean_csv_content(raw_csv_content)
        
        # Create user-specific analyzer
        analyzer = TradeStewardAnalyzer(db, current_user.user_id)
        
        # Analyze data
        result = analyzer.load_data(csv_content)
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=f"Error processing file: {result['error']}")
        
        # Generate plots
        plot_files = analyzer.generate_plots()
        
        # Generate interactive chart data
        try:
            interactive_data = analyzer.generate_interactive_chart_data()
        except Exception as e:
            logger.error(f"Error generating interactive chart data: {e}", exc_info=True)
            interactive_data = {}
        
        # Prepare response
        response = {
            "success": True,
            "filename": file.filename,
            "summary": result,
            "metrics": analyzer.metrics,
            "plots": plot_files,
            "interactive_charts": interactive_data,
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
async def get_metrics(current_user: User = Depends(require_user)):
    """Get current analysis metrics"""
    analyzer = TradeStewardAnalyzer(db, current_user.user_id)
    
    if analyzer.data is None or analyzer.data.empty:
        raise HTTPException(status_code=404, detail="No data loaded. Please upload a file first.")
    
    return analyzer.metrics

@app.get("/database-analysis")
async def get_database_analysis(current_user: User = Depends(require_user)):
    """Get complete analysis of all data in the database for the current user"""
    try:
        # Check if there's data in the database for this user
        total_trades = db.get_trade_count(current_user.user_id)
        if total_trades == 0:
            raise HTTPException(status_code=404, detail="No trading data found in database. Please upload a CSV file first.")
        
        # Create user-specific analyzer and load data
        analyzer = TradeStewardAnalyzer(db, current_user.user_id)
        
        if analyzer.data is None or analyzer.data.empty:
            raise HTTPException(status_code=404, detail="No valid trading data found in database.")
        
        # Generate plots
        plot_files = analyzer.generate_plots()
        
        # Generate interactive chart data
        try:
            interactive_data = analyzer.generate_interactive_chart_data()
        except Exception as e:
            logger.error(f"Error generating interactive chart data: {e}", exc_info=True)
            interactive_data = {}
        
        # Get available accounts for filtering
        available_accounts = db.get_user_accounts(current_user.user_id)
        logger.info(f"Available accounts for user {current_user.username} (ID: {current_user.user_id}): {available_accounts}")
        
        # Prepare response similar to upload but without merge statistics
        response = {
            "success": True,
            "source": "database",
            "metrics": analyzer.metrics,
            "plots": plot_files,
            "interactive_charts": interactive_data,
            "total_trades": len(analyzer.data),
            "available_accounts": available_accounts,
            "date_range": f"{analyzer.data['OpenDate'].min().strftime('%Y-%m-%d')} to {analyzer.data['FinalTradeClosedDate'].max().strftime('%Y-%m-%d')}",
            "total_pnl": float(analyzer.data['TotalNetProfitLoss'].sum())
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting database analysis", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving database analysis: {str(e)}")

@app.post("/database-analysis-filtered")
async def get_filtered_database_analysis(
    request: StrategyFilterRequest,
    current_user: User = Depends(require_user)
):
    """Get analysis of database filtered by selected strategies"""
    try:
        strategies = request.strategies
        if not strategies:
            raise HTTPException(status_code=400, detail="No strategies specified for filtering")
        
        # Check if there's data in the database for this user
        total_trades = db.get_trade_count(current_user.user_id)
        if total_trades == 0:
            raise HTTPException(status_code=404, detail="No trading data found in database. Please upload a CSV file first.")
        
        # Create user-specific analyzer and load data
        analyzer = TradeStewardAnalyzer(db, current_user.user_id)
        
        if analyzer.data is None or analyzer.data.empty:
            raise HTTPException(status_code=404, detail="No valid trading data found in database.")
        
        # Filter data by selected strategies
        original_data = analyzer.data.copy()
        filtered_data = analyzer.data[analyzer.data['Strategy'].isin(strategies)]
        
        if filtered_data.empty:
            raise HTTPException(status_code=404, detail="No trades found for selected strategies.")
        
        # Update analyzer with filtered data
        analyzer.data = filtered_data
        analyzer._calculate_daily_pnl()
        
        # Get user's initial capital setting for accurate calculations
        user_initial_capital = db.get_initial_capital(current_user.user_id)
        analyzer._calculate_metrics(user_initial_capital)
        
        # Generate plots for filtered data
        plot_files = analyzer.generate_plots()
        
        # Generate interactive chart data
        try:
            interactive_data = analyzer.generate_interactive_chart_data()
        except Exception as e:
            logger.error(f"Error generating interactive chart data: {e}", exc_info=True)
            interactive_data = {}
        
        # Prepare response
        response = {
            "success": True,
            "source": "database_filtered",
            "metrics": analyzer.metrics,
            "plots": plot_files,
            "interactive_charts": interactive_data,
            "total_trades": len(filtered_data),
            "filtered_strategies": strategies,
            "original_trade_count": len(original_data),
            "date_range": f"{filtered_data['OpenDate'].min().strftime('%Y-%m-%d')} to {filtered_data['FinalTradeClosedDate'].max().strftime('%Y-%m-%d')}",
            "total_pnl": float(filtered_data['TotalNetProfitLoss'].sum())
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting filtered database analysis", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving filtered analysis: {str(e)}")

@app.post("/commission-analysis-by-account")
async def get_commission_analysis_by_account(
    request: AccountFilterRequest,
    current_user: User = Depends(require_user)
):
    """Get commission analysis filtered by account"""
    try:
        # Check if there's data in the database for this user
        total_trades = db.get_trade_count(current_user.user_id)
        if total_trades == 0:
            raise HTTPException(status_code=404, detail="No trading data found in database. Please upload a CSV file first.")
        
        # Create user-specific analyzer and load data
        analyzer = TradeStewardAnalyzer(db, current_user.user_id)
        
        if analyzer.data is None or analyzer.data.empty:
            raise HTTPException(status_code=404, detail="No valid trading data found in database.")
        
        # Generate commission chart for the specified account
        try:
            commission_plot = analyzer.generate_commission_chart(request.account_num)
            
            response = {
                "success": True,
                "commission_plot": commission_plot,
                "account_filter": request.account_num,
                "filtered": request.account_num is not None
            }
            
            return response
            
        except ValueError as ve:
            raise HTTPException(status_code=404, detail=str(ve))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting commission analysis by account", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving commission analysis: {str(e)}")

@app.delete("/delete-all-trades")
async def delete_all_user_trades(current_user: User = Depends(require_user)):
    """Delete all trades for the current user"""
    try:
        # Get current trade count for logging
        current_count = db.get_trade_count(current_user.user_id)
        
        if current_count == 0:
            return {
                "success": True,
                "deleted_count": 0,
                "message": "No trades found to delete"
            }
        
        # Delete all trades for this user
        deleted_count = db.delete_all_user_trades(current_user.user_id)
        
        logger.info(f"User {current_user.username} (ID: {current_user.user_id}) deleted all {deleted_count} trades")
        
        return {
            "success": True,
            "deleted_count": deleted_count,
            "message": f"Successfully deleted {deleted_count} trades"
        }
        
    except Exception as e:
        logger.error(f"Error deleting all trades for user {current_user.user_id}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error deleting trades: {str(e)}")

@app.get("/data/summary")
async def get_data_summary(current_user: User = Depends(require_user)):
    """Get summary of loaded data for current user"""
    analyzer = TradeStewardAnalyzer(db, current_user.user_id)
    
    if analyzer.data is None or analyzer.data.empty:
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

@app.get("/daily-analysis")
async def get_daily_analysis(current_user: User = Depends(require_user)):
    """Get daily performance analysis for the current user"""
    try:
        # Check if there's data in the database for this user
        total_trades = db.get_trade_count(current_user.user_id)
        if total_trades == 0:
            raise HTTPException(status_code=404, detail="No trading data found in database. Please upload a CSV file first.")
        
        # Create user-specific analyzer and load data
        analyzer = TradeStewardAnalyzer(db, current_user.user_id)
        
        if analyzer.data is None or analyzer.data.empty:
            raise HTTPException(status_code=404, detail="No valid trading data found in database.")
        
        # Calculate daily metrics
        daily_metrics = analyzer.calculate_daily_metrics()
        
        # Get daily performance data
        daily_data = analyzer.get_daily_performance_data()
        
        # Get date range
        start_date = analyzer.data['OpenDate'].min().strftime('%Y-%m-%d')
        end_date = analyzer.data['OpenDate'].max().strftime('%Y-%m-%d')
        
        response = {
            "success": True,
            "daily_metrics": daily_metrics,
            "daily_data": daily_data,
            "date_range": {
                "start": start_date,
                "end": end_date
            },
            "total_days": len(daily_data)
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting daily analysis", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving daily analysis: {str(e)}")

@app.post("/daily-analysis-filtered")
async def get_filtered_daily_analysis(
    request: DateRangeRequest,
    current_user: User = Depends(require_user)
):
    """Get daily performance analysis filtered by date range"""
    try:
        # Check if there's data in the database for this user
        total_trades = db.get_trade_count(current_user.user_id)
        if total_trades == 0:
            raise HTTPException(status_code=404, detail="No trading data found in database. Please upload a CSV file first.")
        
        # Create user-specific analyzer and load data
        analyzer = TradeStewardAnalyzer(db, current_user.user_id)
        
        if analyzer.data is None or analyzer.data.empty:
            raise HTTPException(status_code=404, detail="No valid trading data found in database.")
        
        # Calculate daily metrics for the specified date range
        daily_metrics = analyzer.calculate_daily_metrics(request.start_date, request.end_date)
        
        # Get daily performance data for the specified date range
        daily_data = analyzer.get_daily_performance_data(request.start_date, request.end_date)
        
        response = {
            "success": True,
            "daily_metrics": daily_metrics,
            "daily_data": daily_data,
            "date_range": {
                "start": request.start_date,
                "end": request.end_date
            },
            "total_days": len(daily_data)
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting filtered daily analysis", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving filtered daily analysis: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)