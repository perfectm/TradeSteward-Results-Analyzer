"""
Live Market Data Integration - Phase 3 Implementation
Real-time market data feeds for enhanced risk monitoring and analysis
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import time
import warnings
from abc import ABC, abstractmethod
import websocket
import requests
from concurrent.futures import ThreadPoolExecutor

# Import our risk management modules
from data_loader import TradeStewardDataLoader
from alert_notification_system import AutomatedAlertSystem, RiskAlert, AlertType, AlertSeverity

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class DataProvider(Enum):
    """Supported market data providers"""
    ALPHA_VANTAGE = "alpha_vantage"
    YAHOO_FINANCE = "yahoo_finance" 
    IEX_CLOUD = "iex_cloud"
    POLYGON = "polygon"
    FINNHUB = "finnhub"
    MOCK = "mock"  # For testing without real API keys

class DataType(Enum):
    """Types of market data"""
    EQUITY_PRICES = "equity_prices"
    OPTIONS_PRICES = "options_prices"
    VIX_DATA = "vix_data"
    INTEREST_RATES = "interest_rates"
    ECONOMIC_INDICATORS = "economic_indicators"
    NEWS_SENTIMENT = "news_sentiment"

@dataclass
class MarketDataPoint:
    """Individual market data point"""
    symbol: str
    data_type: DataType
    timestamp: datetime
    price: float
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    open_price: Optional[float] = None
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class MarketDataConfig:
    """Market data configuration"""
    providers: Dict[DataProvider, Dict[str, str]]  # Provider configs with API keys
    update_intervals: Dict[DataType, int]  # Update intervals in seconds
    symbols: Dict[DataType, List[str]]  # Symbols to track by data type
    storage_path: str = "market_data.db"
    max_history_days: int = 30
    retry_attempts: int = 3
    timeout_seconds: int = 10

class MarketDataProvider(ABC):
    """Abstract base class for market data providers"""
    
    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    async def fetch_equity_prices(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch equity price data"""
        pass
    
    @abstractmethod
    async def fetch_options_data(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch options price and Greeks data"""
        pass
    
    @abstractmethod
    async def fetch_vix_data(self) -> MarketDataPoint:
        """Fetch VIX volatility data"""
        pass
    
    async def fetch_interest_rates(self) -> List[MarketDataPoint]:
        """Fetch interest rate data - default implementation"""
        return []
    
    async def fetch_economic_indicators(self) -> List[MarketDataPoint]:
        """Fetch economic indicators - default implementation"""
        return []

class AlphaVantageProvider(MarketDataProvider):
    """Alpha Vantage market data provider"""
    
    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.api_key = config.get('api_key', '')
        self.base_url = "https://www.alphavantage.co/query"
        
    async def fetch_equity_prices(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch equity prices from Alpha Vantage"""
        data_points = []
        
        async with aiohttp.ClientSession() as session:
            for symbol in symbols:
                try:
                    params = {
                        'function': 'GLOBAL_QUOTE',
                        'symbol': symbol,
                        'apikey': self.api_key
                    }
                    
                    async with session.get(self.base_url, params=params) as response:
                        data = await response.json()
                        
                        quote = data.get('Global Quote', {})
                        if quote:
                            data_point = MarketDataPoint(
                                symbol=symbol,
                                data_type=DataType.EQUITY_PRICES,
                                timestamp=datetime.now(),
                                price=float(quote.get('05. price', 0)),
                                volume=int(quote.get('06. volume', 0)),
                                high=float(quote.get('03. high', 0)),
                                low=float(quote.get('04. low', 0)),
                                open_price=float(quote.get('02. open', 0)),
                                metadata={'change_percent': quote.get('10. change percent', '0%')}
                            )
                            data_points.append(data_point)
                            
                except Exception as e:
                    self.logger.error(f"Error fetching {symbol} from Alpha Vantage: {str(e)}")
                    
                # Rate limiting
                await asyncio.sleep(0.2)  # Alpha Vantage rate limit
        
        return data_points
    
    async def fetch_options_data(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Alpha Vantage doesn't provide options data - return empty list"""
        self.logger.warning("Alpha Vantage doesn't support options data")
        return []
    
    async def fetch_vix_data(self) -> MarketDataPoint:
        """Fetch VIX data from Alpha Vantage"""
        try:
            async with aiohttp.ClientSession() as session:
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': 'VIX',
                    'apikey': self.api_key
                }
                
                async with session.get(self.base_url, params=params) as response:
                    data = await response.json()
                    
                    quote = data.get('Global Quote', {})
                    if quote:
                        return MarketDataPoint(
                            symbol='VIX',
                            data_type=DataType.VIX_DATA,
                            timestamp=datetime.now(),
                            price=float(quote.get('05. price', 0)),
                            metadata={'change_percent': quote.get('10. change percent', '0%')}
                        )
                        
        except Exception as e:
            self.logger.error(f"Error fetching VIX from Alpha Vantage: {str(e)}")
            
        return None

class YahooFinanceProvider(MarketDataProvider):
    """Yahoo Finance market data provider (using yfinance)"""
    
    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        try:
            import yfinance as yf
            self.yf = yf
        except ImportError:
            self.logger.error("yfinance package not installed. Install with: pip install yfinance")
            self.yf = None
    
    async def fetch_equity_prices(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch equity prices from Yahoo Finance"""
        if not self.yf:
            return []
            
        data_points = []
        
        try:
            # Fetch data for all symbols at once
            tickers = self.yf.Tickers(' '.join(symbols))
            
            for symbol in symbols:
                try:
                    ticker = tickers.tickers[symbol]
                    info = ticker.info
                    hist = ticker.history(period='1d', interval='1m')
                    
                    if not hist.empty:
                        latest = hist.iloc[-1]
                        data_point = MarketDataPoint(
                            symbol=symbol,
                            data_type=DataType.EQUITY_PRICES,
                            timestamp=datetime.now(),
                            price=float(latest['Close']),
                            volume=int(latest['Volume']),
                            high=float(latest['High']),
                            low=float(latest['Low']),
                            open_price=float(latest['Open']),
                            metadata={
                                'market_cap': info.get('marketCap', 0),
                                'pe_ratio': info.get('trailingPE', 0),
                                'sector': info.get('sector', '')
                            }
                        )
                        data_points.append(data_point)
                        
                except Exception as e:
                    self.logger.error(f"Error fetching {symbol} from Yahoo Finance: {str(e)}")
                    
        except Exception as e:
            self.logger.error(f"Error with Yahoo Finance batch fetch: {str(e)}")
        
        return data_points
    
    async def fetch_options_data(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Fetch options data from Yahoo Finance"""
        if not self.yf:
            return []
            
        data_points = []
        
        for symbol in symbols:
            try:
                ticker = self.yf.Ticker(symbol)
                options = ticker.options
                
                if options:
                    # Get nearest expiry
                    exp_date = options[0]
                    option_chain = ticker.option_chain(exp_date)
                    
                    # Process calls and puts
                    for option_type, chain in [('call', option_chain.calls), ('put', option_chain.puts)]:
                        for _, option in chain.iterrows():
                            data_point = MarketDataPoint(
                                symbol=f"{symbol}_{option['strike']}_{option_type}_{exp_date}",
                                data_type=DataType.OPTIONS_PRICES,
                                timestamp=datetime.now(),
                                price=float(option.get('lastPrice', 0)),
                                bid=float(option.get('bid', 0)),
                                ask=float(option.get('ask', 0)),
                                volume=int(option.get('volume', 0)),
                                implied_volatility=float(option.get('impliedVolatility', 0)),
                                metadata={
                                    'strike': float(option['strike']),
                                    'expiry': exp_date,
                                    'option_type': option_type,
                                    'open_interest': int(option.get('openInterest', 0))
                                }
                            )
                            data_points.append(data_point)
                            
            except Exception as e:
                self.logger.error(f"Error fetching options for {symbol}: {str(e)}")
        
        return data_points
    
    async def fetch_vix_data(self) -> MarketDataPoint:
        """Fetch VIX data from Yahoo Finance"""
        if not self.yf:
            return None
            
        try:
            vix = self.yf.Ticker('^VIX')
            hist = vix.history(period='1d', interval='1m')
            
            if not hist.empty:
                latest = hist.iloc[-1]
                return MarketDataPoint(
                    symbol='VIX',
                    data_type=DataType.VIX_DATA,
                    timestamp=datetime.now(),
                    price=float(latest['Close']),
                    high=float(latest['High']),
                    low=float(latest['Low']),
                    open_price=float(latest['Open'])
                )
                
        except Exception as e:
            self.logger.error(f"Error fetching VIX from Yahoo Finance: {str(e)}")
            
        return None

class MockDataProvider(MarketDataProvider):
    """Mock data provider for testing without API keys"""
    
    def __init__(self, config: Dict[str, str]):
        super().__init__(config)
        self.base_prices = {
            'SPY': 450.0,
            'QQQ': 380.0,
            'IWM': 200.0,
            'GLD': 180.0,
            'VIX': 20.0
        }
    
    async def fetch_equity_prices(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Generate mock equity price data"""
        data_points = []
        
        for symbol in symbols:
            base_price = self.base_prices.get(symbol, 100.0)
            
            # Add some random movement
            price_change = np.random.normal(0, 0.01) * base_price
            current_price = base_price + price_change
            
            volume = np.random.randint(1000000, 10000000)
            
            data_point = MarketDataPoint(
                symbol=symbol,
                data_type=DataType.EQUITY_PRICES,
                timestamp=datetime.now(),
                price=current_price,
                volume=volume,
                high=current_price * (1 + abs(np.random.normal(0, 0.005))),
                low=current_price * (1 - abs(np.random.normal(0, 0.005))),
                open_price=current_price * (1 + np.random.normal(0, 0.002)),
                metadata={'mock_data': True}
            )
            data_points.append(data_point)
        
        return data_points
    
    async def fetch_options_data(self, symbols: List[str]) -> List[MarketDataPoint]:
        """Generate mock options data"""
        data_points = []
        
        for symbol in symbols:
            base_price = self.base_prices.get(symbol, 100.0)
            
            # Generate a few option strikes around current price
            strikes = [base_price * multiplier for multiplier in [0.95, 1.0, 1.05]]
            
            for strike in strikes:
                for option_type in ['call', 'put']:
                    option_price = max(0.1, np.random.uniform(1, 10))
                    iv = np.random.uniform(0.15, 0.4)
                    
                    data_point = MarketDataPoint(
                        symbol=f"{symbol}_{strike}_{option_type}_mock",
                        data_type=DataType.OPTIONS_PRICES,
                        timestamp=datetime.now(),
                        price=option_price,
                        bid=option_price * 0.98,
                        ask=option_price * 1.02,
                        volume=np.random.randint(100, 1000),
                        implied_volatility=iv,
                        delta=np.random.uniform(-1, 1) if option_type == 'put' else np.random.uniform(0, 1),
                        gamma=np.random.uniform(0, 0.1),
                        theta=np.random.uniform(-0.05, 0),
                        vega=np.random.uniform(0, 0.2),
                        metadata={
                            'strike': strike,
                            'option_type': option_type,
                            'mock_data': True
                        }
                    )
                    data_points.append(data_point)
        
        return data_points
    
    async def fetch_vix_data(self) -> MarketDataPoint:
        """Generate mock VIX data"""
        base_vix = self.base_prices['VIX']
        vix_change = np.random.normal(0, 0.1) * base_vix
        current_vix = max(10, base_vix + vix_change)
        
        return MarketDataPoint(
            symbol='VIX',
            data_type=DataType.VIX_DATA,
            timestamp=datetime.now(),
            price=current_vix,
            metadata={'mock_data': True}
        )

class MarketDataStorage:
    """Database storage for market data"""
    
    def __init__(self, db_path: str = "market_data.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize market data database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                price REAL NOT NULL,
                volume INTEGER,
                bid REAL,
                ask REAL,
                high REAL,
                low REAL,
                open_price REAL,
                implied_volatility REAL,
                delta REAL,
                gamma REAL,
                theta REAL,
                vega REAL,
                metadata TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_symbol_timestamp ON market_data(symbol, timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_data_type ON market_data(data_type)
        ''')
        
        conn.commit()
        conn.close()
    
    def store_data_points(self, data_points: List[MarketDataPoint]):
        """Store market data points in database"""
        if not data_points:
            return
            
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for point in data_points:
            cursor.execute('''
                INSERT INTO market_data 
                (symbol, data_type, timestamp, price, volume, bid, ask, high, low, 
                 open_price, implied_volatility, delta, gamma, theta, vega, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                point.symbol,
                point.data_type.value,
                point.timestamp.isoformat(),
                point.price,
                point.volume,
                point.bid,
                point.ask,
                point.high,
                point.low,
                point.open_price,
                point.implied_volatility,
                point.delta,
                point.gamma,
                point.theta,
                point.vega,
                json.dumps(point.metadata) if point.metadata else None
            ))
        
        conn.commit()
        conn.close()
    
    def get_latest_data(self, symbol: str, data_type: DataType) -> Optional[MarketDataPoint]:
        """Get latest data point for symbol and type"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM market_data 
            WHERE symbol = ? AND data_type = ? 
            ORDER BY timestamp DESC 
            LIMIT 1
        '''
        
        cursor = conn.cursor()
        cursor.execute(query, (symbol, data_type.value))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_data_point(row)
        return None
    
    def get_historical_data(self, symbol: str, data_type: DataType, hours: int = 24) -> List[MarketDataPoint]:
        """Get historical data for symbol and type"""
        conn = sqlite3.connect(self.db_path)
        
        since_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        query = '''
            SELECT * FROM market_data 
            WHERE symbol = ? AND data_type = ? AND timestamp >= ?
            ORDER BY timestamp ASC
        '''
        
        cursor = conn.cursor()
        cursor.execute(query, (symbol, data_type.value, since_time))
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_data_point(row) for row in rows]
    
    def _row_to_data_point(self, row) -> MarketDataPoint:
        """Convert database row to MarketDataPoint"""
        metadata = json.loads(row[16]) if row[16] else None
        
        return MarketDataPoint(
            symbol=row[1],
            data_type=DataType(row[2]),
            timestamp=datetime.fromisoformat(row[3]),
            price=row[4],
            volume=row[5],
            bid=row[6],
            ask=row[7],
            high=row[8],
            low=row[9],
            open_price=row[10],
            implied_volatility=row[11],
            delta=row[12],
            gamma=row[13],
            theta=row[14],
            vega=row[15],
            metadata=metadata
        )
    
    def cleanup_old_data(self, days: int = 30):
        """Remove data older than specified days"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        cursor.execute('DELETE FROM market_data WHERE timestamp < ?', (cutoff_date,))
        deleted_rows = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        logger.info(f"Cleaned up {deleted_rows} old market data records")

class LiveMarketDataManager:
    """Main manager for live market data integration"""
    
    def __init__(self, config: MarketDataConfig, alert_system: Optional[AutomatedAlertSystem] = None):
        """
        Initialize live market data manager
        
        Args:
            config: Market data configuration
            alert_system: Optional alert system for notifications
        """
        self.config = config
        self.alert_system = alert_system
        
        # Initialize storage
        self.storage = MarketDataStorage(config.storage_path)
        
        # Initialize providers
        self.providers = {}
        self._initialize_providers()
        
        # Data processing
        self.data_callbacks = {}  # Callbacks for data updates
        self.running = False
        self.update_threads = {}
        
        # Executor for async operations
        self.executor = ThreadPoolExecutor(max_workers=5)
        
        logger.info("Live market data manager initialized")
    
    def _initialize_providers(self):
        """Initialize market data providers"""
        provider_classes = {
            DataProvider.ALPHA_VANTAGE: AlphaVantageProvider,
            DataProvider.YAHOO_FINANCE: YahooFinanceProvider,
            DataProvider.MOCK: MockDataProvider
        }
        
        for provider_type, provider_config in self.config.providers.items():
            if provider_type in provider_classes:
                self.providers[provider_type] = provider_classes[provider_type](provider_config)
                logger.info(f"Initialized {provider_type.value} provider")
    
    def add_data_callback(self, data_type: DataType, callback: Callable[[List[MarketDataPoint]], None]):
        """Add callback for specific data type updates"""
        if data_type not in self.data_callbacks:
            self.data_callbacks[data_type] = []
        self.data_callbacks[data_type].append(callback)
    
    def start_live_feeds(self):
        """Start all live data feeds"""
        if self.running:
            logger.warning("Live feeds already running")
            return
        
        self.running = True
        
        # Start update threads for each data type
        for data_type, interval in self.config.update_intervals.items():
            if data_type in self.config.symbols:
                thread = threading.Thread(
                    target=self._update_loop,
                    args=(data_type, interval),
                    daemon=True
                )
                thread.start()
                self.update_threads[data_type] = thread
                
                logger.info(f"Started {data_type.value} feed with {interval}s interval")
        
        # Start cleanup thread
        cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        cleanup_thread.start()
        
        logger.info("All live market data feeds started")
    
    def stop_live_feeds(self):
        """Stop all live data feeds"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.update_threads.values():
            thread.join(timeout=5)
        
        self.update_threads.clear()
        logger.info("All live market data feeds stopped")
    
    def _update_loop(self, data_type: DataType, interval: int):
        """Update loop for specific data type"""
        while self.running:
            try:
                # Fetch data from available providers
                asyncio.run(self._fetch_and_store_data(data_type))
                
            except Exception as e:
                logger.error(f"Error in {data_type.value} update loop: {str(e)}")
                
                # Send alert for data feed errors
                if self.alert_system:
                    self._send_data_feed_alert(data_type, str(e))
            
            # Wait for next update
            time.sleep(interval)
    
    async def _fetch_and_store_data(self, data_type: DataType):
        """Fetch and store data for specific data type"""
        symbols = self.config.symbols.get(data_type, [])
        if not symbols:
            return
        
        all_data_points = []
        
        # Fetch from all available providers
        for provider_type, provider in self.providers.items():
            try:
                data_points = []
                
                if data_type == DataType.EQUITY_PRICES:
                    data_points = await provider.fetch_equity_prices(symbols)
                elif data_type == DataType.OPTIONS_PRICES:
                    data_points = await provider.fetch_options_data(symbols)
                elif data_type == DataType.VIX_DATA:
                    vix_data = await provider.fetch_vix_data()
                    if vix_data:
                        data_points = [vix_data]
                elif data_type == DataType.INTEREST_RATES:
                    data_points = await provider.fetch_interest_rates()
                elif data_type == DataType.ECONOMIC_INDICATORS:
                    data_points = await provider.fetch_economic_indicators()
                
                all_data_points.extend(data_points)
                
                logger.debug(f"Fetched {len(data_points)} {data_type.value} points from {provider_type.value}")
                
            except Exception as e:
                logger.error(f"Error fetching {data_type.value} from {provider_type.value}: {str(e)}")
        
        # Store data points
        if all_data_points:
            self.storage.store_data_points(all_data_points)
            
            # Trigger callbacks
            await self._trigger_callbacks(data_type, all_data_points)
            
            logger.debug(f"Stored {len(all_data_points)} {data_type.value} data points")
    
    async def _trigger_callbacks(self, data_type: DataType, data_points: List[MarketDataPoint]):
        """Trigger registered callbacks for data type"""
        callbacks = self.data_callbacks.get(data_type, [])
        
        for callback in callbacks:
            try:
                # Run callback in executor to avoid blocking
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, callback, data_points
                )
            except Exception as e:
                logger.error(f"Error in {data_type.value} callback: {str(e)}")
    
    def _cleanup_loop(self):
        """Periodic cleanup of old data"""
        while self.running:
            try:
                # Clean up old data every hour
                time.sleep(3600)
                
                if self.running:  # Check if still running after sleep
                    self.storage.cleanup_old_data(self.config.max_history_days)
                    
            except Exception as e:
                logger.error(f"Error in cleanup loop: {str(e)}")
    
    def _send_data_feed_alert(self, data_type: DataType, error_message: str):
        """Send alert for data feed errors"""
        try:
            alert = RiskAlert(
                alert_id=f"data_feed_error_{data_type.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                timestamp=datetime.now(),
                alert_type=AlertType.SYSTEM_ERROR,
                severity=AlertSeverity.WARNING,
                title=f"Market Data Feed Error",
                message=f"Error in {data_type.value} data feed: {error_message}",
                current_value=1.0,
                threshold_value=0.0,
                metric_name=f"{data_type.value}_feed",
                recommendations=[
                    "Check data provider connectivity",
                    "Verify API keys and credentials",
                    "Review rate limits",
                    "Consider switching to backup provider"
                ]
            )
            
            # Send through alert system if available
            if self.alert_system:
                self.alert_system._send_alert(alert)
                
        except Exception as e:
            logger.error(f"Error sending data feed alert: {str(e)}")
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for symbol"""
        data_point = self.storage.get_latest_data(symbol, DataType.EQUITY_PRICES)
        return data_point.price if data_point else None
    
    def get_latest_vix(self) -> Optional[float]:
        """Get latest VIX value"""
        data_point = self.storage.get_latest_data('VIX', DataType.VIX_DATA)
        return data_point.price if data_point else None
    
    def get_price_history(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Get price history as DataFrame"""
        data_points = self.storage.get_historical_data(symbol, DataType.EQUITY_PRICES, hours)
        
        if not data_points:
            return pd.DataFrame()
        
        data = []
        for point in data_points:
            data.append({
                'timestamp': point.timestamp,
                'price': point.price,
                'volume': point.volume,
                'high': point.high,
                'low': point.low,
                'open': point.open_price
            })
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        return df
    
    def get_options_data(self, symbol: str) -> pd.DataFrame:
        """Get latest options data for symbol"""
        data_points = self.storage.get_historical_data(symbol, DataType.OPTIONS_PRICES, 1)
        
        if not data_points:
            return pd.DataFrame()
        
        data = []
        for point in data_points:
            if point.metadata and 'strike' in point.metadata:
                data.append({
                    'symbol': point.symbol,
                    'strike': point.metadata['strike'],
                    'option_type': point.metadata.get('option_type', ''),
                    'price': point.price,
                    'bid': point.bid,
                    'ask': point.ask,
                    'volume': point.volume,
                    'implied_volatility': point.implied_volatility,
                    'delta': point.delta,
                    'gamma': point.gamma,
                    'theta': point.theta,
                    'vega': point.vega,
                    'timestamp': point.timestamp
                })
        
        return pd.DataFrame(data)
    
    def get_market_data_summary(self) -> Dict[str, Any]:
        """Get summary of market data status"""
        summary = {
            'providers': list(self.providers.keys()),
            'data_types': list(self.config.symbols.keys()),
            'running': self.running,
            'update_intervals': {dt.value: interval for dt, interval in self.config.update_intervals.items()},
            'latest_updates': {}
        }
        
        # Get latest update times for each data type
        for data_type in self.config.symbols.keys():
            symbols = self.config.symbols[data_type]
            if symbols:
                latest_data = self.storage.get_latest_data(symbols[0], data_type)
                if latest_data:
                    summary['latest_updates'][data_type.value] = latest_data.timestamp.isoformat()
        
        return summary


def main():
    """Example usage of Live Market Data Integration"""
    
    print("Live Market Data Integration - Phase 3")
    print("=" * 60)
    
    try:
        # Configure market data settings
        config = MarketDataConfig(
            providers={
                DataProvider.MOCK: {},  # Use mock provider for demo
                # DataProvider.YAHOO_FINANCE: {},  # Uncomment to use Yahoo Finance
                # DataProvider.ALPHA_VANTAGE: {'api_key': 'your_api_key_here'}
            },
            update_intervals={
                DataType.EQUITY_PRICES: 60,    # Update every minute
                DataType.VIX_DATA: 120,        # Update every 2 minutes
                DataType.OPTIONS_PRICES: 300   # Update every 5 minutes
            },
            symbols={
                DataType.EQUITY_PRICES: ['SPY', 'QQQ', 'IWM', 'GLD'],
                DataType.VIX_DATA: ['VIX'],
                DataType.OPTIONS_PRICES: ['SPY', 'QQQ']
            },
            storage_path="market_data.db",
            max_history_days=7
        )
        
        # Initialize market data manager
        market_data_manager = LiveMarketDataManager(config)
        
        print("✅ Market data manager initialized")
        print(f"📊 Providers: {[p.value for p in config.providers.keys()]}")
        print(f"📈 Data types: {[dt.value for dt in config.symbols.keys()]}")
        print(f"🔄 Update intervals: {[(dt.value, interval) for dt, interval in config.update_intervals.items()]}")
        
        # Add data callbacks for demonstration
        def equity_callback(data_points: List[MarketDataPoint]):
            for point in data_points:
                print(f"📈 {point.symbol}: ${point.price:.2f} (Volume: {point.volume:,})")
        
        def vix_callback(data_points: List[MarketDataPoint]):
            for point in data_points:
                print(f"🌪️  VIX: {point.price:.2f}")
        
        market_data_manager.add_data_callback(DataType.EQUITY_PRICES, equity_callback)
        market_data_manager.add_data_callback(DataType.VIX_DATA, vix_callback)
        
        # Start live feeds
        print(f"\n🚀 Starting live market data feeds...")
        market_data_manager.start_live_feeds()
        
        # Run for demonstration
        print(f"Running live feeds for 2 minutes...")
        print(f"Press Ctrl+C to stop early")
        
        try:
            time.sleep(120)  # Run for 2 minutes
        except KeyboardInterrupt:
            print(f"\n🛑 Stopping early...")
        
        # Stop feeds
        market_data_manager.stop_live_feeds()
        
        # Display summary
        summary = market_data_manager.get_market_data_summary()
        print(f"\n📊 Market Data Summary:")
        print(f"  • Providers: {[p.value for p in summary['providers']]}")
        print(f"  • Data Types: {summary['data_types']}")
        print(f"  • Running: {summary['running']}")
        
        # Show some sample data
        print(f"\n📈 Sample Price Data:")
        for symbol in ['SPY', 'QQQ']:
            latest_price = market_data_manager.get_latest_price(symbol)
            if latest_price:
                print(f"  • {symbol}: ${latest_price:.2f}")
        
        vix_value = market_data_manager.get_latest_vix()
        if vix_value:
            print(f"  • VIX: {vix_value:.2f}")
        
        # Show price history for SPY
        spy_history = market_data_manager.get_price_history('SPY', hours=1)
        if not spy_history.empty:
            print(f"\n📊 SPY Price History (last hour): {len(spy_history)} data points")
            print(f"  • Latest: ${spy_history['price'].iloc[-1]:.2f}")
            print(f"  • High: ${spy_history['high'].max():.2f}")
            print(f"  • Low: ${spy_history['low'].min():.2f}")
        
        print(f"\n✅ Live market data integration demonstration complete!")
        
    except Exception as e:
        logger.error(f"Error running market data integration: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()