"""
Alternative Data Integration - Phase 4 Implementation
Advanced alternative data sources for enhanced risk analysis and alpha generation
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import sqlite3
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings
import requests
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import feedparser
import re
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import tweepy
from newsapi import NewsApiClient
from alpha_vantage.timeseries import TimeSeries
import yfinance as yf

# Machine Learning imports for sentiment analysis
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Import our existing modules
from data_loader import TradeStewardDataLoader
from market_data_integration import LiveMarketDataManager, MarketDataPoint, DataType

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class AltDataType(Enum):
    """Types of alternative data"""
    NEWS_SENTIMENT = "news_sentiment"
    SOCIAL_MEDIA_SENTIMENT = "social_media_sentiment"
    OPTIONS_FLOW = "options_flow"
    INSIDER_TRADING = "insider_trading"
    SATELLITE_DATA = "satellite_data"
    CREDIT_DEFAULT_SWAPS = "credit_default_swaps"
    EARNINGS_ESTIMATES = "earnings_estimates"
    SUPPLY_CHAIN_DATA = "supply_chain_data"
    PATENT_FILINGS = "patent_filings"
    ESG_SCORES = "esg_scores"

class SentimentLabel(Enum):
    """Sentiment classification labels"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2

@dataclass
class NewsArticle:
    """News article data structure"""
    title: str
    content: str
    source: str
    published_at: datetime
    url: str
    symbols: List[str]
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[SentimentLabel] = None
    relevance_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SocialMediaPost:
    """Social media post data structure"""
    platform: str
    post_id: str
    content: str
    author: str
    created_at: datetime
    engagement_metrics: Dict[str, int]
    symbols: List[str]
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[SentimentLabel] = None
    influence_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class OptionsFlowData:
    """Options flow data structure"""
    timestamp: datetime
    symbol: str
    option_type: str  # 'call' or 'put'
    strike: float
    expiry: datetime
    volume: int
    open_interest: int
    premium: float
    implied_volatility: float
    delta: float
    unusual_activity_flag: bool
    flow_type: str  # 'sweep', 'block', 'split'
    sentiment: str  # 'bullish', 'bearish', 'neutral'
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class AltDataConfig:
    """Alternative data configuration"""
    news_sources: List[str]
    social_media_platforms: List[str]
    api_keys: Dict[str, str]
    update_intervals: Dict[AltDataType, int]  # seconds
    symbols_to_track: List[str]
    sentiment_threshold: float
    storage_path: str
    max_history_days: int

class SentimentAnalyzer:
    """Advanced sentiment analysis engine"""
    
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.tfidf_vectorizer = None
        self.ml_model = None
        self.logger = logging.getLogger(__name__ + '.SentimentAnalyzer')
        
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment using multiple methods"""
        
        results = {}
        
        # TextBlob sentiment
        blob = TextBlob(text)
        results['textblob_polarity'] = blob.sentiment.polarity
        results['textblob_subjectivity'] = blob.sentiment.subjectivity
        
        # VADER sentiment
        vader_scores = self.vader_analyzer.polarity_scores(text)
        results['vader_compound'] = vader_scores['compound']
        results['vader_positive'] = vader_scores['pos']
        results['vader_negative'] = vader_scores['neg']
        results['vader_neutral'] = vader_scores['neu']
        
        # Combined sentiment score
        combined_score = (results['textblob_polarity'] + results['vader_compound']) / 2
        results['combined_sentiment'] = combined_score
        
        # Classify sentiment
        if combined_score >= 0.6:
            results['sentiment_label'] = SentimentLabel.VERY_POSITIVE
        elif combined_score >= 0.2:
            results['sentiment_label'] = SentimentLabel.POSITIVE
        elif combined_score <= -0.6:
            results['sentiment_label'] = SentimentLabel.VERY_NEGATIVE
        elif combined_score <= -0.2:
            results['sentiment_label'] = SentimentLabel.NEGATIVE
        else:
            results['sentiment_label'] = SentimentLabel.NEUTRAL
        
        return results
    
    def train_custom_sentiment_model(self, training_data: List[Tuple[str, int]]) -> Dict[str, float]:
        """Train custom sentiment model on domain-specific data"""
        
        if len(training_data) < 100:
            self.logger.warning("Insufficient training data for custom sentiment model")
            return {'accuracy': 0.0}
        
        # Prepare data
        texts, labels = zip(*training_data)
        
        # Vectorize text
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        X = self.tfidf_vectorizer.fit_transform(texts)
        y = np.array(labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.ml_model = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        self.ml_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.ml_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"Custom sentiment model trained with accuracy: {accuracy:.3f}")
        
        return {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
    
    def predict_sentiment_ml(self, text: str) -> Dict[str, Any]:
        """Predict sentiment using custom ML model"""
        
        if not self.ml_model or not self.tfidf_vectorizer:
            return {'error': 'Custom model not trained'}
        
        # Vectorize text
        X = self.tfidf_vectorizer.transform([text])
        
        # Predict
        prediction = self.ml_model.predict(X)[0]
        probabilities = self.ml_model.predict_proba(X)[0]
        
        return {
            'prediction': int(prediction),
            'confidence': float(max(probabilities)),
            'class_probabilities': probabilities.tolist()
        }
    
    def extract_financial_keywords(self, text: str) -> Dict[str, List[str]]:
        """Extract financial keywords and entities"""
        
        # Financial keywords
        bullish_keywords = [
            'buy', 'bull', 'bullish', 'long', 'upside', 'gain', 'profit',
            'growth', 'rally', 'surge', 'climb', 'rise', 'increase',
            'outperform', 'beat', 'exceed', 'strong', 'positive'
        ]
        
        bearish_keywords = [
            'sell', 'bear', 'bearish', 'short', 'downside', 'loss', 'decline',
            'fall', 'drop', 'crash', 'plunge', 'tumble', 'underperform',
            'miss', 'weak', 'negative', 'concern', 'risk'
        ]
        
        volatility_keywords = [
            'volatile', 'volatility', 'swing', 'fluctuate', 'uncertain',
            'turbulent', 'choppy', 'erratic', 'unstable'
        ]
        
        text_lower = text.lower()
        
        # Count keyword occurrences
        bullish_count = sum(1 for keyword in bullish_keywords if keyword in text_lower)
        bearish_count = sum(1 for keyword in bearish_keywords if keyword in text_lower)
        volatility_count = sum(1 for keyword in volatility_keywords if keyword in text_lower)
        
        # Extract ticker symbols (basic regex)
        ticker_pattern = r'\$([A-Z]{1,5})\b'
        tickers = re.findall(ticker_pattern, text)
        
        return {
            'bullish_keywords': [kw for kw in bullish_keywords if kw in text_lower],
            'bearish_keywords': [kw for kw in bearish_keywords if kw in text_lower],
            'volatility_keywords': [kw for kw in volatility_keywords if kw in text_lower],
            'tickers_mentioned': list(set(tickers)),
            'keyword_sentiment_score': (bullish_count - bearish_count) / max(1, bullish_count + bearish_count),
            'volatility_score': volatility_count / max(1, len(text.split()))
        }

class NewsDataProvider:
    """News data provider and analyzer"""
    
    def __init__(self, config: AltDataConfig, sentiment_analyzer: SentimentAnalyzer):
        self.config = config
        self.sentiment_analyzer = sentiment_analyzer
        self.logger = logging.getLogger(__name__ + '.NewsDataProvider')
        
        # Initialize news API clients
        self.news_api_client = None
        if 'newsapi_key' in config.api_keys:
            self.news_api_client = NewsApiClient(api_key=config.api_keys['newsapi_key'])
    
    async def fetch_news_articles(self, symbols: List[str], hours_back: int = 24) -> List[NewsArticle]:
        """Fetch news articles for specified symbols"""
        
        articles = []
        
        # Fetch from News API
        if self.news_api_client:
            articles.extend(await self._fetch_from_newsapi(symbols, hours_back))
        
        # Fetch from RSS feeds
        articles.extend(await self._fetch_from_rss_feeds(symbols))
        
        # Analyze sentiment for all articles
        for article in articles:
            sentiment_results = self.sentiment_analyzer.analyze_text_sentiment(
                f"{article.title} {article.content}"
            )
            article.sentiment_score = sentiment_results['combined_sentiment']
            article.sentiment_label = sentiment_results['sentiment_label']
            
            # Calculate relevance score
            article.relevance_score = self._calculate_relevance_score(article, symbols)
        
        return articles
    
    async def _fetch_from_newsapi(self, symbols: List[str], hours_back: int) -> List[NewsArticle]:
        """Fetch articles from News API"""
        
        articles = []
        
        try:
            from_date = (datetime.now() - timedelta(hours=hours_back)).strftime('%Y-%m-%d')
            
            for symbol in symbols:
                # Search for articles mentioning the symbol
                response = self.news_api_client.get_everything(
                    q=f'"{symbol}" OR "{symbol.lower()}"',
                    from_param=from_date,
                    language='en',
                    sort_by='relevancy',
                    page_size=20
                )
                
                if response['status'] == 'ok':
                    for article_data in response['articles']:
                        if article_data['title'] and article_data['description']:
                            article = NewsArticle(
                                title=article_data['title'],
                                content=article_data['description'] or '',
                                source=article_data['source']['name'],
                                published_at=datetime.fromisoformat(
                                    article_data['publishedAt'].replace('Z', '+00:00')
                                ),
                                url=article_data['url'],
                                symbols=[symbol],
                                metadata={
                                    'author': article_data.get('author'),
                                    'url_to_image': article_data.get('urlToImage')
                                }
                            )
                            articles.append(article)
                
                # Rate limiting
                await asyncio.sleep(0.1)
                
        except Exception as e:
            self.logger.error(f"Error fetching from News API: {str(e)}")
        
        return articles
    
    async def _fetch_from_rss_feeds(self, symbols: List[str]) -> List[NewsArticle]:
        """Fetch articles from RSS feeds"""
        
        articles = []
        
        # Common financial RSS feeds
        rss_feeds = [
            'https://feeds.bloomberg.com/markets/news.rss',
            'https://www.reuters.com/business/finance/rss',
            'https://www.cnbc.com/id/100003114/device/rss/rss.html',
            'https://feeds.reuters.com/reuters/businessNews'
        ]
        
        try:
            for feed_url in rss_feeds:
                try:
                    feed = feedparser.parse(feed_url)
                    
                    for entry in feed.entries[:10]:  # Limit per feed
                        # Check if any symbol is mentioned
                        title_content = f"{entry.title} {entry.get('summary', '')}"
                        mentioned_symbols = [
                            symbol for symbol in symbols 
                            if symbol.lower() in title_content.lower()
                        ]
                        
                        if mentioned_symbols:
                            article = NewsArticle(
                                title=entry.title,
                                content=entry.get('summary', ''),
                                source=feed.feed.get('title', 'RSS Feed'),
                                published_at=datetime.fromtimestamp(
                                    time.mktime(entry.published_parsed)
                                ),
                                url=entry.link,
                                symbols=mentioned_symbols,
                                metadata={'feed_url': feed_url}
                            )
                            articles.append(article)
                    
                    # Rate limiting
                    await asyncio.sleep(0.5)
                    
                except Exception as e:
                    self.logger.warning(f"Error parsing RSS feed {feed_url}: {str(e)}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error fetching RSS feeds: {str(e)}")
        
        return articles
    
    def _calculate_relevance_score(self, article: NewsArticle, target_symbols: List[str]) -> float:
        """Calculate relevance score for article"""
        
        text = f"{article.title} {article.content}".lower()
        
        # Base relevance from symbol mentions
        symbol_mentions = sum(
            text.count(symbol.lower()) for symbol in target_symbols
        )
        base_relevance = min(symbol_mentions / 10.0, 1.0)  # Cap at 1.0
        
        # Boost for financial keywords
        financial_keywords = self.sentiment_analyzer.extract_financial_keywords(text)
        keyword_boost = len(financial_keywords['tickers_mentioned']) * 0.1
        
        # Recency boost (more recent = higher relevance)
        hours_old = (datetime.now() - article.published_at).total_seconds() / 3600
        recency_boost = max(0, 1 - (hours_old / 24))  # Decays over 24 hours
        
        # Source credibility (simplified)
        credible_sources = ['bloomberg', 'reuters', 'wsj', 'cnbc', 'marketwatch']
        source_boost = 0.2 if any(source in article.source.lower() for source in credible_sources) else 0
        
        total_relevance = base_relevance + keyword_boost + recency_boost * 0.3 + source_boost
        
        return min(total_relevance, 1.0)

class OptionsFlowProvider:
    """Options flow and unusual activity provider"""
    
    def __init__(self, config: AltDataConfig):
        self.config = config
        self.logger = logging.getLogger(__name__ + '.OptionsFlowProvider')
    
    async def fetch_options_flow_data(self, symbols: List[str]) -> List[OptionsFlowData]:
        """Fetch options flow data (simulated for demo)"""
        
        # In a real implementation, this would connect to options flow APIs
        # such as FlowAlgo, Benzinga, or similar services
        
        flow_data = []
        
        for symbol in symbols:
            # Simulate options flow data
            num_flows = np.random.randint(5, 15)
            
            for _ in range(num_flows):
                # Random options data
                option_type = np.random.choice(['call', 'put'])
                current_price = np.random.uniform(100, 500)
                strike = current_price * np.random.uniform(0.9, 1.1)
                
                # Generate expiry (1-60 days out)
                days_to_expiry = np.random.randint(1, 61)
                expiry = datetime.now() + timedelta(days=days_to_expiry)
                
                # Volume and activity metrics
                volume = np.random.randint(100, 10000)
                open_interest = np.random.randint(volume, volume * 5)
                premium = np.random.uniform(0.5, 50.0)
                
                # Greeks (simplified)
                time_value = days_to_expiry / 365.0
                moneyness = current_price / strike
                delta = self._calculate_approx_delta(option_type, moneyness, time_value)
                
                # Implied volatility
                iv = np.random.uniform(0.15, 0.8)
                
                # Unusual activity detection
                avg_volume = volume * np.random.uniform(0.3, 0.7)
                unusual_activity = volume > avg_volume * 3  # 3x average volume
                
                # Flow type
                flow_type = np.random.choice(['sweep', 'block', 'split'], p=[0.3, 0.4, 0.3])
                
                # Sentiment based on flow characteristics
                if option_type == 'call' and unusual_activity:
                    sentiment = 'bullish'
                elif option_type == 'put' and unusual_activity:
                    sentiment = 'bearish'
                else:
                    sentiment = 'neutral'
                
                flow_data.append(OptionsFlowData(
                    timestamp=datetime.now() - timedelta(minutes=np.random.randint(0, 1440)),
                    symbol=symbol,
                    option_type=option_type,
                    strike=round(strike, 2),
                    expiry=expiry,
                    volume=volume,
                    open_interest=open_interest,
                    premium=round(premium, 2),
                    implied_volatility=round(iv, 3),
                    delta=round(delta, 3),
                    unusual_activity_flag=unusual_activity,
                    flow_type=flow_type,
                    sentiment=sentiment,
                    metadata={
                        'average_volume': avg_volume,
                        'volume_ratio': volume / max(avg_volume, 1),
                        'days_to_expiry': days_to_expiry
                    }
                ))
        
        return flow_data
    
    def _calculate_approx_delta(self, option_type: str, moneyness: float, time_value: float) -> float:
        """Approximate delta calculation"""
        
        if option_type == 'call':
            if moneyness > 1.1:  # ITM
                return np.random.uniform(0.6, 0.9)
            elif moneyness > 0.9:  # ATM
                return np.random.uniform(0.4, 0.6)
            else:  # OTM
                return np.random.uniform(0.1, 0.4)
        else:  # put
            if moneyness < 0.9:  # ITM
                return np.random.uniform(-0.9, -0.6)
            elif moneyness < 1.1:  # ATM
                return np.random.uniform(-0.6, -0.4)
            else:  # OTM
                return np.random.uniform(-0.4, -0.1)
    
    def analyze_options_flow_sentiment(self, flow_data: List[OptionsFlowData]) -> Dict[str, Any]:
        """Analyze overall sentiment from options flow"""
        
        if not flow_data:
            return {}
        
        # Group by symbol
        symbol_analysis = {}
        
        for symbol in set(flow.symbol for flow in flow_data):
            symbol_flows = [flow for flow in flow_data if flow.symbol == symbol]
            
            # Calculate metrics
            total_volume = sum(flow.volume for flow in symbol_flows)
            call_volume = sum(flow.volume for flow in symbol_flows if flow.option_type == 'call')
            put_volume = sum(flow.volume for flow in symbol_flows if flow.option_type == 'put')
            
            put_call_ratio = put_volume / max(call_volume, 1)
            
            # Unusual activity
            unusual_flows = [flow for flow in symbol_flows if flow.unusual_activity_flag]
            unusual_activity_rate = len(unusual_flows) / len(symbol_flows)
            
            # Sentiment scoring
            bullish_flows = len([f for f in symbol_flows if f.sentiment == 'bullish'])
            bearish_flows = len([f for f in symbol_flows if f.sentiment == 'bearish'])
            
            sentiment_score = (bullish_flows - bearish_flows) / len(symbol_flows)
            
            symbol_analysis[symbol] = {
                'total_volume': total_volume,
                'call_volume': call_volume,
                'put_volume': put_volume,
                'put_call_ratio': put_call_ratio,
                'unusual_activity_rate': unusual_activity_rate,
                'sentiment_score': sentiment_score,
                'flow_count': len(symbol_flows),
                'avg_premium': np.mean([flow.premium for flow in symbol_flows]),
                'avg_iv': np.mean([flow.implied_volatility for flow in symbol_flows])
            }
        
        return symbol_analysis

class AltDataStorage:
    """Database storage for alternative data"""
    
    def __init__(self, db_path: str = "alternative_data.db"):
        self.db_path = db_path
        self._initialize_db()
    
    def _initialize_db(self):
        """Initialize alternative data database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # News articles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                source TEXT,
                published_at TEXT NOT NULL,
                url TEXT UNIQUE,
                symbols TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                relevance_score REAL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Social media posts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS social_media_posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                post_id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                author TEXT,
                created_at TEXT NOT NULL,
                engagement_metrics TEXT,
                symbols TEXT,
                sentiment_score REAL,
                sentiment_label TEXT,
                influence_score REAL,
                processed_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Options flow table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS options_flow (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                option_type TEXT NOT NULL,
                strike REAL NOT NULL,
                expiry TEXT NOT NULL,
                volume INTEGER,
                open_interest INTEGER,
                premium REAL,
                implied_volatility REAL,
                delta REAL,
                unusual_activity_flag INTEGER,
                flow_type TEXT,
                sentiment TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                metadata TEXT
            )
        ''')
        
        # Create indexes
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_news_symbols ON news_articles(symbols)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_social_created ON social_media_posts(created_at)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_timestamp ON options_flow(timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_options_symbol ON options_flow(symbol)')
        
        conn.commit()
        conn.close()
    
    def store_news_articles(self, articles: List[NewsArticle]):
        """Store news articles in database"""
        
        if not articles:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for article in articles:
            try:
                cursor.execute('''
                    INSERT OR REPLACE INTO news_articles 
                    (title, content, source, published_at, url, symbols, 
                     sentiment_score, sentiment_label, relevance_score, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    article.title,
                    article.content,
                    article.source,
                    article.published_at.isoformat(),
                    article.url,
                    json.dumps(article.symbols),
                    article.sentiment_score,
                    article.sentiment_label.value if article.sentiment_label else None,
                    article.relevance_score,
                    json.dumps(article.metadata) if article.metadata else None
                ))
            except Exception as e:
                logger.warning(f"Error storing article: {str(e)}")
                continue
        
        conn.commit()
        conn.close()
    
    def store_options_flow(self, flow_data: List[OptionsFlowData]):
        """Store options flow data"""
        
        if not flow_data:
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for flow in flow_data:
            cursor.execute('''
                INSERT INTO options_flow 
                (timestamp, symbol, option_type, strike, expiry, volume, 
                 open_interest, premium, implied_volatility, delta, 
                 unusual_activity_flag, flow_type, sentiment, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                flow.timestamp.isoformat(),
                flow.symbol,
                flow.option_type,
                flow.strike,
                flow.expiry.isoformat(),
                flow.volume,
                flow.open_interest,
                flow.premium,
                flow.implied_volatility,
                flow.delta,
                1 if flow.unusual_activity_flag else 0,
                flow.flow_type,
                flow.sentiment,
                json.dumps(flow.metadata) if flow.metadata else None
            ))
        
        conn.commit()
        conn.close()
    
    def get_recent_news_sentiment(self, symbols: List[str], hours: int = 24) -> Dict[str, float]:
        """Get recent news sentiment for symbols"""
        
        conn = sqlite3.connect(self.db_path)
        
        since_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        sentiment_scores = {}
        
        for symbol in symbols:
            query = '''
                SELECT AVG(sentiment_score) as avg_sentiment, COUNT(*) as article_count
                FROM news_articles 
                WHERE symbols LIKE ? AND published_at >= ? AND sentiment_score IS NOT NULL
            '''
            
            result = conn.execute(query, (f'%"{symbol}"%', since_time)).fetchone()
            
            if result and result[0] is not None:
                sentiment_scores[symbol] = {
                    'avg_sentiment': result[0],
                    'article_count': result[1]
                }
        
        conn.close()
        return sentiment_scores

class AlternativeDataManager:
    """Main manager for alternative data integration"""
    
    def __init__(self, config: AltDataConfig):
        self.config = config
        self.storage = AltDataStorage(config.storage_path)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.news_provider = NewsDataProvider(config, self.sentiment_analyzer)
        self.options_flow_provider = OptionsFlowProvider(config)
        
        self.running = False
        self.update_threads = {}
        
        self.logger = logging.getLogger(__name__ + '.AlternativeDataManager')
    
    def start_data_collection(self):
        """Start collecting alternative data"""
        
        if self.running:
            self.logger.warning("Data collection already running")
            return
        
        self.running = True
        
        # Start collection threads for different data types
        for data_type, interval in self.config.update_intervals.items():
            thread = threading.Thread(
                target=self._collection_loop,
                args=(data_type, interval),
                daemon=True
            )
            thread.start()
            self.update_threads[data_type] = thread
            
            self.logger.info(f"Started {data_type.value} collection with {interval}s interval")
        
        self.logger.info("Alternative data collection started")
    
    def stop_data_collection(self):
        """Stop alternative data collection"""
        
        self.running = False
        
        for thread in self.update_threads.values():
            thread.join(timeout=5)
        
        self.update_threads.clear()
        self.logger.info("Alternative data collection stopped")
    
    def _collection_loop(self, data_type: AltDataType, interval: int):
        """Collection loop for specific data type"""
        
        while self.running:
            try:
                if data_type == AltDataType.NEWS_SENTIMENT:
                    asyncio.run(self._collect_news_data())
                elif data_type == AltDataType.OPTIONS_FLOW:
                    asyncio.run(self._collect_options_flow_data())
                
            except Exception as e:
                self.logger.error(f"Error in {data_type.value} collection: {str(e)}")
            
            time.sleep(interval)
    
    async def _collect_news_data(self):
        """Collect news data"""
        
        try:
            articles = await self.news_provider.fetch_news_articles(
                self.config.symbols_to_track,
                hours_back=1
            )
            
            if articles:
                self.storage.store_news_articles(articles)
                self.logger.info(f"Collected {len(articles)} news articles")
                
        except Exception as e:
            self.logger.error(f"Error collecting news data: {str(e)}")
    
    async def _collect_options_flow_data(self):
        """Collect options flow data"""
        
        try:
            flow_data = await self.options_flow_provider.fetch_options_flow_data(
                self.config.symbols_to_track
            )
            
            if flow_data:
                self.storage.store_options_flow(flow_data)
                self.logger.info(f"Collected {len(flow_data)} options flow records")
                
        except Exception as e:
            self.logger.error(f"Error collecting options flow data: {str(e)}")
    
    def get_sentiment_signals(self, symbols: List[str], hours_back: int = 24) -> Dict[str, Dict[str, Any]]:
        """Get combined sentiment signals for symbols"""
        
        signals = {}
        
        # Get news sentiment
        news_sentiment = self.storage.get_recent_news_sentiment(symbols, hours_back)
        
        # Get options flow sentiment
        # This would fetch from options_flow table and analyze
        
        for symbol in symbols:
            signal_data = {
                'news_sentiment': news_sentiment.get(symbol, {}),
                'options_flow_sentiment': {},  # Would be populated from options analysis
                'combined_sentiment_score': 0.0,
                'signal_strength': 'weak',
                'recommendation': 'neutral'
            }
            
            # Calculate combined sentiment
            if symbol in news_sentiment:
                news_score = news_sentiment[symbol]['avg_sentiment']
                signal_data['combined_sentiment_score'] = news_score
                
                # Determine signal strength and recommendation
                if abs(news_score) > 0.5:
                    signal_data['signal_strength'] = 'strong'
                elif abs(news_score) > 0.2:
                    signal_data['signal_strength'] = 'moderate'
                
                if news_score > 0.2:
                    signal_data['recommendation'] = 'bullish'
                elif news_score < -0.2:
                    signal_data['recommendation'] = 'bearish'
            
            signals[symbol] = signal_data
        
        return signals
    
    def generate_alt_data_report(self) -> Dict[str, Any]:
        """Generate comprehensive alternative data report"""
        
        conn = sqlite3.connect(self.storage.db_path)
        
        # News statistics
        news_stats = conn.execute('''
            SELECT COUNT(*) as total_articles,
                   AVG(sentiment_score) as avg_sentiment,
                   AVG(relevance_score) as avg_relevance,
                   COUNT(DISTINCT source) as unique_sources
            FROM news_articles
            WHERE published_at >= datetime('now', '-24 hours')
        ''').fetchone()
        
        # Options flow statistics
        options_stats = conn.execute('''
            SELECT COUNT(*) as total_flows,
                   SUM(volume) as total_volume,
                   AVG(implied_volatility) as avg_iv,
                   SUM(CASE WHEN unusual_activity_flag = 1 THEN 1 ELSE 0 END) as unusual_flows
            FROM options_flow
            WHERE timestamp >= datetime('now', '-24 hours')
        ''').fetchone()
        
        conn.close()
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'news_analytics': {
                'total_articles': news_stats[0] if news_stats else 0,
                'avg_sentiment': news_stats[1] if news_stats else 0,
                'avg_relevance': news_stats[2] if news_stats else 0,
                'unique_sources': news_stats[3] if news_stats else 0
            },
            'options_flow_analytics': {
                'total_flows': options_stats[0] if options_stats else 0,
                'total_volume': options_stats[1] if options_stats else 0,
                'avg_implied_volatility': options_stats[2] if options_stats else 0,
                'unusual_activity_count': options_stats[3] if options_stats else 0
            },
            'data_quality_score': self._calculate_data_quality_score(),
            'coverage_symbols': self.config.symbols_to_track,
            'active_data_types': [dt.value for dt in self.config.update_intervals.keys()]
        }
    
    def _calculate_data_quality_score(self) -> float:
        """Calculate overall data quality score"""
        
        # Simplified quality scoring
        # In practice, this would be more sophisticated
        
        base_score = 0.7  # Base quality
        
        # News coverage boost
        if len(self.config.news_sources) > 3:
            base_score += 0.1
        
        # API availability boost
        if len(self.config.api_keys) > 2:
            base_score += 0.1
        
        # Update frequency boost
        if any(interval < 300 for interval in self.config.update_intervals.values()):
            base_score += 0.1
        
        return min(base_score, 1.0)


def main():
    """Example usage of Alternative Data Integration"""
    
    print("Alternative Data Integration - Phase 4")
    print("=" * 60)
    
    try:
        # Configure alternative data collection
        alt_data_config = AltDataConfig(
            news_sources=['newsapi', 'rss_feeds'],
            social_media_platforms=['twitter'],  # Would need API access
            api_keys={
                # 'newsapi_key': 'your_newsapi_key_here',
                # 'twitter_bearer_token': 'your_twitter_token_here'
            },
            update_intervals={
                AltDataType.NEWS_SENTIMENT: 300,    # 5 minutes
                AltDataType.OPTIONS_FLOW: 180,      # 3 minutes
            },
            symbols_to_track=['SPY', 'QQQ', 'AAPL', 'TSLA', 'NVDA'],
            sentiment_threshold=0.1,
            storage_path="alternative_data.db",
            max_history_days=30
        )
        
        # Initialize alternative data manager
        alt_data_manager = AlternativeDataManager(alt_data_config)
        
        print("✅ Alternative data system initialized")
        print(f"📊 Tracking symbols: {alt_data_config.symbols_to_track}")
        print(f"📰 News sources: {alt_data_config.news_sources}")
        print(f"💾 Storage path: {alt_data_config.storage_path}")
        
        # Test sentiment analysis
        print(f"\n🧠 Testing sentiment analysis...")
        sample_text = "AAPL stock surges on strong earnings beat, analyst upgrades price target"
        sentiment_results = alt_data_manager.sentiment_analyzer.analyze_text_sentiment(sample_text)
        
        print(f"📝 Sample text: {sample_text}")
        print(f"📊 Sentiment analysis:")
        print(f"   • Combined sentiment: {sentiment_results['combined_sentiment']:.3f}")
        print(f"   • Sentiment label: {sentiment_results['sentiment_label'].name}")
        print(f"   • TextBlob polarity: {sentiment_results['textblob_polarity']:.3f}")
        print(f"   • VADER compound: {sentiment_results['vader_compound']:.3f}")
        
        # Test keyword extraction
        keywords = alt_data_manager.sentiment_analyzer.extract_financial_keywords(sample_text)
        print(f"🔍 Financial keywords:")
        print(f"   • Bullish keywords: {keywords['bullish_keywords']}")
        print(f"   • Tickers mentioned: {keywords['tickers_mentioned']}")
        print(f"   • Keyword sentiment: {keywords['keyword_sentiment_score']:.3f}")
        
        # Collect sample news data
        print(f"\n📰 Collecting sample news data...")
        news_articles = asyncio.run(
            alt_data_manager.news_provider.fetch_news_articles(['AAPL'], hours_back=24)
        )
        
        print(f"✅ Collected {len(news_articles)} news articles")
        
        if news_articles:
            # Display sample articles
            for i, article in enumerate(news_articles[:3]):
                print(f"\n📄 Article {i+1}:")
                print(f"   • Title: {article.title[:80]}...")
                print(f"   • Source: {article.source}")
                print(f"   • Sentiment: {article.sentiment_score:.3f}")
                print(f"   • Relevance: {article.relevance_score:.3f}")
            
            # Store articles
            alt_data_manager.storage.store_news_articles(news_articles)
            print(f"💾 Articles stored in database")
        
        # Collect sample options flow data
        print(f"\n📈 Collecting sample options flow data...")
        options_flow = asyncio.run(
            alt_data_manager.options_flow_provider.fetch_options_flow_data(['SPY', 'QQQ'])
        )
        
        print(f"✅ Collected {len(options_flow)} options flow records")
        
        if options_flow:
            # Analyze options flow sentiment
            flow_analysis = alt_data_manager.options_flow_provider.analyze_options_flow_sentiment(options_flow)
            
            print(f"📊 Options Flow Analysis:")
            for symbol, analysis in flow_analysis.items():
                print(f"   • {symbol}:")
                print(f"     - Total volume: {analysis['total_volume']:,}")
                print(f"     - Put/call ratio: {analysis['put_call_ratio']:.2f}")
                print(f"     - Sentiment score: {analysis['sentiment_score']:.3f}")
                print(f"     - Unusual activity: {analysis['unusual_activity_rate']:.1%}")
            
            # Store options flow data
            alt_data_manager.storage.store_options_flow(options_flow)
            print(f"💾 Options flow data stored")
        
        # Generate sentiment signals
        print(f"\n🎯 Generating sentiment signals...")
        signals = alt_data_manager.get_sentiment_signals(alt_data_config.symbols_to_track, 24)
        
        print(f"📊 Sentiment Signals:")
        for symbol, signal in signals.items():
            if signal['news_sentiment']:
                print(f"   • {symbol}:")
                print(f"     - News sentiment: {signal['combined_sentiment_score']:.3f}")
                print(f"     - Signal strength: {signal['signal_strength']}")
                print(f"     - Recommendation: {signal['recommendation']}")
        
        # Generate comprehensive report
        print(f"\n📋 Generating alternative data report...")
        report = alt_data_manager.generate_alt_data_report()
        
        print(f"✅ Alternative Data Report Generated:")
        print(f"📰 News Analytics:")
        print(f"   • Total articles: {report['news_analytics']['total_articles']}")
        print(f"   • Average sentiment: {report['news_analytics']['avg_sentiment']:.3f}")
        print(f"   • Unique sources: {report['news_analytics']['unique_sources']}")
        
        print(f"📈 Options Flow Analytics:")
        print(f"   • Total flows: {report['options_flow_analytics']['total_flows']}")
        print(f"   • Total volume: {report['options_flow_analytics']['total_volume']:,}")
        print(f"   • Unusual activity: {report['options_flow_analytics']['unusual_activity_count']}")
        
        print(f"🎯 Data Quality Score: {report['data_quality_score']:.2f}")
        
        # Save report
        report_path = "alternative_data_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n🎉 Alternative data integration demonstration complete!")
        print(f"📁 Generated files:")
        print(f"   • {alt_data_config.storage_path} (database)")
        print(f"   • {report_path} (report)")
        
        print(f"\n💡 Next steps:")
        print(f"   • Configure real API keys for live data")
        print(f"   • Start continuous data collection")
        print(f"   • Integrate with risk management system")
        
    except Exception as e:
        logger.error(f"Error in alternative data integration: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()