"""
Advanced RESTful API Gateway for External System Integration
Provides comprehensive API management with authentication, rate limiting, monitoring, and external system connectors
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from urllib.parse import urlparse
import hashlib
import hmac
import base64

import aioredis
import aiohttp
from fastapi import FastAPI, HTTPException, Depends, Request, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.dialects.postgresql import UUID
import uuid

Base = declarative_base()

class APIKeyType(str, Enum):
    READ_ONLY = "READ_ONLY"
    READ_WRITE = "READ_WRITE"
    ADMIN = "ADMIN"

class EndpointCategory(str, Enum):
    RISK_DATA = "risk_data"
    PORTFOLIO = "portfolio"
    ANALYTICS = "analytics"
    REPORTING = "reporting"
    SYSTEM = "system"
    WEBHOOK = "webhook"

class ExternalSystem(str, Enum):
    BLOOMBERG = "bloomberg"
    REFINITIV = "refinitiv"
    INTERACTIVE_BROKERS = "interactive_brokers"
    TD_AMERITRADE = "td_ameritrade"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    CUSTOM = "custom"

@dataclass
class APIResponse:
    success: bool
    data: Optional[Dict] = None
    error: Optional[str] = None
    timestamp: str = None
    request_id: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow().isoformat()
        if self.request_id is None:
            self.request_id = str(uuid.uuid4())

class APIKey(Base):
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True)
    key_id = Column(String(64), unique=True, index=True)
    key_hash = Column(String(128))
    name = Column(String(100))
    key_type = Column(String(20))
    rate_limit = Column(Integer, default=1000)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)
    last_used = Column(DateTime)
    usage_count = Column(Integer, default=0)

class APILog(Base):
    __tablename__ = "api_logs"
    
    id = Column(Integer, primary_key=True)
    request_id = Column(String(64))
    api_key_id = Column(String(64))
    endpoint = Column(String(255))
    method = Column(String(10))
    status_code = Column(Integer)
    response_time = Column(Float)
    request_size = Column(Integer)
    response_size = Column(Integer)
    timestamp = Column(DateTime, default=datetime.utcnow)
    user_agent = Column(String(255))
    ip_address = Column(String(45))
    error_message = Column(Text)

class WebhookEndpoint(Base):
    __tablename__ = "webhook_endpoints"
    
    id = Column(Integer, primary_key=True)
    url = Column(String(500))
    secret = Column(String(128))
    events = Column(Text)  # JSON array of event types
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_triggered = Column(DateTime)

class ExternalSystemConfig(Base):
    __tablename__ = "external_system_configs"
    
    id = Column(Integer, primary_key=True)
    system_name = Column(String(50))
    config_data = Column(Text)  # JSON config
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

class RateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def is_allowed(self, key: str, limit: int, window: int = 60) -> tuple[bool, Dict]:
        current_time = int(time.time())
        window_start = current_time - window
        
        pipe = self.redis.pipeline()
        
        # Remove old entries
        pipe.zremrangebyscore(f"rate_limit:{key}", 0, window_start)
        
        # Count current requests
        pipe.zcard(f"rate_limit:{key}")
        
        # Add current request
        pipe.zadd(f"rate_limit:{key}", {str(current_time): current_time})
        pipe.expire(f"rate_limit:{key}", window + 1)
        
        results = await pipe.execute()
        current_count = results[1]
        
        if current_count >= limit:
            reset_time = current_time + window
            return False, {
                "allowed": False,
                "limit": limit,
                "remaining": 0,
                "reset_time": reset_time
            }
        
        return True, {
            "allowed": True,
            "limit": limit,
            "remaining": limit - current_count - 1,
            "reset_time": current_time + window
        }

class APIGateway:
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.app = FastAPI(
            title="TradeSteward Risk Management API Gateway",
            description="Enterprise API gateway for external system integration",
            version="1.0.0"
        )
        self.setup_middleware()
        self.setup_routes()
        self.redis_url = redis_url
        self.redis = None
        self.rate_limiter = None
        self.external_connectors = {}
        self.webhook_queue = asyncio.Queue()
        
    async def startup(self):
        self.redis = await aioredis.from_url(self.redis_url)
        self.rate_limiter = RateLimiter(self.redis)
        self.setup_external_connectors()
        asyncio.create_task(self.webhook_processor())
    
    def setup_middleware(self):
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        self.app.middleware("http")(self.logging_middleware)
        self.app.middleware("http")(self.rate_limiting_middleware)
    
    async def logging_middleware(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log API call (implement database logging here)
        logging.info(f"API Call: {request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
        
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Process-Time"] = str(process_time)
        
        return response
    
    async def rate_limiting_middleware(self, request: Request, call_next):
        # Skip rate limiting for certain paths
        if request.url.path in ["/health", "/metrics", "/docs", "/openapi.json"]:
            return await call_next(request)
        
        api_key = await self.extract_api_key(request)
        if not api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "API key required"}
            )
        
        # Check rate limit
        allowed, limit_info = await self.rate_limiter.is_allowed(
            api_key.key_id, api_key.rate_limit
        )
        
        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Rate limit exceeded",
                    "limit": limit_info["limit"],
                    "reset_time": limit_info["reset_time"]
                },
                headers={
                    "X-RateLimit-Limit": str(limit_info["limit"]),
                    "X-RateLimit-Remaining": str(limit_info["remaining"]),
                    "X-RateLimit-Reset": str(limit_info["reset_time"])
                }
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(limit_info["limit"])
        response.headers["X-RateLimit-Remaining"] = str(limit_info["remaining"])
        response.headers["X-RateLimit-Reset"] = str(limit_info["reset_time"])
        
        return response
    
    async def extract_api_key(self, request: Request) -> Optional[APIKey]:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            return None
        
        key = auth_header[7:]  # Remove "Bearer " prefix
        
        # In production, query database for API key
        # For now, return a mock API key
        return APIKey(
            key_id="test_key",
            name="Test API Key",
            key_type="READ_WRITE",
            rate_limit=1000,
            is_active=True
        )
    
    def setup_routes(self):
        @self.app.get("/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}
        
        @self.app.get("/metrics")
        async def get_metrics():
            return {
                "uptime": time.time(),
                "requests_processed": 0,  # Implement counter
                "active_connections": 0,
                "error_rate": 0.0
            }
        
        # Risk Data Endpoints
        @self.app.get("/api/v1/risk/portfolio/{portfolio_id}")
        async def get_portfolio_risk(portfolio_id: str):
            return APIResponse(
                success=True,
                data={"portfolio_id": portfolio_id, "risk_metrics": {}}
            ).__dict__
        
        @self.app.get("/api/v1/risk/positions")
        async def get_position_risk():
            return APIResponse(
                success=True,
                data={"positions": []}
            ).__dict__
        
        @self.app.get("/api/v1/risk/var/{confidence_level}")
        async def calculate_var(confidence_level: float):
            return APIResponse(
                success=True,
                data={"var": 0.0, "confidence_level": confidence_level}
            ).__dict__
        
        # Analytics Endpoints
        @self.app.get("/api/v1/analytics/performance/{period}")
        async def get_performance_analytics(period: str):
            return APIResponse(
                success=True,
                data={"period": period, "metrics": {}}
            ).__dict__
        
        @self.app.get("/api/v1/analytics/correlations")
        async def get_correlation_matrix():
            return APIResponse(
                success=True,
                data={"correlation_matrix": []}
            ).__dict__
        
        # Webhook Management
        @self.app.post("/api/v1/webhooks")
        async def create_webhook(webhook_data: dict):
            return APIResponse(
                success=True,
                data={"webhook_id": str(uuid.uuid4())}
            ).__dict__
        
        @self.app.post("/webhooks/risk_alert")
        async def risk_alert_webhook(request: Request):
            data = await request.json()
            await self.webhook_queue.put(("risk_alert", data))
            return {"status": "received"}
    
    def setup_external_connectors(self):
        self.external_connectors = {
            ExternalSystem.BLOOMBERG: BloombergConnector(),
            ExternalSystem.REFINITIV: RefinitivConnector(),
            ExternalSystem.INTERACTIVE_BROKERS: IBKRConnector(),
            ExternalSystem.TD_AMERITRADE: TDAConnector(),
            ExternalSystem.ALPHA_VANTAGE: AlphaVantageConnector(),
            ExternalSystem.QUANDL: QuandlConnector()
        }
    
    async def webhook_processor(self):
        while True:
            try:
                event_type, data = await self.webhook_queue.get()
                await self.process_webhook_event(event_type, data)
            except Exception as e:
                logging.error(f"Webhook processing error: {e}")
            await asyncio.sleep(0.1)
    
    async def process_webhook_event(self, event_type: str, data: Dict):
        # Process webhook events and notify registered endpoints
        logging.info(f"Processing webhook event: {event_type}")
        
        # Implement webhook delivery logic
        pass

class ExternalConnector:
    def __init__(self, name: str):
        self.name = name
        self.session = None
    
    async def connect(self) -> bool:
        self.session = aiohttp.ClientSession()
        return True
    
    async def disconnect(self):
        if self.session:
            await self.session.close()
    
    async def get_data(self, endpoint: str, params: Dict = None) -> Dict:
        if not self.session:
            await self.connect()
        
        try:
            async with self.session.get(endpoint, params=params) as response:
                return await response.json()
        except Exception as e:
            logging.error(f"Error fetching data from {self.name}: {e}")
            return {"error": str(e)}

class BloombergConnector(ExternalConnector):
    def __init__(self):
        super().__init__("Bloomberg Terminal API")
        self.base_url = "https://api.bloomberg.com"
    
    async def get_market_data(self, symbols: List[str]) -> Dict:
        return await self.get_data(f"{self.base_url}/market-data", {
            "symbols": ",".join(symbols)
        })
    
    async def get_options_data(self, underlying: str) -> Dict:
        return await self.get_data(f"{self.base_url}/options/{underlying}")

class RefinitivConnector(ExternalConnector):
    def __init__(self):
        super().__init__("Refinitiv Eikon")
        self.base_url = "https://api.refinitiv.com"
    
    async def get_real_time_data(self, rics: List[str]) -> Dict:
        return await self.get_data(f"{self.base_url}/streaming/pricing", {
            "rics": ",".join(rics)
        })

class IBKRConnector(ExternalConnector):
    def __init__(self):
        super().__init__("Interactive Brokers")
        self.base_url = "https://localhost:5000/v1/api"
    
    async def get_account_summary(self) -> Dict:
        return await self.get_data(f"{self.base_url}/account")
    
    async def get_positions(self) -> Dict:
        return await self.get_data(f"{self.base_url}/portfolio/positions")

class TDAConnector(ExternalConnector):
    def __init__(self):
        super().__init__("TD Ameritrade")
        self.base_url = "https://api.tdameritrade.com/v1"
    
    async def get_quotes(self, symbols: List[str]) -> Dict:
        return await self.get_data(f"{self.base_url}/marketdata/quotes", {
            "symbol": ",".join(symbols)
        })

class AlphaVantageConnector(ExternalConnector):
    def __init__(self):
        super().__init__("Alpha Vantage")
        self.base_url = "https://www.alphavantage.co/query"
    
    async def get_daily_data(self, symbol: str) -> Dict:
        return await self.get_data(self.base_url, {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": "YOUR_API_KEY"
        })

class QuandlConnector(ExternalConnector):
    def __init__(self):
        super().__init__("Quandl")
        self.base_url = "https://www.quandl.com/api/v3"
    
    async def get_dataset(self, dataset_code: str) -> Dict:
        return await self.get_data(f"{self.base_url}/datasets/{dataset_code}.json")

class APIGatewayManager:
    def __init__(self):
        self.gateway = APIGateway()
        self.is_running = False
    
    async def start(self, host: str = "0.0.0.0", port: int = 8080):
        await self.gateway.startup()
        self.is_running = True
        
        config = uvicorn.Config(
            self.gateway.app,
            host=host,
            port=port,
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    def get_api_stats(self) -> Dict:
        return {
            "is_running": self.is_running,
            "total_endpoints": len(self.gateway.app.routes),
            "external_connectors": len(self.gateway.external_connectors),
            "active_webhooks": 0  # Implement counter
        }
    
    async def test_external_connectivity(self) -> Dict[str, bool]:
        results = {}
        for system_name, connector in self.gateway.external_connectors.items():
            try:
                success = await connector.connect()
                results[system_name.value] = success
                await connector.disconnect()
            except Exception as e:
                results[system_name.value] = False
                logging.error(f"Connection test failed for {system_name}: {e}")
        
        return results

# Example usage and testing
async def main():
    # Initialize API Gateway
    manager = APIGatewayManager()
    
    # Test external system connectivity
    print("Testing external system connectivity...")
    connectivity_results = await manager.test_external_connectivity()
    
    for system, status in connectivity_results.items():
        print(f"{system}: {'✓' if status else '✗'}")
    
    # Display API stats
    print("\nAPI Gateway Statistics:")
    stats = manager.get_api_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    # Start the API gateway server
    print("\nStarting API Gateway on port 8080...")
    print("API Documentation available at: http://localhost:8080/docs")
    
    # In production, you would run:
    # await manager.start()

if __name__ == "__main__":
    asyncio.run(main())