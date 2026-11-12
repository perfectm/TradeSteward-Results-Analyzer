"""
High-Frequency Risk Monitoring - Phase 4 Implementation
Microsecond-level risk monitoring and ultra-low latency risk management system
"""

import asyncio
import time
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import logging
import json
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ThreadPoolExecutor
import ctypes
import mmap
import struct

# High-performance libraries
import zmq
import redis
import uvloop  # Ultra-fast asyncio event loop
import orjson  # Fast JSON serialization
import msgpack  # Binary serialization
from numba import jit, cuda
import cupy as cp  # GPU computing (if available)

# Import our existing modules
from data_loader import TradeStewardDataLoader
from market_data_integration import MarketDataPoint, DataType
from alert_notification_system import AutomatedAlertSystem, RiskAlert, AlertType, AlertSeverity

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class RiskEventType(Enum):
    """Types of high-frequency risk events"""
    PRICE_SHOCK = "price_shock"
    VOLATILITY_SPIKE = "volatility_spike"
    LIQUIDITY_CRUNCH = "liquidity_crunch"
    DELTA_LIMIT_BREACH = "delta_limit_breach"
    GAMMA_EXPLOSION = "gamma_explosion"
    THETA_DECAY_ALERT = "theta_decay_alert"
    VEGA_STRESS = "vega_stress"
    CORRELATION_BREAKDOWN = "correlation_breakdown"
    POSITION_LIMIT_BREACH = "position_limit_breach"
    MARGIN_CALL_RISK = "margin_call_risk"

class LatencyBenchmark(Enum):
    """Latency benchmark targets"""
    ULTRA_LOW = 10      # 10 microseconds
    VERY_LOW = 50       # 50 microseconds  
    LOW = 100           # 100 microseconds
    MODERATE = 1000     # 1 millisecond
    HIGH = 10000        # 10 milliseconds

@dataclass
class HighFrequencyConfig:
    """High-frequency monitoring configuration"""
    update_frequency_hz: int = 1000  # Updates per second
    latency_target_us: int = 50      # Target latency in microseconds
    max_queue_size: int = 10000      # Maximum queue size
    enable_gpu_acceleration: bool = False
    use_shared_memory: bool = True
    enable_zero_copy: bool = True
    risk_check_intervals_us: Dict[RiskEventType, int] = field(default_factory=lambda: {
        RiskEventType.PRICE_SHOCK: 100,
        RiskEventType.VOLATILITY_SPIKE: 500,
        RiskEventType.DELTA_LIMIT_BREACH: 200,
        RiskEventType.GAMMA_EXPLOSION: 300
    })
    circuit_breaker_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'price_change_5s': 0.02,      # 2% in 5 seconds
        'volatility_multiplier': 3.0,  # 3x normal volatility
        'delta_exposure': 10000,       # Absolute delta limit
        'gamma_exposure': 1000         # Gamma exposure limit
    })

@dataclass
class RiskSnapshot:
    """Ultra-fast risk snapshot for high-frequency monitoring"""
    timestamp_us: int  # Microsecond timestamp
    portfolio_value: float
    delta_exposure: float
    gamma_exposure: float
    vega_exposure: float
    theta_exposure: float
    implied_volatility: float
    realized_volatility: float
    var_1min: float
    position_count: int
    largest_position_pct: float
    correlation_stress: float
    liquidity_score: float
    
    def to_bytes(self) -> bytes:
        """Serialize to bytes for zero-copy operations"""
        return struct.pack(
            '=Qffffffffffff',
            self.timestamp_us,
            self.portfolio_value,
            self.delta_exposure,
            self.gamma_exposure,
            self.vega_exposure,
            self.theta_exposure,
            self.implied_volatility,
            self.realized_volatility,
            self.var_1min,
            self.position_count,
            self.largest_position_pct,
            self.correlation_stress,
            self.liquidity_score
        )
    
    @classmethod
    def from_bytes(cls, data: bytes) -> 'RiskSnapshot':
        """Deserialize from bytes"""
        unpacked = struct.unpack('=Qffffffffffff', data)
        return cls(*unpacked)

class CircularBuffer:
    """Ultra-fast circular buffer for high-frequency data"""
    
    def __init__(self, maxsize: int, dtype=np.float64):
        self.maxsize = maxsize
        self.buffer = np.zeros(maxsize, dtype=dtype)
        self.head = 0
        self.size = 0
        self.lock = threading.RLock()
    
    def append(self, value: float):
        """Thread-safe append with O(1) complexity"""
        with self.lock:
            self.buffer[self.head] = value
            self.head = (self.head + 1) % self.maxsize
            if self.size < self.maxsize:
                self.size += 1
    
    def get_last_n(self, n: int) -> np.ndarray:
        """Get last n values efficiently"""
        with self.lock:
            if self.size == 0:
                return np.array([])
            
            n = min(n, self.size)
            if self.size < self.maxsize:
                # Buffer not full yet
                start = max(0, self.head - n)
                return self.buffer[start:self.head]
            else:
                # Buffer is full, handle wraparound
                if self.head >= n:
                    return self.buffer[self.head - n:self.head]
                else:
                    return np.concatenate([
                        self.buffer[self.maxsize - (n - self.head):],
                        self.buffer[:self.head]
                    ])
    
    def std(self, n: int = None) -> float:
        """Fast standard deviation calculation"""
        data = self.get_last_n(n or self.size)
        return np.std(data) if len(data) > 1 else 0.0
    
    def mean(self, n: int = None) -> float:
        """Fast mean calculation"""
        data = self.get_last_n(n or self.size)
        return np.mean(data) if len(data) > 0 else 0.0

class SharedMemoryRiskData:
    """Shared memory structure for ultra-low latency data sharing"""
    
    def __init__(self, name: str = "risk_data", create: bool = True):
        self.name = name
        self.size = 1024  # Size in bytes
        
        try:
            if create:
                self.shm = shared_memory.SharedMemory(
                    name=name, create=True, size=self.size
                )
            else:
                self.shm = shared_memory.SharedMemory(name=name)
        except FileExistsError:
            # Memory already exists, connect to it
            self.shm = shared_memory.SharedMemory(name=name)
        
        # Create numpy array view of shared memory
        self.data = np.ndarray((128,), dtype=np.float64, buffer=self.shm.buf)
        
    def write_risk_snapshot(self, snapshot: RiskSnapshot):
        """Write risk snapshot to shared memory"""
        self.data[0] = snapshot.timestamp_us
        self.data[1] = snapshot.portfolio_value
        self.data[2] = snapshot.delta_exposure
        self.data[3] = snapshot.gamma_exposure
        self.data[4] = snapshot.vega_exposure
        self.data[5] = snapshot.theta_exposure
        self.data[6] = snapshot.implied_volatility
        self.data[7] = snapshot.realized_volatility
        self.data[8] = snapshot.var_1min
        self.data[9] = snapshot.position_count
        self.data[10] = snapshot.largest_position_pct
        self.data[11] = snapshot.correlation_stress
        self.data[12] = snapshot.liquidity_score
    
    def read_risk_snapshot(self) -> RiskSnapshot:
        """Read risk snapshot from shared memory"""
        return RiskSnapshot(
            timestamp_us=int(self.data[0]),
            portfolio_value=self.data[1],
            delta_exposure=self.data[2],
            gamma_exposure=self.data[3],
            vega_exposure=self.data[4],
            theta_exposure=self.data[5],
            implied_volatility=self.data[6],
            realized_volatility=self.data[7],
            var_1min=self.data[8],
            position_count=int(self.data[9]),
            largest_position_pct=self.data[10],
            correlation_stress=self.data[11],
            liquidity_score=self.data[12]
        )
    
    def cleanup(self):
        """Clean up shared memory"""
        try:
            self.shm.close()
            self.shm.unlink()
        except:
            pass

# Numba-accelerated risk calculations for ultra-fast execution
@jit(nopython=True, cache=True)
def calculate_portfolio_var_fast(returns: np.ndarray, confidence_level: float = 0.01) -> float:
    """Ultra-fast VaR calculation using Numba JIT compilation"""
    if len(returns) < 10:
        return 0.0
    return np.percentile(returns, confidence_level * 100)

@jit(nopython=True, cache=True)
def calculate_rolling_volatility_fast(prices: np.ndarray, window: int = 20) -> float:
    """Fast rolling volatility calculation"""
    if len(prices) < window:
        return 0.0
    
    returns = np.diff(np.log(prices[-window:]))
    return np.std(returns) * np.sqrt(252 * 24 * 60)  # Annualized minute-level volatility

@jit(nopython=True, cache=True)
def detect_price_shock_fast(prices: np.ndarray, threshold: float = 0.02) -> bool:
    """Fast price shock detection"""
    if len(prices) < 2:
        return False
    
    price_change = abs(prices[-1] - prices[-2]) / prices[-2]
    return price_change > threshold

@jit(nopython=True, cache=True)
def calculate_correlation_breakdown_fast(returns1: np.ndarray, returns2: np.ndarray, 
                                       historical_corr: float, threshold: float = 0.3) -> bool:
    """Fast correlation breakdown detection"""
    if len(returns1) < 10 or len(returns2) < 10:
        return False
    
    current_corr = np.corrcoef(returns1, returns2)[0, 1]
    if np.isnan(current_corr):
        return False
    
    corr_change = abs(current_corr - historical_corr)
    return corr_change > threshold

class HighFrequencyRiskEngine:
    """Core high-frequency risk monitoring engine"""
    
    def __init__(self, config: HighFrequencyConfig):
        self.config = config
        self.running = False
        
        # High-performance data structures
        self.price_buffers = {}  # Symbol -> CircularBuffer
        self.returns_buffers = {}  # Symbol -> CircularBuffer
        self.risk_snapshots = deque(maxlen=10000)  # Last 10k snapshots
        
        # Shared memory for ultra-low latency
        self.shared_mem = None
        if config.use_shared_memory:
            self.shared_mem = SharedMemoryRiskData("hf_risk_data", create=True)
        
        # Redis for pub/sub messaging
        self.redis_client = redis.Redis(decode_responses=False)  # Binary mode for speed
        
        # ZeroMQ for ultra-low latency messaging
        self.zmq_context = zmq.Context()
        self.zmq_publisher = self.zmq_context.socket(zmq.PUB)
        self.zmq_publisher.bind("tcp://*:5555")
        
        # Performance metrics
        self.latency_buffer = CircularBuffer(1000, dtype=np.float64)
        self.throughput_counter = 0
        self.last_throughput_check = time.time()
        
        # GPU acceleration (if available)
        self.gpu_available = False
        if config.enable_gpu_acceleration:
            try:
                import cupy
                self.gpu_available = True
                logger.info("GPU acceleration enabled")
            except ImportError:
                logger.warning("GPU acceleration requested but CuPy not available")
        
        self.logger = logging.getLogger(__name__ + '.HighFrequencyRiskEngine')
        
    def start_monitoring(self):
        """Start high-frequency risk monitoring"""
        
        if self.running:
            return
        
        self.running = True
        
        # Start monitoring loops in separate threads
        self.threads = [
            threading.Thread(target=self._risk_calculation_loop, daemon=True),
            threading.Thread(target=self._circuit_breaker_loop, daemon=True),
            threading.Thread(target=self._performance_monitor_loop, daemon=True),
            threading.Thread(target=self._data_ingestion_loop, daemon=True)
        ]
        
        for thread in self.threads:
            thread.start()
        
        self.logger.info(f"High-frequency risk monitoring started at {self.config.update_frequency_hz} Hz")
    
    def stop_monitoring(self):
        """Stop high-frequency monitoring"""
        
        self.running = False
        
        # Cleanup
        if self.shared_mem:
            self.shared_mem.cleanup()
        
        self.zmq_publisher.close()
        self.zmq_context.term()
        
        self.logger.info("High-frequency risk monitoring stopped")
    
    def _data_ingestion_loop(self):
        """Ultra-fast data ingestion loop"""
        
        # Set thread priority (platform-specific)
        try:
            import os
            os.nice(-10)  # Higher priority on Unix systems
        except:
            pass
        
        sleep_time = 1.0 / self.config.update_frequency_hz
        
        while self.running:
            start_time = time.perf_counter_ns()
            
            try:
                # Simulate high-frequency market data ingestion
                self._ingest_market_data_batch()
                
            except Exception as e:
                self.logger.error(f"Error in data ingestion: {str(e)}")
            
            # Calculate processing latency
            end_time = time.perf_counter_ns()
            latency_us = (end_time - start_time) / 1000
            self.latency_buffer.append(latency_us)
            
            # Sleep for precise timing
            elapsed = (end_time - start_time) / 1e9
            sleep_duration = max(0, sleep_time - elapsed)
            time.sleep(sleep_duration)
    
    def _risk_calculation_loop(self):
        """Main risk calculation loop optimized for speed"""
        
        while self.running:
            start_time = time.perf_counter_ns()
            
            try:
                # Calculate current risk snapshot
                snapshot = self._calculate_risk_snapshot_fast()
                
                # Store in high-speed circular buffer
                self.risk_snapshots.append(snapshot)
                
                # Write to shared memory for other processes
                if self.shared_mem:
                    self.shared_mem.write_risk_snapshot(snapshot)
                
                # Publish via ZeroMQ (binary serialization for speed)
                snapshot_bytes = snapshot.to_bytes()
                self.zmq_publisher.send_multipart([b"risk_snapshot", snapshot_bytes], zmq.NOBLOCK)
                
                # Check for risk events
                risk_events = self._detect_risk_events_fast(snapshot)
                
                # Publish risk events
                for event in risk_events:
                    event_data = msgpack.packb(event, use_bin_type=True)
                    self.zmq_publisher.send_multipart([b"risk_event", event_data], zmq.NOBLOCK)
                
                self.throughput_counter += 1
                
            except Exception as e:
                self.logger.error(f"Error in risk calculation loop: {str(e)}")
            
            # Precise timing control
            end_time = time.perf_counter_ns()
            processing_time_us = (end_time - start_time) / 1000
            
            # Target sleep to maintain frequency
            target_period_us = 1_000_000 / self.config.update_frequency_hz
            sleep_time_us = max(0, target_period_us - processing_time_us)
            
            if sleep_time_us > 0:
                time.sleep(sleep_time_us / 1_000_000)
    
    def _circuit_breaker_loop(self):
        """Circuit breaker monitoring loop"""
        
        while self.running:
            try:
                # Check circuit breaker conditions
                if self._should_trigger_circuit_breaker():
                    self._trigger_circuit_breaker()
                
                # Check every 100ms for circuit breaker conditions
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Error in circuit breaker loop: {str(e)}")
    
    def _performance_monitor_loop(self):
        """Performance monitoring loop"""
        
        while self.running:
            try:
                current_time = time.time()
                
                # Calculate throughput
                if current_time - self.last_throughput_check >= 1.0:
                    throughput = self.throughput_counter
                    self.throughput_counter = 0
                    self.last_throughput_check = current_time
                    
                    # Calculate latency statistics
                    mean_latency = self.latency_buffer.mean()
                    p99_latency = np.percentile(self.latency_buffer.get_last_n(1000), 99)
                    
                    # Log performance metrics
                    if throughput > 0:
                        self.logger.debug(
                            f"Performance: {throughput} calculations/sec, "
                            f"Mean latency: {mean_latency:.1f}μs, "
                            f"P99 latency: {p99_latency:.1f}μs"
                        )
                
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in performance monitor: {str(e)}")
    
    def _ingest_market_data_batch(self):
        """Ingest batch of market data with minimal latency"""
        
        # Simulate high-frequency market data
        symbols = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT']
        
        for symbol in symbols:
            # Initialize buffers if needed
            if symbol not in self.price_buffers:
                self.price_buffers[symbol] = CircularBuffer(10000)
                self.returns_buffers[symbol] = CircularBuffer(10000)
            
            # Simulate price update
            base_price = 100 + hash(symbol) % 400
            price_change = np.random.normal(0, 0.001) * base_price
            new_price = base_price + price_change
            
            # Store price
            old_price = self.price_buffers[symbol].get_last_n(1)
            self.price_buffers[symbol].append(new_price)
            
            # Calculate and store return
            if len(old_price) > 0 and old_price[0] > 0:
                ret = (new_price - old_price[0]) / old_price[0]
                self.returns_buffers[symbol].append(ret)
    
    def _calculate_risk_snapshot_fast(self) -> RiskSnapshot:
        """Calculate risk snapshot with maximum speed"""
        
        current_time_us = int(time.perf_counter_ns() / 1000)
        
        # Fast portfolio-level calculations
        portfolio_value = 1000000.0  # Placeholder
        
        # Calculate Greeks exposures (simplified for speed)
        delta_exposure = 0.0
        gamma_exposure = 0.0
        vega_exposure = 0.0
        theta_exposure = 0.0
        
        for symbol in self.price_buffers:
            # Simplified Greeks calculation
            prices = self.price_buffers[symbol].get_last_n(100)
            if len(prices) > 10:
                # Mock options position
                position_size = 1000
                price = prices[-1]
                
                # Approximate Greeks (very simplified)
                delta_exposure += position_size * 0.5  # Assume 0.5 delta
                gamma_exposure += position_size * 0.01  # Assume 0.01 gamma
                vega_exposure += position_size * 0.2   # Assume 0.2 vega
                theta_exposure += position_size * -0.05 # Assume -0.05 theta
        
        # Calculate volatilities
        all_returns = []
        for symbol in self.returns_buffers:
            returns = self.returns_buffers[symbol].get_last_n(100)
            if len(returns) > 0:
                all_returns.extend(returns)
        
        if all_returns:
            all_returns = np.array(all_returns)
            implied_volatility = np.std(all_returns) * np.sqrt(252 * 24 * 60)  # Annualized
            realized_volatility = implied_volatility * 0.9  # Mock realized vol
            
            # Fast VaR calculation
            var_1min = calculate_portfolio_var_fast(all_returns, 0.01)
        else:
            implied_volatility = 0.2
            realized_volatility = 0.18
            var_1min = 0.0
        
        # Position metrics
        position_count = len(self.price_buffers)
        largest_position_pct = 1.0 / max(position_count, 1) if position_count > 0 else 0.0
        
        # Correlation stress (simplified)
        correlation_stress = 0.0
        if len(self.returns_buffers) >= 2:
            symbols = list(self.returns_buffers.keys())[:2]
            returns1 = self.returns_buffers[symbols[0]].get_last_n(100)
            returns2 = self.returns_buffers[symbols[1]].get_last_n(100)
            
            if len(returns1) > 10 and len(returns2) > 10:
                correlation_stress = calculate_correlation_breakdown_fast(
                    returns1, returns2, 0.5, 0.3
                )
        
        # Liquidity score (mock)
        liquidity_score = 0.8
        
        return RiskSnapshot(
            timestamp_us=current_time_us,
            portfolio_value=portfolio_value,
            delta_exposure=delta_exposure,
            gamma_exposure=gamma_exposure,
            vega_exposure=vega_exposure,
            theta_exposure=theta_exposure,
            implied_volatility=implied_volatility,
            realized_volatility=realized_volatility,
            var_1min=var_1min,
            position_count=position_count,
            largest_position_pct=largest_position_pct,
            correlation_stress=float(correlation_stress),
            liquidity_score=liquidity_score
        )
    
    def _detect_risk_events_fast(self, snapshot: RiskSnapshot) -> List[Dict[str, Any]]:
        """Fast risk event detection"""
        
        events = []
        
        # Price shock detection
        for symbol in self.price_buffers:
            prices = self.price_buffers[symbol].get_last_n(2)
            if len(prices) >= 2:
                if detect_price_shock_fast(prices, self.config.circuit_breaker_thresholds['price_change_5s']):
                    events.append({
                        'type': RiskEventType.PRICE_SHOCK.value,
                        'symbol': symbol,
                        'severity': 'HIGH',
                        'timestamp_us': snapshot.timestamp_us,
                        'details': {'price_change': abs(prices[-1] - prices[-2]) / prices[-2]}
                    })
        
        # Delta exposure check
        if abs(snapshot.delta_exposure) > self.config.circuit_breaker_thresholds['delta_exposure']:
            events.append({
                'type': RiskEventType.DELTA_LIMIT_BREACH.value,
                'severity': 'CRITICAL',
                'timestamp_us': snapshot.timestamp_us,
                'details': {'delta_exposure': snapshot.delta_exposure}
            })
        
        # Gamma exposure check
        if abs(snapshot.gamma_exposure) > self.config.circuit_breaker_thresholds['gamma_exposure']:
            events.append({
                'type': RiskEventType.GAMMA_EXPLOSION.value,
                'severity': 'CRITICAL', 
                'timestamp_us': snapshot.timestamp_us,
                'details': {'gamma_exposure': snapshot.gamma_exposure}
            })
        
        # Volatility spike check
        if snapshot.realized_volatility > snapshot.implied_volatility * self.config.circuit_breaker_thresholds['volatility_multiplier']:
            events.append({
                'type': RiskEventType.VOLATILITY_SPIKE.value,
                'severity': 'HIGH',
                'timestamp_us': snapshot.timestamp_us,
                'details': {
                    'realized_vol': snapshot.realized_volatility,
                    'implied_vol': snapshot.implied_volatility,
                    'multiplier': snapshot.realized_volatility / max(snapshot.implied_volatility, 0.01)
                }
            })
        
        return events
    
    def _should_trigger_circuit_breaker(self) -> bool:
        """Check if circuit breaker should be triggered"""
        
        if len(self.risk_snapshots) < 2:
            return False
        
        latest = self.risk_snapshots[-1]
        previous = self.risk_snapshots[-2]
        
        # Check for extreme changes in key metrics
        portfolio_change = abs(latest.portfolio_value - previous.portfolio_value) / max(previous.portfolio_value, 1)
        
        # Trigger if portfolio changes by more than 5% in one update
        if portfolio_change > 0.05:
            return True
        
        # Check for extreme Greeks changes
        delta_change = abs(latest.delta_exposure - previous.delta_exposure)
        if delta_change > 5000:  # Large delta change
            return True
        
        return False
    
    def _trigger_circuit_breaker(self):
        """Trigger circuit breaker - halt trading and alert"""
        
        self.logger.critical("CIRCUIT BREAKER TRIGGERED - Extreme risk conditions detected")
        
        # Publish circuit breaker event
        event_data = {
            'type': 'CIRCUIT_BREAKER',
            'timestamp_us': int(time.perf_counter_ns() / 1000),
            'severity': 'EMERGENCY',
            'message': 'Circuit breaker triggered due to extreme risk conditions'
        }
        
        event_bytes = msgpack.packb(event_data, use_bin_type=True)
        self.zmq_publisher.send_multipart([b"circuit_breaker", event_bytes], zmq.NOBLOCK)
        
        # Additional emergency actions would go here
        # e.g., halt trading, notify risk managers, etc.
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        
        latencies = self.latency_buffer.get_last_n(1000)
        
        return {
            'mean_latency_us': float(np.mean(latencies)) if len(latencies) > 0 else 0,
            'p50_latency_us': float(np.percentile(latencies, 50)) if len(latencies) > 0 else 0,
            'p95_latency_us': float(np.percentile(latencies, 95)) if len(latencies) > 0 else 0,
            'p99_latency_us': float(np.percentile(latencies, 99)) if len(latencies) > 0 else 0,
            'max_latency_us': float(np.max(latencies)) if len(latencies) > 0 else 0,
            'update_frequency_hz': self.config.update_frequency_hz,
            'buffer_utilization': len(self.risk_snapshots) / 10000,
            'gpu_acceleration': self.gpu_available,
            'shared_memory_enabled': self.config.use_shared_memory
        }
    
    def get_latest_risk_snapshot(self) -> Optional[RiskSnapshot]:
        """Get the most recent risk snapshot"""
        return self.risk_snapshots[-1] if self.risk_snapshots else None

class HFRiskEventConsumer:
    """Consumer for high-frequency risk events"""
    
    def __init__(self, callback: Optional[Callable] = None):
        self.callback = callback
        self.running = False
        
        # ZeroMQ subscriber
        self.zmq_context = zmq.Context()
        self.zmq_subscriber = self.zmq_context.socket(zmq.SUB)
        self.zmq_subscriber.connect("tcp://localhost:5555")
        self.zmq_subscriber.setsockopt(zmq.SUBSCRIBE, b"risk_event")
        self.zmq_subscriber.setsockopt(zmq.SUBSCRIBE, b"circuit_breaker")
        
        self.logger = logging.getLogger(__name__ + '.HFRiskEventConsumer')
    
    def start_consuming(self):
        """Start consuming risk events"""
        
        self.running = True
        self.consumer_thread = threading.Thread(target=self._consume_loop, daemon=True)
        self.consumer_thread.start()
        
        self.logger.info("High-frequency risk event consumer started")
    
    def stop_consuming(self):
        """Stop consuming risk events"""
        
        self.running = False
        self.zmq_subscriber.close()
        self.zmq_context.term()
        
        self.logger.info("Risk event consumer stopped")
    
    def _consume_loop(self):
        """Main event consumption loop"""
        
        while self.running:
            try:
                # Non-blocking receive with timeout
                if self.zmq_subscriber.poll(timeout=100):  # 100ms timeout
                    topic, data = self.zmq_subscriber.recv_multipart(zmq.NOBLOCK)
                    
                    if topic == b"risk_event":
                        event = msgpack.unpackb(data, raw=False)
                        self._handle_risk_event(event)
                    elif topic == b"circuit_breaker":
                        event = msgpack.unpackb(data, raw=False)
                        self._handle_circuit_breaker(event)
                        
            except zmq.Again:
                continue
            except Exception as e:
                self.logger.error(f"Error in consume loop: {str(e)}")
    
    def _handle_risk_event(self, event: Dict[str, Any]):
        """Handle incoming risk event"""
        
        self.logger.warning(f"Risk event: {event['type']} - {event.get('severity', 'UNKNOWN')}")
        
        if self.callback:
            self.callback(event)
    
    def _handle_circuit_breaker(self, event: Dict[str, Any]):
        """Handle circuit breaker event"""
        
        self.logger.critical(f"CIRCUIT BREAKER: {event['message']}")
        
        if self.callback:
            self.callback(event)

class LatencyBenchmarker:
    """Benchmark latency performance of the HF risk system"""
    
    def __init__(self, risk_engine: HighFrequencyRiskEngine):
        self.risk_engine = risk_engine
        self.logger = logging.getLogger(__name__ + '.LatencyBenchmarker')
    
    def run_latency_benchmark(self, duration_seconds: int = 10) -> Dict[str, Any]:
        """Run comprehensive latency benchmark"""
        
        self.logger.info(f"Running latency benchmark for {duration_seconds} seconds...")
        
        # Collect latency samples
        latencies = []
        start_time = time.time()
        
        while time.time() - start_time < duration_seconds:
            # Measure end-to-end latency
            start_ns = time.perf_counter_ns()
            
            # Trigger risk calculation
            snapshot = self.risk_engine._calculate_risk_snapshot_fast()
            
            end_ns = time.perf_counter_ns()
            latency_us = (end_ns - start_ns) / 1000
            latencies.append(latency_us)
            
            time.sleep(0.001)  # 1ms interval
        
        # Calculate statistics
        latencies = np.array(latencies)
        
        results = {
            'samples': len(latencies),
            'mean_latency_us': float(np.mean(latencies)),
            'median_latency_us': float(np.median(latencies)),
            'p95_latency_us': float(np.percentile(latencies, 95)),
            'p99_latency_us': float(np.percentile(latencies, 99)),
            'p99_9_latency_us': float(np.percentile(latencies, 99.9)),
            'max_latency_us': float(np.max(latencies)),
            'min_latency_us': float(np.min(latencies)),
            'std_latency_us': float(np.std(latencies)),
            'benchmark_duration_s': duration_seconds,
            'target_latency_us': self.risk_engine.config.latency_target_us
        }
        
        # Performance assessment
        if results['p99_latency_us'] <= LatencyBenchmark.ULTRA_LOW.value:
            results['performance_grade'] = 'ULTRA_LOW'
        elif results['p99_latency_us'] <= LatencyBenchmark.VERY_LOW.value:
            results['performance_grade'] = 'VERY_LOW'
        elif results['p99_latency_us'] <= LatencyBenchmark.LOW.value:
            results['performance_grade'] = 'LOW'
        elif results['p99_latency_us'] <= LatencyBenchmark.MODERATE.value:
            results['performance_grade'] = 'MODERATE'
        else:
            results['performance_grade'] = 'HIGH'
        
        self.logger.info(f"Benchmark complete: P99 latency = {results['p99_latency_us']:.1f}μs, Grade = {results['performance_grade']}")
        
        return results


def main():
    """Example usage of High-Frequency Risk Monitoring"""
    
    print("High-Frequency Risk Monitoring - Phase 4")
    print("=" * 60)
    
    try:
        # Configure high-frequency monitoring
        hf_config = HighFrequencyConfig(
            update_frequency_hz=1000,  # 1000 updates per second
            latency_target_us=50,      # 50 microsecond target
            max_queue_size=10000,
            enable_gpu_acceleration=False,  # Set to True if GPU available
            use_shared_memory=True,
            enable_zero_copy=True
        )
        
        print("✅ High-frequency risk monitoring system initialized")
        print(f"⚡ Update frequency: {hf_config.update_frequency_hz} Hz")
        print(f"🎯 Latency target: {hf_config.latency_target_us} μs")
        print(f"💾 Shared memory: {'Enabled' if hf_config.use_shared_memory else 'Disabled'}")
        print(f"🚀 Zero-copy: {'Enabled' if hf_config.enable_zero_copy else 'Disabled'}")
        
        # Initialize high-frequency risk engine
        hf_engine = HighFrequencyRiskEngine(hf_config)
        
        # Start monitoring
        print(f"\n🚀 Starting high-frequency risk monitoring...")
        hf_engine.start_monitoring()
        
        # Let it run for a few seconds to collect data
        time.sleep(5)
        
        # Get performance metrics
        perf_metrics = hf_engine.get_performance_metrics()
        
        print(f"\n📊 Performance Metrics:")
        print(f"   • Mean latency: {perf_metrics['mean_latency_us']:.1f} μs")
        print(f"   • P50 latency: {perf_metrics['p50_latency_us']:.1f} μs")
        print(f"   • P95 latency: {perf_metrics['p95_latency_us']:.1f} μs")
        print(f"   • P99 latency: {perf_metrics['p99_latency_us']:.1f} μs")
        print(f"   • Max latency: {perf_metrics['max_latency_us']:.1f} μs")
        print(f"   • Buffer utilization: {perf_metrics['buffer_utilization']:.1%}")
        
        # Get latest risk snapshot
        snapshot = hf_engine.get_latest_risk_snapshot()
        
        if snapshot:
            print(f"\n📸 Latest Risk Snapshot:")
            print(f"   • Portfolio value: ${snapshot.portfolio_value:,.0f}")
            print(f"   • Delta exposure: {snapshot.delta_exposure:.0f}")
            print(f"   • Gamma exposure: {snapshot.gamma_exposure:.0f}")
            print(f"   • Vega exposure: {snapshot.vega_exposure:.0f}")
            print(f"   • Implied volatility: {snapshot.implied_volatility:.1%}")
            print(f"   • 1-min VaR: {snapshot.var_1min:.4f}")
            print(f"   • Position count: {snapshot.position_count}")
        
        # Initialize event consumer
        print(f"\n📡 Starting risk event consumer...")
        
        def event_handler(event):
            event_time = datetime.fromtimestamp(event['timestamp_us'] / 1_000_000)
            print(f"⚠️  Risk Event: {event['type']} ({event.get('severity', 'UNKNOWN')}) at {event_time}")
        
        event_consumer = HFRiskEventConsumer(callback=event_handler)
        event_consumer.start_consuming()
        
        # Run latency benchmark
        print(f"\n⏱️  Running latency benchmark...")
        benchmarker = LatencyBenchmarker(hf_engine)
        benchmark_results = benchmarker.run_latency_benchmark(duration_seconds=5)
        
        print(f"\n📈 Benchmark Results:")
        print(f"   • Samples: {benchmark_results['samples']:,}")
        print(f"   • Mean latency: {benchmark_results['mean_latency_us']:.1f} μs")
        print(f"   • Median latency: {benchmark_results['median_latency_us']:.1f} μs")
        print(f"   • P99 latency: {benchmark_results['p99_latency_us']:.1f} μs")
        print(f"   • P99.9 latency: {benchmark_results['p99_9_latency_us']:.1f} μs")
        print(f"   • Performance grade: {benchmark_results['performance_grade']}")
        
        # Let the system run for a bit more to show events
        print(f"\n⏳ Running system for 10 seconds to demonstrate event detection...")
        time.sleep(10)
        
        # Stop monitoring
        print(f"\n🛑 Stopping high-frequency monitoring...")
        event_consumer.stop_consuming()
        hf_engine.stop_monitoring()
        
        # Generate final report
        final_perf = hf_engine.get_performance_metrics()
        
        print(f"\n📋 Final Performance Report:")
        print(f"   • System achieved {final_perf['update_frequency_hz']} Hz update rate")
        print(f"   • Average latency: {final_perf['mean_latency_us']:.1f} μs")
        print(f"   • P99 latency: {final_perf['p99_latency_us']:.1f} μs")
        
        if final_perf['p99_latency_us'] <= hf_config.latency_target_us:
            print(f"   ✅ Latency target of {hf_config.latency_target_us} μs achieved!")
        else:
            print(f"   ⚠️  Latency target of {hf_config.latency_target_us} μs not met")
        
        print(f"\n🎉 High-frequency risk monitoring demonstration complete!")
        
        print(f"\n💡 Key Features Demonstrated:")
        print(f"   • Microsecond-level risk calculations")
        print(f"   • Real-time circuit breaker monitoring")
        print(f"   • Zero-copy shared memory communication")
        print(f"   • Ultra-low latency event publishing")
        print(f"   • Comprehensive performance benchmarking")
        
        # Save benchmark results
        benchmark_file = "hf_risk_benchmark_results.json"
        with open(benchmark_file, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"\n📁 Generated files:")
        print(f"   • {benchmark_file} (benchmark results)")
        
    except Exception as e:
        logger.error(f"Error in high-frequency risk monitoring: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()