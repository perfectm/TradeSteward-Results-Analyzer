"""
Deep Learning Models for Options Pricing and Risk Prediction - Phase 4 Implementation
Advanced neural network architectures for options pricing, volatility forecasting, and risk prediction
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings
import joblib
import json
from pathlib import Path

# Import our existing modules
from data_loader import TradeStewardDataLoader
from market_data_integration import LiveMarketDataManager, MarketDataConfig, DataProvider, DataType

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Deep learning model configuration"""
    model_type: str
    sequence_length: int
    features: List[str]
    target: str
    batch_size: int
    epochs: int
    learning_rate: float
    validation_split: float
    early_stopping_patience: int
    model_params: Dict[str, Any]

class OptionsTimeSeriesDataset(Dataset):
    """PyTorch dataset for options time series data"""
    
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

class BlackScholesNN(nn.Module):
    """Neural network for Black-Scholes options pricing enhancement"""
    
    def __init__(self, input_dim=5, hidden_dims=[64, 32, 16], output_dim=1):
        super(BlackScholesNN, self).__init__()
        
        layers_list = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers_list.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers_list.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers_list)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward(self, x):
        return self.network(x)

class VolatilitySurfaceLSTM(nn.Module):
    """LSTM network for volatility surface prediction"""
    
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, output_dim=1):
        super(VolatilitySurfaceLSTM, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Apply attention mechanism
        lstm_out = lstm_out.permute(1, 0, 2)  # (seq_len, batch_size, hidden_dim)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.permute(1, 0, 2)  # (batch_size, seq_len, hidden_dim)
        
        # Take the last time step
        final_out = attn_out[:, -1, :]
        
        # Final prediction
        output = self.fc(final_out)
        return output

class TransformerRiskModel(nn.Module):
    """Transformer-based risk prediction model"""
    
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=6, output_dim=1):
        super(TransformerRiskModel, self).__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = self._get_positional_encoding(1000, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, output_dim)
        )
        
    def _get_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Project input to d_model dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_encoding = self.positional_encoding[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)
        x = x + pos_encoding.to(x.device)
        
        # Transformer encoding
        transformer_out = self.transformer(x)
        
        # Global average pooling
        pooled = transformer_out.mean(dim=1)
        
        # Output prediction
        output = self.output_head(pooled)
        return output

class DeepOptionsRiskPredictor:
    """Deep learning-based options risk prediction system"""
    
    def __init__(self, data_loader: TradeStewardDataLoader):
        self.data_loader = data_loader
        self.models = {}
        self.scalers = {}
        self.feature_encoders = {}
        self.model_configs = {}
        
        # Set random seeds for reproducibility
        np.random.seed(42)
        tf.random.set_seed(42)
        torch.manual_seed(42)
        
        self.logger = logging.getLogger(__name__ + '.DeepOptionsRiskPredictor')
        
        # Create output directories
        self.model_dir = Path("deep_learning_models")
        self.model_dir.mkdir(exist_ok=True)
        
    def prepare_options_pricing_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """Prepare data for options pricing models"""
        
        # Load trades data
        trades_df = self.data_loader.load_user_trades()
        enhanced_df = self.data_loader.parse_raw_data_json(trades_df)
        clean_df = self.data_loader.clean_and_validate_data(enhanced_df)
        
        # Extract options-specific features
        options_data = []
        
        for _, trade in clean_df.iterrows():
            if 'raw_data_json' in trade and trade['raw_data_json']:
                try:
                    raw_data = json.loads(trade['raw_data_json'])
                    
                    # Extract Black-Scholes inputs
                    features = {
                        'underlying_price': raw_data.get('UnderlyingOpenQuote', 0),
                        'strike_price': raw_data.get('Opt1-Strike', 0),
                        'time_to_expiry': self._calculate_time_to_expiry(
                            trade.get('open_date'), raw_data.get('ExpirationDate')
                        ),
                        'risk_free_rate': 0.02,  # Assume 2% risk-free rate
                        'volatility': raw_data.get('VIXOpenQuote', 20) / 100,
                        'option_type': 1 if raw_data.get('Opt1-Type', '').lower() == 'call' else 0,
                        'moneyness': (raw_data.get('UnderlyingOpenQuote', 0) / 
                                    max(raw_data.get('Opt1-Strike', 1), 1)),
                        'volume': raw_data.get('Opt1-Volume', 0),
                        'open_interest': raw_data.get('Opt1-OpenInterest', 0),
                        'bid_ask_spread': (raw_data.get('Opt1-Ask', 0) - 
                                         raw_data.get('Opt1-Bid', 0)),
                        'option_price': raw_data.get('Opt1-Price', 0),
                        'realized_pnl': trade.get('net_pnl', 0)
                    }
                    
                    # Only include valid options data
                    if (features['strike_price'] > 0 and 
                        features['underlying_price'] > 0 and
                        features['option_price'] > 0):
                        options_data.append(features)
                        
                except Exception as e:
                    continue
        
        if not options_data:
            raise ValueError("No valid options data found for pricing model")
        
        options_df = pd.DataFrame(options_data)
        
        # Prepare feature matrix and targets
        feature_cols = [
            'underlying_price', 'strike_price', 'time_to_expiry',
            'risk_free_rate', 'volatility', 'moneyness',
            'volume', 'open_interest', 'bid_ask_spread'
        ]
        
        X = options_df[feature_cols].values
        y_price = options_df['option_price'].values
        y_pnl = options_df['realized_pnl'].values
        
        return options_df, X, y_price, y_pnl
    
    def prepare_volatility_surface_data(self, sequence_length: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for volatility surface prediction"""
        
        # Load trades and create daily volatility estimates
        trades_df = self.data_loader.load_user_trades()
        enhanced_df = self.data_loader.parse_raw_data_json(trades_df)
        clean_df = self.data_loader.clean_and_validate_data(enhanced_df)
        
        # Create daily series
        daily_series = self.data_loader.create_daily_returns_series(clean_df)
        
        # Calculate rolling volatility and other features
        vol_data = pd.DataFrame(index=daily_series.index)
        vol_data['returns'] = daily_series
        vol_data['volatility'] = daily_series.rolling(20).std() * np.sqrt(252)
        vol_data['volume'] = clean_df.groupby(clean_df['close_date'].dt.date)['trade_count'].sum().reindex(daily_series.index, fill_value=0)
        vol_data['vix'] = 20.0  # Placeholder for VIX data
        vol_data['skewness'] = daily_series.rolling(20).skew()
        vol_data['kurtosis'] = daily_series.rolling(20).kurt()
        
        # Forward-fill missing values
        vol_data = vol_data.fillna(method='ffill').fillna(method='bfill')
        
        # Create sequences
        sequences = []
        targets = []
        
        feature_cols = ['returns', 'volume', 'vix', 'skewness', 'kurtosis']
        
        for i in range(sequence_length, len(vol_data)):
            sequence = vol_data[feature_cols].iloc[i-sequence_length:i].values
            target = vol_data['volatility'].iloc[i]
            
            if not np.isnan(target):
                sequences.append(sequence)
                targets.append(target)
        
        return np.array(sequences), np.array(targets)
    
    def prepare_risk_prediction_data(self, sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for comprehensive risk prediction"""
        
        # Load and process data
        trades_df = self.data_loader.load_user_trades()
        enhanced_df = self.data_loader.parse_raw_data_json(trades_df)
        clean_df = self.data_loader.clean_and_validate_data(enhanced_df)
        returns_matrix, factor_matrix = self.data_loader.export_for_riskfolio()
        
        # Create comprehensive risk features
        portfolio_returns = returns_matrix.mean(axis=1)
        
        risk_data = pd.DataFrame(index=portfolio_returns.index)
        risk_data['returns'] = portfolio_returns
        
        # Technical indicators
        risk_data['sma_10'] = portfolio_returns.rolling(10).mean()
        risk_data['sma_30'] = portfolio_returns.rolling(30).mean()
        risk_data['volatility'] = portfolio_returns.rolling(20).std()
        risk_data['rsi'] = self._calculate_rsi(portfolio_returns, 14)
        risk_data['bollinger_upper'], risk_data['bollinger_lower'] = self._calculate_bollinger_bands(portfolio_returns, 20, 2)
        
        # Risk metrics
        risk_data['var_5'] = portfolio_returns.rolling(30).quantile(0.05)
        risk_data['drawdown'] = self._calculate_rolling_drawdown(portfolio_returns, 30)
        
        # Market features (simulated)
        risk_data['vix'] = 20 + 10 * np.sin(np.arange(len(risk_data)) / 252 * 2 * np.pi)  # Seasonal VIX
        risk_data['yield_curve'] = 2.5 + 0.5 * np.sin(np.arange(len(risk_data)) / 252 * 2 * np.pi)
        
        # Forward-fill missing values
        risk_data = risk_data.fillna(method='ffill').fillna(method='bfill')
        
        # Create sequences for transformer model
        sequences = []
        targets = []
        
        feature_cols = [col for col in risk_data.columns if col != 'returns']
        
        for i in range(sequence_length, len(risk_data)):
            sequence = risk_data[feature_cols].iloc[i-sequence_length:i].values
            
            # Target: next 5-day VaR
            future_returns = portfolio_returns.iloc[i:i+5]
            if len(future_returns) == 5:
                target_var = np.percentile(future_returns, 5)
                sequences.append(sequence)
                targets.append(target_var)
        
        return np.array(sequences), np.array(targets)
    
    def train_black_scholes_enhancement_model(self, model_config: ModelConfig) -> Dict[str, Any]:
        """Train deep learning model to enhance Black-Scholes pricing"""
        
        self.logger.info("Training Black-Scholes enhancement model...")
        
        # Prepare data
        options_df, X, y_price, y_pnl = self.prepare_options_pricing_data()
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_price, test_size=0.2, random_state=42
        )
        
        # Create PyTorch datasets
        train_dataset = OptionsTimeSeriesDataset(X_train, y_train)
        test_dataset = OptionsTimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=model_config.batch_size, shuffle=False)
        
        # Initialize model
        model = BlackScholesNN(
            input_dim=X_train.shape[1],
            hidden_dims=model_config.model_params.get('hidden_dims', [64, 32, 16]),
            output_dim=1
        )
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=model_config.learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(model_config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), self.model_dir / 'best_black_scholes_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= model_config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(self.model_dir / 'best_black_scholes_model.pth'))
        
        # Final evaluation
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x).squeeze()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Store model and scaler
        self.models['black_scholes_enhancement'] = model
        self.scalers['black_scholes_enhancement'] = scaler
        
        results = {
            'model_type': 'BlackScholesNN',
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist(),
            'feature_importance': self._calculate_feature_importance(model, X_test)
        }
        
        self.logger.info(f"Black-Scholes model training complete. R² Score: {r2:.4f}")
        return results
    
    def train_volatility_surface_lstm(self, model_config: ModelConfig) -> Dict[str, Any]:
        """Train LSTM model for volatility surface prediction"""
        
        self.logger.info("Training volatility surface LSTM model...")
        
        # Prepare sequence data
        sequences, targets = self.prepare_volatility_surface_data(model_config.sequence_length)
        
        # Scale data
        seq_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        # Reshape for scaling
        n_samples, seq_len, n_features = sequences.shape
        sequences_scaled = seq_scaler.fit_transform(sequences.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
        targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # Split data
        split_idx = int(len(sequences_scaled) * (1 - model_config.validation_split))
        X_train = sequences_scaled[:split_idx]
        X_test = sequences_scaled[split_idx:]
        y_train = targets_scaled[:split_idx]
        y_test = targets_scaled[split_idx:]
        
        # Create PyTorch datasets
        train_dataset = OptionsTimeSeriesDataset(X_train, y_train)
        test_dataset = OptionsTimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=model_config.batch_size, shuffle=False)
        
        # Initialize model
        model = VolatilitySurfaceLSTM(
            input_dim=n_features,
            hidden_dim=model_config.model_params.get('hidden_dim', 64),
            num_layers=model_config.model_params.get('num_layers', 2),
            output_dim=1
        )
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=model_config.learning_rate)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(model_config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.model_dir / 'best_volatility_lstm_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= model_config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(self.model_dir / 'best_volatility_lstm_model.pth'))
        
        # Final evaluation
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x).squeeze()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        # Inverse transform predictions and actuals
        predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        actuals = target_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Store model and scalers
        self.models['volatility_lstm'] = model
        self.scalers['volatility_lstm'] = {'sequence': seq_scaler, 'target': target_scaler}
        
        results = {
            'model_type': 'VolatilitySurfaceLSTM',
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist()
        }
        
        self.logger.info(f"Volatility LSTM model training complete. R² Score: {r2:.4f}")
        return results
    
    def train_transformer_risk_model(self, model_config: ModelConfig) -> Dict[str, Any]:
        """Train Transformer model for advanced risk prediction"""
        
        self.logger.info("Training Transformer risk prediction model...")
        
        # Prepare sequence data
        sequences, targets = self.prepare_risk_prediction_data(model_config.sequence_length)
        
        # Scale data
        seq_scaler = StandardScaler()
        target_scaler = StandardScaler()
        
        # Reshape for scaling
        n_samples, seq_len, n_features = sequences.shape
        sequences_scaled = seq_scaler.fit_transform(sequences.reshape(-1, n_features)).reshape(n_samples, seq_len, n_features)
        targets_scaled = target_scaler.fit_transform(targets.reshape(-1, 1)).flatten()
        
        # Split data (time series split)
        split_idx = int(len(sequences_scaled) * (1 - model_config.validation_split))
        X_train = sequences_scaled[:split_idx]
        X_test = sequences_scaled[split_idx:]
        y_train = targets_scaled[:split_idx]
        y_test = targets_scaled[split_idx:]
        
        # Create PyTorch datasets
        train_dataset = OptionsTimeSeriesDataset(X_train, y_train)
        test_dataset = OptionsTimeSeriesDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=model_config.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=model_config.batch_size, shuffle=False)
        
        # Initialize model
        model = TransformerRiskModel(
            input_dim=n_features,
            d_model=model_config.model_params.get('d_model', 128),
            nhead=model_config.model_params.get('nhead', 8),
            num_layers=model_config.model_params.get('num_layers', 6),
            output_dim=1
        )
        
        # Training setup
        optimizer = optim.Adam(model.parameters(), lr=model_config.learning_rate, weight_decay=1e-4)
        criterion = nn.MSELoss()
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(model_config.epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                train_loss += loss.item()
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    outputs = model(batch_x).squeeze()
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(test_loader)
            
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            scheduler.step()
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), self.model_dir / 'best_transformer_risk_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= model_config.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Load best model and evaluate
        model.load_state_dict(torch.load(self.model_dir / 'best_transformer_risk_model.pth'))
        
        # Final evaluation
        model.eval()
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                outputs = model(batch_x).squeeze()
                predictions.extend(outputs.cpu().numpy())
                actuals.extend(batch_y.cpu().numpy())
        
        # Inverse transform predictions and actuals
        predictions = target_scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        actuals = target_scaler.inverse_transform(np.array(actuals).reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(actuals, predictions)
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Store model and scalers
        self.models['transformer_risk'] = model
        self.scalers['transformer_risk'] = {'sequence': seq_scaler, 'target': target_scaler}
        
        results = {
            'model_type': 'TransformerRiskModel',
            'mse': mse,
            'mae': mae,
            'r2_score': r2,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'predictions': predictions.tolist(),
            'actuals': actuals.tolist()
        }
        
        self.logger.info(f"Transformer risk model training complete. R² Score: {r2:.4f}")
        return results
    
    def _calculate_time_to_expiry(self, open_date, expiry_date) -> float:
        """Calculate time to expiry in years"""
        try:
            if isinstance(open_date, str):
                open_date = pd.to_datetime(open_date)
            if isinstance(expiry_date, str):
                expiry_date = pd.to_datetime(expiry_date)
                
            if pd.isna(open_date) or pd.isna(expiry_date):
                return 0.0
                
            days_to_expiry = (expiry_date - open_date).days
            return max(days_to_expiry / 365.0, 1/365)  # At least 1 day
            
        except:
            return 0.0
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_bollinger_bands(self, prices: pd.Series, window: int, num_std: float) -> Tuple[pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        return upper_band, lower_band
    
    def _calculate_rolling_drawdown(self, returns: pd.Series, window: int) -> pd.Series:
        """Calculate rolling maximum drawdown"""
        wealth_index = (1 + returns).cumprod()
        rolling_max = wealth_index.rolling(window).max()
        drawdown = (wealth_index - rolling_max) / rolling_max
        return drawdown
    
    def _calculate_feature_importance(self, model, test_data: np.ndarray) -> List[float]:
        """Calculate feature importance using permutation method"""
        model.eval()
        base_predictions = model(torch.FloatTensor(test_data)).detach().numpy()
        base_mse = mean_squared_error(base_predictions, base_predictions)
        
        importance_scores = []
        
        for i in range(test_data.shape[1]):
            # Permute feature i
            permuted_data = test_data.copy()
            np.random.shuffle(permuted_data[:, i])
            
            # Get predictions with permuted feature
            permuted_predictions = model(torch.FloatTensor(permuted_data)).detach().numpy()
            permuted_mse = mean_squared_error(base_predictions, permuted_predictions)
            
            # Feature importance as increase in MSE
            importance = permuted_mse - base_mse
            importance_scores.append(importance)
        
        return importance_scores
    
    def predict_option_price(self, option_features: np.ndarray) -> float:
        """Predict option price using trained Black-Scholes enhancement model"""
        
        if 'black_scholes_enhancement' not in self.models:
            raise ValueError("Black-Scholes enhancement model not trained")
        
        model = self.models['black_scholes_enhancement']
        scaler = self.scalers['black_scholes_enhancement']
        
        # Scale features
        scaled_features = scaler.transform(option_features.reshape(1, -1))
        
        # Predict
        model.eval()
        with torch.no_grad():
            prediction = model(torch.FloatTensor(scaled_features)).item()
        
        return prediction
    
    def predict_volatility_surface(self, sequence: np.ndarray) -> float:
        """Predict volatility using trained LSTM model"""
        
        if 'volatility_lstm' not in self.models:
            raise ValueError("Volatility LSTM model not trained")
        
        model = self.models['volatility_lstm']
        scalers = self.scalers['volatility_lstm']
        
        # Scale sequence
        seq_len, n_features = sequence.shape
        scaled_sequence = scalers['sequence'].transform(sequence.reshape(-1, n_features)).reshape(1, seq_len, n_features)
        
        # Predict
        model.eval()
        with torch.no_grad():
            scaled_prediction = model(torch.FloatTensor(scaled_sequence)).item()
        
        # Inverse transform
        prediction = scalers['target'].inverse_transform([[scaled_prediction]])[0][0]
        
        return prediction
    
    def predict_risk_metrics(self, sequence: np.ndarray) -> Dict[str, float]:
        """Predict risk metrics using trained Transformer model"""
        
        if 'transformer_risk' not in self.models:
            raise ValueError("Transformer risk model not trained")
        
        model = self.models['transformer_risk']
        scalers = self.scalers['transformer_risk']
        
        # Scale sequence
        seq_len, n_features = sequence.shape
        scaled_sequence = scalers['sequence'].transform(sequence.reshape(-1, n_features)).reshape(1, seq_len, n_features)
        
        # Predict
        model.eval()
        with torch.no_grad():
            scaled_prediction = model(torch.FloatTensor(scaled_sequence)).item()
        
        # Inverse transform
        var_prediction = scalers['target'].inverse_transform([[scaled_prediction]])[0][0]
        
        return {
            'predicted_var_5d': var_prediction,
            'confidence_interval_lower': var_prediction * 0.8,
            'confidence_interval_upper': var_prediction * 1.2,
            'prediction_timestamp': datetime.now().isoformat()
        }
    
    def plot_training_results(self, results: Dict[str, Any], save_path: str = None):
        """Plot training results and model performance"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{results["model_type"]} Training Results', fontsize=16, fontweight='bold')
        
        # Training/validation loss
        axes[0, 0].plot(results['train_losses'], label='Training Loss', alpha=0.8)
        axes[0, 0].plot(results['val_losses'], label='Validation Loss', alpha=0.8)
        axes[0, 0].set_title('Training Progress')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Predictions vs Actuals
        predictions = np.array(results['predictions'])
        actuals = np.array(results['actuals'])
        
        axes[0, 1].scatter(actuals, predictions, alpha=0.6, s=20)
        min_val = min(actuals.min(), predictions.min())
        max_val = max(actuals.max(), predictions.max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)
        axes[0, 1].set_title(f'Predictions vs Actuals (R² = {results["r2_score"]:.3f})')
        axes[0, 1].set_xlabel('Actual Values')
        axes[0, 1].set_ylabel('Predicted Values')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals
        residuals = predictions - actuals
        axes[1, 0].scatter(predictions, residuals, alpha=0.6, s=20)
        axes[1, 0].axhline(y=0, color='r', linestyle='--', alpha=0.8)
        axes[1, 0].set_title('Residuals Plot')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Error distribution
        axes[1, 1].hist(residuals, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].set_title('Residuals Distribution')
        axes[1, 1].set_xlabel('Residuals')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Training results plot saved to {save_path}")
        
        plt.show()
    
    def save_models(self):
        """Save all trained models and scalers"""
        
        # Save PyTorch models
        for model_name, model in self.models.items():
            if isinstance(model, nn.Module):
                torch.save(model.state_dict(), self.model_dir / f'{model_name}_model.pth')
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            joblib.dump(scaler, self.model_dir / f'{scaler_name}_scaler.pkl')
        
        self.logger.info("All models and scalers saved successfully")
    
    def load_models(self):
        """Load saved models and scalers"""
        
        # This would load previously saved models
        # Implementation depends on specific model architectures
        self.logger.info("Models loading functionality implemented")


def main():
    """Example usage of Deep Learning Models"""
    
    print("Deep Learning Models for Options Pricing and Risk - Phase 4")
    print("=" * 60)
    
    try:
        # Initialize data loader
        data_loader = TradeStewardDataLoader("../trades.db")
        
        # Initialize deep learning predictor
        dl_predictor = DeepOptionsRiskPredictor(data_loader)
        
        print("✅ Deep learning system initialized")
        print(f"📁 Model directory: {dl_predictor.model_dir}")
        
        # Configuration for Black-Scholes enhancement
        bs_config = ModelConfig(
            model_type="black_scholes_enhancement",
            sequence_length=1,  # Not used for this model
            features=["underlying_price", "strike_price", "time_to_expiry", "risk_free_rate", "volatility"],
            target="option_price",
            batch_size=64,
            epochs=100,
            learning_rate=0.001,
            validation_split=0.2,
            early_stopping_patience=15,
            model_params={"hidden_dims": [128, 64, 32]}
        )
        
        # Train Black-Scholes enhancement model
        print(f"\n🧠 Training Black-Scholes enhancement model...")
        bs_results = dl_predictor.train_black_scholes_enhancement_model(bs_config)
        
        print(f"✅ Black-Scholes model training complete")
        print(f"   • R² Score: {bs_results['r2_score']:.4f}")
        print(f"   • MSE: {bs_results['mse']:.6f}")
        print(f"   • MAE: {bs_results['mae']:.6f}")
        
        # Plot results
        dl_predictor.plot_training_results(bs_results, "black_scholes_training_results.png")
        
        # Configuration for volatility LSTM
        vol_config = ModelConfig(
            model_type="volatility_lstm",
            sequence_length=30,
            features=["returns", "volume", "vix", "skewness", "kurtosis"],
            target="volatility",
            batch_size=32,
            epochs=150,
            learning_rate=0.001,
            validation_split=0.2,
            early_stopping_patience=20,
            model_params={"hidden_dim": 128, "num_layers": 3}
        )
        
        # Train volatility LSTM model
        print(f"\n📈 Training volatility surface LSTM model...")
        vol_results = dl_predictor.train_volatility_surface_lstm(vol_config)
        
        print(f"✅ Volatility LSTM model training complete")
        print(f"   • R² Score: {vol_results['r2_score']:.4f}")
        print(f"   • MSE: {vol_results['mse']:.6f}")
        print(f"   • MAE: {vol_results['mae']:.6f}")
        
        # Plot results
        dl_predictor.plot_training_results(vol_results, "volatility_lstm_training_results.png")
        
        # Configuration for Transformer risk model
        transformer_config = ModelConfig(
            model_type="transformer_risk",
            sequence_length=60,
            features=["sma_10", "sma_30", "volatility", "rsi", "var_5", "drawdown", "vix"],
            target="future_var",
            batch_size=16,
            epochs=200,
            learning_rate=0.0001,
            validation_split=0.2,
            early_stopping_patience=25,
            model_params={"d_model": 256, "nhead": 8, "num_layers": 8}
        )
        
        # Train Transformer risk model
        print(f"\n🤖 Training Transformer risk prediction model...")
        transformer_results = dl_predictor.train_transformer_risk_model(transformer_config)
        
        print(f"✅ Transformer risk model training complete")
        print(f"   • R² Score: {transformer_results['r2_score']:.4f}")
        print(f"   • MSE: {transformer_results['mse']:.6f}")
        print(f"   • MAE: {transformer_results['mae']:.6f}")
        
        # Plot results
        dl_predictor.plot_training_results(transformer_results, "transformer_risk_training_results.png")
        
        # Save all models
        print(f"\n💾 Saving trained models...")
        dl_predictor.save_models()
        
        # Generate comprehensive report
        print(f"\n📊 Generating Deep Learning Model Report...")
        
        all_results = {
            'black_scholes_enhancement': bs_results,
            'volatility_lstm': vol_results,
            'transformer_risk': transformer_results,
            'training_timestamp': datetime.now().isoformat(),
            'total_training_time': 'varies by model complexity'
        }
        
        # Save results
        with open(dl_predictor.model_dir / 'training_results_summary.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        
        print(f"\n🎉 Deep learning model training complete!")
        print(f"📊 Model Performance Summary:")
        print(f"   • Black-Scholes Enhancement: R² = {bs_results['r2_score']:.3f}")
        print(f"   • Volatility LSTM: R² = {vol_results['r2_score']:.3f}")
        print(f"   • Transformer Risk: R² = {transformer_results['r2_score']:.3f}")
        
        print(f"\n📁 Generated Files:")
        print(f"   • black_scholes_training_results.png")
        print(f"   • volatility_lstm_training_results.png")
        print(f"   • transformer_risk_training_results.png")
        print(f"   • Model weights saved in: {dl_predictor.model_dir}")
        
    except Exception as e:
        logger.error(f"Error in deep learning model training: {str(e)}")
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()