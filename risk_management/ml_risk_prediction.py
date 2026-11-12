"""
Machine Learning Risk Prediction Models - Phase 3 Implementation
Advanced ML models for predicting risk metrics and portfolio outcomes
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings
import logging
from datetime import datetime, timedelta
import joblib
import json
from dataclasses import dataclass

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Data class for model performance metrics"""
    model_name: str
    mse: float
    mae: float
    r2: float
    mape: float
    directional_accuracy: float
    feature_importance: Dict[str, float]

class MLRiskPredictor:
    """
    Advanced machine learning models for risk prediction
    Supports multiple algorithms and ensemble methods
    """
    
    def __init__(self, returns_data: pd.DataFrame, target_horizon: int = 5):
        """
        Initialize ML risk predictor
        
        Args:
            returns_data: Historical returns data
            target_horizon: Prediction horizon in days
        """
        self.returns_data = returns_data
        self.target_horizon = target_horizon
        
        # Data storage
        self.features_df = None
        self.targets_df = None
        self.feature_names = None
        
        # Models
        self.models = {}
        self.ensemble_model = None
        self.scalers = {}
        
        # Performance tracking
        self.model_performance = {}
        self.feature_importance = {}
        
        # Parameters
        self.lookback_window = 30
        self.train_test_split = 0.8
        
        logger.info(f"ML Risk Predictor initialized with {len(returns_data)} observations")
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer comprehensive features for risk prediction
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Engineering features for ML risk prediction...")
        
        # Start with portfolio returns
        portfolio_returns = self.returns_data.mean(axis=1)
        
        features_list = []
        
        # Historical window for feature engineering
        for i in range(self.lookback_window, len(portfolio_returns)):
            window_returns = portfolio_returns.iloc[i-self.lookback_window:i]
            
            feature_dict = {
                'date': portfolio_returns.index[i],
                
                # Basic statistical features
                'return_mean_5d': window_returns.tail(5).mean(),
                'return_mean_10d': window_returns.tail(10).mean(),
                'return_mean_20d': window_returns.tail(20).mean(),
                'return_mean_30d': window_returns.mean(),
                
                'volatility_5d': window_returns.tail(5).std(),
                'volatility_10d': window_returns.tail(10).std(),
                'volatility_20d': window_returns.tail(20).std(),
                'volatility_30d': window_returns.std(),
                
                # Skewness and kurtosis
                'skewness_10d': window_returns.tail(10).skew(),
                'skewness_20d': window_returns.tail(20).skew(),
                'kurtosis_10d': window_returns.tail(10).kurtosis(),
                'kurtosis_20d': window_returns.tail(20).kurtosis(),
                
                # Momentum features
                'momentum_5d': (window_returns.tail(5).prod() - 1),
                'momentum_10d': (window_returns.tail(10).prod() - 1),
                'momentum_20d': (window_returns.tail(20).prod() - 1),
                
                # Volatility ratios
                'vol_ratio_5_20': window_returns.tail(5).std() / window_returns.tail(20).std(),
                'vol_ratio_10_20': window_returns.tail(10).std() / window_returns.tail(20).std(),
                
                # Autocorrelation features
                'autocorr_1d': window_returns.autocorr(lag=1),
                'autocorr_5d': window_returns.autocorr(lag=5),
                
                # Percentile features
                'return_percentile_10': np.percentile(window_returns, 10),
                'return_percentile_25': np.percentile(window_returns, 25),
                'return_percentile_75': np.percentile(window_returns, 75),
                'return_percentile_90': np.percentile(window_returns, 90),
                
                # Drawdown features
                'max_drawdown_10d': self._calculate_max_drawdown(window_returns.tail(10)),
                'max_drawdown_20d': self._calculate_max_drawdown(window_returns.tail(20)),
                'current_drawdown': self._calculate_current_drawdown(window_returns),
                
                # Trend features
                'trend_5d': self._calculate_trend(window_returns.tail(5)),
                'trend_10d': self._calculate_trend(window_returns.tail(10)),
                'trend_20d': self._calculate_trend(window_returns.tail(20)),
                
                # Volatility clustering
                'vol_clustering': self._calculate_vol_clustering(window_returns),
                
                # Range features
                'range_5d': window_returns.tail(5).max() - window_returns.tail(5).min(),
                'range_10d': window_returns.tail(10).max() - window_returns.tail(10).min(),
                'range_20d': window_returns.tail(20).max() - window_returns.tail(20).min(),
                
                # Cross-sectional features (strategy dispersion)
                'strategy_correlation': self._calculate_strategy_correlation(i),
                'strategy_dispersion': self._calculate_strategy_dispersion(i),
                
                # Technical indicators
                'rsi_10d': self._calculate_rsi(window_returns.tail(10)),
                'rsi_20d': self._calculate_rsi(window_returns.tail(20)),
                
                # Regime indicators
                'vol_regime': 1 if window_returns.std() > window_returns.rolling(20).std().mean() else 0,
                'return_regime': 1 if window_returns.mean() > 0 else 0,
                
                # Calendar features
                'day_of_week': portfolio_returns.index[i].dayofweek,
                'day_of_month': portfolio_returns.index[i].day,
                'month': portfolio_returns.index[i].month,
                'quarter': portfolio_returns.index[i].quarter,
                
                # Lagged returns
                'return_lag_1': window_returns.iloc[-1] if len(window_returns) > 0 else 0,
                'return_lag_2': window_returns.iloc[-2] if len(window_returns) > 1 else 0,
                'return_lag_3': window_returns.iloc[-3] if len(window_returns) > 2 else 0,
                'return_lag_5': window_returns.iloc[-5] if len(window_returns) > 4 else 0,
                
                # Rolling z-scores
                'return_zscore_10d': (window_returns.iloc[-1] - window_returns.tail(10).mean()) / window_returns.tail(10).std(),
                'return_zscore_20d': (window_returns.iloc[-1] - window_returns.tail(20).mean()) / window_returns.tail(20).std(),
                
                # Volatility persistence
                'vol_persistence': self._calculate_vol_persistence(window_returns)
            }
            
            features_list.append(feature_dict)
        
        features_df = pd.DataFrame(features_list).set_index('date')
        
        # Handle NaN and infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(features_df.median())
        
        self.features_df = features_df
        self.feature_names = list(features_df.columns)
        
        logger.info(f"Engineered {len(self.feature_names)} features for {len(features_df)} observations")
        
        return features_df
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown for a return series"""
        if len(returns) == 0:
            return 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def _calculate_current_drawdown(self, returns: pd.Series) -> float:
        """Calculate current drawdown"""
        if len(returns) == 0:
            return 0
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        current_drawdown = (cumulative.iloc[-1] - running_max.iloc[-1]) / running_max.iloc[-1]
        return abs(current_drawdown)
    
    def _calculate_trend(self, returns: pd.Series) -> float:
        """Calculate trend strength using linear regression slope"""
        if len(returns) < 3:
            return 0
        x = np.arange(len(returns))
        y = (1 + returns).cumprod().values
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _calculate_vol_clustering(self, returns: pd.Series) -> float:
        """Calculate volatility clustering metric"""
        if len(returns) < 10:
            return 0
        squared_returns = returns ** 2
        autocorr = squared_returns.autocorr(lag=1)
        return autocorr if not np.isnan(autocorr) else 0
    
    def _calculate_strategy_correlation(self, index: int) -> float:
        """Calculate average correlation between strategies"""
        if index < self.lookback_window:
            return 0
        
        window_data = self.returns_data.iloc[index-10:index]  # 10-day window
        if len(window_data) < 5:
            return 0
        
        corr_matrix = window_data.corr()
        # Get average correlation (excluding diagonal)
        mask = np.triu(np.ones_like(corr_matrix), k=1).astype(bool)
        avg_corr = corr_matrix.where(mask).stack().mean()
        return avg_corr if not np.isnan(avg_corr) else 0
    
    def _calculate_strategy_dispersion(self, index: int) -> float:
        """Calculate dispersion between strategy returns"""
        if index < self.lookback_window:
            return 0
        
        window_data = self.returns_data.iloc[index-5:index]  # 5-day window
        if len(window_data) == 0:
            return 0
        
        daily_spreads = window_data.max(axis=1) - window_data.min(axis=1)
        return daily_spreads.mean()
    
    def _calculate_rsi(self, returns: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator"""
        if len(returns) < period:
            return 50  # Neutral RSI
        
        deltas = returns.diff()
        gains = deltas.where(deltas > 0, 0).rolling(window=period).mean()
        losses = (-deltas.where(deltas < 0, 0)).rolling(window=period).mean()
        
        rs = gains / losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not np.isnan(rsi.iloc[-1]) else 50
    
    def _calculate_vol_persistence(self, returns: pd.Series) -> float:
        """Calculate volatility persistence measure"""
        if len(returns) < 10:
            return 0
        
        volatilities = returns.rolling(5).std()
        if len(volatilities) < 5:
            return 0
        
        autocorr = volatilities.autocorr(lag=1)
        return autocorr if not np.isnan(autocorr) else 0
    
    def create_targets(self) -> pd.DataFrame:
        """
        Create prediction targets for multiple risk metrics
        
        Returns:
            DataFrame with target variables
        """
        logger.info(f"Creating prediction targets with {self.target_horizon}-day horizon...")
        
        portfolio_returns = self.returns_data.mean(axis=1)
        targets_list = []
        
        for i in range(len(portfolio_returns) - self.target_horizon):
            future_returns = portfolio_returns.iloc[i+1:i+1+self.target_horizon]
            
            if len(future_returns) == self.target_horizon:
                target_dict = {
                    'date': portfolio_returns.index[i],
                    
                    # Future volatility (main target)
                    'future_volatility': future_returns.std() * np.sqrt(252),
                    
                    # Future return
                    'future_return': future_returns.mean() * 252,
                    
                    # Future VaR (5%)
                    'future_var_5': abs(np.percentile(future_returns, 5)),
                    
                    # Future maximum drawdown
                    'future_max_drawdown': self._calculate_max_drawdown(future_returns),
                    
                    # Future Sharpe ratio
                    'future_sharpe': (future_returns.mean() / future_returns.std()) * np.sqrt(252) if future_returns.std() > 0 else 0,
                    
                    # Binary classification targets
                    'high_vol_regime': 1 if future_returns.std() > portfolio_returns.rolling(30).std().iloc[i] else 0,
                    'positive_return': 1 if future_returns.sum() > 0 else 0,
                    'large_drawdown': 1 if self._calculate_max_drawdown(future_returns) > 0.02 else 0,
                    
                    # Extreme event indicators
                    'extreme_negative': 1 if future_returns.min() < np.percentile(portfolio_returns.rolling(60).quantile(0.05).iloc[:i], 10) else 0,
                    'extreme_positive': 1 if future_returns.max() > np.percentile(portfolio_returns.rolling(60).quantile(0.95).iloc[:i], 90) else 0
                }
                
                targets_list.append(target_dict)
        
        targets_df = pd.DataFrame(targets_list).set_index('date')
        
        # Handle NaN and infinite values
        targets_df = targets_df.replace([np.inf, -np.inf], np.nan)
        targets_df = targets_df.fillna(0)
        
        self.targets_df = targets_df
        
        logger.info(f"Created {len(targets_df.columns)} target variables for {len(targets_df)} observations")
        
        return targets_df
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare aligned training data
        
        Returns:
            Tuple of (features, targets) DataFrames
        """
        if self.features_df is None:
            self.engineer_features()
        
        if self.targets_df is None:
            self.create_targets()
        
        # Align features and targets
        common_dates = self.features_df.index.intersection(self.targets_df.index)
        
        aligned_features = self.features_df.loc[common_dates]
        aligned_targets = self.targets_df.loc[common_dates]
        
        logger.info(f"Prepared training data: {len(aligned_features)} samples, {len(aligned_features.columns)} features")
        
        return aligned_features, aligned_targets
    
    def train_models(self, target_variable: str = 'future_volatility') -> Dict[str, Any]:
        """
        Train multiple ML models for risk prediction
        
        Args:
            target_variable: Target variable to predict
            
        Returns:
            Dictionary with training results
        """
        logger.info(f"Training ML models for {target_variable} prediction...")
        
        # Prepare data
        X, y_df = self.prepare_training_data()
        
        if target_variable not in y_df.columns:
            logger.error(f"Target variable {target_variable} not found")
            return {}
        
        y = y_df[target_variable]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Split into train/test
        split_idx = int(len(X) * self.train_test_split)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        # Scale features
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_variable] = scaler
        
        # Define models
        models_config = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
            'xgboost': xgb.XGBRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42, n_jobs=-1
            ),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.01, max_iter=2000),
            'elastic_net': ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=2000),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale'),
            'mlp': MLPRegressor(
                hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42, early_stopping=True
            )
        }
        
        trained_models = {}
        performance_results = {}
        
        # Train each model
        for model_name, model in models_config.items():
            logger.info(f"Training {model_name}...")
            
            try:
                # Use scaled data for some models
                if model_name in ['ridge', 'lasso', 'elastic_net', 'svr', 'mlp']:
                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_test_scaled)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=tscv, scoring='neg_mean_squared_error')
                else:
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='neg_mean_squared_error')
                
                # Calculate performance metrics
                mse = mean_squared_error(y_test, y_pred)
                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
                
                # Directional accuracy
                y_test_diff = np.diff(y_test)
                y_pred_diff = np.diff(y_pred)
                directional_accuracy = np.mean(np.sign(y_test_diff) == np.sign(y_pred_diff))
                
                # Feature importance
                feature_importance = {}
                if hasattr(model, 'feature_importances_'):
                    importance_scores = model.feature_importances_
                    feature_importance = dict(zip(X.columns, importance_scores))
                elif hasattr(model, 'coef_'):
                    importance_scores = np.abs(model.coef_)
                    feature_importance = dict(zip(X.columns, importance_scores))
                
                performance = ModelPerformance(
                    model_name=model_name,
                    mse=mse,
                    mae=mae,
                    r2=r2,
                    mape=mape,
                    directional_accuracy=directional_accuracy,
                    feature_importance=feature_importance
                )
                
                trained_models[model_name] = model
                performance_results[model_name] = performance
                
                logger.info(f"{model_name} - R²: {r2:.3f}, MAE: {mae:.4f}, Directional Accuracy: {directional_accuracy:.3f}")
                
            except Exception as e:
                logger.warning(f"Error training {model_name}: {str(e)}")
                continue
        
        # Store models and performance
        self.models[target_variable] = trained_models
        self.model_performance[target_variable] = performance_results
        
        # Create ensemble model
        if len(trained_models) >= 3:
            ensemble_models = []
            
            # Select top 3 models by R²
            sorted_models = sorted(performance_results.items(), key=lambda x: x[1].r2, reverse=True)
            
            for model_name, _ in sorted_models[:3]:
                if model_name in ['ridge', 'lasso', 'elastic_net', 'svr', 'mlp']:
                    # For scaled models, we need to wrap them
                    ensemble_models.append((model_name, trained_models[model_name]))
                else:
                    ensemble_models.append((model_name, trained_models[model_name]))
            
            if ensemble_models:
                try:
                    ensemble = VotingRegressor(estimators=ensemble_models)
                    
                    # Fit ensemble (handle scaling appropriately)
                    if any(name in ['ridge', 'lasso', 'elastic_net', 'svr', 'mlp'] for name, _ in ensemble_models):
                        # Mix of scaled and unscaled models - use scaled data
                        ensemble.fit(X_train_scaled, y_train)
                        ensemble_pred = ensemble.predict(X_test_scaled)
                    else:
                        ensemble.fit(X_train, y_train)
                        ensemble_pred = ensemble.predict(X_test)
                    
                    # Ensemble performance
                    ensemble_r2 = r2_score(y_test, ensemble_pred)
                    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
                    
                    self.ensemble_model = ensemble
                    
                    logger.info(f"Ensemble model - R²: {ensemble_r2:.3f}, MAE: {ensemble_mae:.4f}")
                    
                except Exception as e:
                    logger.warning(f"Error creating ensemble: {str(e)}")
        
        training_results = {
            'target_variable': target_variable,
            'n_models_trained': len(trained_models),
            'best_model': max(performance_results.items(), key=lambda x: x[1].r2)[0] if performance_results else None,
            'performance_summary': {name: {'r2': perf.r2, 'mae': perf.mae} for name, perf in performance_results.items()},
            'feature_count': len(X.columns),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Model training complete for {target_variable}")
        
        return training_results
    
    def predict_risk_metrics(self, model_name: str = None, target_variable: str = 'future_volatility') -> Dict:
        """
        Make risk predictions using trained models
        
        Args:
            model_name: Specific model to use, None for best model
            target_variable: Target variable to predict
            
        Returns:
            Prediction results
        """
        if target_variable not in self.models:
            logger.error(f"No trained models for {target_variable}")
            return {}
        
        # Get latest features
        if self.features_df is None:
            self.engineer_features()
        
        latest_features = self.features_df.iloc[-1:].values
        
        # Select model
        if model_name is None:
            # Use best performing model
            best_model_name = max(
                self.model_performance[target_variable].items(),
                key=lambda x: x[1].r2
            )[0]
            model = self.models[target_variable][best_model_name]
            model_name = best_model_name
        else:
            if model_name not in self.models[target_variable]:
                logger.error(f"Model {model_name} not found for {target_variable}")
                return {}
            model = self.models[target_variable][model_name]
        
        # Scale features if needed
        if model_name in ['ridge', 'lasso', 'elastic_net', 'svr', 'mlp']:
            if target_variable in self.scalers:
                latest_features = self.scalers[target_variable].transform(latest_features)
        
        # Make prediction
        try:
            prediction = model.predict(latest_features)[0]
            
            # Get prediction confidence (if possible)
            confidence = None
            if hasattr(model, 'predict_proba'):
                # For classifiers
                probabilities = model.predict_proba(latest_features)[0]
                confidence = max(probabilities)
            elif model_name == 'random_forest':
                # For random forest, use prediction variance
                predictions = [tree.predict(latest_features)[0] for tree in model.estimators_]
                confidence = 1 - (np.std(predictions) / np.mean(predictions)) if np.mean(predictions) != 0 else 0
            
            prediction_results = {
                'prediction': prediction,
                'model_used': model_name,
                'target_variable': target_variable,
                'confidence': confidence,
                'prediction_date': datetime.now().isoformat(),
                'horizon_days': self.target_horizon
            }
            
            logger.info(f"Risk prediction: {target_variable} = {prediction:.4f} using {model_name}")
            
            return prediction_results
            
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return {}
    
    def plot_model_performance(self, target_variable: str = 'future_volatility', save_path: str = None) -> None:
        """
        Create comprehensive model performance visualization
        
        Args:
            target_variable: Target variable to visualize
            save_path: Path to save the plot
        """
        if target_variable not in self.model_performance:
            logger.error(f"No performance data for {target_variable}")
            return
        
        performance_data = self.model_performance[target_variable]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'ML Model Performance - {target_variable.replace("_", " ").title()}', 
                     fontsize=16, fontweight='bold')
        
        # 1. R² Comparison
        models = list(performance_data.keys())
        r2_scores = [performance_data[m].r2 for m in models]
        
        ax1 = axes[0, 0]
        bars = ax1.bar(models, r2_scores, alpha=0.7, color='steelblue')
        ax1.set_title('R² Score Comparison', fontweight='bold')
        ax1.set_ylabel('R² Score')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{score:.3f}', ha='center', va='bottom')
        
        # 2. MAE Comparison
        mae_scores = [performance_data[m].mae for m in models]
        
        ax2 = axes[0, 1]
        ax2.bar(models, mae_scores, alpha=0.7, color='orange')
        ax2.set_title('Mean Absolute Error', fontweight='bold')
        ax2.set_ylabel('MAE')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Directional Accuracy
        dir_accuracy = [performance_data[m].directional_accuracy for m in models]
        
        ax3 = axes[0, 2]
        ax3.bar(models, dir_accuracy, alpha=0.7, color='green')
        ax3.set_title('Directional Accuracy', fontweight='bold')
        ax3.set_ylabel('Accuracy')
        ax3.tick_params(axis='x', rotation=45)
        
        # 4. Feature Importance (for best model)
        best_model = max(performance_data.items(), key=lambda x: x[1].r2)[0]
        feature_imp = performance_data[best_model].feature_importance
        
        if feature_imp:
            # Top 10 features
            sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:10]
            feature_names = [f[0] for f in sorted_features]
            importance_values = [f[1] for f in sorted_features]
            
            ax4 = axes[1, 0]
            ax4.barh(feature_names, importance_values, alpha=0.7, color='purple')
            ax4.set_title(f'Top Features ({best_model})', fontweight='bold')
            ax4.set_xlabel('Importance')
        
        # 5. Model Performance Radar Chart
        ax5 = axes[1, 1]
        
        # Normalize metrics for radar chart
        max_r2 = max(r2_scores) if max(r2_scores) > 0 else 1
        max_mae = max(mae_scores) if max(mae_scores) > 0 else 1
        max_dir = max(dir_accuracy) if max(dir_accuracy) > 0 else 1
        
        angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        angles = np.concatenate((angles, [angles[0]]))
        
        for i, model in enumerate(models[:5]):  # Show top 5 models
            values = [
                performance_data[model].r2 / max_r2,
                1 - (performance_data[model].mae / max_mae),  # Invert MAE (lower is better)
                performance_data[model].directional_accuracy / max_dir
            ]
            values += values[:1]  # Complete the circle
            
            ax5.plot(angles, values, 'o-', linewidth=2, label=model)
            ax5.fill(angles, values, alpha=0.25)
        
        ax5.set_xticks(angles[:-1])
        ax5.set_xticklabels(['R²', 'MAE (inv)', 'Dir. Acc.'])
        ax5.set_title('Model Performance Radar', fontweight='bold')
        ax5.legend(bbox_to_anchor=(1.1, 1.1))
        ax5.grid(True)
        
        # 6. Performance Summary Table
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        # Create performance summary
        summary_text = f"Performance Summary - {target_variable.replace('_', ' ').title()}\n\n"
        summary_text += f"Best Model: {best_model}\n"
        summary_text += f"Best R²: {max(r2_scores):.3f}\n"
        summary_text += f"Best MAE: {min(mae_scores):.4f}\n"
        summary_text += f"Best Dir. Acc.: {max(dir_accuracy):.3f}\n\n"
        summary_text += f"Models Trained: {len(models)}\n"
        summary_text += f"Prediction Horizon: {self.target_horizon} days\n"
        summary_text += f"Feature Count: {len(self.feature_names) if self.feature_names else 0}"
        
        ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Model performance plot saved to {save_path}")
        
        plt.show()
    
    def save_models(self, filepath: str = 'ml_risk_models.joblib') -> None:
        """Save trained models to disk"""
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_names,
            'model_performance': self.model_performance,
            'target_horizon': self.target_horizon,
            'ensemble_model': self.ensemble_model
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str = 'ml_risk_models.joblib') -> None:
        """Load trained models from disk"""
        try:
            model_data = joblib.load(filepath)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.feature_names = model_data['feature_names']
            self.model_performance = model_data['model_performance']
            self.target_horizon = model_data['target_horizon']
            self.ensemble_model = model_data.get('ensemble_model')
            
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def generate_ml_report(self) -> Dict:
        """
        Generate comprehensive ML analysis report
        
        Returns:
            Detailed ML analysis report
        """
        logger.info("Generating ML risk prediction report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_summary': {
                'target_variables': list(self.models.keys()) if self.models else [],
                'feature_count': len(self.feature_names) if self.feature_names else 0,
                'prediction_horizon': self.target_horizon,
                'data_observations': len(self.features_df) if self.features_df is not None else 0
            },
            'model_performance': {},
            'feature_analysis': {},
            'predictions': {},
            'recommendations': []
        }
        
        # Model performance summary
        for target_var, performance_data in self.model_performance.items():
            report['model_performance'][target_var] = {}
            
            for model_name, perf in performance_data.items():
                report['model_performance'][target_var][model_name] = {
                    'r2_score': perf.r2,
                    'mae': perf.mae,
                    'mse': perf.mse,
                    'mape': perf.mape,
                    'directional_accuracy': perf.directional_accuracy
                }
        
        # Feature importance analysis
        if self.model_performance:
            for target_var, performance_data in self.model_performance.items():
                best_model = max(performance_data.items(), key=lambda x: x[1].r2)[0]
                feature_imp = performance_data[best_model].feature_importance
                
                if feature_imp:
                    # Top 10 features
                    sorted_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:10]
                    report['feature_analysis'][target_var] = {
                        'best_model': best_model,
                        'top_features': dict(sorted_features)
                    }
        
        # Current predictions
        for target_var in self.models.keys():
            try:
                prediction_result = self.predict_risk_metrics(target_variable=target_var)
                if prediction_result:
                    report['predictions'][target_var] = prediction_result
            except Exception as e:
                logger.warning(f"Error generating prediction for {target_var}: {str(e)}")
        
        # Generate recommendations
        recommendations = self._generate_ml_recommendations()
        report['recommendations'] = recommendations
        
        return report
    
    def _generate_ml_recommendations(self) -> List[str]:
        """Generate ML-based recommendations"""
        recommendations = []
        
        if not self.model_performance:
            return ["No model performance data available for recommendations"]
        
        # Model performance recommendations
        for target_var, performance_data in self.model_performance.items():
            best_r2 = max(perf.r2 for perf in performance_data.values())
            
            if best_r2 < 0.3:
                recommendations.append(f"Low predictive power for {target_var} (R²={best_r2:.3f}) - Consider feature engineering")
            elif best_r2 > 0.7:
                recommendations.append(f"Strong predictive model for {target_var} (R²={best_r2:.3f}) - Suitable for production use")
            
            # Directional accuracy recommendations
            best_dir_acc = max(perf.directional_accuracy for perf in performance_data.values())
            if best_dir_acc < 0.6:
                recommendations.append(f"Poor directional accuracy for {target_var} - Model may not be suitable for trend prediction")
        
        # Feature importance recommendations
        if 'future_volatility' in self.model_performance:
            perf_data = self.model_performance['future_volatility']
            best_model = max(perf_data.items(), key=lambda x: x[1].r2)[0]
            
            if best_model in ['random_forest', 'xgboost', 'gradient_boosting']:
                recommendations.append("Tree-based models performing best - Consider ensemble methods")
            elif best_model in ['ridge', 'lasso', 'elastic_net']:
                recommendations.append("Linear models performing best - Feature selection may be beneficial")
        
        # General recommendations
        recommendations.extend([
            "Retrain models monthly to maintain prediction accuracy",
            "Monitor model performance degradation over time",
            "Consider alternative data sources for feature enhancement",
            "Implement real-time prediction pipeline for dynamic risk management"
        ])
        
        return recommendations


def main():
    """Example usage of ML Risk Prediction"""
    
    try:
        # Install XGBoost if not available
        try:
            import xgboost
        except ImportError:
            print("Installing XGBoost...")
            import subprocess
            subprocess.check_call(["pip", "install", "xgboost"])
            import xgboost
        
        # Load data from previous phases
        returns_matrix = pd.read_csv('returns_matrix.csv', index_col=0, parse_dates=True)
        
        print("Machine Learning Risk Prediction - Phase 3")
        print("=" * 60)
        
        # Initialize ML predictor
        ml_predictor = MLRiskPredictor(returns_matrix, target_horizon=5)
        
        # Engineer features
        print("🔬 Engineering features for ML models...")
        features = ml_predictor.engineer_features()
        print(f"✅ Engineered {len(features.columns)} features")
        
        # Create targets
        print("🎯 Creating prediction targets...")
        targets = ml_predictor.create_targets()
        print(f"✅ Created {len(targets.columns)} target variables")
        
        # Train models for multiple targets
        target_variables = ['future_volatility', 'future_var_5', 'future_max_drawdown']
        
        for target_var in target_variables:
            print(f"\n🤖 Training models for {target_var}...")
            training_results = ml_predictor.train_models(target_variable=target_var)
            
            if training_results:
                print(f"✅ Trained {training_results['n_models_trained']} models")
                print(f"  • Best model: {training_results['best_model']}")
                
                # Show top 3 model performances
                perf_summary = training_results['performance_summary']
                sorted_models = sorted(perf_summary.items(), key=lambda x: x[1]['r2'], reverse=True)
                
                for i, (model_name, metrics) in enumerate(sorted_models[:3], 1):
                    print(f"  {i}. {model_name}: R²={metrics['r2']:.3f}, MAE={metrics['mae']:.4f}")
        
        # Make predictions
        print(f"\n📊 Making risk predictions...")
        for target_var in target_variables:
            prediction = ml_predictor.predict_risk_metrics(target_variable=target_var)
            if prediction:
                print(f"  • {target_var}: {prediction['prediction']:.4f} (model: {prediction['model_used']})")
        
        # Create visualizations
        ml_predictor.plot_model_performance('future_volatility', 'phase3_ml_performance.png')
        
        # Save models
        ml_predictor.save_models('phase3_ml_models.joblib')
        
        # Generate report
        report = ml_predictor.generate_ml_report()
        
        # Save report
        with open('phase3_ml_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display recommendations
        print(f"\n💡 ML-based recommendations:")
        for rec in report['recommendations']:
            print(f"  • {rec}")
        
        print("\n✅ ML risk prediction analysis complete!")
        print("📁 Results saved to:")
        print("  • phase3_ml_performance.png")
        print("  • phase3_ml_models.joblib")
        print("  • phase3_ml_report.json")
        
    except Exception as e:
        logger.error(f"Error in ML risk prediction: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()