"""
Regime-Switching Models - Phase 3 Implementation
Dynamic risk management with market regime detection and adaptation
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import warnings
import logging
from datetime import datetime, timedelta
import json
from dataclasses import dataclass

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

@dataclass
class RegimeMetrics:
    """Data class for regime-specific metrics"""
    regime_id: int
    regime_name: str
    probability: float
    duration: float
    volatility: float
    return_mean: float
    var_5: float
    max_drawdown: float
    transition_prob: Dict[int, float]

class RegimeSwitchingModel:
    """
    Advanced regime-switching model for dynamic risk management
    Implements multiple regime detection algorithms and adaptive risk measures
    """
    
    def __init__(self, returns_data: pd.DataFrame, market_indicators: pd.DataFrame = None):
        """
        Initialize regime-switching model
        
        Args:
            returns_data: Portfolio/strategy returns data
            market_indicators: Additional market indicators (VIX, term spreads, etc.)
        """
        self.returns_data = returns_data
        self.market_indicators = market_indicators
        
        # Model components
        self.regime_models = {}
        self.current_regime = None
        self.regime_history = None
        self.transition_matrix = None
        self.regime_metrics = {}
        
        # Features for regime detection
        self.features = None
        self.scaler = StandardScaler()
        
        # Model parameters
        self.n_regimes = 3  # Default: Low vol, Normal, High vol
        self.lookback_window = 30  # Days for feature calculation
        
        logger.info(f"Regime-switching model initialized with {len(returns_data)} observations")
    
    def prepare_features(self) -> pd.DataFrame:
        """
        Prepare features for regime detection
        
        Returns:
            DataFrame with regime detection features
        """
        logger.info("Preparing features for regime detection...")
        
        # Start with returns data
        features_df = self.returns_data.copy()
        
        # Portfolio-level features
        portfolio_returns = features_df.mean(axis=1)  # Equal-weighted portfolio
        
        # Rolling volatility features
        features_df['volatility_5d'] = portfolio_returns.rolling(5).std()
        features_df['volatility_10d'] = portfolio_returns.rolling(10).std()
        features_df['volatility_20d'] = portfolio_returns.rolling(20).std()
        
        # Rolling correlation features
        features_df['avg_correlation'] = features_df.rolling(self.lookback_window).corr().groupby(level=0).mean().mean(axis=1)
        
        # Momentum features
        features_df['return_5d'] = portfolio_returns.rolling(5).mean()
        features_df['return_10d'] = portfolio_returns.rolling(10).mean()
        features_df['return_20d'] = portfolio_returns.rolling(20).mean()
        
        # Skewness and kurtosis
        features_df['skewness_20d'] = portfolio_returns.rolling(20).apply(lambda x: stats.skew(x))
        features_df['kurtosis_20d'] = portfolio_returns.rolling(20).apply(lambda x: stats.kurtosis(x))
        
        # Drawdown features
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        features_df['current_drawdown'] = drawdown
        features_df['max_drawdown_20d'] = drawdown.rolling(20).min()
        
        # VaR estimation (rolling)
        features_df['var_5_20d'] = portfolio_returns.rolling(20).quantile(0.05)
        features_df['var_1_20d'] = portfolio_returns.rolling(20).quantile(0.01)
        
        # Add market indicators if available
        if self.market_indicators is not None:
            # Align dates
            common_dates = features_df.index.intersection(self.market_indicators.index)
            
            for col in self.market_indicators.columns:
                if col in ['VIX', 'vix', 'volatility']:
                    features_df.loc[common_dates, 'vix_level'] = self.market_indicators.loc[common_dates, col]
                    features_df['vix_change'] = features_df['vix_level'].pct_change()
                    features_df['vix_ma_ratio'] = features_df['vix_level'] / features_df['vix_level'].rolling(20).mean()
        
        # Technical indicators
        features_df['return_autocorr'] = portfolio_returns.rolling(20).apply(
            lambda x: x.autocorr(lag=1) if len(x.dropna()) > 5 else 0
        )
        
        # Regime persistence features
        features_df['vol_persistence'] = features_df['volatility_20d'].rolling(10).apply(
            lambda x: 1 if (x.iloc[-1] > x.mean()) == (x.iloc[-2] > x.mean()) else 0
        )
        
        # Remove infinite and NaN values
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.dropna()
        
        # Select final feature set for regime detection
        regime_features = [
            'volatility_5d', 'volatility_10d', 'volatility_20d',
            'avg_correlation', 'return_5d', 'return_10d', 'return_20d',
            'skewness_20d', 'kurtosis_20d', 'current_drawdown',
            'max_drawdown_20d', 'var_5_20d', 'var_1_20d',
            'return_autocorr', 'vol_persistence'
        ]
        
        # Add VIX features if available
        if 'vix_level' in features_df.columns:
            regime_features.extend(['vix_level', 'vix_change', 'vix_ma_ratio'])
        
        # Filter features that exist
        available_features = [f for f in regime_features if f in features_df.columns]
        
        self.features = features_df[available_features].copy()
        
        logger.info(f"Prepared {len(available_features)} features for {len(self.features)} observations")
        return self.features
    
    def fit_gaussian_mixture_model(self, n_regimes: int = None) -> Dict:
        """
        Fit Gaussian Mixture Model for regime detection
        
        Args:
            n_regimes: Number of regimes (default from class)
            
        Returns:
            Dictionary with model results
        """
        if n_regimes is None:
            n_regimes = self.n_regimes
        
        logger.info(f"Fitting Gaussian Mixture Model with {n_regimes} regimes...")
        
        if self.features is None:
            self.prepare_features()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(self.features)
        
        # Fit Gaussian Mixture Model
        gmm = GaussianMixture(
            n_components=n_regimes,
            covariance_type='full',
            max_iter=100,
            random_state=42
        )
        
        regime_labels = gmm.fit_predict(X_scaled)
        regime_probabilities = gmm.predict_proba(X_scaled)
        
        # Store model
        self.regime_models['gmm'] = gmm
        
        # Create regime history
        self.regime_history = pd.DataFrame({
            'regime': regime_labels,
            'date': self.features.index
        }).set_index('date')
        
        # Add regime probabilities
        for i in range(n_regimes):
            self.regime_history[f'prob_regime_{i}'] = regime_probabilities[:, i]
        
        # Calculate regime metrics
        regime_metrics = self._calculate_regime_metrics(regime_labels, n_regimes)
        
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(regime_labels, n_regimes)
        
        results = {
            'model': 'gaussian_mixture',
            'n_regimes': n_regimes,
            'aic': gmm.aic(X_scaled),
            'bic': gmm.bic(X_scaled),
            'log_likelihood': gmm.score(X_scaled),
            'regime_metrics': regime_metrics,
            'transition_matrix': transition_matrix,
            'regime_history': self.regime_history
        }
        
        self.regime_metrics = regime_metrics
        self.transition_matrix = transition_matrix
        
        logger.info(f"GMM fitted successfully - AIC: {results['aic']:.2f}, BIC: {results['bic']:.2f}")
        
        return results
    
    def fit_kmeans_regimes(self, n_regimes: int = None) -> Dict:
        """
        Fit K-means clustering for regime detection
        
        Args:
            n_regimes: Number of regimes
            
        Returns:
            Dictionary with model results
        """
        if n_regimes is None:
            n_regimes = self.n_regimes
        
        logger.info(f"Fitting K-means clustering with {n_regimes} regimes...")
        
        if self.features is None:
            self.prepare_features()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(self.features)
        
        # Fit K-means
        kmeans = KMeans(
            n_clusters=n_regimes,
            random_state=42,
            n_init=10
        )
        
        regime_labels = kmeans.fit_predict(X_scaled)
        
        # Store model
        self.regime_models['kmeans'] = kmeans
        
        # Create regime history
        self.regime_history = pd.DataFrame({
            'regime': regime_labels,
            'date': self.features.index
        }).set_index('date')
        
        # Calculate distances to centroids (proxy for probabilities)
        distances = kmeans.transform(X_scaled)
        probabilities = 1 / (1 + distances)  # Convert distances to probabilities
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        
        for i in range(n_regimes):
            self.regime_history[f'prob_regime_{i}'] = probabilities[:, i]
        
        # Calculate regime metrics
        regime_metrics = self._calculate_regime_metrics(regime_labels, n_regimes)
        
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(regime_labels, n_regimes)
        
        # Calculate inertia (within-cluster sum of squares)
        inertia = kmeans.inertia_
        
        results = {
            'model': 'kmeans',
            'n_regimes': n_regimes,
            'inertia': inertia,
            'regime_metrics': regime_metrics,
            'transition_matrix': transition_matrix,
            'regime_history': self.regime_history
        }
        
        self.regime_metrics = regime_metrics
        self.transition_matrix = transition_matrix
        
        logger.info(f"K-means fitted successfully - Inertia: {inertia:.2f}")
        
        return results
    
    def fit_volatility_regimes(self) -> Dict:
        """
        Fit simple volatility-based regime model
        
        Returns:
            Dictionary with model results
        """
        logger.info("Fitting volatility-based regime model...")
        
        if self.features is None:
            self.prepare_features()
        
        # Use 20-day volatility as the main regime indicator
        vol_col = 'volatility_20d'
        if vol_col not in self.features.columns:
            logger.error("Volatility feature not available")
            return {}
        
        volatility = self.features[vol_col]
        
        # Define regime thresholds based on volatility percentiles
        low_threshold = volatility.quantile(0.33)
        high_threshold = volatility.quantile(0.67)
        
        # Assign regimes
        regime_labels = np.zeros(len(volatility))
        regime_labels[volatility <= low_threshold] = 0  # Low volatility
        regime_labels[(volatility > low_threshold) & (volatility <= high_threshold)] = 1  # Normal
        regime_labels[volatility > high_threshold] = 2  # High volatility
        
        # Create regime history
        self.regime_history = pd.DataFrame({
            'regime': regime_labels.astype(int),
            'date': self.features.index
        }).set_index('date')
        
        # Calculate probabilities based on distance from thresholds
        for i in range(3):
            if i == 0:  # Low vol regime
                prob = np.maximum(0, (low_threshold - volatility) / low_threshold)
            elif i == 1:  # Normal regime
                prob = 1 - np.abs(volatility - volatility.median()) / volatility.std()
            else:  # High vol regime
                prob = np.maximum(0, (volatility - high_threshold) / (volatility.max() - high_threshold))
            
            prob = np.clip(prob, 0, 1)
            self.regime_history[f'prob_regime_{i}'] = prob
        
        # Normalize probabilities
        prob_cols = [f'prob_regime_{i}' for i in range(3)]
        prob_sum = self.regime_history[prob_cols].sum(axis=1)
        for col in prob_cols:
            self.regime_history[col] = self.regime_history[col] / prob_sum
        
        # Calculate regime metrics
        regime_metrics = self._calculate_regime_metrics(regime_labels.astype(int), 3)
        
        # Calculate transition matrix
        transition_matrix = self._calculate_transition_matrix(regime_labels.astype(int), 3)
        
        results = {
            'model': 'volatility_based',
            'n_regimes': 3,
            'thresholds': {'low': low_threshold, 'high': high_threshold},
            'regime_metrics': regime_metrics,
            'transition_matrix': transition_matrix,
            'regime_history': self.regime_history
        }
        
        self.regime_metrics = regime_metrics
        self.transition_matrix = transition_matrix
        
        logger.info("Volatility-based regime model fitted successfully")
        
        return results
    
    def _calculate_regime_metrics(self, regime_labels: np.ndarray, n_regimes: int) -> Dict[int, RegimeMetrics]:
        """Calculate metrics for each regime"""
        
        metrics = {}
        portfolio_returns = self.returns_data.mean(axis=1)
        aligned_returns = portfolio_returns.loc[self.features.index]
        
        for regime_id in range(n_regimes):
            regime_mask = regime_labels == regime_id
            
            if regime_mask.sum() == 0:
                continue
                
            regime_returns = aligned_returns[regime_mask]
            
            # Calculate regime-specific metrics
            volatility = regime_returns.std() * np.sqrt(252)  # Annualized
            return_mean = regime_returns.mean() * 252  # Annualized
            var_5 = np.percentile(regime_returns, 5)
            
            # Drawdown calculation
            cumulative = (1 + regime_returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = abs(drawdown.min())
            
            # Duration calculation
            regime_periods = self._calculate_regime_durations(regime_labels, regime_id)
            avg_duration = np.mean(regime_periods) if regime_periods else 0
            
            # Transition probabilities
            transition_probs = {}
            for next_regime in range(n_regimes):
                transition_probs[next_regime] = self._calculate_transition_prob(
                    regime_labels, regime_id, next_regime
                )
            
            # Regime name
            if n_regimes == 3:
                regime_names = {0: "Low Volatility", 1: "Normal Market", 2: "High Volatility"}
            else:
                regime_names = {i: f"Regime {i}" for i in range(n_regimes)}
            
            metrics[regime_id] = RegimeMetrics(
                regime_id=regime_id,
                regime_name=regime_names.get(regime_id, f"Regime {regime_id}"),
                probability=regime_mask.mean(),
                duration=avg_duration,
                volatility=volatility,
                return_mean=return_mean,
                var_5=abs(var_5),
                max_drawdown=max_drawdown,
                transition_prob=transition_probs
            )
        
        return metrics
    
    def _calculate_regime_durations(self, regime_labels: np.ndarray, regime_id: int) -> List[int]:
        """Calculate duration of each regime period"""
        durations = []
        current_duration = 0
        in_regime = False
        
        for label in regime_labels:
            if label == regime_id:
                if not in_regime:
                    in_regime = True
                    current_duration = 1
                else:
                    current_duration += 1
            else:
                if in_regime:
                    durations.append(current_duration)
                    in_regime = False
                    current_duration = 0
        
        # Add final duration if we end in the regime
        if in_regime:
            durations.append(current_duration)
        
        return durations
    
    def _calculate_transition_prob(self, regime_labels: np.ndarray, from_regime: int, to_regime: int) -> float:
        """Calculate transition probability between regimes"""
        transitions = 0
        from_regime_count = 0
        
        for i in range(len(regime_labels) - 1):
            if regime_labels[i] == from_regime:
                from_regime_count += 1
                if regime_labels[i + 1] == to_regime:
                    transitions += 1
        
        return transitions / from_regime_count if from_regime_count > 0 else 0
    
    def _calculate_transition_matrix(self, regime_labels: np.ndarray, n_regimes: int) -> np.ndarray:
        """Calculate full transition matrix"""
        transition_matrix = np.zeros((n_regimes, n_regimes))
        
        for i in range(n_regimes):
            for j in range(n_regimes):
                transition_matrix[i, j] = self._calculate_transition_prob(regime_labels, i, j)
        
        return transition_matrix
    
    def predict_current_regime(self, model_type: str = 'gmm') -> Tuple[int, np.ndarray]:
        """
        Predict current regime
        
        Args:
            model_type: Type of model to use ('gmm', 'kmeans', 'volatility')
            
        Returns:
            Tuple of (regime_id, probabilities)
        """
        if model_type not in self.regime_models and model_type != 'volatility':
            logger.error(f"Model {model_type} not fitted")
            return -1, np.array([])
        
        if self.features is None or len(self.features) == 0:
            logger.error("No features available for prediction")
            return -1, np.array([])
        
        # Get latest features
        latest_features = self.features.iloc[-1:].values
        
        if model_type == 'gmm':
            X_scaled = self.scaler.transform(latest_features)
            regime_probs = self.regime_models['gmm'].predict_proba(X_scaled)[0]
            regime_id = np.argmax(regime_probs)
            
        elif model_type == 'kmeans':
            X_scaled = self.scaler.transform(latest_features)
            regime_id = self.regime_models['kmeans'].predict(X_scaled)[0]
            
            # Get probabilities from distances
            distances = self.regime_models['kmeans'].transform(X_scaled)[0]
            regime_probs = 1 / (1 + distances)
            regime_probs = regime_probs / regime_probs.sum()
            
        elif model_type == 'volatility':
            # Use latest volatility to determine regime
            latest_vol = self.features['volatility_20d'].iloc[-1]
            
            # Get thresholds from last fit
            if self.regime_history is not None:
                prob_cols = [col for col in self.regime_history.columns if col.startswith('prob_regime_')]
                regime_probs = self.regime_history[prob_cols].iloc[-1].values
                regime_id = np.argmax(regime_probs)
            else:
                regime_id = 1  # Default to normal regime
                regime_probs = np.array([0.33, 0.34, 0.33])
        
        else:
            logger.error(f"Unknown model type: {model_type}")
            return -1, np.array([])
        
        self.current_regime = regime_id
        
        logger.info(f"Current regime prediction: {regime_id} (confidence: {regime_probs[regime_id]:.3f})")
        
        return regime_id, regime_probs
    
    def get_regime_adjusted_risk_measures(self, regime_id: int = None) -> Dict:
        """
        Get risk measures adjusted for current regime
        
        Args:
            regime_id: Specific regime ID, None for current regime
            
        Returns:
            Regime-adjusted risk measures
        """
        if regime_id is None:
            regime_id = self.current_regime
        
        if regime_id is None or regime_id not in self.regime_metrics:
            logger.warning("No regime information available")
            return {}
        
        regime_metric = self.regime_metrics[regime_id]
        
        # Base risk measures
        base_measures = {
            'regime_id': regime_id,
            'regime_name': regime_metric.regime_name,
            'regime_probability': regime_metric.probability,
            'expected_volatility': regime_metric.volatility,
            'expected_return': regime_metric.return_mean,
            'regime_var_5': regime_metric.var_5,
            'regime_max_drawdown': regime_metric.max_drawdown
        }
        
        # Regime-adjusted risk limits
        if regime_id == 0:  # Low volatility regime
            risk_multiplier = 0.8
        elif regime_id == 2:  # High volatility regime  
            risk_multiplier = 1.5
        else:  # Normal regime
            risk_multiplier = 1.0
        
        adjusted_measures = {
            'adjusted_var_limit': 0.02 * risk_multiplier,  # Base 2% VaR limit
            'adjusted_drawdown_limit': 0.05 * risk_multiplier,  # Base 5% drawdown limit
            'adjusted_position_limit': 1.0 / risk_multiplier,  # Inverse relationship for position size
            'risk_multiplier': risk_multiplier
        }
        
        return {**base_measures, **adjusted_measures}
    
    def simulate_regime_scenarios(self, n_simulations: int = 1000, horizon: int = 252) -> Dict:
        """
        Simulate future regime paths and returns
        
        Args:
            n_simulations: Number of simulation paths
            horizon: Simulation horizon in days
            
        Returns:
            Simulation results
        """
        logger.info(f"Running regime scenario simulation: {n_simulations} paths, {horizon} days")
        
        if self.transition_matrix is None:
            logger.error("No transition matrix available")
            return {}
        
        n_regimes = len(self.transition_matrix)
        current_regime = self.current_regime if self.current_regime is not None else 0
        
        # Simulate regime paths
        regime_paths = np.zeros((n_simulations, horizon), dtype=int)
        return_paths = np.zeros((n_simulations, horizon))
        
        for sim in range(n_simulations):
            current_state = current_regime
            
            for day in range(horizon):
                # Record current regime
                regime_paths[sim, day] = current_state
                
                # Generate return based on current regime
                if current_state in self.regime_metrics:
                    regime_metric = self.regime_metrics[current_state]
                    daily_return_mean = regime_metric.return_mean / 252
                    daily_volatility = regime_metric.volatility / np.sqrt(252)
                    
                    return_paths[sim, day] = np.random.normal(daily_return_mean, daily_volatility)
                else:
                    return_paths[sim, day] = np.random.normal(0, 0.01)
                
                # Transition to next regime
                if day < horizon - 1:
                    transition_probs = self.transition_matrix[current_state]
                    current_state = np.random.choice(n_regimes, p=transition_probs)
        
        # Calculate scenario statistics
        final_returns = np.prod(1 + return_paths, axis=1) - 1
        
        scenario_stats = {
            'final_returns': final_returns,
            'mean_return': np.mean(final_returns),
            'volatility': np.std(final_returns),
            'var_5': np.percentile(final_returns, 5),
            'var_1': np.percentile(final_returns, 1),
            'regime_paths': regime_paths,
            'return_paths': return_paths
        }
        
        # Regime time statistics
        regime_time_spent = {}
        for regime_id in range(n_regimes):
            time_in_regime = (regime_paths == regime_id).mean(axis=1)
            regime_time_spent[regime_id] = {
                'mean_time': np.mean(time_in_regime),
                'median_time': np.median(time_in_regime),
                'std_time': np.std(time_in_regime)
            }
        
        scenario_stats['regime_time_statistics'] = regime_time_spent
        
        logger.info(f"Scenario simulation complete - Mean return: {scenario_stats['mean_return']:.3f}")
        
        return scenario_stats
    
    def plot_regime_analysis(self, save_path: str = None) -> None:
        """
        Create comprehensive regime analysis visualization
        
        Args:
            save_path: Path to save the plot
        """
        if self.regime_history is None:
            logger.error("No regime history available for plotting")
            return
        
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle('Regime-Switching Model Analysis', fontsize=16, fontweight='bold')
        
        # 1. Regime timeline
        ax1 = axes[0, 0]
        regime_colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        for regime_id in self.regime_history['regime'].unique():
            regime_mask = self.regime_history['regime'] == regime_id
            regime_dates = self.regime_history.index[regime_mask]
            
            if len(regime_dates) > 0:
                y_values = [regime_id] * len(regime_dates)
                ax1.scatter(regime_dates, y_values, 
                           c=regime_colors[regime_id % len(regime_colors)], 
                           alpha=0.7, s=20, label=f'Regime {regime_id}')
        
        ax1.set_title('Regime Timeline', fontweight='bold')
        ax1.set_ylabel('Regime ID')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Regime probabilities over time
        ax2 = axes[0, 1]
        prob_cols = [col for col in self.regime_history.columns if col.startswith('prob_regime_')]
        
        for i, col in enumerate(prob_cols):
            ax2.plot(self.regime_history.index, self.regime_history[col], 
                    color=regime_colors[i % len(regime_colors)], 
                    label=f'Regime {i}', alpha=0.8)
        
        ax2.set_title('Regime Probabilities Over Time', fontweight='bold')
        ax2.set_ylabel('Probability')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Regime characteristics
        ax3 = axes[1, 0]
        if self.regime_metrics:
            regime_ids = list(self.regime_metrics.keys())
            volatilities = [self.regime_metrics[r].volatility for r in regime_ids]
            returns = [self.regime_metrics[r].return_mean for r in regime_ids]
            
            scatter = ax3.scatter(volatilities, returns, 
                                s=[self.regime_metrics[r].probability * 1000 for r in regime_ids],
                                c=regime_ids, cmap='viridis', alpha=0.7)
            
            for i, regime_id in enumerate(regime_ids):
                ax3.annotate(f'R{regime_id}', (volatilities[i], returns[i]),
                           xytext=(5, 5), textcoords='offset points')
            
            ax3.set_title('Regime Risk-Return Profile', fontweight='bold')
            ax3.set_xlabel('Volatility')
            ax3.set_ylabel('Expected Return')
            ax3.grid(True, alpha=0.3)
        
        # 4. Transition matrix heatmap
        ax4 = axes[1, 1]
        if self.transition_matrix is not None:
            im = ax4.imshow(self.transition_matrix, cmap='Blues', aspect='auto')
            
            # Add text annotations
            for i in range(len(self.transition_matrix)):
                for j in range(len(self.transition_matrix)):
                    text = ax4.text(j, i, f'{self.transition_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black")
            
            ax4.set_title('Regime Transition Matrix', fontweight='bold')
            ax4.set_xlabel('To Regime')
            ax4.set_ylabel('From Regime')
            plt.colorbar(im, ax=ax4)
        
        # 5. Regime duration distribution
        ax5 = axes[2, 0]
        if self.regime_metrics:
            durations = [self.regime_metrics[r].duration for r in self.regime_metrics.keys()]
            regime_names = [self.regime_metrics[r].regime_name for r in self.regime_metrics.keys()]
            
            ax5.bar(regime_names, durations, alpha=0.7, 
                   color=[regime_colors[i % len(regime_colors)] for i in range(len(durations))])
            ax5.set_title('Average Regime Duration', fontweight='bold')
            ax5.set_ylabel('Duration (days)')
            ax5.tick_params(axis='x', rotation=45)
        
        # 6. Feature importance (if using features)
        ax6 = axes[2, 1]
        if self.features is not None and hasattr(self, 'regime_models') and 'gmm' in self.regime_models:
            # Calculate feature importance based on GMM means separation
            feature_names = self.features.columns
            gmm_means = self.regime_models['gmm'].means_
            
            # Calculate range across regimes for each feature
            feature_ranges = np.ptp(gmm_means, axis=0)  # Peak-to-peak (max - min)
            
            # Sort features by importance
            sorted_indices = np.argsort(feature_ranges)[-10:]  # Top 10 features
            
            ax6.barh(range(len(sorted_indices)), feature_ranges[sorted_indices])
            ax6.set_yticks(range(len(sorted_indices)))
            ax6.set_yticklabels([feature_names[i] for i in sorted_indices])
            ax6.set_title('Feature Importance for Regime Detection', fontweight='bold')
            ax6.set_xlabel('Range Across Regimes')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Regime analysis plot saved to {save_path}")
        
        plt.show()
    
    def generate_regime_report(self) -> Dict:
        """
        Generate comprehensive regime analysis report
        
        Returns:
            Detailed regime analysis report
        """
        logger.info("Generating regime analysis report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_summary': {
                'available_models': list(self.regime_models.keys()),
                'current_regime': self.current_regime,
                'n_observations': len(self.features) if self.features is not None else 0,
                'n_features': len(self.features.columns) if self.features is not None else 0
            },
            'regime_metrics': {},
            'transition_matrix': self.transition_matrix.tolist() if self.transition_matrix is not None else [],
            'current_regime_analysis': {},
            'regime_recommendations': []
        }
        
        # Add regime metrics
        if self.regime_metrics:
            for regime_id, metrics in self.regime_metrics.items():
                report['regime_metrics'][regime_id] = {
                    'regime_name': metrics.regime_name,
                    'probability': metrics.probability,
                    'duration': metrics.duration,
                    'volatility': metrics.volatility,
                    'return_mean': metrics.return_mean,
                    'var_5': metrics.var_5,
                    'max_drawdown': metrics.max_drawdown,
                    'transition_probabilities': metrics.transition_prob
                }
        
        # Current regime analysis
        if self.current_regime is not None:
            current_measures = self.get_regime_adjusted_risk_measures()
            report['current_regime_analysis'] = current_measures
        
        # Generate recommendations
        recommendations = self._generate_regime_recommendations()
        report['regime_recommendations'] = recommendations
        
        return report
    
    def _generate_regime_recommendations(self) -> List[str]:
        """Generate regime-based recommendations"""
        recommendations = []
        
        if self.current_regime is None or not self.regime_metrics:
            return ["Insufficient regime data for recommendations"]
        
        current_metric = self.regime_metrics.get(self.current_regime)
        if not current_metric:
            return ["Current regime metrics unavailable"]
        
        # Volatility-based recommendations
        if current_metric.volatility > 0.3:  # High volatility regime
            recommendations.extend([
                "High volatility regime detected - Consider reducing position sizes",
                "Increase hedging activities and monitor Greeks more frequently",
                "Expect higher VaR and implement tighter risk controls"
            ])
        elif current_metric.volatility < 0.15:  # Low volatility regime
            recommendations.extend([
                "Low volatility regime - Opportunity for increased position sizes",
                "Consider volatility selling strategies",
                "Monitor for potential volatility regime shifts"
            ])
        
        # Return-based recommendations
        if current_metric.return_mean < -0.1:  # Negative return regime
            recommendations.extend([
                "Negative return regime - Implement defensive positioning",
                "Consider tail hedging strategies"
            ])
        
        # Transition probability recommendations
        if self.current_regime in current_metric.transition_prob:
            stay_prob = current_metric.transition_prob[self.current_regime]
            if stay_prob < 0.7:
                recommendations.append(
                    f"Low regime persistence ({stay_prob:.1%}) - Expect regime change soon"
                )
        
        # Duration recommendations
        if current_metric.duration < 10:
            recommendations.append("Short regime duration expected - Monitor for quick changes")
        elif current_metric.duration > 50:
            recommendations.append("Long regime duration expected - Plan for extended period")
        
        return recommendations


def main():
    """Example usage of Regime-Switching Model"""
    
    try:
        # Load data from previous phases
        returns_matrix = pd.read_csv('returns_matrix.csv', index_col=0, parse_dates=True)
        
        print("Regime-Switching Models - Phase 3")
        print("=" * 60)
        
        # Initialize regime-switching model
        regime_model = RegimeSwitchingModel(returns_matrix)
        
        # Prepare features
        print("🔍 Preparing features for regime detection...")
        features = regime_model.prepare_features()
        print(f"✅ Prepared {len(features.columns)} features for {len(features)} observations")
        
        # Fit different regime models
        print("\n🎯 Fitting regime models...")
        
        # 1. Gaussian Mixture Model
        gmm_results = regime_model.fit_gaussian_mixture_model(n_regimes=3)
        print(f"  • GMM: AIC={gmm_results['aic']:.2f}, BIC={gmm_results['bic']:.2f}")
        
        # 2. K-means clustering
        kmeans_results = regime_model.fit_kmeans_regimes(n_regimes=3)
        print(f"  • K-means: Inertia={kmeans_results['inertia']:.2f}")
        
        # 3. Volatility-based regimes
        vol_results = regime_model.fit_volatility_regimes()
        print(f"  • Volatility-based: {len(vol_results['regime_metrics'])} regimes identified")
        
        # Current regime prediction
        print("\n📊 Current regime analysis...")
        current_regime, probabilities = regime_model.predict_current_regime('gmm')
        print(f"  • Current regime: {current_regime}")
        print(f"  • Regime probabilities: {probabilities}")
        
        # Regime-adjusted risk measures
        adjusted_measures = regime_model.get_regime_adjusted_risk_measures()
        print(f"\n⚖️  Regime-adjusted measures:")
        print(f"  • Risk multiplier: {adjusted_measures.get('risk_multiplier', 1.0):.2f}")
        print(f"  • Adjusted VaR limit: {adjusted_measures.get('adjusted_var_limit', 0):.3f}")
        
        # Regime scenario simulation
        print("\n🎲 Running regime scenario simulation...")
        scenario_results = regime_model.simulate_regime_scenarios(n_simulations=1000, horizon=63)
        print(f"  • Mean return (3 months): {scenario_results['mean_return']:.3f}")
        print(f"  • VaR (5%): {scenario_results['var_5']:.3f}")
        
        # Create visualizations
        regime_model.plot_regime_analysis('phase3_regime_analysis.png')
        
        # Generate report
        report = regime_model.generate_regime_report()
        
        # Save report
        with open('phase3_regime_report.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Display recommendations
        print(f"\n💡 Regime-based recommendations:")
        for rec in report['regime_recommendations']:
            print(f"  • {rec}")
        
        print("\n✅ Regime-switching analysis complete!")
        print("📁 Results saved to:")
        print("  • phase3_regime_analysis.png")
        print("  • phase3_regime_report.json")
        
    except Exception as e:
        logger.error(f"Error in regime-switching analysis: {str(e)}", exc_info=True)
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()