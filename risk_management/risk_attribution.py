"""
Risk Attribution and Decomposition Module (Section 6 Priority)
Implements factor risk models and risk budgeting for TradeSteward options trading
"""

import numpy as np
import pandas as pd
import riskfolio as rp
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
import logging
from datetime import datetime

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class FactorRiskModel:
    """
    Factor risk model implementation for options trading portfolios
    Focuses on risk attribution and decomposition (Section 6)
    """
    
    def __init__(self, returns_data: pd.DataFrame, factor_data: pd.DataFrame = None):
        """
        Initialize factor risk model
        
        Args:
            returns_data: DataFrame with strategy/asset returns (date index)
            factor_data: DataFrame with factor returns (date index)
        """
        self.returns_data = returns_data
        self.factor_data = factor_data
        self.factor_loadings = None
        self.factor_exposures = None
        self.idiosyncratic_risk = None
        self.portfolio_weights = None
        self.risk_contributions = None
        
    def estimate_factor_loadings(self, method: str = 'ols') -> pd.DataFrame:
        """
        Estimate factor loadings using regression analysis
        
        Args:
            method: 'ols' for OLS regression, 'pca' for Principal Component Analysis
            
        Returns:
            DataFrame with factor loadings
        """
        logger.info(f"Estimating factor loadings using {method} method...")
        
        if self.factor_data is None:
            logger.warning("No factor data provided, creating synthetic factors")
            self.factor_data = self._create_synthetic_factors()
        
        # Align data
        common_dates = self.returns_data.index.intersection(self.factor_data.index)
        returns_aligned = self.returns_data.loc[common_dates]
        factors_aligned = self.factor_data.loc[common_dates]
        
        if method == 'ols':
            loadings = self._estimate_ols_loadings(returns_aligned, factors_aligned)
        elif method == 'pca':
            loadings = self._estimate_pca_loadings(returns_aligned)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.factor_loadings = loadings
        return loadings
    
    def _estimate_ols_loadings(self, returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        """
        Estimate factor loadings using OLS regression
        
        Args:
            returns: Aligned returns data
            factors: Aligned factor data
            
        Returns:
            Factor loadings DataFrame
        """
        loadings_dict = {}
        r_squared_dict = {}
        
        for asset in returns.columns:
            # Run regression for each asset
            y = returns[asset].values
            X = factors.values
            
            # Remove NaN values
            mask = ~(np.isnan(y) | np.isnan(X).any(axis=1))
            y_clean = y[mask]
            X_clean = X[mask]
            
            if len(y_clean) < 10:  # Need minimum observations
                logger.warning(f"Insufficient data for {asset}, using zero loadings")
                loadings_dict[asset] = {factor: 0.0 for factor in factors.columns}
                r_squared_dict[asset] = 0.0
                continue
            
            # Fit regression
            reg = LinearRegression().fit(X_clean, y_clean)
            
            # Store loadings
            loadings_dict[asset] = {
                factor: loading for factor, loading in zip(factors.columns, reg.coef_)
            }
            
            # Calculate R-squared
            r_squared_dict[asset] = reg.score(X_clean, y_clean)
        
        # Convert to DataFrame
        loadings_df = pd.DataFrame(loadings_dict).T
        loadings_df.index.name = 'Asset'
        
        # Store R-squared values
        self.factor_r_squared = r_squared_dict
        
        logger.info(f"Factor loadings estimated for {len(loadings_df)} assets")
        logger.info(f"Average R-squared: {np.mean(list(r_squared_dict.values())):.3f}")
        
        return loadings_df
    
    def _estimate_pca_loadings(self, returns: pd.DataFrame, n_components: int = 5) -> pd.DataFrame:
        """
        Estimate factor loadings using Principal Component Analysis
        
        Args:
            returns: Returns data
            n_components: Number of principal components
            
        Returns:
            Factor loadings DataFrame
        """
        # Remove NaN values
        returns_clean = returns.dropna()
        
        if len(returns_clean) < 20:
            logger.warning("Insufficient data for PCA, using correlation-based factors")
            return self._correlation_based_factors(returns)
        
        # Standardize returns
        scaler = StandardScaler()
        returns_scaled = scaler.fit_transform(returns_clean)
        
        # Fit PCA
        pca = PCA(n_components=n_components)
        pca.fit(returns_scaled)
        
        # Create factor loadings
        loadings_df = pd.DataFrame(
            pca.components_.T,
            index=returns.columns,
            columns=[f'PC{i+1}' for i in range(n_components)]
        )
        
        # Store explained variance
        self.explained_variance_ratio = pca.explained_variance_ratio_
        
        logger.info(f"PCA loadings estimated with {n_components} components")
        logger.info(f"Explained variance: {pca.explained_variance_ratio_.sum():.3f}")
        
        return loadings_df
    
    def _correlation_based_factors(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Create simple correlation-based factors when PCA fails
        
        Args:
            returns: Returns data
            
        Returns:
            Simple factor loadings
        """
        logger.info("Creating correlation-based factors...")
        
        # Calculate correlation matrix
        corr_matrix = returns.corr()
        
        # Create simple factors
        n_assets = len(returns.columns)
        loadings_dict = {}
        
        for asset in returns.columns:
            # Market factor (equal weights)
            market_loading = 1.0 / n_assets
            
            # Volatility factor (based on volatility)
            vol = returns[asset].std()
            vol_loading = vol / returns.std().mean()
            
            # Momentum factor (based on recent performance)
            momentum = returns[asset].tail(30).mean() if len(returns) >= 30 else 0
            momentum_loading = momentum / returns.mean().mean() if returns.mean().mean() != 0 else 0
            
            loadings_dict[asset] = {
                'Market': market_loading,
                'Volatility': vol_loading,
                'Momentum': momentum_loading
            }
        
        return pd.DataFrame(loadings_dict).T
    
    def _create_synthetic_factors(self) -> pd.DataFrame:
        """
        Create synthetic factor data when real factors are not available
        
        Returns:
            Synthetic factor returns DataFrame
        """
        logger.info("Creating synthetic factor data...")
        
        dates = self.returns_data.index
        n_days = len(dates)
        
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Create realistic factor returns
        factors = pd.DataFrame({
            'Market': np.random.normal(0.0005, 0.015, n_days),  # Market factor
            'Volatility': np.random.normal(-0.0001, 0.05, n_days),  # VIX-like
            'Interest_Rate': np.random.normal(0.0001, 0.002, n_days),  # Rate changes
            'Momentum': np.random.normal(0.0002, 0.01, n_days),  # Momentum
            'Value': np.random.normal(0.0001, 0.008, n_days)  # Value factor
        }, index=dates)
        
        return factors
    
    def calculate_factor_exposures(self, portfolio_weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Calculate portfolio factor exposures
        
        Args:
            portfolio_weights: Dictionary of asset weights, None for equal weights
            
        Returns:
            Portfolio factor exposures
        """
        if self.factor_loadings is None:
            logger.error("Factor loadings not estimated. Call estimate_factor_loadings() first.")
            return pd.DataFrame()
        
        # Default to equal weights if not provided
        if portfolio_weights is None:
            assets = self.factor_loadings.index
            portfolio_weights = {asset: 1.0/len(assets) for asset in assets}
        
        self.portfolio_weights = portfolio_weights
        
        # Calculate weighted factor exposures
        exposures = {}
        for factor in self.factor_loadings.columns:
            exposure = sum(
                portfolio_weights.get(asset, 0) * self.factor_loadings.loc[asset, factor]
                for asset in self.factor_loadings.index
                if asset in portfolio_weights
            )
            exposures[factor] = exposure
        
        exposures_df = pd.DataFrame([exposures], index=['Portfolio'])
        self.factor_exposures = exposures_df
        
        logger.info("Portfolio factor exposures calculated")
        return exposures_df
    
    def decompose_portfolio_risk(self, portfolio_weights: Dict[str, float] = None) -> Dict:
        """
        Decompose portfolio risk into factor and idiosyncratic components
        
        Args:
            portfolio_weights: Dictionary of asset weights
            
        Returns:
            Risk decomposition results
        """
        logger.info("Decomposing portfolio risk...")
        
        if self.factor_loadings is None:
            self.estimate_factor_loadings()
        
        if portfolio_weights is None:
            assets = self.factor_loadings.index
            portfolio_weights = {asset: 1.0/len(assets) for asset in assets}
        
        # Align data for risk calculation
        common_dates = self.returns_data.index.intersection(self.factor_data.index)
        returns_aligned = self.returns_data.loc[common_dates]
        factors_aligned = self.factor_data.loc[common_dates]
        
        # Calculate factor covariance matrix
        factor_cov = factors_aligned.cov().values
        
        # Calculate portfolio factor loadings
        portfolio_loadings = []
        assets_in_portfolio = [asset for asset in self.factor_loadings.index if asset in portfolio_weights]
        
        for factor in self.factor_loadings.columns:
            factor_loading = sum(
                portfolio_weights[asset] * self.factor_loadings.loc[asset, factor]
                for asset in assets_in_portfolio
            )
            portfolio_loadings.append(factor_loading)
        
        portfolio_loadings = np.array(portfolio_loadings)
        
        # Calculate factor risk contribution
        factor_risk_variance = np.dot(portfolio_loadings, np.dot(factor_cov, portfolio_loadings))
        factor_risk_volatility = np.sqrt(factor_risk_variance)
        
        # Calculate total portfolio risk
        portfolio_returns = sum(
            portfolio_weights[asset] * returns_aligned[asset]
            for asset in assets_in_portfolio
        )
        total_risk_variance = portfolio_returns.var()
        total_risk_volatility = np.sqrt(total_risk_variance)
        
        # Calculate idiosyncratic risk
        idiosyncratic_risk_variance = max(0, total_risk_variance - factor_risk_variance)
        idiosyncratic_risk_volatility = np.sqrt(idiosyncratic_risk_variance)
        
        # Risk contributions by factor
        factor_contributions = {}
        for i, factor in enumerate(self.factor_loadings.columns):
            factor_contrib_variance = (
                portfolio_loadings[i] * np.dot(factor_cov[i], portfolio_loadings)
            )
            factor_contributions[factor] = {
                'variance_contribution': factor_contrib_variance,
                'volatility_contribution': factor_contrib_variance / (2 * total_risk_volatility) if total_risk_volatility > 0 else 0,
                'percentage_of_risk': factor_contrib_variance / total_risk_variance * 100 if total_risk_variance > 0 else 0
            }
        
        risk_decomposition = {
            'total_risk_volatility': total_risk_volatility * np.sqrt(252),  # Annualized
            'factor_risk_volatility': factor_risk_volatility * np.sqrt(252),
            'idiosyncratic_risk_volatility': idiosyncratic_risk_volatility * np.sqrt(252),
            'factor_risk_percentage': factor_risk_variance / total_risk_variance * 100 if total_risk_variance > 0 else 0,
            'idiosyncratic_risk_percentage': idiosyncratic_risk_variance / total_risk_variance * 100 if total_risk_variance > 0 else 0,
            'factor_contributions': factor_contributions,
            'portfolio_loadings': dict(zip(self.factor_loadings.columns, portfolio_loadings))
        }
        
        logger.info(f"Risk decomposition complete - Total risk: {risk_decomposition['total_risk_volatility']:.2%}")
        logger.info(f"Factor risk: {risk_decomposition['factor_risk_percentage']:.1f}%, "
                   f"Idiosyncratic: {risk_decomposition['idiosyncratic_risk_percentage']:.1f}%")
        
        return risk_decomposition
    
    def calculate_risk_budgets(self, portfolio_weights: Dict[str, float] = None) -> pd.DataFrame:
        """
        Calculate risk budgets for each asset and strategy
        
        Args:
            portfolio_weights: Dictionary of asset weights
            
        Returns:
            Risk budget DataFrame
        """
        logger.info("Calculating risk budgets...")
        
        if portfolio_weights is None:
            assets = self.factor_loadings.index if self.factor_loadings is not None else self.returns_data.columns
            portfolio_weights = {asset: 1.0/len(assets) for asset in assets}
        
        # Calculate portfolio returns
        common_dates = self.returns_data.index
        portfolio_returns = pd.Series(0, index=common_dates)
        
        for asset, weight in portfolio_weights.items():
            if asset in self.returns_data.columns:
                portfolio_returns += weight * self.returns_data[asset]
        
        portfolio_risk = portfolio_returns.std() * np.sqrt(252)  # Annualized
        
        # Calculate marginal risk contributions
        risk_budgets = []
        
        for asset in portfolio_weights.keys():
            if asset not in self.returns_data.columns:
                continue
                
            weight = portfolio_weights[asset]
            
            # Calculate marginal contribution to risk
            asset_returns = self.returns_data[asset]
            covariance_with_portfolio = asset_returns.cov(portfolio_returns)
            marginal_risk = covariance_with_portfolio / portfolio_risk if portfolio_risk > 0 else 0
            
            # Risk contribution
            risk_contribution = weight * marginal_risk
            risk_contribution_percentage = risk_contribution / portfolio_risk * 100 if portfolio_risk > 0 else 0
            
            risk_budgets.append({
                'Asset': asset,
                'Weight': weight,
                'Marginal_Risk': marginal_risk,
                'Risk_Contribution': risk_contribution,
                'Risk_Contribution_Percentage': risk_contribution_percentage,
                'Risk_Budget_Ratio': risk_contribution_percentage / (weight * 100) if weight > 0 else 0
            })
        
        risk_budget_df = pd.DataFrame(risk_budgets)
        self.risk_contributions = risk_budget_df
        
        logger.info("Risk budgets calculated")
        return risk_budget_df
    
    def plot_risk_attribution(self, save_path: str = None) -> None:
        """
        Create visualizations for risk attribution analysis
        
        Args:
            save_path: Path to save the plot
        """
        if self.factor_loadings is None:
            logger.error("Factor loadings not available for plotting")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Factor Loadings Heatmap
        sns.heatmap(self.factor_loadings, annot=True, cmap='RdYlBu', center=0, ax=ax1)
        ax1.set_title('Factor Loadings Matrix', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Assets/Strategies')
        
        # 2. Factor Exposures (if available)
        if self.factor_exposures is not None:
            exposures = self.factor_exposures.iloc[0]
            ax2.bar(exposures.index, exposures.values, color='steelblue', alpha=0.7)
            ax2.set_title('Portfolio Factor Exposures', fontsize=14, fontweight='bold')
            ax2.set_ylabel('Exposure')
            ax2.tick_params(axis='x', rotation=45)
        else:
            ax2.text(0.5, 0.5, 'No Factor Exposures\nCalculated', 
                    transform=ax2.transAxes, ha='center', va='center', fontsize=12)
            ax2.set_title('Portfolio Factor Exposures', fontsize=14, fontweight='bold')
        
        # 3. Risk Contributions (if available)
        if self.risk_contributions is not None:
            risk_contrib = self.risk_contributions.set_index('Asset')['Risk_Contribution_Percentage']
            ax3.pie(risk_contrib.values, labels=risk_contrib.index, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Risk Contribution by Asset', fontsize=14, fontweight='bold')
        else:
            ax3.text(0.5, 0.5, 'No Risk Contributions\nCalculated', 
                    transform=ax3.transAxes, ha='center', va='center', fontsize=12)
            ax3.set_title('Risk Contribution by Asset', fontsize=14, fontweight='bold')
        
        # 4. R-squared by Asset (if available)
        if hasattr(self, 'factor_r_squared'):
            r_squared_series = pd.Series(self.factor_r_squared)
            ax4.bar(r_squared_series.index, r_squared_series.values, color='green', alpha=0.7)
            ax4.set_title('Factor Model R-squared by Asset', fontsize=14, fontweight='bold')
            ax4.set_ylabel('R-squared')
            ax4.tick_params(axis='x', rotation=45)
            ax4.set_ylim(0, 1)
        else:
            ax4.text(0.5, 0.5, 'No R-squared\nData Available', 
                    transform=ax4.transAxes, ha='center', va='center', fontsize=12)
            ax4.set_title('Factor Model R-squared by Asset', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Risk attribution plot saved to {save_path}")
        
        plt.show()
    
    def generate_risk_report(self) -> Dict:
        """
        Generate comprehensive risk attribution report
        
        Returns:
            Dictionary with risk attribution results
        """
        logger.info("Generating comprehensive risk attribution report...")
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'total_assets': len(self.returns_data.columns),
                'analysis_period': {
                    'start': self.returns_data.index.min().strftime('%Y-%m-%d'),
                    'end': self.returns_data.index.max().strftime('%Y-%m-%d')
                }
            }
        }
        
        # Factor model results
        if self.factor_loadings is not None:
            report['factor_model'] = {
                'number_of_factors': len(self.factor_loadings.columns),
                'factor_names': list(self.factor_loadings.columns),
                'factor_loadings': self.factor_loadings.to_dict(),
                'average_r_squared': np.mean(list(self.factor_r_squared.values())) if hasattr(self, 'factor_r_squared') else None
            }
        
        # Portfolio factor exposures
        if self.factor_exposures is not None:
            report['factor_exposures'] = self.factor_exposures.to_dict()
        
        # Risk contributions
        if self.risk_contributions is not None:
            report['risk_contributions'] = self.risk_contributions.to_dict()
        
        # Risk decomposition (calculate if not done)
        if self.portfolio_weights is not None:
            risk_decomp = self.decompose_portfolio_risk(self.portfolio_weights)
            report['risk_decomposition'] = risk_decomp
        
        logger.info("Risk attribution report generated")
        return report


def main():
    """Example usage of the Risk Attribution module"""
    
    # This would typically be called after data_loader.py
    try:
        # Load sample data (in practice, use data_loader output)
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        
        # Create sample strategy returns
        returns_data = pd.DataFrame({
            'Iron_Condor': np.random.normal(0.0005, 0.02, len(dates)),
            'Call_Side': np.random.normal(0.0008, 0.025, len(dates)),
            'Put_Side': np.random.normal(0.0003, 0.018, len(dates)),
            'Strangle': np.random.normal(0.0006, 0.022, len(dates))
        }, index=dates)
        
        # Create sample factor data
        factor_data = pd.DataFrame({
            'Market': np.random.normal(0.0005, 0.015, len(dates)),
            'Volatility': np.random.normal(-0.0001, 0.05, len(dates)),
            'Interest_Rate': np.random.normal(0.0001, 0.002, len(dates))
        }, index=dates)
        
        # Initialize risk attribution model
        risk_model = FactorRiskModel(returns_data, factor_data)
        
        # Estimate factor loadings
        loadings = risk_model.estimate_factor_loadings(method='ols')
        print("Factor Loadings:")
        print(loadings)
        
        # Calculate portfolio exposures
        portfolio_weights = {
            'Iron_Condor': 0.4,
            'Call_Side': 0.3,
            'Put_Side': 0.2,
            'Strangle': 0.1
        }
        
        exposures = risk_model.calculate_factor_exposures(portfolio_weights)
        print("\nPortfolio Factor Exposures:")
        print(exposures)
        
        # Risk decomposition
        risk_decomp = risk_model.decompose_portfolio_risk(portfolio_weights)
        print(f"\nRisk Decomposition:")
        print(f"Total Risk: {risk_decomp['total_risk_volatility']:.2%}")
        print(f"Factor Risk: {risk_decomp['factor_risk_percentage']:.1f}%")
        print(f"Idiosyncratic Risk: {risk_decomp['idiosyncratic_risk_percentage']:.1f}%")
        
        # Risk budgets
        risk_budgets = risk_model.calculate_risk_budgets(portfolio_weights)
        print("\nRisk Budgets:")
        print(risk_budgets)
        
        # Generate comprehensive report
        report = risk_model.generate_risk_report()
        
        # Create visualization
        risk_model.plot_risk_attribution('risk_management/risk_attribution_analysis.png')
        
        logger.info("Risk attribution analysis complete!")
        
    except Exception as e:
        logger.error(f"Error in risk attribution analysis: {str(e)}")


if __name__ == "__main__":
    main()