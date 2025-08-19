import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import os
import logging
from datetime import datetime
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

class BiasCorrection:
    """
    Implements various bias correction methods for time series data.
    
    This class provides methods to correct systematic biases in forecasted or modeled data
    by comparing with reference/observed data. It supports multiple correction methods
    including linear scaling, quantile mapping, and distribution mapping.
    """
    
    def __init__(self, config=None):
        """
        Initialize the bias correction processor.
        
        Args:
            config (dict): Configuration parameters for bias correction
        """
        self.config = config or {}
        self.method = self.config.get('method', 'linear_scaling')
        self.reference_period = self.config.get('reference_period', None)
        self.target_variable = self.config.get('target_variable', None)
        self.output_dir = self.config.get('output_dir', 'outputs/generated_data/bias_corrected')
        self.quantile_mapping_params = {}
        self.linear_params = {}
        self.distribution_params = {}
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def fit(self, reference_data, model_data, method=None):
        """
        Fit the bias correction parameters based on reference and model data.
        
        Args:
            reference_data (pd.DataFrame/Series): Reference/observed data
            model_data (pd.DataFrame/Series): Model data to be corrected
            method (str, optional): Bias correction method to use, overrides the default
            
        Returns:
            self: Returns the instance for method chaining
        """
        if method:
            self.method = method
            
        logger.info(f"Fitting bias correction using {self.method} method")
        
        # Ensure data are pandas Series and aligned
        reference_data, model_data = self._prepare_data(reference_data, model_data)
        
        # Filter to reference period if specified
        if self.reference_period:
            start_date, end_date = self.reference_period
            reference_data = reference_data.loc[start_date:end_date]
            model_data = model_data.loc[start_date:end_date]
        
        # Apply the selected bias correction method
        if self.method == 'linear_scaling':
            self._fit_linear_scaling(reference_data, model_data)
        elif self.method == 'quantile_mapping':
            self._fit_quantile_mapping(reference_data, model_data)
        elif self.method == 'distribution_mapping':
            self._fit_distribution_mapping(reference_data, model_data)
        elif self.method == 'delta_change':
            self._fit_delta_change(reference_data, model_data)
        elif self.method == 'ratio_adjustment':
            self._fit_ratio_adjustment(reference_data, model_data)
        elif self.method == 'regression':
            self._fit_regression(reference_data, model_data)
        else:
            raise ValueError(f"Unknown bias correction method: {self.method}")
            
        # Calculate basic bias statistics
        mean_bias = (model_data - reference_data).mean()
        max_bias = (model_data - reference_data).max()
        min_bias = (model_data - reference_data).min()
        
        logger.info(f"Bias statistics - Mean: {mean_bias:.4f}, Min: {min_bias:.4f}, Max: {max_bias:.4f}")
        
        return self
        
    def transform(self, data_to_correct, return_bias=False):
        """
        Apply the fitted bias correction to new data.
        
        Args:
            data_to_correct (pd.DataFrame/Series): Data to apply bias correction to
            return_bias (bool): If True, also return the calculated bias
            
        Returns:
            pd.Series: Bias-corrected data
            pd.Series: Bias values (if return_bias=True)
        """
        logger.info(f"Applying {self.method} bias correction to data")
        
        # Ensure data is a pandas Series
        if isinstance(data_to_correct, pd.DataFrame):
            if self.target_variable and self.target_variable in data_to_correct.columns:
                data = data_to_correct[self.target_variable].copy()
            else:
                data = data_to_correct.iloc[:, 0].copy()
        else:
            data = data_to_correct.copy()
        
        # Apply the selected bias correction method
        if self.method == 'linear_scaling':
            corrected_data = self._apply_linear_scaling(data)
        elif self.method == 'quantile_mapping':
            corrected_data = self._apply_quantile_mapping(data)
        elif self.method == 'distribution_mapping':
            corrected_data = self._apply_distribution_mapping(data)
        elif self.method == 'delta_change':
            corrected_data = self._apply_delta_change(data)
        elif self.method == 'ratio_adjustment':
            corrected_data = self._apply_ratio_adjustment(data)
        elif self.method == 'regression':
            corrected_data = self._apply_regression(data)
        else:
            raise ValueError(f"Unknown bias correction method: {self.method}")
        
        # Calculate bias if requested
        if return_bias:
            bias = data - corrected_data
            return corrected_data, bias
        
        return corrected_data
    
    def save_to_csv(self, corrected_data, original_data=None, filename=None):
        """
        Save the corrected data to a CSV file.
        
        Args:
            corrected_data (pd.Series): Bias-corrected data
            original_data (pd.Series, optional): Original data for comparison
            filename (str, optional): Custom filename
            
        Returns:
            str: Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bias_corrected_{self.method}_{timestamp}.csv"
        
        output_path = os.path.join(self.output_dir, filename)
        
        # Create DataFrame with original and corrected data
        if original_data is not None:
            df = pd.DataFrame({
                'original': original_data,
                'corrected': corrected_data,
                'bias': original_data - corrected_data
            })
        else:
            df = pd.DataFrame({
                'corrected': corrected_data
            })
        
        # Save to CSV
        df.to_csv(output_path)
        logger.info(f"Bias corrected data saved to {output_path}")
        
        return output_path
    
    def plot_correction(self, original_data, corrected_data, reference_data=None, output_path=None):
        """
        Plot the original and bias-corrected data for comparison.
        
        Args:
            original_data (pd.Series): Original uncorrected data
            corrected_data (pd.Series): Bias-corrected data
            reference_data (pd.Series, optional): Reference data for comparison
            output_path (str, optional): Path to save the plot
            
        Returns:
            str: Path to the saved plot
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot data
        ax.plot(original_data.index, original_data.values, 'b-', alpha=0.7, label='Original')
        ax.plot(corrected_data.index, corrected_data.values, 'r-', label='Bias Corrected')
        
        if reference_data is not None:
            ax.plot(reference_data.index, reference_data.values, 'g-', alpha=0.7, label='Reference')
        
        # Add details
        ax.set_title(f'Bias Correction Results ({self.method})')
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        
        # Save plot if path provided
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.output_dir, f"bias_correction_plot_{timestamp}.png")
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        logger.info(f"Bias correction plot saved to {output_path}")
        return output_path
    
    def evaluate_correction(self, reference_data, original_data, corrected_data, verbose=True):
        """
        Evaluate the performance of the bias correction.
        
        Args:
            reference_data (pd.Series): Reference/observed data
            original_data (pd.Series): Original uncorrected data
            corrected_data (pd.Series): Bias-corrected data
            verbose (bool): Whether to print evaluation results
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        # Align time series
        common_index = reference_data.index.intersection(original_data.index).intersection(corrected_data.index)
        reference = reference_data.loc[common_index]
        original = original_data.loc[common_index]
        corrected = corrected_data.loc[common_index]
        
        # Calculate metrics
        original_rmse = np.sqrt(((original - reference) ** 2).mean())
        corrected_rmse = np.sqrt(((corrected - reference) ** 2).mean())
        
        original_mae = (original - reference).abs().mean()
        corrected_mae = (corrected - reference).abs().mean()
        
        original_bias = (original - reference).mean()
        corrected_bias = (corrected - reference).mean()
        
        # Calculate correlation coefficients
        original_corr = original.corr(reference)
        corrected_corr = corrected.corr(reference)
        
        # Compile results
        results = {
            'method': self.method,
            'original_rmse': original_rmse,
            'corrected_rmse': corrected_rmse,
            'rmse_improvement': original_rmse - corrected_rmse,
            'rmse_improvement_pct': 100 * (original_rmse - corrected_rmse) / original_rmse,
            'original_mae': original_mae,
            'corrected_mae': corrected_mae,
            'mae_improvement': original_mae - corrected_mae,
            'mae_improvement_pct': 100 * (original_mae - corrected_mae) / original_mae,
            'original_bias': original_bias,
            'corrected_bias': corrected_bias,
            'bias_reduction': abs(original_bias) - abs(corrected_bias),
            'bias_reduction_pct': 100 * (abs(original_bias) - abs(corrected_bias)) / abs(original_bias) if original_bias != 0 else 0,
            'original_corr': original_corr,
            'corrected_corr': corrected_corr
        }
        
        # Print results if verbose
        if verbose:
            print(f"\nBias Correction Evaluation ({self.method}):")
            print(f"RMSE: {original_rmse:.4f} -> {corrected_rmse:.4f} ({results['rmse_improvement_pct']:.1f}% improvement)")
            print(f"MAE: {original_mae:.4f} -> {corrected_mae:.4f} ({results['mae_improvement_pct']:.1f}% improvement)")
            print(f"Bias: {original_bias:.4f} -> {corrected_bias:.4f} ({results['bias_reduction_pct']:.1f}% reduction)")
            print(f"Correlation: {original_corr:.4f} -> {corrected_corr:.4f}")
        
        return results
        
    def _prepare_data(self, reference_data, model_data):
        """Convert input data to pandas Series with datetime index and align them"""
        # Convert to Series if DataFrame
        if isinstance(reference_data, pd.DataFrame):
            if self.target_variable and self.target_variable in reference_data.columns:
                reference_data = reference_data[self.target_variable]
            else:
                reference_data = reference_data.iloc[:, 0]
                
        if isinstance(model_data, pd.DataFrame):
            if self.target_variable and self.target_variable in model_data.columns:
                model_data = model_data[self.target_variable]
            else:
                model_data = model_data.iloc[:, 0]
        
        # Ensure indices are datetime
        if not isinstance(reference_data.index, pd.DatetimeIndex):
            try:
                reference_data.index = pd.to_datetime(reference_data.index)
            except:
                logger.warning("Could not convert reference data index to datetime")
                
        if not isinstance(model_data.index, pd.DatetimeIndex):
            try:
                model_data.index = pd.to_datetime(model_data.index)
            except:
                logger.warning("Could not convert model data index to datetime")
        
        # Align data on common dates
        common_index = reference_data.index.intersection(model_data.index)
        if len(common_index) == 0:
            raise ValueError("No common dates between reference and model data")
            
        return reference_data.loc[common_index], model_data.loc[common_index]
    
    #---------------------------------------------------------------------------
    # Linear Scaling / Shift Method
    #---------------------------------------------------------------------------
    def _fit_linear_scaling(self, reference_data, model_data):
        """Fit linear scaling parameters"""
        # Calculate mean of reference and model data
        ref_mean = reference_data.mean()
        model_mean = model_data.mean()
        
        # Calculate standard deviation of reference and model data
        ref_std = reference_data.std()
        model_std = model_data.std()
        
        # Store parameters
        self.linear_params = {
            'ref_mean': ref_mean,
            'model_mean': model_mean,
            'ref_std': ref_std,
            'model_std': model_std,
            'additive_factor': ref_mean - model_mean,
            'scaling_factor': ref_std / model_std if model_std > 0 else 1.0
        }
        
        logger.info(f"Linear scaling parameters fitted: additive={self.linear_params['additive_factor']:.4f}, scaling={self.linear_params['scaling_factor']:.4f}")
    
    def _apply_linear_scaling(self, data):
        """Apply linear scaling correction"""
        if not self.linear_params:
            raise ValueError("Linear scaling parameters not fitted. Call fit() first.")
        
        # Apply additive and scaling correction
        correction_type = self.config.get('linear_correction_type', 'both')
        
        if correction_type == 'additive':
            # Add the bias
            corrected = data + self.linear_params['additive_factor']
        elif correction_type == 'scaling':
            # Scale the data
            corrected = self.linear_params['model_mean'] + (data - self.linear_params['model_mean']) * self.linear_params['scaling_factor']
        else:  # 'both'
            # Apply both corrections
            corrected = self.linear_params['ref_mean'] + (data - self.linear_params['model_mean']) * self.linear_params['scaling_factor']
        
        return corrected
    
    #---------------------------------------------------------------------------
    # Quantile Mapping Method
    #---------------------------------------------------------------------------
    def _fit_quantile_mapping(self, reference_data, model_data):
        """Fit quantile mapping parameters"""
        # Number of quantiles
        n_quantiles = self.config.get('n_quantiles', 100)
        
        # Calculate quantiles for both datasets
        ref_quantiles = np.percentile(reference_data, np.linspace(0, 100, n_quantiles))
        model_quantiles = np.percentile(model_data, np.linspace(0, 100, n_quantiles))
        
        # Create interpolation function
        self.quantile_mapping_params = {
            'ref_quantiles': ref_quantiles,
            'model_quantiles': model_quantiles,
            'n_quantiles': n_quantiles,
            'interpolator': interp1d(
                model_quantiles, 
                ref_quantiles, 
                bounds_error=False, 
                fill_value=(ref_quantiles[0], ref_quantiles[-1])
            )
        }
        
        logger.info(f"Quantile mapping fitted with {n_quantiles} quantiles")
    
    def _apply_quantile_mapping(self, data):
        """Apply quantile mapping correction"""
        if not self.quantile_mapping_params:
            raise ValueError("Quantile mapping parameters not fitted. Call fit() first.")
        
        # Apply the interpolation function to correct values
        corrected = self.quantile_mapping_params['interpolator'](data)
        
        return pd.Series(corrected, index=data.index, name=data.name)
    
    #---------------------------------------------------------------------------
    # Distribution Mapping Method
    #---------------------------------------------------------------------------
    def _fit_distribution_mapping(self, reference_data, model_data):
        """Fit distribution mapping parameters"""
        # Select distribution
        dist_name = self.config.get('distribution', 'gamma')
        
        if dist_name == 'normal':
            # Fit normal distribution to both datasets
            ref_mean, ref_std = stats.norm.fit(reference_data)
            model_mean, model_std = stats.norm.fit(model_data)
            
            self.distribution_params = {
                'distribution': 'normal',
                'ref_params': (ref_mean, ref_std),
                'model_params': (model_mean, model_std)
            }
            
        elif dist_name == 'gamma':
            # For gamma, ensure all values are positive
            ref_data = reference_data - reference_data.min() + 0.01 if reference_data.min() <= 0 else reference_data
            model_data = model_data - model_data.min() + 0.01 if model_data.min() <= 0 else model_data
            
            # Fit gamma distribution to both datasets
            ref_alpha, _, ref_loc, ref_scale = stats.gamma.fit(ref_data)
            model_alpha, _, model_loc, model_scale = stats.gamma.fit(model_data)
            
            self.distribution_params = {
                'distribution': 'gamma',
                'ref_params': (ref_alpha, ref_loc, ref_scale),
                'model_params': (model_alpha, model_loc, model_scale),
                'ref_min': reference_data.min(),
                'model_min': model_data.min()
            }
            
        else:
            raise ValueError(f"Unsupported distribution: {dist_name}")
            
        logger.info(f"Distribution mapping fitted using {dist_name} distribution")
    
    def _apply_distribution_mapping(self, data):
        """Apply distribution mapping correction"""
        if not self.distribution_params:
            raise ValueError("Distribution parameters not fitted. Call fit() first.")
        
        dist_name = self.distribution_params['distribution']
        
        if dist_name == 'normal':
            # Get parameters
            ref_mean, ref_std = self.distribution_params['ref_params']
            model_mean, model_std = self.distribution_params['model_params']
            
            # Calculate probabilities using model distribution
            probs = stats.norm.cdf(data, loc=model_mean, scale=model_std)
            
            # Calculate corrected values using reference distribution
            corrected = stats.norm.ppf(probs, loc=ref_mean, scale=ref_std)
            
        elif dist_name == 'gamma':
            # Handle negative values
            model_min = self.distribution_params['model_min']
            data_adjusted = data - model_min + 0.01 if model_min <= 0 else data
            
            # Get parameters
            ref_alpha, ref_loc, ref_scale = self.distribution_params['ref_params']
            model_alpha, model_loc, model_scale = self.distribution_params['model_params']
            
            # Calculate probabilities using model distribution
            probs = stats.gamma.cdf(data_adjusted, model_alpha, loc=model_loc, scale=model_scale)
            
            # Calculate corrected values using reference distribution
            corrected = stats.gamma.ppf(probs, ref_alpha, loc=ref_loc, scale=ref_scale)
            
            # Adjust back if needed
            ref_min = self.distribution_params['ref_min']
            if ref_min <= 0:
                corrected = corrected + ref_min - 0.01
        
        return pd.Series(corrected, index=data.index, name=data.name)
    
    #---------------------------------------------------------------------------
    # Delta Change Method
    #---------------------------------------------------------------------------
    def _fit_delta_change(self, reference_data, model_data):
        """Fit delta change parameters"""
        # Calculate mean change
        mean_delta = (model_data.mean() - reference_data.mean())
        
        # Calculate relative change
        if reference_data.mean() != 0:
            relative_delta = model_data.mean() / reference_data.mean()
        else:
            relative_delta = 1.0
            
        # Calculate monthly/seasonal changes if data has datetime index
        monthly_delta = {}
        if isinstance(reference_data.index, pd.DatetimeIndex):
            # Group by month and calculate mean delta
            ref_monthly = reference_data.groupby(reference_data.index.month).mean()
            model_monthly = model_data.groupby(model_data.index.month).mean()
            
            # Calculate monthly deltas for each month
            for month in range(1, 13):
                if month in ref_monthly.index and month in model_monthly.index:
                    if ref_monthly[month] != 0:
                        monthly_delta[month] = model_monthly[month] / ref_monthly[month]
                    else:
                        monthly_delta[month] = 1.0
        
        self.delta_params = {
            'mean_delta': mean_delta,
            'relative_delta': relative_delta,
            'monthly_delta': monthly_delta,
            'correction_type': self.config.get('delta_correction_type', 'additive')
        }
        
        logger.info(f"Delta change parameters fitted: mean_delta={mean_delta:.4f}, relative_delta={relative_delta:.4f}")
    
    def _apply_delta_change(self, data):
        """Apply delta change correction"""
        if not hasattr(self, 'delta_params'):
            raise ValueError("Delta change parameters not fitted. Call fit() first.")
        
        correction_type = self.delta_params['correction_type']
        
        # Apply monthly correction if available and data has datetime index
        if self.delta_params['monthly_delta'] and isinstance(data.index, pd.DatetimeIndex):
            corrected = data.copy()
            
            for month, delta in self.delta_params['monthly_delta'].items():
                # Get month mask
                month_mask = data.index.month == month
                
                if correction_type == 'additive':
                    corrected.loc[month_mask] = data.loc[month_mask] - self.delta_params['mean_delta']
                else:  # multiplicative
                    corrected.loc[month_mask] = data.loc[month_mask] / delta
                    
            return corrected
            
        # Apply global correction
        if correction_type == 'additive':
            return data - self.delta_params['mean_delta']
        else:  # multiplicative
            return data / self.delta_params['relative_delta']
    
    #---------------------------------------------------------------------------
    # Ratio Adjustment Method
    #---------------------------------------------------------------------------
    def _fit_ratio_adjustment(self, reference_data, model_data):
        """Fit ratio adjustment parameters"""
        # Calculate ratio between reference and model data
        if model_data.mean() != 0:
            ratio = reference_data.mean() / model_data.mean()
        else:
            ratio = 1.0
            
        # Calculate monthly ratios if data has datetime index
        monthly_ratio = {}
        if isinstance(reference_data.index, pd.DatetimeIndex):
            # Group by month and calculate mean 
            ref_monthly = reference_data.groupby(reference_data.index.month).mean()
            model_monthly = model_data.groupby(model_data.index.month).mean()
            
            # Calculate monthly ratios for each month
            for month in range(1, 13):
                if month in ref_monthly.index and month in model_monthly.index:
                    if model_monthly[month] != 0:
                        monthly_ratio[month] = ref_monthly[month] / model_monthly[month]
                    else:
                        monthly_ratio[month] = 1.0
        
        self.ratio_params = {
            'ratio': ratio,
            'monthly_ratio': monthly_ratio
        }
        
        logger.info(f"Ratio adjustment parameters fitted: ratio={ratio:.4f}")
    
    def _apply_ratio_adjustment(self, data):
        """Apply ratio adjustment correction"""
        if not hasattr(self, 'ratio_params'):
            raise ValueError("Ratio adjustment parameters not fitted. Call fit() first.")
        
        # Apply monthly ratio if available and data has datetime index
        if self.ratio_params['monthly_ratio'] and isinstance(data.index, pd.DatetimeIndex):
            corrected = data.copy()
            
            for month, ratio in self.ratio_params['monthly_ratio'].items():
                # Get month mask
                month_mask = data.index.month == month
                corrected.loc[month_mask] = data.loc[month_mask] * ratio
                    
            return corrected
            
        # Apply global ratio
        return data * self.ratio_params['ratio']
    
    #---------------------------------------------------------------------------
    # Regression Method
    #---------------------------------------------------------------------------
    def _fit_regression(self, reference_data, model_data):
        """Fit regression parameters"""
        # Create X (model) and y (reference) arrays
        X = model_data.values.reshape(-1, 1)
        y = reference_data.values
        
        # Fit regression model
        self.regression_model = LinearRegression()
        self.regression_model.fit(X, y)
        
        # Store parameters
        self.regression_params = {
            'slope': self.regression_model.coef_[0],
            'intercept': self.regression_model.intercept_,
            'r2': self.regression_model.score(X, y)
        }
        
        logger.info(f"Regression parameters fitted: slope={self.regression_params['slope']:.4f}, intercept={self.regression_params['intercept']:.4f}, R²={self.regression_params['r2']:.4f}")
    
    def _apply_regression(self, data):
        """Apply regression correction"""
        if not hasattr(self, 'regression_model'):
            raise ValueError("Regression parameters not fitted. Call fit() first.")
        
        # Reshape data for prediction
        X = data.values.reshape(-1, 1)
        
        # Apply regression model
        corrected_values = self.regression_model.predict(X)
        
        return pd.Series(corrected_values, index=data.index, name=data.name)
