import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import pickle

from darts import TimeSeries
from darts.models import ExponentialSmoothing
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, rmse
from darts.backtesting import backtest_forecasting, plot_residuals_analysis

logger = logging.getLogger(__name__)

class ETSModel:
    """Exponential Smoothing Time Series Forecasting Model using Darts"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.train_series = None
        self.val_series = None
        self.test_series = None
        self.predictions = None
        self.best_model_type = None
        self.optimized = False
        
    def create_series(self, data, time_col='date', value_col=None):
        """
        Create a Darts TimeSeries object
        
        Args:
            data: Input time series data
            time_col: Column name for timestamps
            value_col: Column name for values
            
        Returns:
            Darts TimeSeries object
        """
        try:
            # If it's already a TimeSeries object, return it
            if isinstance(data, TimeSeries):
                return data
                
            # If it's a pandas Series, convert to DataFrame
            if isinstance(data, pd.Series):
                df = data.reset_index()
                value_col = value_col or data.name
                time_col = time_col or df.columns[0]
            else:
                df = data.copy()
                
            # Create TimeSeries object
            series = TimeSeries.from_dataframe(df, time_col, value_col)
            return series
            
        except Exception as e:
            logger.error(f"Error creating TimeSeries: {str(e)}")
            raise
    
    def scale_data(self, series, scaler=None):
        """
        Scale the time series data
        
        Args:
            series: Input TimeSeries
            scaler: Optional pre-fitted scaler
            
        Returns:
            Scaled TimeSeries and scaler
        """
        try:
            # Create a new scaler if not provided
            if scaler is None:
                scaler_type = self.config.get('scaler_type', 'standard')
                if scaler_type.lower() == 'minmax':
                    scaler = Scaler(scaler_type='minmax')
                else:
                    scaler = Scaler(scaler_type='standard')
                    
            # Scale the series
            scaled_series = scaler.fit_transform(series) if not scaler.fitted else scaler.transform(series)
            
            return scaled_series, scaler
            
        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            raise
    
    def build_model(self, model_type=None):
        """
        Build and configure the ETS model
        
        Args:
            model_type: Optional tuple of (error_type, trend_type, seasonal_type, damped_trend)
            
        Returns:
            Configured Darts ExponentialSmoothing model
        """
        try:
            # Use default model type if not provided
            if model_type is None:
                # Get model parameters from config
                error_type = self.config.get('error_type', 'add')
                trend_type = self.config.get('trend_type', 'add')
                seasonal_type = self.config.get('seasonal_type', 'add')
                damped_trend = self.config.get('damped_trend', None)
                
                model_type = (error_type, trend_type, seasonal_type, damped_trend)
            
            # Get other parameters from config
            seasonal_periods = self.config.get('seasonal_periods', None)
            
            # Build the model
            model = ExponentialSmoothing(
                error_type=model_type[0],
                trend_type=model_type[1],
                seasonal_type=model_type[2],
                damped_trend=model_type[3],
                seasonal_periods=seasonal_periods
            )
            
            logger.info(f"ETS model built with parameters: {model_type}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def optimize_model_type(self, series):
        """
        Find the best ETS model type using backtesting
        
        Args:
            series: Input TimeSeries
            
        Returns:
            Best model type as tuple (error_type, trend_type, seasonal_type, damped_trend)
        """
        try:
            # Define model types to try
            if self.config.get('model_types'):
                model_types = self.config.get('model_types')
            else:
                model_types = [
                    ('add', 'add', 'add', None),
                    ('mul', 'add', 'add', None),
                    ('add', 'add', 'mul', None),
                    ('mul', 'add', 'mul', None),
                    ('add', 'mul', 'add', None),
                    ('mul', 'mul', 'add', None),
                    ('add', 'mul', 'mul', None),
                    ('mul', 'mul', 'mul', None),
                    ('add', 'add', None, None),
                    ('mul', 'add', None, None),
                    ('add', 'mul', None, None),
                    ('mul', 'mul', None, None),
                    ('add', None, None, None),
                    ('mul', None, None, None)
                ]
            
            # Set validation split for backtesting
            validation_split = self.config.get('validation_split', 0.2)
            n_samples = len(series)
            val_size = int(n_samples * validation_split)
            
            # Calculate the split date for backtesting
            split_index = n_samples - val_size
            split_date = series.time_index[split_index]
            
            # Initialize tracking variables
            min_error = float('inf')
            best_model_type = None
            
            # Test each model type
            for model_type in model_types:
                logger.info(f"Testing ETS model type: {model_type}")
                
                try:
                    # Create model with current type
                    model = ExponentialSmoothing(
                        error_type=model_type[0],
                        trend_type=model_type[1],
                        seasonal_type=model_type[2],
                        damped_trend=model_type[3],
                        seasonal_periods=self.config.get('seasonal_periods', None)
                    )
                    
                    # Use backtesting to evaluate model
                    forecast_horizon = self.config.get('forecast_horizon', 1)
                    stride = self.config.get('backtest_stride', 1)
                    metric = self.config.get('optimization_metric', 'mae')
                    
                    errors = backtest_forecasting(
                        series,
                        model,
                        start=pd.Timestamp(split_date),
                        forecast_horizon=forecast_horizon,
                        stride=stride,
                        metric=metric,
                        verbose=False
                    )
                    
                    # Calculate mean error
                    mean_error = np.mean(errors)
                    
                    logger.info(f"Model type {model_type} has mean {metric}: {mean_error:.4f}")
                    
                    # Update best model if this one is better
                    if mean_error < min_error:
                        min_error = mean_error
                        best_model_type = model_type
                        
                except Exception as e:
                    logger.warning(f"Error testing model type {model_type}: {str(e)}")
                    continue
            
            if best_model_type:
                logger.info(f"Best ETS model type: {best_model_type} with mean error: {min_error:.4f}")
                return best_model_type
            else:
                # Fallback to default model if all optimization attempts fail
                default_model = ('add', 'add', 'add', None)
                logger.warning(f"Model optimization failed. Using default model type: {default_model}")
                return default_model
                
        except Exception as e:
            logger.error(f"Error in model optimization: {str(e)}")
            default_model = ('add', 'add', 'add', None)
            logger.warning(f"Using default model type: {default_model}")
            return default_model
    
    def fit(self, data, validation_split=0.2, optimize=None):
        """
        Prepare data and train the ETS model
        
        Args:
            data: Input time series data
            validation_split: Fraction of data to use for validation
            optimize: Whether to optimize model type (if None, use config)
            
        Returns:
            Trained model
        """
        try:
            # Create TimeSeries object if not already
            series = self.create_series(data)
            
            # Create a scaler and scale the data
            self.scaler = Scaler(scaler_type=self.config.get('scaler_type', 'standard'))
            series_scaled, self.scaler = self.scale_data(series, self.scaler)
            
            # Split the data
            # First determine validation and test splits
            total_split = validation_split
            if isinstance(total_split, float):
                val_split_idx = int(len(series_scaled) * (1 - total_split))
                split_date = series_scaled.time_index[val_split_idx]
            else:
                split_date = pd.Timestamp(total_split)
            
            # Split into training and validation
            train_series, val_series = series_scaled.split_after(split_date)
            
            # Store the series for later use
            self.train_series = train_series
            self.val_series = val_series
            self.original_series = series
            
            logger.info(f"Data split into training ({len(train_series)} points) and "
                      f"validation ({len(val_series)} points)")
            
            # Determine whether to optimize
            if optimize is None:
                optimize = self.config.get('optimize_model', True)
            
            # Optimize model type if requested
            if optimize:
                logger.info("Optimizing ETS model type...")
                self.best_model_type = self.optimize_model_type(train_series)
                self.optimized = True
            else:
                # Use model type from config
                error_type = self.config.get('error_type', 'add')
                trend_type = self.config.get('trend_type', 'add')
                seasonal_type = self.config.get('seasonal_type', 'add')
                damped_trend = self.config.get('damped_trend', None)
                
                self.best_model_type = (error_type, trend_type, seasonal_type, damped_trend)
            
            # Build the model with the best or specified type
            self.model = self.build_model(self.best_model_type)
            
            # Train the model
            logger.info(f"Training ETS model with type: {self.best_model_type}...")
            self.model.fit(train_series)
            
            logger.info("ETS model trained successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, n_steps=None, series=None):
        """
        Generate predictions using the fitted model
        
        Args:
            n_steps: Number of steps to predict
            series: Optional TimeSeries for prediction (uses train_series if None)
            
        Returns:
            Predicted TimeSeries
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before predict()")
            
        try:
            # Use validation series length if n_steps not specified
            if n_steps is None:
                n_steps = len(self.val_series) if self.val_series is not None else 1
            
            # Generate predictions
            logger.info(f"Generating predictions for {n_steps} steps")
            
            # Use provided series or training series
            input_series = series if series is not None else self.train_series
            
            # Make predictions
            predictions = self.model.predict(n_steps)
            
            # Invert scaling
            predictions_unscaled = self.scaler.inverse_transform(predictions)
            
            # Store predictions
            self.predictions = predictions_unscaled
            
            return predictions_unscaled
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def evaluate(self, predictions=None, actuals=None):
        """
        Evaluate model performance
        
        Args:
            predictions: Predicted TimeSeries (if None, use stored predictions)
            actuals: Actual TimeSeries for comparison (if None, use val_series)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before evaluate()")
            
        try:
            # Use stored predictions if not provided
            if predictions is None:
                if self.predictions is None:
                    raise ValueError("No predictions available. Call predict() before evaluate()")
                predictions = self.predictions
            
            # Use validation data if actuals not provided
            if actuals is None:
                if self.val_series is None:
                    raise ValueError("No validation data available and no actuals provided")
                    
                # Ensure actuals are unscaled
                actuals = self.scaler.inverse_transform(self.val_series)
            elif isinstance(actuals, pd.Series) or isinstance(actuals, pd.DataFrame):
                # Convert to TimeSeries if it's a pandas object
                actuals = self.create_series(actuals)
            
            # Truncate series to match lengths if needed
            min_length = min(len(predictions), len(actuals))
            predictions_trunc = predictions[:min_length]
            actuals_trunc = actuals[:min_length]
            
            # Calculate metrics
            rmse_value = rmse(actuals_trunc, predictions_trunc)
            mae_value = mae(actuals_trunc, predictions_trunc)
            mape_value = mape(actuals_trunc, predictions_trunc)
            
            # Calculate MBE (mean bias error)
            # Convert to numpy arrays for calculation
            y_true = actuals_trunc.values().flatten()
            y_pred = predictions_trunc.values().flatten()
            mbe_value = np.mean(y_true - y_pred)
            
            metrics = {
                'rmse': rmse_value,
                'mae': mae_value,
                'mape': mape_value,
                'mbe': mbe_value
            }
            
            logger.info(f"Model evaluation: RMSE={rmse_value:.4f}, MAE={mae_value:.4f}, "
                      f"MAPE={mape_value:.4f}%, MBE={mbe_value:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def plot_predictions(self, predictions=None, actuals=None, start_date=None, end_date=None, save_path=None):
        """
        Plot model predictions against actual values
        
        Args:
            predictions: Predicted TimeSeries (if None, use stored predictions)
            actuals: Actual TimeSeries for comparison (if None, use original_series)
            start_date: Start date for plotting
            end_date: End date for plotting
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            # Use stored predictions if not provided
            if predictions is None:
                if self.predictions is None:
                    raise ValueError("No predictions available. Call predict() before plot_predictions()")
                predictions = self.predictions
            
            # Use original data if actuals not provided
            if actuals is None:
                if not hasattr(self, 'original_series'):
                    raise ValueError("No original data available and no actuals provided")
                actuals = self.original_series
            
            # Create the plot
            plt.figure(figsize=(14, 6))
            
            # Plot actual data
            label = 'Observed'
            if start_date is not None or end_date is not None:
                actuals_slice = actuals.slice(start_date, end_date)
                actuals_slice.plot(label=label)
            else:
                actuals.plot(label=label)
            
            # Plot predictions
            label = 'Predicted'
            if start_date is not None or end_date is not None:
                predictions_slice = predictions.slice(start_date, end_date)
                predictions_slice.plot(label=label, color='r')
            else:
                predictions.plot(label=label, color='r')
            
            plt.title('ETS Model: Observed vs Predicted')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Predictions plot saved to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting predictions: {str(e)}")
            raise
    
    def plot_residuals(self, save_path=None):
        """
        Plot residuals analysis
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.model is None or self.predictions is None:
            raise ValueError("Model must be fitted and predictions generated before plotting residuals")
            
        try:
            # Get validation data
            val_unscaled = self.scaler.inverse_transform(self.val_series)
            
            # Calculate residuals
            residuals = val_unscaled.values() - self.predictions.values()
            
            # Create the figure
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot residual values
            axes[0, 0].plot(self.predictions.time_index, residuals)
            axes[0, 0].set_title('Residuals Over Time')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Residual')
            axes[0, 0].grid(True)
            
            # Plot histogram of residuals
            axes[0, 1].hist(residuals, bins=20)
            axes[0, 1].set_title('Histogram of Residuals')
            axes[0, 1].set_xlabel('Residual')
            axes[0, 1].set_ylabel('Frequency')
            axes[0, 1].grid(True)
            
            # Plot residuals vs fitted values
            axes[1, 0].scatter(self.predictions.values(), residuals)
            axes[1, 0].set_title('Residuals vs Fitted Values')
            axes[1, 0].set_xlabel('Fitted Value')
            axes[1, 0].set_ylabel('Residual')
            axes[1, 0].grid(True)
            
            # QQ plot of residuals
            from scipy import stats
            import numpy as np
            
            residuals_flat = residuals.flatten()
            stats.probplot(residuals_flat, dist="norm", plot=axes[1, 1])
            axes[1, 1].set_title('Normal Q-Q Plot')
            axes[1, 1].grid(True)
            
            plt.tight_layout()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Residuals analysis plot saved to {save_path}")
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting residuals: {str(e)}")
            raise
    
    def forecast(self, n_steps):
        """
        Generate future forecasts
        
        Args:
            n_steps: Number of steps to forecast
            
        Returns:
            Forecasted TimeSeries
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before forecast()")
            
        try:
            # Generate forecast
            logger.info(f"Generating {n_steps}-step forecast")
            
            # Make predictions
            forecast = self.model.predict(n_steps)
            
            # Invert scaling
            forecast_unscaled = self.scaler.inverse_transform(forecast)
            
            return forecast_unscaled
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def save(self, path):
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save Darts ETS model
            with open(f"{path}/ets_model.pkl", 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save scaler
            self.scaler.save(f"{path}/ets_scaler.pkl")
            
            # Save best model type
            with open(f"{path}/ets_model_type.txt", 'w') as f:
                f.write(str(self.best_model_type))
            
            # Save config
            import json
            with open(f"{path}/ets_config.json", "w") as f:
                # Convert any non-serializable types to strings
                config_serializable = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) 
                                     else v for k, v in self.config.items()}
                json.dump(config_serializable, f)
                
            logger.info(f"ETS model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path):
        """
        Load a trained model
        
        Args:
            path: Path to load the model from
        """
        try:
            # Load Darts ETS model
            with open(f"{path}/ets_model.pkl", 'rb') as f:
                self.model = pickle.load(f)
            
            # Load scaler
            self.scaler = Scaler.load(f"{path}/ets_scaler.pkl")
            
            # Load best model type
            with open(f"{path}/ets_model_type.txt", 'r') as f:
                self.best_model_type = eval(f.read())
            
            # Load config
            import json
            with open(f"{path}/ets_config.json", "r") as f:
                self.config = json.load(f)
                
            logger.info(f"ETS model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
