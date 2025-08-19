import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import logging
import os
import joblib

logger = logging.getLogger(__name__)

class ProphetModel:
    """Prophet Time Series Forecasting Model"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.forecast = None
        self.train_data = None
        self.test_data = None
        
    def build_model(self):
        """
        Build and configure the Prophet model
        
        Returns:
            Configured Prophet model
        """
        # Get model parameters from config
        seasonality_mode = self.config.get('seasonality_mode', 'additive')
        yearly_seasonality = self.config.get('yearly_seasonality', True)
        weekly_seasonality = self.config.get('weekly_seasonality', True)
        daily_seasonality = self.config.get('daily_seasonality', False)
        changepoint_prior_scale = self.config.get('changepoint_prior_scale', 0.1)
        n_changepoints = self.config.get('n_changepoints', 100)
        seasonality_prior_scale = self.config.get('seasonality_prior_scale', 10.0)
        holidays_prior_scale = self.config.get('holidays_prior_scale', 10.0)
        
        # Build model
        model = Prophet(
            seasonality_mode=seasonality_mode,
            yearly_seasonality=yearly_seasonality,
            weekly_seasonality=weekly_seasonality,
            daily_seasonality=daily_seasonality,
            changepoint_prior_scale=changepoint_prior_scale,
            n_changepoints=n_changepoints,
            seasonality_prior_scale=seasonality_prior_scale,
            holidays_prior_scale=holidays_prior_scale
        )
        
        # Add country holidays if specified
        country_name = self.config.get('country_holidays')
        if country_name:
            model.add_country_holidays(country_name=country_name)
        
        # Add custom seasonality if specified
        custom_seasonalities = self.config.get('custom_seasonalities', [])
        for seasonality in custom_seasonalities:
            model.add_seasonality(
                name=seasonality['name'],
                period=seasonality['period'],
                fourier_order=seasonality['fourier_order']
            )
        
        logger.info(f"Prophet model built with changepoint_prior_scale={changepoint_prior_scale}, n_changepoints={n_changepoints}")
        
        return model
    
    def fit(self, data, validation_split=0.2):
        """
        Prepare data and fit the Prophet model
        
        Args:
            data: Input time series data
            validation_split: Fraction of data to use for validation
            
        Returns:
            Fitted model
        """
        try:
            # Convert data to Prophet format if needed
            if isinstance(data, pd.Series):
                df = data.reset_index().rename(columns={'index': 'ds', data.name: 'y'})
            else:
                df = data.copy()
                
            # Ensure required columns are present
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("Data must have 'ds' (date) and 'y' (value) columns")
            
            # Split data into training and validation sets
            train_size = int(len(df) * (1 - validation_split))
            self.train_data = df[:train_size]
            self.test_data = df[train_size:]
            
            logger.info(f"Data split into {len(self.train_data)} training and {len(self.test_data)} validation samples")
            
            # Build model if not already built
            if self.model is None:
                self.model = self.build_model()
            
            # Fit model on training data
            logger.info("Fitting Prophet model...")
            self.model.fit(self.train_data)
            logger.info("Prophet model fitted successfully")
            
            return self.model
            
        except Exception as e:
            logger.error(f"Error fitting Prophet model: {str(e)}")
            raise
    
    def predict(self, periods=None, data=None):
        """
        Generate predictions with the fitted model
        
        Args:
            periods: Number of periods to forecast
            data: Optional dataframe with ds column for specific dates
            
        Returns:
            Predictions dataframe
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before predict()")
            
        try:
            # Use test data periods if periods not specified
            if periods is None and self.test_data is not None:
                periods = len(self.test_data)
            elif periods is None:
                periods = 30  # Default forecast horizon
            
            # Create future dataframe
            if data is not None:
                future = data
            else:
                # Use frequency from config or default to business days
                freq = self.config.get('frequency', 'B')
                future = self.model.make_future_dataframe(
                    periods=periods, 
                    freq=freq,
                    include_history=True
                )
            
            # Make predictions
            logger.info(f"Generating predictions for {periods} periods")
            self.forecast = self.model.predict(future)
            
            return self.forecast
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def forecast_future(self, periods, freq='D'):
        """
        Generate future forecasts
        
        Args:
            periods: Number of periods to forecast
            freq: Frequency of forecasting ('D' for daily, 'W' for weekly, etc.)
            
        Returns:
            Forecast dataframe with future predictions only
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before forecast_future()")
            
        try:
            # Create future dataframe
            future = self.model.make_future_dataframe(
                periods=periods, 
                freq=freq,
                include_history=False
            )
            
            # Make predictions
            logger.info(f"Generating future forecast for {periods} periods with frequency '{freq}'")
            future_forecast = self.model.predict(future)
            
            return future_forecast
            
        except Exception as e:
            logger.error(f"Error generating future forecast: {str(e)}")
            raise
    
    def evaluate(self, test_data=None):
        """
        Evaluate model performance
        
        Args:
            test_data: Test data for evaluation (if None, use validation data)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before evaluate()")
            
        try:
            # Use validation data if not provided
            if test_data is None:
                if self.test_data is None:
                    raise ValueError("No test data provided and no validation data available")
                test_data = self.test_data
            
            # Ensure forecast has been generated
            if self.forecast is None:
                self.predict()
            
            # Get predictions for test period
            test_pred = self.forecast[self.forecast['ds'].isin(test_data['ds'])]
            
            # Merge actual and predicted values
            eval_df = pd.merge(test_data[['ds', 'y']], test_pred[['ds', 'yhat']], on='ds')
            
            # Calculate metrics
            mse = np.mean((eval_df['y'] - eval_df['yhat']) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(eval_df['y'] - eval_df['yhat']))
            mbe = np.mean(eval_df['y'] - eval_df['yhat'])
            
            # Calculate MAPE with handling for zero values
            epsilon = 1e-10
            mape = np.mean(np.abs((eval_df['y'] - eval_df['yhat']) / np.maximum(np.abs(eval_df['y']), epsilon))) * 100
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mbe': mbe,
                'mape': mape
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def plot_forecast(self, include_components=False, save_path=None):
        """
        Plot the forecast
        
        Args:
            include_components: Whether to plot the components
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.model is None or self.forecast is None:
            raise ValueError("Model has not been fitted or no forecast available")
            
        try:
            # Plot the forecast
            fig = self.model.plot(self.forecast)
            
            # Highlight test data if available
            if self.test_data is not None:
                ax = fig.get_axes()[0]
                ax.plot(self.test_data['ds'], self.test_data['y'], 'r.', alpha=0.7, label='Validation')
                ax.legend()
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path)
                logger.info(f"Forecast plot saved to {save_path}")
            
            # Plot components if requested
            if include_components:
                fig_comp = self.model.plot_components(self.forecast)
                
                if save_path:
                    components_path = save_path.replace('.png', '_components.png')
                    fig_comp.savefig(components_path)
                    logger.info(f"Components plot saved to {components_path}")
                
                return fig, fig_comp
            
            return fig
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {str(e)}")
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
            
            # Save Prophet model
            with open(f"{path}/prophet_model.pkl", 'wb') as f:
                joblib.dump(self.model, f)
            
            # Save other attributes
            if self.train_data is not None:
                self.train_data.to_csv(f"{path}/train_data.csv", index=False)
            
            if self.test_data is not None:
                self.test_data.to_csv(f"{path}/test_data.csv", index=False)
            
            # Save config
            import json
            with open(f"{path}/prophet_config.json", "w") as f:
                json.dump(self.config, f)
                
            logger.info(f"Prophet model saved to {path}")
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
            # Load Prophet model
            with open(f"{path}/prophet_model.pkl", 'rb') as f:
                self.model = joblib.load(f)
            
            # Load other attributes
            if os.path.exists(f"{path}/train_data.csv"):
                self.train_data = pd.read_csv(f"{path}/train_data.csv")
            
            if os.path.exists(f"{path}/test_data.csv"):
                self.test_data = pd.read_csv(f"{path}/test_data.csv")
            
            # Load config
            import json
            with open(f"{path}/prophet_config.json", "r") as f:
                self.config = json.load(f)
                
            logger.info(f"Prophet model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
