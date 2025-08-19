import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from datetime import timedelta

from models.core.arima.model import ARIMAModel
from models.utils.forecast_metrics import calculate_metrics

logger = logging.getLogger(__name__)

class ARIMAAdapter:
    """Adapter for the ARIMA model to fit into the pipeline"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = ARIMAModel(config)
        
    def preprocess_data(self, data_path):
        """
        Preprocess the input data for ARIMA modeling
        
        Args:
            data_path: Path to the input data
            
        Returns:
            Preprocessed data
        """
        logger.info(f"Preprocessing data from {data_path}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
            
            # Fill in missing data using forward filling or other methods defined in config
            fill_method = self.config.get('fill_method', 'ffill')
            df.fillna(method=fill_method, inplace=True)
            
            # Apply any additional preprocessing steps
            if self.config.get('remove_outliers', False):
                # Implement outlier removal logic
                pass
                
            logger.info(f"Data preprocessing completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
            
    def split_data(self, data):
        """
        Split the data into training and testing sets
        
        Args:
            data: Preprocessed data
            
        Returns:
            Tuple of (train_data, test_data)
        """
        test_size = self.config.get('test_size', 10)
        
        train_data = data.iloc[:-test_size]
        test_data = data.iloc[-test_size:]
        
        logger.info(f"Data split: train={train_data.shape}, test={test_data.shape}")
        
        return train_data, test_data
        
    def visualize_forecast(self, data, forecast, output_path=None):
        """
        Visualize the forecast against actual data
        
        Args:
            data: Original time series data
            forecast: Forecasted values
            output_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 6))
        
        # Plot the last 100 points of original data for better visibility
        history_points = self.config.get('visualization_history', 100)
        plt.plot(data.index[-history_points:], data.iloc[-history_points:], label='Historical Data')
        
        # Create forecast dates
        forecast_dates = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=len(forecast),
            freq='D'  # Assuming daily data
        )
        
        plt.plot(forecast_dates, forecast, label='ARIMA Forecast', linestyle='--', color='red')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.title('Time Series Data with ARIMA Forecast')
        plt.legend()
        plt.grid(True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Forecast visualization saved to {output_path}")
        
        plt.close()
        
    def run_pipeline(self, data_path):
        """
        Run the complete ARIMA forecasting pipeline
        
        Args:
            data_path: Path to the input data
            
        Returns:
            DataFrame with forecast results
        """
        logger.info(f"Starting ARIMA forecasting pipeline with data from {data_path}")
        
        # Step 1: Preprocess the data
        data = self.preprocess_data(data_path)
        
        # Step 2: Split the data
        train_data, test_data = self.split_data(data)
        
        # Step 3: Train the model
        self.model.fit(train_data)
        
        # Step 4: Generate forecast
        forecast_horizon = self.config.get('forecast_horizon', 10)
        forecast = self.model.predict(forecast_horizon)
        
        # Step 5: Evaluate the forecast if test data is available
        metrics = None
        if not test_data.empty and len(test_data) == len(forecast):
            metrics = calculate_metrics(test_data.values.flatten(), forecast)
            logger.info(f"Forecast metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}%")
        
        # Step 6: Visualize the results
        output_dir = self.config.get('output_dir', 'outputs/visualizations')
        self.visualize_forecast(
            data, 
            forecast, 
            output_path=f"{output_dir}/arima_forecast.png"
        )
        
        # Step 7: Prepare the output
        forecast_index = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=len(forecast),
            freq='D'  # Assuming daily data
        )
        
        forecast_df = pd.DataFrame({
            'date': forecast_index,
            'forecast': forecast
        })
        forecast_df.set_index('date', inplace=True)
        
        if metrics:
            # Add metrics as attributes to the DataFrame for later use
            for key, value in metrics.items():
                forecast_df.attrs[key] = value
        
        return forecast_df
