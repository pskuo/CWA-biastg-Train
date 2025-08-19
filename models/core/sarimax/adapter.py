import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from datetime import timedelta
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.core.sarimax.model import SARIMAXModel
from models.utils.forecast_metrics import calculate_metrics

logger = logging.getLogger(__name__)

class SARIMAXAdapter:
    """Adapter for the SARIMAX model to fit into the pipeline"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = SARIMAXModel(config)
        
    def preprocess_data(self, data_path):
        """
        Preprocess the input data for SARIMAX modeling
        
        Args:
            data_path: Path to the input data
            
        Returns:
            Preprocessed data
        """
        logger.info(f"Preprocessing data from {data_path}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
            
            # Get the target column from config
            target_column = self.config.get('target_column', 'biastg')
            
            # Ensure the target column exists
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Handle frequency and missing values
            freq = self.config.get('frequency', 'B')  # 'B' for business days
            fill_method = self.config.get('fill_method', 'ffill')
            
            df = df.asfreq(freq).fillna(method=fill_method)
            
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
        split_ratio = self.config.get('train_test_split', 0.8)
        
        # Split point
        split_idx = int(split_ratio * len(data))
        
        train_data = data.iloc[:split_idx]
        test_data = data.iloc[split_idx:]
        
        logger.info(f"Data split: train={train_data.shape}, test={test_data.shape}")
        
        return train_data, test_data
        
    def visualize_forecast(self, original_data, predictions, test_data=None, output_path=None):
        """
        Visualize the forecast against actual data
        
        Args:
            original_data: Original time series data
            predictions: Forecasted values
            test_data: Actual test data for comparison (if available)
            output_path: Path to save the visualization
        """
        plt.figure(figsize=(12, 6))
        
        # Get the target column
        target_column = self.config.get('target_column', 'biastg')
        
        # Plot the original data
        history_points = self.config.get('visualization_history', 100)
        if len(original_data) > history_points:
            plt.plot(original_data.index[-history_points:], 
                    original_data[target_column][-history_points:], 
                    label='Historical Data')
        else:
            plt.plot(original_data.index, 
                    original_data[target_column], 
                    label='Historical Data')
        
        # Plot test data if provided
        if test_data is not None:
            plt.plot(test_data.index, 
                    test_data[target_column], 
                    label='Actual Values', 
                    color='green')
        
        # Plot predictions
        if isinstance(predictions.index[0], pd.Timestamp):
            plt.plot(predictions.index, 
                    predictions, 
                    label='SARIMAX Forecast', 
                    linestyle='--', 
                    color='red')
        else:
            # Create forecast dates if predictions don't have dates
            last_date = original_data.index[-1]
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=len(predictions),
                freq='D'  # Assuming daily data
            )
            plt.plot(forecast_dates, 
                    predictions, 
                    label='SARIMAX Forecast', 
                    linestyle='--', 
                    color='red')
        
        plt.xlabel('Date')
        plt.ylabel(target_column)
        plt.title('Time Series Data with SARIMAX Forecast')
        plt.legend()
        plt.grid(True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Forecast visualization saved to {output_path}")
        
        plt.close()
        
    def run_pipeline(self, data_path):
        """
        Run the complete SARIMAX forecasting pipeline
        
        Args:
            data_path: Path to the input data
            
        Returns:
            DataFrame with forecast results
        """
        logger.info(f"Starting SARIMAX forecasting pipeline with data from {data_path}")
        
        # Step 1: Preprocess the data
        data = self.preprocess_data(data_path)
        
        # Get target column
        target_column = self.config.get('target_column', 'biastg')
        
        # Step 2: Split the data
        train_data, test_data = self.split_data(data)
        
        # Step 3: Train the model
        self.model.fit(train_data, target_column=target_column)
        
        # Step 4: Generate forecast
        forecast_horizon = self.config.get('forecast_horizon', 10)
        
        # Get exogenous variables for forecasting
        exog_test = test_data.drop(columns=[target_column])
        
        # Generate predictions for test period
        test_predictions = self.model.predict(exog_test, steps=len(test_data))
        
        # Step 5: Evaluate the forecast
        metrics = calculate_metrics(test_data[target_column].values, test_predictions)
        logger.info(f"Test metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}%")
        
        # Step 6: Generate future forecast (beyond the test period)
        # In a real scenario, you would need to generate exogenous variables for future
        # For now, we'll just predict the next 'forecast_horizon' steps
        future_forecast = self.model.predict(steps=forecast_horizon)
        
        # Step 7: Visualize the results
        output_dir = self.config.get('output_dir', 'outputs/visualizations')
        self.visualize_forecast(
            original_data=train_data, 
            predictions=pd.Series(test_predictions, index=test_data.index),
            test_data=test_data,
            output_path=f"{output_dir}/sarimax_test_forecast.png"
        )
        
        # Step 8: Prepare the output
        forecast_index = pd.date_range(
            start=data.index[-1] + timedelta(days=1),
            periods=len(future_forecast),
            freq='D'  # Assuming daily data
        )
        
        forecast_df = pd.DataFrame({
            'date': forecast_index,
            'forecast': future_forecast
        })
        forecast_df.set_index('date', inplace=True)
        
        # Add metrics as attributes to the DataFrame for later use
        for key, value in metrics.items():
            forecast_df.attrs[key] = value
        
        return forecast_df
