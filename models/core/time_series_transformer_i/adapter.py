import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from datetime import timedelta

from models.core.prophet.model import ProphetModel
from models.utils.forecast_metrics import calculate_metrics

logger = logging.getLogger(__name__)

class ProphetAdapter:
    """Adapter for the Prophet model to fit into the pipeline"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = ProphetModel(config)
        
    def preprocess_data(self, data_path):
        """
        Preprocess the input data for Prophet modeling
        
        Args:
            data_path: Path to the input data
            
        Returns:
            Preprocessed data in Prophet format
        """
        logger.info(f"Preprocessing data from {data_path}")
        
        try:
            # Read the CSV file
            df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
            
            # Get the target column from config
            target_column = self.config.get('target_column', 'biastg')
            
            # Try to determine target column from file if not specified
            filename = os.path.basename(data_path)
            if 'biastg' in filename and target_column == 'biastg':
                target_column = 'biastg'
                logger.info(f"Target column detected from filename: {target_column}")
            
            # Ensure the target column exists
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Extract target variable
            target_data = df[target_column]
            
            # Fill in missing data
            freq = self.config.get('frequency', 'B')  # Business day frequency
            fill_method = self.config.get('fill_method', 'ffill')
            
            target_data = target_data.asfreq(freq).fillna(method=fill_method)
            
            # Convert to Prophet format
            prophet_df = target_data.reset_index().rename(columns={
                'date': 'ds',
                target_column: 'y'
            })
            
            logger.info(f"Data preprocessing completed. Shape: {prophet_df.shape}")
            return prophet_df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def run_pipeline(self, data_path):
        """
        Run the complete Prophet forecasting pipeline
        
        Args:
            data_path: Path to the input data
            
        Returns:
            DataFrame with forecast results
        """
        logger.info(f"Starting Prophet forecasting pipeline with data from {data_path}")
        
        # Step 1: Preprocess the data
        data = self.preprocess_data(data_path)
        
        # Step 2: Train the model
        validation_split = self.config.get('validation_split', 0.2)
        self.model.fit(data, validation_split=validation_split)
        
        # Step 3: Generate predictions for validation
        self.model.predict()
        
        # Step 4: Generate future forecast
        forecast_horizon = self.config.get('forecast_horizon', 10)
        forecast_freq = self.config.get('forecast_frequency', 'D')
        future_forecast = self.model.forecast_future(periods=forecast_horizon, freq=forecast_freq)
        
        # Step 5: Evaluate the model
        metrics = self.model.evaluate()
        logger.info(f"Model metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}%")
        
        # Step 6: Visualize the results
        output_dir = self.config.get('output_dir', 'outputs/visualizations/prophet')
        os.makedirs(output_dir, exist_ok=True)
        
        include_components = self.config.get('include_components', True)
        self.model.plot_forecast(
            include_components=include_components,
            save_path=f"{output_dir}/prophet_forecast.png"
        )
        
        # Step 7: Prepare the output
        # Extract future predictions only
        future_results = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        future_results = future_results.rename(columns={
            'yhat': 'forecast',
            'yhat_lower': 'lower_bound',
            'yhat_upper': 'upper_bound'
        })
        future_results.set_index('ds', inplace=True)
        
        # Add metrics as attributes to the DataFrame
        for key, value in metrics.items():
            future_results.attrs[key] = value
        
        # Save model if requested
        if self.config.get('save_model', True):
            model_dir = self.config.get('model_dir', 'outputs/models/prophet')
            os.makedirs(model_dir, exist_ok=True)
            self.model.save(model_dir)
        
        # Save results to CSV if requested
        if self.config.get('save_results', True):
            results_dir = self.config.get('results_dir', 'outputs/results/prophet')
            os.makedirs(results_dir, exist_ok=True)
            csv_path = f"{results_dir}/prophet_forecast.csv"
            future_results.to_csv(csv_path)
            logger.info(f"Forecast results saved to {csv_path}")
        
        return future_results
