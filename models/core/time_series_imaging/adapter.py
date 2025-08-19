import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from datetime import timedelta

from models.core.cnn_lstm.model import CNNLSTMTimeSeriesModel
from models.utils.forecast_metrics import calculate_metrics

logger = logging.getLogger(__name__)

class CNNLSTMAdapter:
    """Adapter for the CNN-LSTM Time Series Imaging model to fit into the pipeline"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = CNNLSTMTimeSeriesModel(config)
        
    def preprocess_data(self, data_path):
        """
        Preprocess the input data for CNN-LSTM modeling
        
        Args:
            data_path: Path to the input data
            
        Returns:
            Preprocessed data DataFrame
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
            
            # Apply preprocessing as specified in config
            # Fill missing values
            fill_method = self.config.get('fill_method', 'ffill')
            df = df.fillna(method=fill_method)
            
            logger.info(f"Data preprocessing completed. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def run_pipeline(self, data_path):
        """
        Run the complete CNN-LSTM forecasting pipeline
        
        Args:
            data_path: Path to the input data
            
        Returns:
            DataFrame with forecast results
        """
        logger.info(f"Starting CNN-LSTM forecasting pipeline with data from {data_path}")
        
        # Step 1: Preprocess the data
        data = self.preprocess_data(data_path)
        
        # Step 2: Train the model
        validation_split = self.config.get('validation_split', 0.2)
        test_split = self.config.get('test_split', 0.1)
        self.model.fit(data, validation_split=validation_split, test_split=test_split)
        
        # Step 3: Plot training history
        output_dir = self.config.get('output_dir', 'outputs/visualizations/cnn_lstm')
        os.makedirs(output_dir, exist_ok=True)
        
        history_plot_path = f"{output_dir}/training_history.png"
        self.model.plot_training_history(save_path=history_plot_path)
        
        # Step 4: Evaluate the model
        metrics = self.model.evaluate()
        logger.info(f"Model metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}%")
        
        # Step 5: Generate and plot forecast
        forecast_horizon = self.config.get('forecast_horizon', 10)
        forecast_plot_path = f"{output_dir}/forecast.png"
        
        _, forecast_values, forecast_dates = self.model.plot_forecast(
            data,
            steps=forecast_horizon,
            plot_history=self.config.get('plot_history', 50),
            save_path=forecast_plot_path
        )
        
        # Step 6: Prepare the output
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecast_values
        })
        forecast_df.set_index('date', inplace=True)
        
        # Add metrics as attributes to the DataFrame
        for key, value in metrics.items():
            forecast_df.attrs[key] = value
        
        # Save model if requested
        if self.config.get('save_model', True):
            model_dir = self.config.get('model_dir', 'outputs/models/cnn_lstm')
            os.makedirs(model_dir, exist_ok=True)
            self.model.save(model_dir)
        
        # Save results to CSV if requested
        if self.config.get('save_results', True):
            results_dir = self.config.get('results_dir', 'outputs/results/cnn_lstm')
            os.makedirs(results_dir, exist_ok=True)
            
            # Save forecast
            csv_path = f"{results_dir}/cnn_lstm_forecast.csv"
            forecast_df.to_csv(csv_path)
            logger.info(f"Forecast results saved to {csv_path}")
            
            # Save metrics
            metrics_df = pd.DataFrame([metrics])
            metrics_path = f"{results_dir}/cnn_lstm_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Metrics saved to {metrics_path}")
        
        return forecast_df
