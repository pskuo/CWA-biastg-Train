import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from datetime import timedelta

from models.core.rnn.model import RNNModel
from models.utils.forecast_metrics import calculate_metrics

logger = logging.getLogger(__name__)

class RNNAdapter:
    """Adapter for the RNN model to fit into the pipeline"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = RNNModel(config)
        
    def preprocess_data(self, data_path):
        """
        Preprocess the input data for RNN modeling
        
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
            
            logger.info(f"Data preprocessing completed. Shape: {target_data.shape}")
            return target_data
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def visualize_results(self, data, predictions, forecast=None, output_path=None):
        """
        Visualize the model results
        
        Args:
            data: Original time series data
            predictions: Model predictions
            forecast: Future forecast values (optional)
            output_path: Path to save the visualization
        """
        plt.figure(figsize=(14, 6))
        
        # Plot original data
        plt.plot(data.index, data, label='Observed')
        
        # Plot predictions
        pred_index = None
        if isinstance(predictions, pd.Series) or isinstance(predictions, pd.DataFrame):
            pred_index = predictions.index
            plt.plot(pred_index, predictions, color='r', label='Predicted')
        else:
            # If predictions don't have an index, use the last part of the data index
            pred_len = len(predictions)
            if pred_len <= len(data):
                pred_index = data.index[-pred_len:]
                plt.plot(pred_index, predictions, color='r', label='Predicted')
        
        # Plot future forecast if provided
        if forecast is not None:
            if pred_index is not None:
                last_date = pred_index[-1]
            else:
                last_date = data.index[-1]
                
            forecast_dates = pd.date_range(
                start=last_date + timedelta(days=1),
                periods=len(forecast),
                freq='D'  # Assuming daily data
            )
            plt.plot(forecast_dates, forecast, 'g--', label='Forecast')
        
        plt.title('RNN GRU: Observed vs Predicted vs Forecast')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Results visualization saved to {output_path}")
        
        plt.close()
    
    def run_pipeline(self, data_path):
        """
        Run the complete RNN forecasting pipeline
        
        Args:
            data_path: Path to the input data
            
        Returns:
            DataFrame with forecast results
        """
        logger.info(f"Starting RNN forecasting pipeline with data from {data_path}")
        
        # Step 1: Preprocess the data
        data = self.preprocess_data(data_path)
        
        # Step 2: Train the model
        validation_split = self.config.get('validation_split', 0.2)
        self.model.fit(data, validation_split=validation_split)
        
        # Step 3: Generate predictions for validation
        predictions = self.model.predict()
        
        # Step 4: Generate future forecast
        forecast_horizon = self.config.get('forecast_horizon', 10)
        forecast = self.model.forecast(data, steps=forecast_horizon)
        
        # Step 5: Evaluate the model
        metrics = self.model.evaluate()
        logger.info(f"Model metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}%")
        
        # Step 6: Visualize the results
        output_dir = self.config.get('output_dir', 'outputs/visualizations/rnn')
        self.visualize_results(
            data=data,
            predictions=predictions,
            forecast=forecast,
            output_path=f"{output_dir}/rnn_forecast.png"
        )
        
        # Plot training history
        self.model.plot_training_history(
            save_path=f"{output_dir}/rnn_training_history.png"
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
        
        # Add metrics as attributes to the DataFrame for later use
        for key, value in metrics.items():
            forecast_df.attrs[key] = value
        
        # Save model if requested
        if self.config.get('save_model', True):
            model_dir = self.config.get('model_dir', 'outputs/models/rnn')
            os.makedirs(model_dir, exist_ok=True)
            self.model.save(model_dir)
        
        # Save results to CSV if requested
        if self.config.get('save_results', True):
            results_dir = self.config.get('results_dir', 'outputs/results/rnn')
            os.makedirs(results_dir, exist_ok=True)
            csv_path = f"{results_dir}/rnn_forecast.csv"
            forecast_df.to_csv(csv_path)
            logger.info(f"Forecast results saved to {csv_path}")
        
        return forecast_df
