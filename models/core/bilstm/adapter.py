import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from datetime import timedelta
import re

from models.core.bilstm_pytorch.model import BiLSTMModel
from models.utils.forecast_metrics import calculate_metrics

logger = logging.getLogger(__name__)

class BiLSTMPytorchAdapter:
    """Adapter for the PyTorch BiLSTM model to fit into the pipeline"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = BiLSTMModel(config)
        
    def extract_metadata_from_filename(self, filename):
        """
        Extract metadata (lat, lon, date) from filename
        
        Args:
            filename: Filename to extract metadata from
            
        Returns:
            Dictionary with extracted metadata
        """
        metadata = {}
        
        # Extract latitude and longitude if present
        try:
            lat_match = re.search(r"lat_([0-9.-]+)", filename)
            lon_match = re.search(r"lon_([0-9.-]+)", filename)
            
            if lat_match:
                metadata['latitude'] = float(lat_match.group(1))
            
            if lon_match:
                metadata['longitude'] = float(lon_match.group(1))
        except Exception as e:
            logger.warning(f"Error extracting lat/lon from filename: {e}")
        
        # Extract date if present
        try:
            date_match = re.search(r"(\d+)d", filename)
            if date_match:
                metadata['forecast_days'] = int(date_match.group(1))
        except Exception as e:
            logger.warning(f"Error extracting date from filename: {e}")
        
        return metadata
    
    def preprocess_data(self, data_path):
        """
        Preprocess the input data for BiLSTM modeling
        
        Args:
            data_path: Path to the input data
            
        Returns:
            Preprocessed data and metadata
        """
        logger.info(f"Preprocessing data from {data_path}")
        
        try:
            # Extract metadata from filename
            filename = os.path.basename(data_path)
            metadata = self.extract_metadata_from_filename(filename)
            
            # Read the CSV file
            df = pd.read_csv(data_path, parse_dates=['date'], index_col='date')
            
            # Get the target column from config
            target_column = self.config.get('target_column', 'biastg')
            
            # Determine target column from file if not in config
            if 'biastg' in filename and target_column == 'biastg':
                target_column = 'biastg'
                metadata['target_type'] = 'biastg'
            else:
                metadata['target_type'] = 'biastg'
            
            # Ensure the target column exists
            if target_column not in df.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Extract target variable
            target_data = df[target_column]
            
            # Fill in missing data
            freq = self.config.get('frequency', 'B')  # Business day frequency
            fill_method = self.config.get('fill_method', 'ffill')
            
            target_data = target_data.asfreq(freq).fillna(method=fill_method)
            
            # Drop any remaining NaN values
            target_data = target_data.dropna()
            
            logger.info(f"Data preprocessing completed. Shape: {target_data.shape}")
            return target_data, metadata
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            raise
    
    def visualize_forecast(self, data, predictions, forecast=None, metadata=None, output_path=None):
        """
        Visualize the model results
        
        Args:
            data: Original time series data
            predictions: Model predictions
            forecast: Future forecast values (optional)
            metadata: Metadata about the data (optional)
            output_path: Path to save the visualization
        """
        plt.figure(figsize=(14, 6))
        
        # Create title with metadata if available
        title = "BiLSTM PyTorch: Observed vs Predicted"
        if metadata:
            if 'latitude' in metadata and 'longitude' in metadata:
                title += f" (Lat: {metadata['latitude']}, Lon: {metadata['longitude']})"
            if 'forecast_days' in metadata:
                title += f" - {metadata['forecast_days']} Days"
            if 'target_type' in metadata:
                title += f" - {metadata['target_type']}"
        
        plt.title(title)
        
        # Plot original data
        plt.plot(data.index, data.values, label='Observed', alpha=0.7)
        
        # Plot predictions
        pred_index = None
        if isinstance(predictions, pd.Series) or isinstance(predictions, pd.DataFrame):
            pred_index = predictions.index
            plt.plot(pred_index, predictions.values, color='r', label='Predicted')
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
        
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.savefig(output_path)
            logger.info(f"Forecast visualization saved to {output_path}")
        
        plt.close()
    
    def run_pipeline(self, data_path):
        """
        Run the complete BiLSTM forecasting pipeline
        
        Args:
            data_path: Path to the input data
            
        Returns:
            DataFrame with forecast results
        """
        logger.info(f"Starting BiLSTM PyTorch forecasting pipeline with data from {data_path}")
        
        # Step 1: Preprocess the data
        data, metadata = self.preprocess_data(data_path)
        
        # Step 2: Train the model
        self.model.fit(data)
        
        # Step 3: Generate predictions for validation
        predictions = self.model.predict()
        
        # Step 4: Generate future forecast
        forecast_horizon = self.config.get('forecast_horizon', 10)
        if metadata and 'forecast_days' in metadata:
            forecast_horizon = metadata['forecast_days']
        
        forecast = self.model.forecast(data, steps=forecast_horizon)
        
        # Step 5: Evaluate the model
        metrics = self.model.evaluate()
        logger.info(f"Model metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}%")
        
        # Step 6: Visualize the results
        output_dir = self.config.get('output_dir', 'outputs/visualizations/bilstm_pytorch')
        
        # Create filename base from metadata
        filename_base = "bilstm_pytorch"
        if metadata:
            if 'target_type' in metadata:
                filename_base += f"_{metadata['target_type']}"
            if 'latitude' in metadata and 'longitude' in metadata:
                filename_base += f"_lat{metadata['latitude']}_lon{metadata['longitude']}"
            if 'forecast_days' in metadata:
                filename_base += f"_{metadata['forecast_days']}d"
        
        self.visualize_forecast(
            data=data,
            predictions=predictions,
            forecast=forecast,
            metadata=metadata,
            output_path=f"{output_dir}/{filename_base}_forecast.png"
        )
        
        # Plot training history
        self.model.plot_training_history(
            save_path=f"{output_dir}/{filename_base}_training_history.png"
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
        
        # Add metadata to the DataFrame
        if metadata:
            for key, value in metadata.items():
                forecast_df.attrs[key] = value
        
        # Add metrics as attributes to the DataFrame for later use
        for key, value in metrics.items():
            forecast_df.attrs[key] = value
        
        # Save model if requested
        if self.config.get('save_model', True):
            model_dir = self.config.get('model_dir', 'outputs/models/bilstm_pytorch')
            model_path = f"{model_dir}/{filename_base}"
            os.makedirs(model_path, exist_ok=True)
            self.model.save(model_path)
        
        # Save results to CSV if requested
        if self.config.get('save_results', True):
            results_dir = self.config.get('results_dir', 'outputs/results/bilstm_pytorch')
            os.makedirs(results_dir, exist_ok=True)
            csv_path = f"{results_dir}/{filename_base}_forecast.csv"
            forecast_df.to_csv(csv_path)
            logger.info(f"Forecast results saved to {csv_path}")
            
            # Save metrics to a separate CSV file
            metrics_df = pd.DataFrame([metrics])
            metrics_csv_path = f"{results_dir}/{filename_base}_metrics.csv"
            metrics_df.to_csv(metrics_csv_path, index=False)
            logger.info(f"Metrics saved to {metrics_csv_path}")
        
        return forecast_df
