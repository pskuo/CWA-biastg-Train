import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from datetime import timedelta

from models.core.transformer_ts.model import TransformerTSModel
from models.utils.forecast_metrics import calculate_metrics

logger = logging.getLogger(__name__)

class TransformerTSAdapter:
    """Adapter for the Darts Transformer model to fit into the pipeline"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = TransformerTSModel(config)
        
    def preprocess_data(self, data_path):
        """
        Preprocess the input data for Transformer modeling
        
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
    
    def run_pipeline(self, data_path):
        """
        Run the complete Transformer forecasting pipeline
        
        Args:
            data_path: Path to the input data
            
        Returns:
            DataFrame with forecast results
        """
        logger.info(f"Starting Transformer forecasting pipeline with data from {data_path}")
        
        # Step 1: Preprocess the data
        data = self.preprocess_data(data_path)
        
        # Step 2: Train the model
        validation_split = self.config.get('validation_split', 0.2)
        self.model.fit(data, validation_split=validation_split)
        
        # Step 3: Generate predictions for validation
        predictions = self.model.predict()
        
        # Step 4: Evaluate the model
        metrics = self.model.evaluate()
        logger.info(f"Model metrics: RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, MAPE={metrics['mape']:.4f}%")
        
        # Step 5: Visualize the results
        output_dir = self.config.get('output_dir', 'outputs/visualizations/transformer_ts')
        os.makedirs(output_dir, exist_ok=True)
        
        self.model.plot_predictions(
            save_path=f"{output_dir}/transformer_predictions.png"
        )
        
        # Step 6: Generate future forecast
        forecast_horizon = self.config.get('forecast_horizon', 10)
        
        # Create a forecast dataset by using the last input_chunk_length of data
        input_chunk_length = self.config.get('input_chunk_length', 24)
        last_data = data.iloc[-input_chunk_length:]
        
        # Predict future values
        future_predictions = self.model.predict(n_steps=forecast_horizon, series=last_data)
        
        # Step 7: Prepare the output
        # Extract forecasted values
        forecast_values = future_predictions.values().flatten()
        
        # Create date range for forecast
        last_date = data.index[-1]
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=len(forecast_values),
            freq='D'  # Assuming daily data
        )
        
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
            model_dir = self.config.get('model_dir', 'outputs/models/transformer_ts')
            os.makedirs(model_dir, exist_ok=True)
            self.model.save(model_dir)
        
        # Save results to CSV if requested
        if self.config.get('save_results', True):
            results_dir = self.config.get('results_dir', 'outputs/results/transformer_ts')
            os.makedirs(results_dir, exist_ok=True)
            csv_path = f"{results_dir}/transformer_forecast.csv"
            forecast_df.to_csv(csv_path)
            logger.info(f"Forecast results saved to {csv_path}")
            
            # Also save metrics to a separate file
            metrics_df = pd.DataFrame([metrics])
            metrics_path = f"{results_dir}/transformer_metrics.csv"
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Metrics saved to {metrics_path}")
        
        # If backtesting is enabled, perform and save backtesting results
        if self.config.get('perform_backtesting', True):
            logger.info("Performing backtesting...")
            backtest_results = self.model.backtest(
                forecast_horizon=self.config.get('backtest_horizon', forecast_horizon),
                stride=self.config.get('backtest_stride', 1),
                retrain=self.config.get('backtest_retrain', False)
            )
            
            # Plot backtesting results
            plt.figure(figsize=(14, 6))
            self.model.original_series.plot(label='Actual')
            backtest_results.plot(label='Backtest Predictions', color='r')
            plt.title('Transformer Model: Backtesting Results')
            plt.xlabel('Date')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
            
            # Save backtest plot
            backtest_plot_path = f"{output_dir}/transformer_backtest.png"
            plt.savefig(backtest_plot_path)
            logger.info(f"Backtest plot saved to {backtest_plot_path}")
            plt.close()
            
            # Evaluate backtest results
            backtest_metrics = {}
            try:
                # Calculate metrics for backtesting
                backtest_metrics = {
                    'rmse': rmse(self.model.original_series, backtest_results),
                    'mae': mae(self.model.original_series, backtest_results),
                    'mape': mape(self.model.original_series, backtest_results),
                    'mbe': np.mean(self.model.original_series.values().flatten() - 
                                  backtest_results.values().flatten())
                }
                
                # Save backtest metrics
                backtest_metrics_df = pd.DataFrame([backtest_metrics])
                backtest_metrics_path = f"{results_dir}/transformer_backtest_metrics.csv"
                backtest_metrics_df.to_csv(backtest_metrics_path, index=False)
                logger.info(f"Backtest metrics saved to {backtest_metrics_path}")
                
            except Exception as e:
                logger.warning(f"Error calculating backtest metrics: {str(e)}")
        
        return forecast_df
