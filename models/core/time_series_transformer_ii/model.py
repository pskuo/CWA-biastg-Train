import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.models import TransformerModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, mae, rmse
from darts.utils.likelihood_models import GaussianLikelihood

logger = logging.getLogger(__name__)

class TransformerTSModel:
    """Transformer-based Time Series Forecasting Model using Darts"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.scaler = None
        self.train_series = None
        self.val_series = None
        self.test_series = None
        self.predictions = None
        
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
            scaled_series = scaler.fit_transform(series) if scaler.fitted else scaler.transform(series)
            
            return scaled_series, scaler
            
        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            raise
    
    def build_model(self):
        """
        Build and configure the Transformer model
        
        Returns:
            Configured Darts TransformerModel
        """
        try:
            # Get model parameters from config
            input_chunk_length = self.config.get('input_chunk_length', 24)
            output_chunk_length = self.config.get('output_chunk_length', 1)
            n_epochs = self.config.get('n_epochs', 100)
            batch_size = self.config.get('batch_size', 32)
            n_head = self.config.get('n_head', 4)
            dropout = self.config.get('dropout', 0.1)
            d_model = self.config.get('d_model', 64)
            num_encoder_layers = self.config.get('num_encoder_layers', 3)
            num_decoder_layers = self.config.get('num_decoder_layers', 3)
            activation = self.config.get('activation', 'relu')
            optimizer = self.config.get('optimizer', 'adam')
            lr = self.config.get('learning_rate', 1e-3)
            
            # Configure the likelihood model
            use_gaussian = self.config.get('use_gaussian_likelihood', False)
            likelihood = GaussianLikelihood() if use_gaussian else None
            
            # Check for GPU availability and get force CPU setting
            force_cpu = self.config.get('force_cpu', False)
            device = torch.device('cuda' if torch.cuda.is_available() and not force_cpu else 'cpu')
            
            # Build the model
            model = TransformerModel(
                input_chunk_length=input_chunk_length,
                output_chunk_length=output_chunk_length,
                n_epochs=n_epochs,
                batch_size=batch_size,
                n_head=n_head,
                dropout=dropout,
                d_model=d_model,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                activation=activation,
                optimizer_kwargs={"lr": lr},
                optimizer=optimizer,
                likelihood=likelihood,
                force_reset=True,
                random_state=self.config.get('random_state', 42),
                pl_trainer_kwargs={
                    "accelerator": "gpu" if device.type == 'cuda' else "cpu",
                    "devices": 1 if device.type == 'cuda' else None
                }
            )
            
            logger.info(f"Transformer model built with input_chunk_length={input_chunk_length}, "
                      f"output_chunk_length={output_chunk_length}, device={device}")
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def fit(self, data, validation_split=0.2):
        """
        Prepare data and train the Transformer model
        
        Args:
            data: Input time series data
            validation_split: Fraction of data to use for validation
            
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
            
            # Build the model if it doesn't exist
            if self.model is None:
                self.model = self.build_model()
            
            # Train the model
            logger.info("Training Transformer model...")
            verbose = self.config.get('verbose', True)
            
            # Set up early stopping if enabled
            pl_trainer_kwargs = {}
            if self.config.get('use_early_stopping', True):
                from pytorch_lightning.callbacks import EarlyStopping
                early_stopping = EarlyStopping(
                    monitor=self.config.get('early_stopping_monitor', 'val_loss'),
                    patience=self.config.get('early_stopping_patience', 10),
                    mode=self.config.get('early_stopping_mode', 'min')
                )
                pl_trainer_kwargs['callbacks'] = [early_stopping]
                
            # Set up model checkpoint if enabled
            if self.config.get('use_model_checkpoint', True):
                from pytorch_lightning.callbacks import ModelCheckpoint
                checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints/transformer_ts')
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint = ModelCheckpoint(
                    dirpath=checkpoint_dir,
                    filename=f"transformer-{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}",
                    save_top_k=1,
                    monitor=self.config.get('checkpoint_monitor', 'val_loss')
                )
                if 'callbacks' in pl_trainer_kwargs:
                    pl_trainer_kwargs['callbacks'].append(checkpoint)
                else:
                    pl_trainer_kwargs['callbacks'] = [checkpoint]
            
            # Add trainer kwargs from config if any
            additional_trainer_kwargs = self.config.get('pl_trainer_kwargs', {})
            pl_trainer_kwargs.update(additional_trainer_kwargs)
                
            # Train the model
            self.model.fit(
                train_series,
                val_series=val_series,
                verbose=verbose,
                pl_trainer_kwargs=pl_trainer_kwargs
            )
            
            logger.info("Transformer model trained successfully")
            return self.model
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def predict(self, n_steps=None, series=None, prediction_type='point'):
        """
        Generate predictions using the fitted model
        
        Args:
            n_steps: Number of steps to predict
            series: Optional TimeSeries for prediction (uses val_series if None)
            prediction_type: Type of prediction ('point' or 'quantiles')
            
        Returns:
            Predicted TimeSeries
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before predict()")
            
        try:
            # Use validation series if not provided
            if series is None:
                if self.val_series is None:
                    raise ValueError("No validation data available and no series provided")
                series = self.val_series
            else:
                # Ensure it's a TimeSeries and scale it
                series = self.create_series(series)
                series, _ = self.scale_data(series, self.scaler)
                
            # Use validation length if n_steps not specified
            if n_steps is None:
                n_steps = len(self.val_series) if self.val_series is not None else 1
            
            # Generate predictions
            logger.info(f"Generating predictions for {n_steps} steps")
            
            if prediction_type == 'quantiles' and self.config.get('use_gaussian_likelihood', False):
                # Generate quantile predictions (for uncertainty)
                quantiles = self.config.get('quantiles', [0.05, 0.5, 0.95])
                predictions = self.model.predict(
                    n=n_steps,
                    series=series,
                    num_samples=self.config.get('num_samples', 500),
                    quantiles=quantiles
                )
            else:
                # Generate point predictions
                predictions = self.model.predict(n=n_steps, series=series)
            
            # Invert scaling
            predictions_unscaled = self.scaler.inverse_transform(predictions)
            
            # Store predictions
            self.predictions = predictions_unscaled
            
            return predictions_unscaled
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def backtest(self, start_date=None, forecast_horizon=None, stride=None, retrain=False):
        """
        Perform backtesting of the model
        
        Args:
            start_date: Start date for backtesting
            forecast_horizon: Horizon for each forecast
            stride: Stride between forecasts
            retrain: Whether to retrain the model for each forecast
            
        Returns:
            Backtest results
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before backtest()")
            
        try:
            # Set default parameters if not provided
            if forecast_horizon is None:
                forecast_horizon = self.config.get('output_chunk_length', 1)
                
            if stride is None:
                stride = self.config.get('backtest_stride', forecast_horizon)
                
            # Determine start date
            if start_date is None and self.val_series is not None:
                # Use validation start by default
                start_date = self.val_series.start_time()
            elif start_date is None:
                # Use 80% of data if no validation set
                start_idx = int(0.8 * len(self.train_series))
                start_date = self.train_series.time_index[start_idx]
                
            # Perform backtesting
            logger.info(f"Performing backtesting from {start_date} with "
                      f"forecast_horizon={forecast_horizon}, stride={stride}")
            
            backtest_series = self.model.historical_forecasts(
                self.train_series,
                start=start_date,
                forecast_horizon=forecast_horizon,
                stride=stride,
                retrain=retrain,
                verbose=self.config.get('verbose', True)
            )
            
            # Invert scaling
            backtest_series_unscaled = self.scaler.inverse_transform(backtest_series)
            
            return backtest_series_unscaled
            
        except Exception as e:
            logger.error(f"Error during backtesting: {str(e)}")
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
            
            plt.title('Transformer Model: Observed vs Predicted')
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
            
            # Save Darts Transformer model
            self.model.save(f"{path}/transformer_model.pt")
            
            # Save scaler
            self.scaler.save(f"{path}/transformer_scaler.pkl")
            
            # Save config
            import json
            with open(f"{path}/transformer_config.json", "w") as f:
                # Convert any non-serializable types to strings
                config_serializable = {k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) 
                                     else v for k, v in self.config.items()}
                json.dump(config_serializable, f)
                
            logger.info(f"Transformer model saved to {path}")
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
            # Load Darts Transformer model
            self.model = TransformerModel.load(f"{path}/transformer_model.pt")
            
            # Load scaler
            self.scaler = Scaler.load(f"{path}/transformer_scaler.pkl")
            
            # Load config
            import json
            with open(f"{path}/transformer_config.json", "r") as f:
                self.config = json.load(f)
                
            logger.info(f"Transformer model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
