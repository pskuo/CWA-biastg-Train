import numpy as np
import pandas as pd
import logging
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, LSTM, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import STL

logger = logging.getLogger(__name__)

class CNNLSTMTimeSeriesModel:
    """CNN-LSTM hybrid model for time series forecasting using time series imaging approach"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.stl_result = None
        self.history = None
        self.window_size = self.config.get('window_size', 30)
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.scaler = None
        
    def decompose_time_series(self, series):
        """
        Decompose time series using STL decomposition
        
        Args:
            series: Time series data
            
        Returns:
            Tuple of (residual, trend, seasonal) components
        """
        try:
            # Get STL parameters from config
            seasonal_period = self.config.get('seasonal_period', 7)
            seasonal_deg = self.config.get('seasonal_deg', 1)
            trend_deg = self.config.get('trend_deg', 1)
            low_pass_deg = self.config.get('low_pass_deg', 1)
            robust = self.config.get('robust', True)
            
            # Apply STL decomposition
            stl = STL(
                series, 
                seasonal=seasonal_period,
                seasonal_deg=seasonal_deg,
                trend_deg=trend_deg,
                low_pass_deg=low_pass_deg,
                robust=robust
            )
            self.stl_result = stl.fit()
            
            logger.info(f"Time series decomposed using STL with seasonal period {seasonal_period}")
            
            return self.stl_result.resid, self.stl_result.trend, self.stl_result.seasonal
            
        except Exception as e:
            logger.error(f"Error in time series decomposition: {str(e)}")
            raise
    
    def create_images(self, time_series, window_size=None):
        """
        Create time series "images" using sliding window approach
        
        Args:
            time_series: Input time series
            window_size: Size of the sliding window
            
        Returns:
            Numpy array of time series images
        """
        if window_size is None:
            window_size = self.window_size
            
        try:
            # Create images
            images = []
            targets = []
            
            for i in range(len(time_series) - window_size):
                window = time_series[i:i + window_size].values
                target = time_series.iloc[i + window_size]
                images.append(window)
                targets.append(target)
                
            # Reshape to simulate 2D images (window_size, 1, 1)
            X = np.array(images).reshape(-1, window_size, 1, 1)
            y = np.array(targets)
            
            logger.info(f"Created {len(images)} time series images with window size {window_size}")
            
            return X, y
            
        except Exception as e:
            logger.error(f"Error creating time series images: {str(e)}")
            raise
    
    def scale_data(self, data):
        """
        Scale the data using Min-Max scaling
        
        Args:
            data: Input data
            
        Returns:
            Scaled data
        """
        try:
            from sklearn.preprocessing import MinMaxScaler
            
            # Create scaler if it doesn't exist
            if self.scaler is None:
                self.scaler = MinMaxScaler()
                data_scaled = self.scaler.fit_transform(data.values.reshape(-1, 1)).flatten()
            else:
                data_scaled = self.scaler.transform(data.values.reshape(-1, 1)).flatten()
                
            # Convert back to pandas Series
            data_scaled = pd.Series(data_scaled, index=data.index)
            
            return data_scaled
            
        except Exception as e:
            logger.error(f"Error scaling data: {str(e)}")
            raise
    
    def build_model(self):
        """
        Build the CNN-LSTM hybrid model
        
        Returns:
            Compiled Keras model
        """
        try:
            # Get model parameters from config
            cnn_filters = self.config.get('cnn_filters', 64)
            cnn_kernel_size = self.config.get('cnn_kernel_size', 3)
            pool_size = self.config.get('pool_size', 2)
            dropout_rate = self.config.get('dropout_rate', 0.3)
            lstm_units = self.config.get('lstm_units', 64)
            dense_units = self.config.get('dense_units', 64)
            l2_reg = self.config.get('l2_reg', 0.01)
            learning_rate = self.config.get('learning_rate', 0.001)
            
            # Build model
            model = Sequential()
            
            # CNN Layers
            model.add(Conv2D(
                cnn_filters, 
                (cnn_kernel_size, 1), 
                activation='relu', 
                input_shape=(self.window_size, 1, 1)
            ))
            model.add(MaxPooling2D((pool_size, 1)))
            model.add(Dropout(dropout_rate))
            model.add(BatchNormalization())
            
            # LSTM Layer
            model.add(Flatten())
            model.add(LSTM(lstm_units, return_sequences=False))
            
            # Dense Layers
            model.add(Dense(
                dense_units, 
                activation='relu', 
                kernel_regularizer=regularizers.l2(l2_reg)
            ))
            model.add(Dropout(dropout_rate))
            model.add(Dense(1))
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=learning_rate), 
                loss='mean_squared_error'
            )
            
            logger.info(f"Built CNN-LSTM model with {cnn_filters} CNN filters and {lstm_units} LSTM units")
            model.summary(print_fn=logger.info)
            
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {str(e)}")
            raise
    
    def prepare_data(self, data, validation_split=0.2, test_split=0.1):
        """
        Preprocess and prepare data for training
        
        Args:
            data: Input data DataFrame
            validation_split: Proportion of data for validation
            test_split: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        try:
            # Get target column from config or use the first column
            target_column = self.config.get('target_column', data.columns[0])
            
            # Ensure target column exists
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Extract target data
            target_data = data[target_column]
            
            # Fill missing values
            fill_method = self.config.get('fill_method', 'ffill')
            target_data = target_data.fillna(method=fill_method)
            
            # Optionally scale the data
            if self.config.get('scale_data', True):
                target_data = self.scale_data(target_data)
            
            # Decompose time series if specified
            if self.config.get('use_decomposition', True):
                residual, _, _ = self.decompose_time_series(target_data)
                processed_data = pd.Series(residual, index=target_data.index)
            else:
                processed_data = target_data
            
            # Create time series images
            X, y = self.create_images(processed_data, self.window_size)
            
            # Calculate split indices
            n_samples = X.shape[0]
            test_size = int(n_samples * test_split)
            val_size = int(n_samples * validation_split)
            train_size = n_samples - val_size - test_size
            
            # Split the data
            X_train = X[:train_size]
            y_train = y[:train_size]
            
            X_val = X[train_size:train_size+val_size]
            y_val = y[train_size:train_size+val_size]
            
            X_test = X[train_size+val_size:]
            y_test = y[train_size+val_size:]
            
            # Store data for later use
            self.train_data = (X_train, y_train)
            self.val_data = (X_val, y_val)
            self.test_data = (X_test, y_test)
            
            logger.info(f"Data prepared with {train_size} training, {val_size} validation, and {test_size} test samples")
            
            return X_train, y_train, X_val, y_val, X_test, y_test
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
    
    def fit(self, data, validation_split=0.2, test_split=0.1):
        """
        Train the CNN-LSTM model
        
        Args:
            data: Input data
            validation_split: Proportion of data for validation
            test_split: Proportion of data for testing
            
        Returns:
            Training history
        """
        try:
            # Prepare the data
            X_train, y_train, X_val, y_val, _, _ = self.prepare_data(
                data, 
                validation_split=validation_split,
                test_split=test_split
            )
            
            # Build the model if it doesn't exist
            if self.model is None:
                self.model = self.build_model()
            
            # Get training parameters from config
            epochs = self.config.get('epochs', 50)
            batch_size = self.config.get('batch_size', 32)
            patience = self.config.get('patience', 10)
            
            # Set up callbacks
            callbacks = []
            
            # Early stopping
            if self.config.get('use_early_stopping', True):
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=patience,
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
            
            # Model checkpoint
            if self.config.get('use_checkpoint', True):
                checkpoint_dir = self.config.get('checkpoint_dir', 'checkpoints/cnn_lstm')
                os.makedirs(checkpoint_dir, exist_ok=True)
                
                checkpoint = ModelCheckpoint(
                    filepath=f"{checkpoint_dir}/model.h5",
                    save_best_only=True,
                    monitor='val_loss'
                )
                callbacks.append(checkpoint)
            
            # Train the model
            logger.info(f"Training model for {epochs} epochs with batch size {batch_size}")
            
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=self.config.get('verbose', 1)
            )
            
            logger.info("Model training completed")
            
            return self.history
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def evaluate(self, data=None):
        """
        Evaluate the model on test data
        
        Args:
            data: Optional test data to evaluate on (if None, use stored test data)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() before evaluate()")
            
        try:
            # Use stored test data if not provided
            if data is None:
                if self.test_data is None:
                    raise ValueError("No test data available")
                X_test, y_test = self.test_data
            else:
                # Prepare the provided data
                processed_data = data[self.config.get('target_column', data.columns[0])]
                
                if self.config.get('scale_data', True) and self.scaler is not None:
                    processed_data = self.scale_data(processed_data)
                
                if self.config.get('use_decomposition', True) and self.stl_result is not None:
                    processed_data = pd.Series(self.stl_result.resid, index=processed_data.index)
                
                X_test, y_test = self.create_images(processed_data, self.window_size)
            
            # Make predictions
            y_pred = self.model.predict(X_test).flatten()
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            mbe = np.mean(y_test - y_pred)
            
            # Calculate MAPE with handling for zero values
            epsilon = 1e-10
            mape = np.mean(np.abs((y_test - y_pred) / np.maximum(np.abs(y_test), epsilon))) * 100
            
            metrics = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mbe': mbe,
                'mape': mape
            }
            
            logger.info(f"Model evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}%, MBE={mbe:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise
    
    def predict_next_steps(self, data, steps=10):
        """
        Generate recursive multi-step forecasts
        
        Args:
            data: Input data containing the recent values
            steps: Number of steps to forecast
            
        Returns:
            Numpy array of forecasted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained. Call fit() before predict_next_steps()")
            
        try:
            # Extract and process the target data
            target_column = self.config.get('target_column', data.columns[0])
            target_data = data[target_column]
            
            # Fill missing values
            target_data = target_data.fillna(method='ffill')
            
            # Scale the data if needed
            if self.config.get('scale_data', True) and self.scaler is not None:
                scaled_data = self.scale_data(target_data)
            else:
                scaled_data = target_data
            
            # Decompose if configured
            if self.config.get('use_decomposition', True) and self.stl_result is not None:
                # Get the last values from the decomposition
                residual = self.stl_result.resid
                trend = self.stl_result.trend
                seasonal = self.stl_result.seasonal
                
                # For forecasting, use the last window_size residuals
                current_residuals = pd.Series(residual, index=target_data.index)[-self.window_size:]
            else:
                # Use the original data
                current_residuals = scaled_data[-self.window_size:]
                trend = None
                seasonal = None
            
            # Prepare input for prediction
            current_input = current_residuals.values.reshape(1, self.window_size, 1, 1)
            
            # Generate recursive forecasts
            forecast_residuals = []
            
            for _ in range(steps):
                # Predict next residual value
                next_residual = self.model.predict(current_input)[0][0]
                forecast_residuals.append(next_residual)
                
                # Slide the window and update with the predicted value
                current_input = np.roll(current_input, -1, axis=1)
                current_input[0, -1, 0, 0] = next_residual
            
            # If we're using decomposition, add back trend and seasonality
            if self.config.get('use_decomposition', True) and trend is not None and seasonal is not None:
                # Get the trend and seasonal components for the future periods
                if len(trend) >= len(target_data) + steps:
                    future_trend = trend[len(target_data):len(target_data)+steps]
                else:
                    # Extrapolate trend if needed
                    from scipy import signal
                    trend_diff = np.diff(trend[-10:])
                    avg_trend_diff = np.mean(trend_diff)
                    future_trend = trend[-1] + np.arange(1, steps+1) * avg_trend_diff
                
                # For seasonality, use the appropriate seasonal pattern
                season_length = self.config.get('seasonal_period', 7)
                future_seasonal = np.array([])
                for i in range(steps):
                    idx = (len(seasonal) - season_length + (i % season_length)) % len(seasonal)
                    future_seasonal = np.append(future_seasonal, seasonal[idx])
                
                # Add components back
                forecast = np.array(forecast_residuals) + future_trend + future_seasonal
            else:
                forecast = np.array(forecast_residuals)
            
            # Inverse scale if needed
            if self.config.get('scale_data', True) and self.scaler is not None:
                forecast = self.scaler.inverse_transform(forecast.reshape(-1, 1)).flatten()
            
            logger.info(f"Generated {steps}-step forecast")
            
            return forecast
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def plot_forecast(self, data, steps=10, plot_history=50, save_path=None):
        """
        Generate and plot forecast
        
        Args:
            data: Input data
            steps: Number of steps to forecast
            plot_history: Number of historical points to plot
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        try:
            # Generate forecast
            forecast = self.predict_next_steps(data, steps)
            
            # Extract the target data
            target_column = self.config.get('target_column', data.columns[0])
            target_data = data[target_column]
            
            # Create date range for forecast
            last_date = target_data.index[-1]
            date_freq = pd.infer_freq(target_data.index)
            if date_freq is None:
                date_freq = 'D'  # Default to daily frequency
                
            future_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=steps, 
                freq=date_freq
            )
            
            # Create the plot
            plt.figure(figsize=(12, 6))
            
            # Plot the historical data
            history_slice = slice(-plot_history, None) if plot_history else slice(None, None)
            plt.plot(
                target_data.index[history_slice], 
                target_data.iloc[history_slice], 
                label='Historical Data'
            )
            
            # Plot the forecast
            plt.plot(
                future_dates, 
                forecast, 
                label='Forecast', 
                linestyle='--', 
                color='red'
            )
            
            plt.xlabel('Date')
            plt.ylabel(target_column)
            plt.title('CNN-LSTM Time Series Forecast')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Forecast plot saved to {save_path}")
            
            return plt.gcf(), forecast, future_dates
            
        except Exception as e:
            logger.error(f"Error plotting forecast: {str(e)}")
            raise
    
    def plot_training_history(self, save_path=None):
        """
        Plot the training history
        
        Args:
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
            
        try:
            plt.figure(figsize=(12, 5))
            plt.plot(self.history.history['loss'], label='Training Loss')
            plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Training history plot saved to {save_path}")
            
            return plt.gcf()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
            raise
    
    def save(self, path):
        """
        Save the model and related objects
        
        Args:
            path: Directory path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        try:
            # Create directory if it doesn't exist
            os.makedirs(path, exist_ok=True)
            
            # Save the Keras model
            model_path = os.path.join(path, "cnn_lstm_model.h5")
            self.model.save(model_path)
            
            # Save the scaler if exists
            if self.scaler is not None:
                import joblib
                scaler_path = os.path.join(path, "scaler.pkl")
                joblib.dump(self.scaler, scaler_path)
            
            # Save the model configuration
            import json
            config_path = os.path.join(path, "config.json")
            with open(config_path, 'w') as f:
                # Convert any non-serializable objects to strings
                config_serializable = {
                    k: str(v) if not isinstance(v, (int, float, str, bool, list, dict, type(None))) 
                    else v for k, v in self.config.items()
                }
                json.dump(config_serializable, f)
            
            # Save the window size
            window_path = os.path.join(path, "window_size.txt")
            with open(window_path, 'w') as f:
                f.write(str(self.window_size))
            
            logger.info(f"Model saved to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def load(self, path):
        """
        Load a saved model and related objects
        
        Args:
            path: Directory path to load the model from
        """
        try:
            # Load the Keras model
            model_path = os.path.join(path, "cnn_lstm_model.h5")
            self.model = load_model(model_path)
            
            # Load the scaler if exists
            scaler_path = os.path.join(path, "scaler.pkl")
            if os.path.exists(scaler_path):
                import joblib
                self.scaler = joblib.load(scaler_path)
            
            # Load the model configuration
            import json
            config_path = os.path.join(path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
            
            # Load the window size
            window_path = os.path.join(path, "window_size.txt")
            if os.path.exists(window_path):
                with open(window_path, 'r') as f:
                    self.window_size = int(f.read())
            
            logger.info(f"Model loaded from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
