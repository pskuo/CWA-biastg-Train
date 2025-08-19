import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging
import os
import joblib

logger = logging.getLogger(__name__)

class RNNModel:
    """GRU-based RNN Model for time series forecasting"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def create_dataset(self, dataset, look_back=1):
        """
        Create time-lagged dataset for RNN
        
        Args:
            dataset: Input time series data
            look_back: Number of time lags to use as input features
            
        Returns:
            Tuple of (X, y) where X is the input sequences and y is the target values
        """
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i:(i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)
    
    def build_model(self, look_back=3):
        """
        Build and compile the GRU model
        
        Args:
            look_back: Number of time lags to use as input features
            
        Returns:
            Compiled Keras model
        """
        # Get model parameters from config
        gru_units_1 = self.config.get('gru_units_1', 32)
        gru_units_2 = self.config.get('gru_units_2', 16)
        dropout_rate = self.config.get('dropout_rate', 0.2)
        optimizer = self.config.get('optimizer', 'nadam')
        learning_rate = self.config.get('learning_rate', 0.001)
        
        # Build model
        model = Sequential()
        
        # First GRU layer with return_sequences for stacking
        model.add(GRU(
            gru_units_1, 
            input_shape=(1, look_back), 
            return_sequences=True
        ))
        
        # Add dropout for regularization if specified
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        # Second GRU layer
        model.add(GRU(gru_units_2))
        
        # Add dropout for regularization if specified
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model with specified optimizer
        if optimizer.lower() == 'adam':
            opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == 'nadam':
            opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        elif optimizer.lower() == 'rmsprop':
            opt = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
        else:
            # Default to nadam
            opt = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
        
        model.compile(
            loss=self.config.get('loss_function', 'mean_squared_error'),
            optimizer=opt
        )
        
        logger.info(f"GRU model built with {gru_units_1} and {gru_units_2} units")
        model.summary(print_fn=logger.info)
        
        return model
    
    def fit(self, data, validation_split=0.2):
        """
        Prepare data and train the RNN model
        
        Args:
            data: Input time series data
            validation_split: Fraction of data to use for validation
            
        Returns:
            Training history
        """
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
            
            # Split the data into training and testing sets
            train_size = int(len(scaled_data) * (1 - validation_split))
            train_data = scaled_data[:train_size]
            val_data = scaled_data[train_size:]
            
            # Get the look-back window size
            look_back = self.config.get('look_back', 3)
            
            # Create sequences
            X_train, y_train = self.create_dataset(train_data, look_back)
            X_val, y_val = self.create_dataset(val_data, look_back)
            
            # Reshape input data for GRU [samples, timesteps, features]
            X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
            X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1]))
            
            # Build model if not already built
            if self.model is None:
                self.model = self.build_model(look_back)
            
            # Set up callbacks
            callbacks = []
            
            # Early stopping
            if self.config.get('use_early_stopping', True):
                early_stopping = EarlyStopping(
                    monitor='val_loss',
                    patience=self.config.get('early_stopping_patience', 10),
                    restore_best_weights=True
                )
                callbacks.append(early_stopping)
            
            # Learning rate reduction
            if self.config.get('use_reduce_lr', True):
                reduce_lr = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.get('lr_reduction_factor', 0.2),
                    patience=self.config.get('lr_patience', 5),
                    min_lr=self.config.get('min_lr', 0.0001)
                )
                callbacks.append(reduce_lr)
            
            # Model checkpoint
            checkpoint_path = self.config.get('checkpoint_path')
            if checkpoint_path:
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                checkpoint = ModelCheckpoint(
                    checkpoint_path,
                    monitor='val_loss',
                    save_best_only=True
                )
                callbacks.append(checkpoint)
            
            # Train model
            epochs = self.config.get('epochs', 100)
            batch_size = self.config.get('batch_size', 1)
            
            logger.info(f"Training RNN model with {epochs} epochs and batch size {batch_size}")
            
            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val, y_val),
                callbacks=callbacks,
                verbose=self.config.get('verbose', 1)
            )
            
            logger.info("RNN model trained successfully")
            
            # Store data for later use
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            
            return self.history
            
        except Exception as e:
            logger.error(f"Error training RNN model: {str(e)}")
            raise
    
    def predict(self, data=None):
        """
        Generate predictions using the fitted model
        
        Args:
            data: Input data for prediction (if None, use validation data)
            
        Returns:
            Predicted values in the original scale
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before predict()")
            
        try:
            # Use validation data if no data provided
            if data is None:
                if not hasattr(self, 'X_val'):
                    raise ValueError("No validation data available and no data provided")
                X_pred = self.X_val
            else:
                # Scale the data
                scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
                
                # Create sequences
                look_back = self.config.get('look_back', 3)
                X_pred, _ = self.create_dataset(scaled_data, look_back)
                
                # Reshape input data for GRU [samples, timesteps, features]
                X_pred = np.reshape(X_pred, (X_pred.shape[0], 1, X_pred.shape[1]))
            
            # Generate predictions
            predictions = self.model.predict(X_pred)
            
            # Inverse transform to get original scale
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
    
    def forecast(self, data, steps):
        """
        Generate multi-step forecast
        
        Args:
            data: Input time series data
            steps: Number of steps to forecast
            
        Returns:
            Forecasted values in the original scale
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before forecast()")
            
        try:
            # Scale the data
            scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
            
            # Get the look-back window size
            look_back = self.config.get('look_back', 3)
            
            # Extract the most recent data points for the forecast
            recent_data = scaled_data[-look_back:].flatten()
            
            # Initialize forecast values
            forecast_values = []
            
            # Create a rolling window for forecasting
            window = recent_data.copy()
            
            # Generate forecasts step by step
            for _ in range(steps):
                # Reshape window for prediction
                X = np.reshape(window, (1, 1, look_back))
                
                # Generate the next forecast step
                next_step = self.model.predict(X)[0, 0]
                
                # Add the prediction to the forecast values
                forecast_values.append(next_step)
                
                # Update the window by dropping the oldest value and adding the new prediction
                window = np.append(window[1:], next_step)
            
            # Convert forecast values to array and reshape for inverse transform
            forecast_values = np.array(forecast_values).reshape(-1, 1)
            
            # Inverse transform to get original scale
            forecast_values = self.scaler.inverse_transform(forecast_values)
            
            return forecast_values.flatten()
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def evaluate(self, X_test=None, y_test=None):
        """
        Evaluate model performance
        
        Args:
            X_test: Test input data (if None, use validation data)
            y_test: Test target data (if None, use validation data)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before evaluate()")
            
        try:
            # Use validation data if not provided
            if X_test is None and y_test is None:
                if hasattr(self, 'X_val') and hasattr(self, 'y_val'):
                    X_test = self.X_val
                    y_test = self.y_val
                else:
                    raise ValueError("No test data provided and no validation data available")
            
            # Generate predictions
            predictions = self.model.predict(X_test)
            
            # Inverse transform for metrics in original scale
            y_test_orig = self.scaler.inverse_transform(y_test.reshape(-1, 1))
            predictions_orig = self.scaler.inverse_transform(predictions)
            
            # Calculate metrics
            mse = np.mean((y_test_orig - predictions_orig) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(y_test_orig - predictions_orig))
            mbe = np.mean(y_test_orig - predictions_orig)
            
            # Calculate MAPE with handling for zero values
            epsilon = 1e-10
            mape = np.mean(np.abs((y_test_orig - predictions_orig) / np.maximum(np.abs(y_test_orig), epsilon))) * 100
            
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
    
    def plot_training_history(self, save_path=None):
        """
        Plot training history
        
        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
            
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history.history['loss'], label='Training Loss')
            if 'val_loss' in self.history.history:
                plt.plot(self.history.history['val_loss'], label='Validation Loss')
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend()
            plt.grid(True)
            
            if save_path:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                logger.info(f"Training history plot saved to {save_path}")
                
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting training history: {str(e)}")
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
            
            # Save Keras model
            self.model.save(f"{path}/rnn_model.h5")
            
            # Save scaler
            joblib.dump(self.scaler, f"{path}/rnn_scaler.pkl")
            
            # Save config
            import json
            with open(f"{path}/rnn_config.json", "w") as f:
                json.dump(self.config, f)
                
            logger.info(f"RNN model saved to {path}")
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
            # Load Keras model
            self.model = load_model(f"{path}/rnn_model.h5")
            
            # Load scaler
            self.scaler = joblib.load(f"{path}/rnn_scaler.pkl")
            
            # Load config
            import json
            with open(f"{path}/rnn_config.json", "r") as f:
                self.config = json.load(f)
                
            logger.info(f"RNN model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
