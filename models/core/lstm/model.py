import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
import logging
import os
import joblib

logger = logging.getLogger(__name__)

class LSTMModel:
    """LSTM Time Series Forecasting Model with hyperparameter tuning"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        self.best_params = None
        
    def create_dataset(self, dataset, look_back=1, time_steps=1):
        """
        Create time series sequences for LSTM
        
        Args:
            dataset: Input time series data
            look_back: Number of time lag
            time_steps: Number of time steps
            
        Returns:
            Tuple of (X, y) where X is the input sequences and y is the target values
        """
        X, Y = [], []
        for i in range(len(dataset) - time_steps - look_back + 1):
            a = dataset[i:(i + time_steps * look_back), 0].reshape(time_steps, look_back)
            X.append(a)
            Y.append(dataset[i + time_steps * look_back - 1, 0])
        return np.array(X), np.array(Y)
    
    def lr_scheduler(self, epoch, lr):
        """
        Learning rate scheduler function
        
        Args:
            epoch: Current epoch
            lr: Current learning rate
            
        Returns:
            New learning rate
        """
        decay_after = self.config.get('lr_decay_after', 10)
        decay_rate = self.config.get('lr_decay_rate', 0.9)
        
        if epoch < decay_after:
            return lr
        else:
            return lr * decay_rate
    
    def build_model(self, units=64, dropout_rate=0.2, time_steps=3, look_back=1):
        """
        Build and compile the LSTM model
        
        Args:
            units: Number of units in the first LSTM layer
            dropout_rate: Dropout rate
            time_steps: Number of time steps
            look_back: Number of time lag
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(units, 
                      return_sequences=True, 
                      input_shape=(time_steps, look_back)))
        model.add(Dropout(dropout_rate))
        
        # Second LSTM layer
        model.add(LSTM(units // 2, 
                      return_sequences=True))
        model.add(Dropout(dropout_rate))
        
        # Third LSTM layer
        model.add(LSTM(units // 4))
        model.add(Dropout(dropout_rate))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.config.get('learning_rate', 0.001))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        
        return model
    
    def tune_hyperparameters(self, X_train, y_train):
        """
        Perform hyperparameter tuning using RandomizedSearchCV
        
        Args:
            X_train: Training input sequences
            y_train: Training target values
            
        Returns:
            Best hyperparameters
        """
        # Get hyperparameter search space from config
        param_grid = self.config.get('param_grid', {
            'units': [32, 64, 128],
            'dropout_rate': [0.2, 0.3, 0.5],
            'batch_size': [16, 32, 64],
            'epochs': [50, 100]
        })
        
        # Number of search iterations
        n_iter = self.config.get('tuning_iterations', 5)
        
        # Create the model
        time_steps = self.config.get('time_steps', 3)
        look_back = self.config.get('look_back', 1)
        
        def create_model(units=64, dropout_rate=0.2):
            return self.build_model(units=units, dropout_rate=dropout_rate, 
                                  time_steps=time_steps, look_back=look_back)
        
        model = KerasRegressor(build_fn=create_model, verbose=0)
        
        # Perform randomized search
        logger.info("Starting hyperparameter tuning")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=3,
            verbose=1,
            random_state=42
        )
        
        try:
            random_search.fit(X_train, y_train)
            
            best_params = random_search.best_params_
            logger.info(f"Best hyperparameters: {best_params}")
            
            return best_params
            
        except Exception as e:
            logger.error(f"Error during hyperparameter tuning: {str(e)}")
            # Return default parameters
            return {
                'units': 64,
                'dropout_rate': 0.2,
                'batch_size': 32,
                'epochs': 100
            }
    
    def fit(self, data, tune_hyperparameters=True):
        """
        Prepare data, tune hyperparameters, and train the LSTM model
        
        Args:
            data: Input time series data
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Training history
        """
        try:
            # Scale the data
            scaled_data = self.scaler.fit_transform(data.values.reshape(-1, 1))
            
            # Get parameters
            look_back = self.config.get('look_back', 1)
            time_steps = self.config.get('time_steps', 3)
            
            # Split the data into training and testing sets
            train_size = int(len(scaled_data) * self.config.get('train_size', 0.8))
            train_data = scaled_data[:train_size]
            test_data = scaled_data[train_size:]
            
            # Create sequences
            X_train, y_train = self.create_dataset(train_data, look_back, time_steps)
            X_test, y_test = self.create_dataset(test_data, look_back, time_steps)
            
            # Reshape the input to be [samples, time steps, features]
            X_train = np.reshape(X_train, (X_train.shape[0], time_steps, look_back))
            X_test = np.reshape(X_test, (X_test.shape[0], time_steps, look_back))
            
            # Tune hyperparameters if requested
            if tune_hyperparameters:
                self.best_params = self.tune_hyperparameters(X_train, y_train)
            else:
                # Use default or provided parameters
                self.best_params = {
                    'units': self.config.get('units', 64),
                    'dropout_rate': self.config.get('dropout_rate', 0.2),
                    'batch_size': self.config.get('batch_size', 32),
                    'epochs': self.config.get('epochs', 100)
                }
            
            # Build model with best parameters
            self.model = self.build_model(
                units=self.best_params['units'],
                dropout_rate=self.best_params['dropout_rate'],
                time_steps=time_steps,
                look_back=look_back
            )
            
            # Setup callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=self.config.get('early_stopping_patience', 10),
                restore_best_weights=True
            )
            
            lr_scheduler = LearningRateScheduler(self.lr_scheduler)
            
            # Train the model
            logger.info("Training LSTM model")
            self.history = self.model.fit(
                X_train, y_train,
                epochs=self.best_params['epochs'],
                batch_size=self.best_params['batch_size'],
                verbose=self.config.get('verbose', 1),
                validation_data=(X_test, y_test),
                callbacks=[early_stopping, lr_scheduler]
            )
            
            logger.info("LSTM model trained successfully")
            
            # Store the train/test data for later use
            self.X_train = X_train
            self.y_train = y_train
            self.X_test = X_test
            self.y_test = y_test
            
            return self.history
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
            raise
    
    def predict(self, data=None):
        """
        Generate predictions using the fitted model
        
        Args:
            data: Input data for prediction (if None, use test data)
            
        Returns:
            Predicted values in the original scale
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before predict()")
            
        try:
            # Use test data if no data provided
            if data is None:
                if not hasattr(self, 'X_test'):
                    raise ValueError("No test data available and no data provided")
                predictions = self.model.predict(self.X_test)
                return self.scaler.inverse_transform(predictions)
            
            # Process the provided data
            look_back = self.config.get('look_back', 1)
            time_steps = self.config.get('time_steps', 3)
            
            # Scale the data
            scaled_data = self.scaler.transform(data.values.reshape(-1, 1))
            
            # Create sequences
            X, _ = self.create_dataset(scaled_data, look_back, time_steps)
            
            # Reshape
            X = np.reshape(X, (X.shape[0], time_steps, look_back))
            
            # Generate predictions
            predictions = self.model.predict(X)
            
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
            
            # Get parameters
            look_back = self.config.get('look_back', 1)
            time_steps = self.config.get('time_steps', 3)
            
            # Initialize forecast with the last sequence of data
            forecast_input = scaled_data[-time_steps:].reshape(1, time_steps, look_back)
            forecast_values = []
            
            # Generate forecast step by step
            for _ in range(steps):
                # Predict the next value
                next_pred = self.model.predict(forecast_input)
                
                # Add prediction to forecast values
                forecast_values.append(next_pred[0, 0])
                
                # Update forecast input
                # Remove the first value and append the new prediction
                forecast_input = np.roll(forecast_input, -1, axis=1)
                forecast_input[0, -1, 0] = next_pred[0, 0]
            
            # Convert forecast to array and inverse transform
            forecast_values = np.array(forecast_values).reshape(-1, 1)
            forecast_values = self.scaler.inverse_transform(forecast_values)
            
            return forecast_values.flatten()
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def evaluate(self, actual=None, predicted=None):
        """
        Evaluate model performance
        
        Args:
            actual: Actual values (if None, use test data)
            predicted: Predicted values (if None, generate predictions)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been fitted. Call fit() before evaluate()")
            
        try:
            # Use test data if no data provided
            if actual is None and hasattr(self, 'y_test'):
                actual = self.scaler.inverse_transform(self.y_test.reshape(-1, 1))
            
            if predicted is None and hasattr(self, 'X_test'):
                predicted = self.predict(None)
            
            if actual is None or predicted is None:
                raise ValueError("Both actual and predicted values must be provided")
            
            # Flatten arrays if needed
            if isinstance(actual, np.ndarray) and actual.ndim > 1:
                actual = actual.flatten()
            if isinstance(predicted, np.ndarray) and predicted.ndim > 1:
                predicted = predicted.flatten()
            
            # Calculate metrics
            mse = np.mean((actual - predicted) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual - predicted))
            mbe = np.mean(actual - predicted)
            
            # Calculate MAPE with handling for zero values
            epsilon = 1e-10
            mape = np.mean(np.abs((actual - predicted) / np.maximum(np.abs(actual), epsilon))) * 100
            
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
            self.model.save(f"{path}/lstm_model.h5")
            
            # Save scaler and other attributes
            joblib.dump(self.scaler, f"{path}/lstm_scaler.pkl")
            
            # Save best parameters
            if self.best_params:
                import json
                with open(f"{path}/lstm_params.json", "w") as f:
                    json.dump({
                        "best_params": self.best_params,
                        "config": self.config
                    }, f)
                
            logger.info(f"LSTM model saved to {path}")
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
            self.model = load_model(f"{path}/lstm_model.h5")
            
            # Load scaler
            self.scaler = joblib.load(f"{path}/lstm_scaler.pkl")
            
            # Load parameters
            import json
            try:
                with open(f"{path}/lstm_params.json", "r") as f:
                    params = json.load(f)
                    self.best_params = params.get("best_params")
                    self.config = params.get("config", {})
            except FileNotFoundError:
                logger.warning("No parameters file found")
                
            logger.info(f"LSTM model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
