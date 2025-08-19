import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import logging
import os

logger = logging.getLogger(__name__)

class BiLSTMNet(nn.Module):
    """Bidirectional LSTM neural network implementation in PyTorch"""
    
    def __init__(self, input_size, hidden_size, num_layers):
        """
        Initialize the BiLSTM network
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of stacked LSTM layers
        """
        super(BiLSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True, 
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            
        Returns:
            Output tensor
        """
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        # Initialize cell state
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class BiLSTMModel:
    """BiLSTM Time Series Forecasting Model using PyTorch"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.history = {'train_loss': [], 'val_loss': []}
        self.criterion = nn.MSELoss()
    
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
    
    def build_model(self, input_size=1, hidden_size=None, num_layers=None):
        """
        Build the BiLSTM model
        
        Args:
            input_size: Number of input features
            hidden_size: Number of hidden units in LSTM
            num_layers: Number of stacked LSTM layers
            
        Returns:
            Initialized BiLSTM model
        """
        # Get parameters from config if not specified
        if hidden_size is None:
            hidden_size = self.config.get('hidden_size', 4)
        if num_layers is None:
            num_layers = self.config.get('num_layers', 3)
        
        model = BiLSTMNet(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers
        ).to(self.device)
        
        logger.info(f"Built BiLSTM model with {num_layers} layers and {hidden_size} hidden units")
        
        return model
    
    def fit(self, data, validation_split=0.2):
        """
        Prepare data and train the BiLSTM model
        
        Args:
            data: Input time series data
            validation_split: Fraction of data for validation
            
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
            train_size = int(len(scaled_data) * (1 - validation_split))
            train_data = scaled_data[:train_size]
            val_data = scaled_data[train_size:]
            
            # Create sequences
            X_train, y_train = self.create_dataset(train_data, look_back, time_steps)
            X_val, y_val = self.create_dataset(val_data, look_back, time_steps)
            
            # Convert to PyTorch tensors
            X_train = torch.tensor(X_train, dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.float32)
            X_val = torch.tensor(X_val, dtype=torch.float32)
            y_val = torch.tensor(y_val, dtype=torch.float32)
            
            # Create dataloaders
            batch_size = self.config.get('batch_size', 32)
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            
            train_loader = DataLoader(
                train_dataset, 
                batch_size=batch_size, 
                shuffle=self.config.get('shuffle_training', True)
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=batch_size, 
                shuffle=False
            )
            
            # Build model if not already built
            if self.model is None:
                self.model = self.build_model(input_size=look_back)
            
            # Setup optimizer
            learning_rate = self.config.get('learning_rate', 0.001)
            optimizer = torch.optim.Adam(
                self.model.parameters(), 
                lr=learning_rate,
                weight_decay=self.config.get('weight_decay', 0)
            )
            
            # Setup learning rate scheduler if enabled
            scheduler = None
            if self.config.get('use_lr_scheduler', False):
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=self.config.get('lr_factor', 0.5),
                    patience=self.config.get('lr_patience', 5),
                    min_lr=self.config.get('min_lr', 1e-6)
                )
            
            # Train the model
            num_epochs = self.config.get('epochs', 50)
            patience = self.config.get('early_stopping_patience', 10)
            best_val_loss = float('inf')
            patience_counter = 0
            
            logger.info(f"Training BiLSTM model for {num_epochs} epochs")
            
            for epoch in range(num_epochs):
                # Training
                self.model.train()
                train_loss = 0
                for inputs, targets in train_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device).unsqueeze(1)
                    
                    optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    train_loss += loss.item() * inputs.size(0)
                
                train_loss /= len(train_loader.dataset)
                self.history['train_loss'].append(train_loss)
                
                # Validation
                self.model.eval()
                val_loss = 0
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        inputs = inputs.to(self.device)
                        targets = targets.to(self.device).unsqueeze(1)
                        
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, targets)
                        
                        val_loss += loss.item() * inputs.size(0)
                
                val_loss /= len(val_loader.dataset)
                self.history['val_loss'].append(val_loss)
                
                # Update learning rate if scheduler is used
                if scheduler is not None:
                    scheduler.step(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    # Save best model if specified
                    if self.config.get('save_best_model', True):
                        checkpoint_path = self.config.get('checkpoint_path')
                        if checkpoint_path:
                            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                            torch.save(self.model.state_dict(), checkpoint_path)
                else:
                    patience_counter += 1
                
                # Log progress
                if (epoch + 1) % 10 == 0 or epoch == 0:
                    logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping check
                if patience_counter >= patience and self.config.get('use_early_stopping', True):
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            logger.info("BiLSTM model training completed")
            
            # Load best model if saved
            if self.config.get('save_best_model', True) and self.config.get('checkpoint_path'):
                self.model.load_state_dict(torch.load(self.config.get('checkpoint_path')))
                logger.info("Loaded best model from checkpoint")
            
            # Store data for later use
            self.X_train = X_train
            self.y_train = y_train
            self.X_val = X_val
            self.y_val = y_val
            
            return self.history
            
        except Exception as e:
            logger.error(f"Error during model training: {str(e)}")
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
                look_back = self.config.get('look_back', 1)
                time_steps = self.config.get('time_steps', 3)
                X_pred, _ = self.create_dataset(scaled_data, look_back, time_steps)
                
                # Convert to PyTorch tensor
                X_pred = torch.tensor(X_pred, dtype=torch.float32)
            
            # Make predictions
            self.model.eval()
            predictions = []
            
            # Process in batches to avoid memory issues with large datasets
            batch_size = self.config.get('prediction_batch_size', 128)
            
            # Create DataLoader for prediction
            pred_dataset = TensorDataset(X_pred, torch.zeros(X_pred.shape[0]))
            pred_loader = DataLoader(pred_dataset, batch_size=batch_size, shuffle=False)
            
            with torch.no_grad():
                for inputs, _ in pred_loader:
                    inputs = inputs.to(self.device)
                    outputs = self.model(inputs)
                    predictions.append(outputs.cpu().numpy())
            
            # Concatenate batches
            predictions = np.concatenate(predictions, axis=0)
            
            # Inverse transform to get original scale
            predictions = self.scaler.inverse_transform(predictions)
            
            return predictions.flatten()
            
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
            
            # Extract last sequence for forecasting
            last_sequence = scaled_data[-time_steps:].reshape(1, time_steps, look_back)
            last_sequence = torch.tensor(last_sequence, dtype=torch.float32).to(self.device)
            
            # Generate forecast step by step
            forecasts = []
            current_sequence = last_sequence.clone()
            
            self.model.eval()
            with torch.no_grad():
                for _ in range(steps):
                    # Predict next value
                    next_value = self.model(current_sequence)
                    forecasts.append(next_value.item())
                    
                    # Update sequence for next prediction
                    # Remove oldest value and add new prediction
                    new_sequence = torch.cat([
                        current_sequence[:, 1:, :],
                        next_value.reshape(1, 1, 1)
                    ], dim=1)
                    current_sequence = new_sequence
            
            # Convert forecasts to numpy array
            forecasts = np.array(forecasts).reshape(-1, 1)
            
            # Inverse transform to get original scale
            forecasts = self.scaler.inverse_transform(forecasts)
            
            return forecasts.flatten()
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def evaluate(self, X_test=None, y_test=None):
        """
        Evaluate model on test data
        
        Args:
            X_test: Test input data
            y_test: Test target data
            
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
            
            # Make predictions
            self.model.eval()
            with torch.no_grad():
                X_test_tensor = X_test.to(self.device) if isinstance(X_test, torch.Tensor) else torch.tensor(X_test, dtype=torch.float32).to(self.device)
                predictions = self.model(X_test_tensor).cpu().numpy()
            
            # Get actual values
            y_test_array = y_test.cpu().numpy() if isinstance(y_test, torch.Tensor) else y_test
            
            # Inverse transform for metrics in original scale
            y_test_orig = self.scaler.inverse_transform(y_test_array.reshape(-1, 1))
            predictions_orig = self.scaler.inverse_transform(predictions)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            mse = mean_squared_error(y_test_orig, predictions_orig)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_orig, predictions_orig)
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
        if not self.history['train_loss']:
            raise ValueError("No training history available. Train the model first.")
            
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history['train_loss'], label='Training Loss')
            plt.plot(self.history['val_loss'], label='Validation Loss')
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
            
            # Save PyTorch model
            torch.save(self.model.state_dict(), f"{path}/bilstm_model.pt")
            
            # Save model architecture parameters
            model_params = {
                'input_size': self.config.get('look_back', 1),
                'hidden_size': self.config.get('hidden_size', 4),
                'num_layers': self.config.get('num_layers', 3)
            }
            
            # Save scaler and other metadata
            import joblib
            joblib.dump(self.scaler, f"{path}/bilstm_scaler.pkl")
            
            # Save configuration
            import json
            with open(f"{path}/bilstm_config.json", "w") as f:
                json.dump({
                    "model_params": model_params,
                    "config": self.config,
                    "device": str(self.device)
                }, f)
                
            logger.info(f"BiLSTM model saved to {path}")
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
            # Load configuration
            import json
            with open(f"{path}/bilstm_config.json", "r") as f:
                saved_data = json.load(f)
                model_params = saved_data["model_params"]
                self.config = saved_data["config"]
            
            # Load scaler
            import joblib
            self.scaler = joblib.load(f"{path}/bilstm_scaler.pkl")
            
            # Build model with saved parameters
            self.model = self.build_model(
                input_size=model_params['input_size'],
                hidden_size=model_params['hidden_size'],
                num_layers=model_params['num_layers']
            )
            
            # Load model weights
            self.model.load_state_dict(torch.load(
                f"{path}/bilstm_model.pt",
                map_location=self.device
            ))
            self.model.to(self.device)
            self.model.eval()
                
            logger.info(f"BiLSTM model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
