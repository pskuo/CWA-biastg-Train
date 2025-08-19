import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def calculate_metrics(y_true, y_pred):
    """
    Calculate various forecast performance metrics
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary with different metrics
    """
    # Ensure inputs are numpy arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Calculate Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Calculate Mean Absolute Error (MAE)
    mae = mean_absolute_error(y_true, y_pred)
    
    # Calculate Mean Bias Error (MBE)
    mbe = np.mean(y_true - y_pred)
    
    # Calculate Mean Absolute Percentage Error (MAPE)
    # Handle division by zero by adding a small epsilon
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), epsilon)))
    
    # Calculate R-squared (coefficient of determination)
    if np.var(y_true) == 0:
        r2 = 0  # Undefined, but set to 0
    else:
        ss_total = np.sum((y_true - np.mean(y_true))**2)
        ss_residual = np.sum((y_true - y_pred)**2)
        r2 = 1 - (ss_residual / ss_total)
    
    return {
        'rmse': rmse,
        'mae': mae,
        'mbe': mbe,
        'mape': mape,
        'r2': r2
    }
