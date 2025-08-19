import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.statespace.sarimax import SARIMAX
import pmdarima as pm
from scipy.stats import boxcox
from scipy.special import inv_boxcox
from scipy import stats

logger = logging.getLogger(__name__)

class SARIMAXModel:
    """SARIMAX Time Series Forecasting Model with additional features"""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.model = None
        self.results = None
        self.order = None
        self.seasonal_order = None
        self.lambda_value = None
        self.exog_columns = None
        
    def remove_outliers(self, data, column_name=None, threshold=3):
        """
        Remove outliers from the data using Z-score
        
        Args:
            data: Input data (DataFrame or Series)
            column_name: Name of the column to check for outliers (if DataFrame)
            threshold: Z-score threshold for outlier detection
            
        Returns:
            Data with outliers removed
        """
        if column_name is None and isinstance(data, pd.DataFrame):
            # If no column is specified but data is DataFrame, use the first column
            column_name = data.columns[0]
        
        if isinstance(data, pd.DataFrame) and column_name:
            target = data[column_name]
            z_scores = np.abs(stats.zscore(target))
            return data[(z_scores < threshold)]
        else:
            # For Series
            z_scores = np.abs(stats.zscore(data))
            return data[(z_scores < threshold)]
    
    def create_features(self, data, target_column):
        """
        Create additional features for SARIMAX model
        
        Args:
            data: Input data (Series or DataFrame)
            target_column: Name of the target column
            
        Returns:
            DataFrame with additional features
        """
        if isinstance(data, pd.Series):
            data = data.to_frame(name=target_column)
        
        # Number of lags to use
        n_lags = self.config.get('n_lags', 3)
        
        # Create lagged features and rolling statistics
        for i in range(1, n_lags + 1):
            data[f"lag_{i}"] = data[target_column].shift(i)
            data[f"rolling_mean_{i}"] = data[target_column].rolling(window=i).mean()
            data[f"rolling_std_{i}"] = data[target_column].rolling(window=i).std()
        
        # Optional: add more sophisticated features
        if self.config.get('add_extra_features', False):
            # Add day of week, month, etc. if your data has a datetime index
            if isinstance(data.index, pd.DatetimeIndex):
                data['day_of_week'] = data.index.dayofweek
                data['month'] = data.index.month
                data['quarter'] = data.index.quarter
        
        # Remove NaN values created by lags and rolling windows
        data = data.dropna()
        
        return data
    
    def transform_target(self, data, target_column):
        """
        Apply Box-Cox transformation to the target variable
        
        Args:
            data: DataFrame with the target column
            target_column: Name of the target column
            
        Returns:
            Tuple of (transformed data, lambda value)
        """
        if not self.config.get('use_boxcox', True):
            logger.info("Box-Cox transformation skipped due to configuration")
            return data[target_column].values, None
        
        try:
            data_transformed, lambda_value = boxcox(data[target_column])
            logger.info(f"Box-Cox transformation applied with lambda={lambda_value}")
            return data_transformed, lambda_value
        except Exception as e:
            logger.warning(f"Box-Cox transformation failed: {str(e)}. Proceeding without transformation.")
            return data[target_column].values, None
    
    def inverse_transform(self, data):
        """
        Apply inverse Box-Cox transformation
        
        Args:
            data: Transformed data
            
        Returns:
            Original scale data
        """
        if self.lambda_value is None:
            return data
        
        return inv_boxcox(data, self.lambda_value)
    
    def find_best_parameters(self, train_data, exog=None):
        """
        Find the best SARIMAX parameters using pmdarima's auto_arima
        
        Args:
            train_data: Training data (transformed if applicable)
            exog: Exogenous variables
            
        Returns:
            Tuple of (order, seasonal_order)
        """
        try:
            # Get auto_arima parameters from config
            start_p = self.config.get('start_p', 0)
            start_q = self.config.get('start_q', 0)
            start_P = self.config.get('start_P', 0)
            start_Q = self.config.get('start_Q', 0)
            max_p = self.config.get('max_p', 3)
            max_q = self.config.get('max_q', 3)
            max_P = self.config.get('max_P', 3)
            max_Q = self.config.get('max_Q', 3)
            max_d = self.config.get('max_d', 1)
            max_D = self.config.get('max_D', 1)
            seasonal = self.config.get('seasonal', True)
            m = self.config.get('seasonal_period', 12)
            
            logger.info("Performing auto_arima parameter search")
            
            stepwise_model = pm.auto_arima(
                train_data, 
                start_p=start_p, start_q=start_q,
                start_P=start_P, start_Q=start_Q,
                max_p=max_p, max_q=max_q,
                max_P=max_P, max_Q=max_Q, 
                seasonal=seasonal,
                stepwise=True, 
                suppress_warnings=True, 
                D=1, max_D=max_D,
                error_action='ignore',
                trace=True,
                m=m,
                exogenous=exog
            )
            
            order = stepwise_model.order
            seasonal_order = stepwise_model.seasonal_order
            
            logger.info(f"Best SARIMAX parameters - Order: {order}, Seasonal Order: {seasonal_order}")
            
            return order, seasonal_order
            
        except Exception as e:
            logger.error(f"Failed to find best parameters: {str(e)}")
            # Return default parameters
            return (1, 1, 1), (1, 1, 1, 12)
    
    def fit(self, data, target_column='biastg', exog_columns=None):
        """
        Prepare data and train the SARIMAX model
        
        Args:
            data: Input data (DataFrame)
            target_column: Name of the target column
            exog_columns: Names of exogenous variables columns
            
        Returns:
            Fitted model results
        """
        try:
            # Store exog columns for prediction
            self.exog_columns = exog_columns
            
            # Step 1: Create features from the data
            data_with_features = self.create_features(data, target_column)
            
            # Step 2: Outlier removal (if enabled)
            if self.config.get('remove_outliers', True):
                outlier_threshold = self.config.get('outlier_threshold', 3)
                data_no_outliers = self.remove_outliers(
                    data_with_features, 
                    column_name=target_column, 
                    threshold=outlier_threshold
                )
                logger.info(f"Removed outliers: {len(data_with_features) - len(data_no_outliers)} points")
            else:
                data_no_outliers = data_with_features
            
            # Step 3: Transform the target variable
            data_transformed, self.lambda_value = self.transform_target(data_no_outliers, target_column)
            
            # Step 4: Prepare exogenous variables
            if exog_columns is None:
                # Use all columns except the target as exog variables
                exog = data_no_outliers.drop(columns=[target_column])
            else:
                exog = data_no_outliers[exog_columns]
            
            # Step 5: Find the best model parameters
            if self.order is None or self.seasonal_order is None:
                self.order, self.seasonal_order = self.find_best_parameters(data_transformed, exog)
            
            # Step 6: Fit the model
            self.model = SARIMAX(
                data_transformed, 
                order=self.order, 
                seasonal_order=self.seasonal_order,
                exog=exog
            )
            
            self.results = self.model.fit(disp=0)
            logger.info("SARIMAX model fitted successfully")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error fitting SARIMAX model: {str(e)}")
            raise
    
    def predict(self, data=None, horizon=None, steps=None):
        """
        Generate predictions using the fitted model
        
        Args:
            data: Data containing exogenous variables (for forecasting)
            horizon: Forecast horizon (alternative to steps)
            steps: Number of steps to forecast (alternative to horizon)
            
        Returns:
            Forecasted values in the original scale
        """
        if not self.results:
            raise ValueError("Model has not been fitted. Call fit() before predict()")
            
        try:
            # Determine the number of steps
            if steps is None and horizon is None:
                steps = self.config.get('forecast_horizon', 10)
            elif steps is None:
                steps = horizon
            
            logger.info(f"Generating forecasts for {steps} steps")
            
            if data is not None:
                # Create features for the forecast period
                if self.exog_columns:
                    exog = data[self.exog_columns]
                else:
                    # Use all columns as exog
                    exog = data
                
                # Generate predictions
                predictions = self.results.forecast(steps=steps, exog=exog)
            else:
                # No exogenous variables
                predictions = self.results.forecast(steps=steps)
            
            # Inverse transform the predictions
            if self.lambda_value is not None:
                predictions = self.inverse_transform(predictions)
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error generating forecast: {str(e)}")
            raise
    
    def save(self, path):
        """
        Save the trained model
        
        Args:
            path: Path to save the model
        """
        if not self.results:
            raise ValueError("No model results to save")
            
        try:
            # Save model results
            self.results.save(f"{path}/sarimax_model.pkl")
            
            # Save additional parameters
            import pickle
            with open(f"{path}/sarimax_params.pkl", "wb") as f:
                pickle.dump({
                    "order": self.order,
                    "seasonal_order": self.seasonal_order,
                    "lambda_value": self.lambda_value,
                    "exog_columns": self.exog_columns,
                    "config": self.config
                }, f)
                
            logger.info(f"SARIMAX model saved to {path}")
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
            from statsmodels.tsa.statespace.sarimax import SARIMAXResults
            self.results = SARIMAXResults.load(f"{path}/sarimax_model.pkl")
            
            # Load parameters
            import pickle
            with open(f"{path}/sarimax_params.pkl", "rb") as f:
                params = pickle.load(f)
                self.order = params["order"]
                self.seasonal_order = params["seasonal_order"]
                self.lambda_value = params["lambda_value"]
                self.exog_columns = params["exog_columns"]
                self.config = params["config"]
                
            logger.info(f"SARIMAX model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
