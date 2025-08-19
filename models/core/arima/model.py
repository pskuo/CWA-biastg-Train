import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Function to calculate Mean Bias Error (MBE)
def mean_bias_error(y_true, y_pred):
    return np.mean(y_true - y_pred)

# Function to calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def rmse()

# Read the CSV file
file_name = 'SeaData\Cold_Tongue_Equatorial Current_s_dbias.csv'
df = pd.read_csv(file_name, parse_dates=['date'], index_col='date')

# Fill in missing data using forward filling
df.fillna(method='ffill', inplace=True)

# Split the dataset into training and testing datasets
train_data = df.iloc[:-10]
test_data = df.iloc[-10:]
# Inspect the data
print(df.head())

# Plot the time series data
plt.figure(figsize=(10, 5))
plt.plot(df)
plt.xlabel('Date')
plt.ylabel('biastg')
plt.title('Time Series Data')
plt.show()

# Determine the ARIMA order (p, d, q) - You may need to manually adjust these values
# Define the parameter grid for the grid search
p_values = range(0, 3)
d_values = range(0, 3)
q_values = range(0, 3)

# Perform grid search
best_aic = np.inf
best_params = None

for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = ARIMA(df, order=(p, d, q))
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_params = (p, d, q)
            except:
                continue

print(f"Best ARIMA parameters: {best_params}")

# Fit the ARIMA model
model = ARIMA(df, order=best_params)
results = model.fit()

# Summarize the model
print(results.summary())

# Forecast the next 10 days
forecast_steps = 10
forecast = results.forecast(steps=forecast_steps)
print(forecast)

# Plot the original data and the forecast
plt.figure(figsize=(10, 5))
plt.plot(df.index[-100:], df['biastg'].iloc[-100:], label='Original Data')
plt.plot(pd.date_range(df.index[-1], periods=forecast_steps+1, closed='right'), forecast, label='Forecast', linestyle='--', color='red')
plt.xlabel('Date')
plt.ylabel('biastg')
plt.title('Time Series Data with ARIMA Forecast')
plt.legend()
plt.show()

# Extract 'biastg' values from the test_data DataFrame
test_data_values = test_data['biastg'].values

# Ensure the lengths of the arrays are equal
assert len(test_data_values) == len(forecast), "The lengths of test_data_values and forecast arrays do not match."

# Calculate performance metrics
rmse = np.sqrt(mean_squared_error(test_data_values, forecast))
mae = mean_absolute_error(test_data_values, forecast)
mbe = mean_bias_error(test_data_values, forecast)
mape = mean_absolute_percentage_error(test_data_values, forecast)

print(f"RMSE: {rmse}")
print(f"MAE: {mae}")
print(f"MBE: {mbe}")
print(f"MAPE: {mape}")