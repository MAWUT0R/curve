import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load data from CSV file
df = pd.read_csv('training_data.csv')

# Feature engineering (you can add more features if needed)
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.dayofweek

# ARIMA model (adjust order as needed)
order = (1, 1, 1)  # Example order, adjust as needed
arima_model = sm.tsa.ARIMA(df['volume_usd'], order=order)
arima_results = arima_model.fit()

# SVM model
X = df[['active_users', 'tx_count', 'day_of_week']]
y = df['volume_usd']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

svm_model = SVR()
svm_model.fit(X_train, y_train)

# Make predictions
arima_pred = arima_results.predict(start=len(df), end=len(df) + 6)
svm_pred = svm_model.predict(df[['active_users', 'tx_count', 'day_of_week']].tail(7))

# Ensemble predictions (simple average for demonstration)
ensemble_pred = (arima_pred + svm_pred) / 2

# Display predictions
predictions = pd.DataFrame({'date': pd.date_range(start=df['date'].max() + pd.Timedelta(days=1), periods=7),
                             'predicted_volume': ensemble_pred})

print(predictions)

# Plotting ARIMA predictions
plt.plot(df['date'], df['volume_usd'], label='Actual')
plt.plot(predictions['date'], predictions['predicted_volume'], label='Ensemble Prediction')
plt.legend()
plt.show()
