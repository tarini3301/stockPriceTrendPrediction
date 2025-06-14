import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import datetime

st.set_page_config(page_title="Stock Price Prediction Dashboard", layout="wide")
st.title("📈 LSTM Stock Price Predictor")

# Sidebar - User Inputs
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
future_days = st.sidebar.slider("Forecast days into the future", 1, 90, 30)

# Fetch Data
data = yf.download(ticker, start=start_date, end=end_date)

# ✅ FIX: Flatten MultiIndex columns if present
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data[['Close']]

# Moving Averages
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['SMA_50'] = data['Close'].rolling(window=50).mean()

# RSI Indicator
delta = data['Close'].diff()
up = delta.clip(lower=0)
down = -1 * delta.clip(upper=0)
avg_gain = up.rolling(window=14).mean()
avg_loss = down.rolling(window=14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# Normalize for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close']].dropna())

# Prepare sequences
window_size = 60
X = []
for i in range(window_size, len(scaled_data)):
    X.append(scaled_data[i - window_size:i])
X = np.array(X)

# Load LSTM Model
model = load_model('lstm_model.h5')

# Predict past values
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)
actual_prices = scaler.inverse_transform(scaled_data[window_size:])

# Plot prediction vs actual
st.subheader(f"{ticker} - Historical Prediction")
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(actual_prices, label='Actual')
ax1.plot(predicted_prices, label='Predicted')
ax1.set_xlabel("Days")
ax1.set_ylabel("Price (USD)")
ax1.set_title("Actual vs Predicted Prices")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# Forecast future prices
last_sequence = scaled_data[-window_size:]
future_predictions = []
current_input = last_sequence.copy()

for _ in range(future_days):
    current_input_reshaped = np.reshape(current_input, (1, window_size, 1))
    next_pred = model.predict(current_input_reshaped, verbose=0)
    future_predictions.append(next_pred[0, 0])
    current_input = np.append(current_input[1:], [[next_pred[0, 0]]], axis=0)

# Inverse transform forecast
future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
future_dates = pd.date_range(data.index[-1] + pd.Timedelta(days=1), periods=future_days)
forecast_df = pd.DataFrame(data=future_prices, index=future_dates, columns=['Forecast'])

# Plot forecast
st.subheader(f"{ticker} Forecast for Next {future_days} Days")
fig2, ax2 = plt.subplots(figsize=(12, 6))
ax2.plot(data['Close'], label='Historical')
ax2.plot(forecast_df, label='Forecast', color='orange')
ax2.set_xlabel("Date")
ax2.set_ylabel("Price (USD)")
ax2.legend()
ax2.grid()
st.pyplot(fig2)

# Plot Indicators
st.subheader(f"{ticker} Closing Price & Moving Averages")
st.line_chart(data[['Close', 'SMA_20', 'SMA_50']].dropna())

# Plot RSI
st.subheader("RSI Indicator")
st.line_chart(data['RSI'].dropna())
