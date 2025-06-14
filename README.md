# 📈 Stock Price Trend Prediction using LSTM

Predict future stock prices using historical data and LSTM neural networks. Visualize predictions, moving averages, and RSI on an interactive Streamlit dashboard.

---

## 🚀 Features

- Fetch stock data using `yfinance`
- Normalize and prepare data for time series
- Build and train LSTM model with Keras
- Plot actual vs predicted prices
- Calculate SMA (20, 50) and RSI indicators
- Deploy with Streamlit for interactive use

---

## 🛠 Tools & Libraries

- Python
- Keras (TensorFlow backend)
- Pandas, NumPy, Matplotlib
- yfinance
- Streamlit

---

## 📂 Files

- `stock_prediction_lstm.ipynb` – Jupyter notebook for model training
- `lstm_model.h5` – Trained model weights
- `streamlit_dashboard.py` – Streamlit dashboard for visualization
- `requirements.txt` – Python dependencies

---

## ▶️ How to Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run notebook (training)
jupyter notebook stock_prediction_lstm.ipynb

# Run Streamlit dashboard
streamlit run streamlit_dashboard.py
