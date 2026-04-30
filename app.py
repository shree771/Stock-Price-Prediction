import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model import get_data, train_lstm, predict_lstm

st.title("📊 Stock Price Prediction ")

# =========================
# Sidebar UI
# =========================
st.sidebar.header("Settings")

stock_options = {
    "Apple (AAPL)": "AAPL",
    "Tesla (TSLA)": "TSLA",
    "Microsoft (MSFT)": "MSFT",
    "Google (GOOGL)": "GOOGL",
    "Amazon (AMZN)": "AMZN",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "TCS (TCS.NS)": "TCS.NS",
    "Infosys (INFY.NS)": "INFY.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "SBI (SBIN.NS)": "SBIN.NS"
}

selected_stock = st.sidebar.selectbox("Select Stock", list(stock_options.keys()))
stock = stock_options[selected_stock]

days = st.sidebar.slider("Days to Predict", 1, 30, 10)

if st.button("Predict"):

    data = get_data(stock)

    # =========================
    # Live Chart
    # =========================
    st.subheader("📈 Live Stock Price")
    st.line_chart(data['Close'])

    st.subheader("Raw Data")
    st.write(data.tail())

    # =========================
    # Moving Averages
    # =========================
    data['MA50'] = data['Close'].rolling(50).mean()
    data['MA100'] = data['Close'].rolling(100).mean()

    # =========================
    # Train Model
    # =========================
    model, scaler = train_lstm(data)

    # =========================
    # Full Prediction
    # =========================
    dataset = data[['Close']].values
    scaled_data = scaler.transform(dataset)

    X = []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])

    X = np.array(X)

    predicted_full = model.predict(X, verbose=0)
    predicted_full = scaler.inverse_transform(predicted_full)

    pred_plot = [None]*60 + list(predicted_full.flatten())

    # =========================
    # Accuracy
    # =========================
    actual = data['Close'][60:]
    predicted_vals = predicted_full.flatten()

    rmse = np.sqrt(mean_squared_error(actual, predicted_vals))
    mae = mean_absolute_error(actual, predicted_vals)

    st.subheader("📊 Model Accuracy")
    st.write(f"RMSE: {rmse:.2f}")
    st.write(f"MAE: {mae:.2f}")

    # =========================
    # Future Prediction
    # =========================
    future_preds = []
    temp_data = data.copy()

    for _ in range(days):
        pred = predict_lstm(model, scaler, temp_data)
        future_preds.append(pred)

        new_row = temp_data.iloc[-1:].copy()
        new_row['Close'] = pred
        temp_data = pd.concat([temp_data, new_row], ignore_index=True)

    last_date = data.index[-1]
    future_dates = pd.date_range(start=last_date, periods=days+1, freq='D')[1:]

    # =========================
    # Graph
    # =========================
    st.subheader("📉 Stock Prediction Graph")

    fig, ax = plt.subplots()

    ax.plot(data.index, data['Close'], label="Actual Price", linewidth=2)
    ax.plot(data.index, pred_plot, label="Model Prediction", linestyle='dashed')
    ax.plot(data.index, data['MA50'], label="MA50")
    ax.plot(data.index, data['MA100'], label="MA100")

    ax.plot(future_dates, future_preds, label="Future Prediction", marker='o')

    ax.legend()
    st.pyplot(fig)

    # =========================
    # Download CSV
    # =========================
    df_pred = pd.DataFrame({
        "Date": future_dates,
        "Predicted Price": future_preds
    })

    st.download_button(
        "⬇ Download Predictions CSV",
        df_pred.to_csv(index=False),
        file_name="predictions.csv"
    )