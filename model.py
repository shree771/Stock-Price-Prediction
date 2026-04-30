import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Fetch data
def get_data(stock):
    data = yf.download(stock, start="2018-01-01", end="2024-01-01")
    return data

# Train model
def train_lstm(data):
    dataset = data[['Close']].values

    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    X, y = [], []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i])
        y.append(scaled_data[i])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=10, batch_size=32, verbose=0)

    return model, scaler

# Predict next value
def predict_lstm(model, scaler, data):
    last_60 = data[['Close']].values[-60:]
    last_60_scaled = scaler.transform(last_60)

    X_test = []
    X_test.append(last_60_scaled)
    X_test = np.array(X_test)

    pred = model.predict(X_test, verbose=0)
    pred = scaler.inverse_transform(pred)

    return pred[0][0]