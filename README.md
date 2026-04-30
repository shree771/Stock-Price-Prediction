Introduction:

A Stock Price Prediction Web Application built using Streamlit (Python) that analyzes historical stock data and predicts future prices using an LSTM (Long Short-Term Memory) model. Users can easily select stocks, visualize trends, and view predicted prices through a clean and interactive interface.

Project Overview:

This project collects historical stock data from financial APIs and uses deep learning techniques to forecast future prices.

The application:

Fetches historical stock data using API requests
Processes and normalizes the dataset
Predicts future stock prices using LSTM model
Displays graphs for visualization
Uses Streamlit backend with interactive frontend

Features:
Real-time stock data visualization
LSTM-based prediction model
Clean and simple UI design
Dropdown for selecting different stocks
Slider to choose prediction days
Graph showing actual vs predicted prices
Model accuracy metrics (RMSE, MAE)

Technologies Used:
Python
Streamlit
NumPy
Pandas
Matplotlib
Scikit-learn
TensorFlow / Keras (LSTM)
Yahoo Finance API (or similar)

Project Structure:

Stock-Price-Prediction/
│
├── app.py # Streamlit frontend application
├── model.py # LSTM model logic
├── requirements.txt # Python dependencies
│
├── data/ # Dataset (optional)
│
└── README.md # Project documentation

Installation and Setup:

Follow these steps to run the project locally.

1️) Clone Repository:

git clone https://github.com/your-username/stock-price-prediction.git

cd stock-price-prediction

2️) Create Virtual Environment (Recommended)

Windows:
python -m venv venv
venv\Scripts\activate

Mac/Linux:
python3 -m venv venv
source venv/bin/activate

3️) Install Dependencies:

pip install -r requirements.txt

4️) Run Application:

streamlit run app.py

5️) Open Browser:

Visit: http://localhost:8501/

Author:

Sowmiya V
