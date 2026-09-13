# 📈 Stock Price Prediction using LSTM

A **Stock Price Prediction web application** built using **Python, TensorFlow/Keras, LSTM, and Flask**. The project uses historical stock market data to train a Long Short-Term Memory (LSTM) neural network and provides a web interface for predicting stock prices.

---

## 📌 Project Overview

Stock prices are time-series data that contain sequential patterns and dependencies. This project uses an **LSTM (Long Short-Term Memory)** neural network, a type of Recurrent Neural Network (RNN), to learn patterns from historical stock-price data and generate predictions.

The trained model is integrated into a **Flask web application**, allowing users to interact with the prediction system through a browser.

---

## ✨ Features

- 📊 Historical stock price analysis
- 🧠 LSTM-based deep learning model
- 📈 Time-series forecasting
- 🔄 Data normalization using Scikit-learn
- 💾 Pre-trained model saved in `.h5` format
- 💾 Saved scaler using Pickle
- 🌐 Flask-based web interface
- 📱 Simple and interactive frontend
- ☁️ Deployment configuration for Render
- 🐍 Python-based machine learning pipeline

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python | Core programming language |
| TensorFlow | Deep learning framework |
| Keras | Building and training the LSTM model |
| NumPy | Numerical computation |
| Pandas | Data processing |
| Scikit-learn | Data preprocessing and scaling |
| Flask | Web application backend |
| HTML/CSS | Frontend |
| Jupyter Notebook | Model development and experimentation |
| Render | Deployment |

---

## 📂 Project Structure

```text
Stock-Prediction/
│
├── static/
│   └──              # Static files such as CSS and other assets
│
├── templates/
│   └──              # HTML templates
│
├── .render.yaml     # Render deployment configuration
├── Procfile         # Application startup configuration
├── Stock.ipynb      # Model development and training notebook
├── app.py           # Flask application
├── lstm_model1.h5   # Trained LSTM model
├── requirements.txt # Python dependencies
├── runtime.txt      # Python runtime version
└── scaler.pkl       # Saved data scaler
