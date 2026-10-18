---
title: Hybrid HCLA Stock & Volatility Predictor
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: 1.38.0
app_file: frontend.py
pinned: false
license: mit
---

# 📈 Hybrid HCLA Volatility & Stock Price Predictor

An advanced AI-powered stock forecasting and volatility prediction dashboard. This project implements a **Hybrid CNN-LSTM Attention (HCLA)** deep learning architecture combined with **Variational Mode Decomposition (VMD)** for noise reduction, alongside multi-model benchmarking (Vanilla LSTM & ARIMA) and real-time news sentiment analysis.

---

## 🌟 Key Features

- **🧠 Hybrid HCLA Architecture**:
  - **Conv1D**: Extracts high-level spatial and local pattern features from multi-variable time series.
  - **LSTM**: Captures long-term temporal dependencies.
  - **Custom Attention Layer**: Dynamically assigns importance weights to critical historical time steps.
  - **Linear Trend Residual Extrapolation**: Fuses deep representations with baseline trend dynamics for superior convergence and stability.
- **🌊 VMD Signal Denoising**:
  - Employs **Variational Mode Decomposition** (`vmdpy`) to decompose complex stock time series into discrete modes, filtering out high-frequency market noise before model training.
- **📊 21+ Technical Indicators**:
  - Automatically calculates SMA (10/20/50), EMA (10/20/50), RSI, MACD (Line, Signal, Histogram), Bollinger Bands (Upper/Middle/Lower), Momentum, Rate of Change (ROC), Historical & EWMA Volatility, Log Returns, and Disparity Index.
- **⚖️ Multi-Model Benchmarking**:
  - Compares the **HCLA** model against a **Vanilla LSTM** and a classical **ARIMA** baseline across $R^2$, **RMSE**, and **MDA (Mean Directional Accuracy)**.
- **🔮 1-Month (21 Trading Days) Future Projection**:
  - Iteratively rolls forward indicators and forecasts out-of-sample future price trajectories.
- **📰 Live Sentiment Analysis**:
  - Fetches the latest stock news via Yahoo Finance and evaluates market sentiment using **VADER Sentiment Analysis**.
- **🎨 Interactive Streamlit Web UI**:
  - Modern dashboard with Plotly interactive charts (zoom, pan, range slider).
  - Dark / Light mode toggle.
  - One-click Excel export (`.xlsx`) of the denoised feature dataset.

---

## 🏗️ Architecture & Pipeline

```
 [ Yahoo Finance Data (yfinance) ]
                │
                ▼
 [ 21 Technical Indicators Calculation ]
 (SMA, EMA, RSI, MACD, Bollinger Bands, Volatility, etc.)
                │
                ▼
 [ Variational Mode Decomposition (VMD) ]
 (Noise reduction by filtering high-frequency modes)
                │
                ▼
 [ MinMax Scaling & Sliding Window (60 Days) ]
                │
                ├───────────────────────┬───────────────────────┐
                ▼                       ▼                       ▼
         [ HCLA Model ]          [ Vanilla LSTM ]        [ ARIMA Baseline ]
     (Conv1D + LSTM + Attn)
                │                       │                       │
                └───────────────────────┴───────────────────────┘
                                        │
                                        ▼
                   [ Model Evaluation: R², RMSE, MDA ]
                                        │
                                        ▼
             [ Out-of-Sample 1-Month Multi-Step Future Forecast ]
                                        │
                                        ▼
              [ Streamlit Dashboard & News Sentiment Display ]
```

---

## 📁 Project Structure

```plaintext
├── frontend.py          # Streamlit web dashboard application
├── model_engine.py      # Core ML engine (Data fetching, VMD, HCLA, LSTM, ARIMA, Future forecasting)
├── test_news.py         # Utility script to test Yahoo Finance news fetching & parsing
├── test_vmd.py          # Utility script to test Variational Mode Decomposition
├── requirements.txt     # Python package dependencies
└── README.md            # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites
- **Python 3.9+** (Python 3.10 / 3.11 recommended)
- `pip` package manager

### 2. Clone / Open Directory
Open your terminal in the project directory:
```powershell
cd c:\Users\Asus\OneDrive\Desktop\FRP
```

### 3. Create & Activate a Virtual Environment (Recommended)
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# Activate on Windows (CMD)
.\venv\Scripts\activate.bat

# Activate on Linux/macOS
source venv/bin/activate
```

### 4. Install Dependencies
```powershell
pip install -r requirements.txt
```

---

## 🖥️ Running the Application

### Launch the Web Dashboard (Streamlit)
```powershell
streamlit run frontend.py
```
After running, open your browser at **`http://localhost:8501`**.

#### Inside the Dashboard:
1. Enter any valid ticker symbol in the sidebar (e.g., `TSLA`, `AAPL`, `NVDA`, `BTC-USD`, `MSFT`).
2. Adjust the number of **Training Epochs** using the slider (default: 20).
3. Click **"Train & Predict"**.
4. Browse through the 3 tabs:
   - **📈 Forecast Overview**: Actual vs. fitted prices, 1-month forecast curve, and Excel export.
   - **⚖️ Model Comparison**: Evaluation table highlighting the best performing model.
   - **📰 News & Sentiment**: Real-time articles with compound sentiment polarity scores.

---

### Command-Line Execution (Standalone Test)
To run the model directly from the terminal without the UI:
```powershell
python model_engine.py
```

### Run Component Tests
- **Test News Sentiment API**:
  ```powershell
  python test_news.py
  ```
- **Test VMD Denoising**:
  ```powershell
  python test_vmd.py
  ```

---

## 📐 Evaluation Metrics

| Metric | Full Name | Description | Target |
| :--- | :--- | :--- | :--- |
| **$R^2$** | Coefficient of Determination | Measures the proportion of variance explained by the model. | Closer to $1.0$ is better |
| **RMSE** | Root Mean Squared Error | Measures the standard deviation of prediction residuals. | Lower is better |
| **MDA** | Mean Directional Accuracy | Percentage of times the model correctly predicts the price direction (Up/Down). | Closer to $100\%$ ($1.0$) is better |

---

## 📦 Dependencies

- **UI & Visualization**: `streamlit`, `plotly`, `openpyxl`
- **Machine Learning & Deep Learning**: `tensorflow`, `scikit-learn`, `statsmodels`
- **Signal Processing**: `vmdpy`
- **Financial Data & NLP**: `yfinance`, `vaderSentiment`
- **Core Scientific Computation**: `numpy`, `pandas`

---

## ⚠️ Disclaimer

This project is built for **educational and research purposes only**. Financial markets are inherently stochastic; predictions provided by these models should not be used as financial or investment advice.
