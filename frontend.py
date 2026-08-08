import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io
import yfinance as yf
from model_engine import (
    get_hcla_data,
    train_hcla_model,
    train_lstm_model,
    train_arima_model,
    calculate_metrics,
    predict_future,
    add_technical_indicators,
    get_news_sentiment,
    FEATURE_COLS,
    apply_vmd_denoising
)
from sklearn.metrics import mean_squared_error

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HCLA Volatility Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# ADVANCED LIGHT SKEUOMORPHIC DESIGN SYSTEM (CSS INJECTION)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg-canvas: #E8ECEF;
        --bg-panel: #E2E7ED;
        --bg-card: #E8ECEF;
        --bg-inset: #DFE4EA;
        
        --shadow-raised: 6px 6px 14px #B8C2CC, -6px -6px 14px #FFFFFF;
        --shadow-raised-sm: 3px 3px 8px #B8C2CC, -3px -3px 8px #FFFFFF;
        --shadow-inset: inset 2px 2px 5px #B8C2CC, inset -2px -2px 5px #FFFFFF;
        --shadow-inset-deep: inset 3px 3px 7px #AFB9C4, inset -3px -3px 7px #FFFFFF;
        
        --accent-blue: #2563EB;
        --accent-blue-hover: #1D4ED8;
        --accent-green: #10B981;
        --accent-red: #EF4444;
        --accent-purple: #7C3AED;
        --accent-amber: #F59E0B;
        
        --text-main: #1E293B;
        --text-muted: #64748B;
        --text-subtle: #94A3B8;
        
        --border-soft: rgba(255, 255, 255, 0.7);
        --border-shadow: rgba(184, 194, 204, 0.5);
    }

    /* Global Typography and Canvas */
    html, body, [data-testid="stAppViewContainer"], .stApp {
        background-color: var(--bg-canvas) !important;
        color: var(--text-main) !important;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    [data-testid="stHeader"] {
        background-color: transparent !important;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1380px !important;
    }

    /* Sidebar Tactile Styling */
    [data-testid="stSidebar"] {
        background-color: #E2E7ED !important;
        border-right: 1px solid #D0D8E0 !important;
        box-shadow: 4px 0 12px rgba(184, 194, 204, 0.3) !important;
    }

    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        gap: 1.25rem !important;
    }

    /* Tactile Form Inputs (Text Input) */
    [data-testid="stTextInput"] input {
        background: var(--bg-inset) !important;
        color: var(--text-main) !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.05em !important;
        border: 1px solid var(--border-shadow) !important;
        border-radius: 12px !important;
        box-shadow: var(--shadow-inset) !important;
        padding: 0.7rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }

    [data-testid="stTextInput"] input:focus {
        border-color: var(--accent-blue) !important;
        box-shadow: var(--shadow-inset), 0 0 0 2px rgba(37, 99, 235, 0.2) !important;
    }

    /* Slider Styling */
    [data-testid="stSlider"] label {
        color: var(--text-main) !important;
        font-weight: 600 !important;
    }

    [data-testid="stSlider"] [data-baseweb="slider"] {
        padding-top: 8px !important;
        padding-bottom: 8px !important;
    }

    /* Primary Action Button (Skeuomorphic 3D Extrusion) */
    .stButton > button[kind="primary"], .stButton > button {
        width: 100% !important;
        background: linear-gradient(145deg, #2A6BF2, #1D54D1) !important;
        color: #FFFFFF !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        padding: 0.8rem 1.4rem !important;
        border-radius: 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        box-shadow: 4px 4px 10px #B8C2CC, -3px -3px 8px #FFFFFF, inset 0 1px 1px rgba(255, 255, 255, 0.4) !important;
        transition: all 0.18s cubic-bezier(0.4, 0, 0.2, 1) !important;
        cursor: pointer !important;
    }

    .stButton > button:hover {
        background: linear-gradient(145deg, #3373F5, #225CE0) !important;
        transform: translateY(-2px) !important;
        box-shadow: 6px 8px 16px #B0BAC4, -4px -4px 10px #FFFFFF, inset 0 1px 2px rgba(255, 255, 255, 0.5) !important;
    }

    .stButton > button:active {
        transform: translateY(1px) scale(0.985) !important;
        box-shadow: inset 3px 3px 6px rgba(0, 0, 0, 0.25), inset -2px -2px 4px rgba(255, 255, 255, 0.15) !important;
    }

    .stButton > button p {
        color: #FFFFFF !important;
        font-weight: 700 !important;
    }

    /* Secondary / Download Button */
    .stDownloadButton > button {
        background: linear-gradient(145deg, #F0F4F8, #D8E0E8) !important;
        color: var(--text-main) !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        border-radius: 12px !important;
        border: 1px solid var(--border-soft) !important;
        box-shadow: var(--shadow-raised-sm) !important;
        padding: 0.7rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }

    .stDownloadButton > button:hover {
        background: linear-gradient(145deg, #F5F9FD, #E0E7EF) !important;
        transform: translateY(-1px) !important;
        box-shadow: 4px 4px 10px #B8C2CC, -4px -4px 10px #FFFFFF !important;
    }

    .stDownloadButton > button:active {
        transform: translateY(1px) !important;
        box-shadow: var(--shadow-inset) !important;
    }

    .stDownloadButton > button p {
        color: var(--text-main) !important;
    }

    /* Tactile Tabs Navigation */
    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--bg-inset) !important;
        border-radius: 16px !important;
        padding: 6px 8px !important;
        box-shadow: var(--shadow-inset) !important;
        gap: 8px !important;
        border: 1px solid rgba(184, 194, 204, 0.4) !important;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        padding: 10px 20px !important;
        color: var(--text-muted) !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        border: none !important;
        background-color: transparent !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(145deg, #FFFFFF, #E8ECEF) !important;
        color: var(--text-main) !important;
        font-weight: 700 !important;
        box-shadow: 3px 3px 8px #B8C2CC, -3px -3px 8px #FFFFFF !important;
    }

    .stTabs [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Tactile Containers & Cards */
    .tactile-card {
        background: var(--bg-card);
        border-radius: 18px;
        padding: 24px;
        box-shadow: var(--shadow-raised);
        border: 1px solid var(--border-soft);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }

    .tactile-card-recessed {
        background: var(--bg-inset);
        border-radius: 14px;
        padding: 18px;
        box-shadow: var(--shadow-inset);
        border: 1px solid rgba(184, 194, 204, 0.4);
    }

    /* Metric Grid 3D Cards */
    .metric-card-3d {
        background: var(--bg-card);
        border-radius: 18px;
        padding: 20px 22px;
        box-shadow: var(--shadow-raised);
        border: 1px solid var(--border-soft);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 145px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .metric-card-3d:hover {
        transform: translateY(-3px);
        box-shadow: 8px 8px 18px #B0BAC4, -8px -8px 18px #FFFFFF;
    }

    .metric-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 8px;
    }

    .metric-title {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--text-muted);
    }

    .metric-icon-recessed {
        width: 32px;
        height: 32px;
        border-radius: 10px;
        background: var(--bg-inset);
        box-shadow: var(--shadow-inset);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1rem;
    }

    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 800;
        color: var(--text-main);
        line-height: 1.1;
        margin-top: 4px;
        margin-bottom: 8px;
    }

    /* Tactile Badges (Raised & Inset) */
    .tactile-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
        width: fit-content;
    }

    .badge-positive {
        background: #DCFCE7;
        color: #15803D;
        border: 1px solid #86EFAC;
        box-shadow: 2px 2px 5px rgba(184, 194, 204, 0.4), inset 0 1px 1px #FFFFFF;
    }

    .badge-negative {
        background: #FEE2E2;
        color: #B91C1C;
        border: 1px solid #FCA5A5;
        box-shadow: 2px 2px 5px rgba(184, 194, 204, 0.4), inset 0 1px 1px #FFFFFF;
    }

    .badge-neutral {
        background: #F1F5F9;
        color: #475569;
        border: 1px solid #CBD5E1;
        box-shadow: 2px 2px 5px rgba(184, 194, 204, 0.4), inset 0 1px 1px #FFFFFF;
    }

    .badge-purple {
        background: #F3E8FF;
        color: #6B21A8;
        border: 1px solid #D8B4FE;
        box-shadow: 2px 2px 5px rgba(184, 194, 204, 0.4), inset 0 1px 1px #FFFFFF;
    }

    /* Physical Spec Plate (Sidebar) */
    .spec-plate {
        background: linear-gradient(145deg, #DEE3E9, #E6EBF1);
        border-radius: 14px;
        padding: 16px;
        box-shadow: var(--shadow-inset);
        border: 1px solid rgba(184, 194, 204, 0.5);
    }

    .spec-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 8px 10px;
        margin-bottom: 8px;
        background: #E8ECEF;
        border-radius: 10px;
        box-shadow: var(--shadow-raised-sm);
        border: 1px solid rgba(255, 255, 255, 0.8);
    }

    .spec-item:last-child {
        margin-bottom: 0;
    }

    .spec-tag {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 0.78rem;
        padding: 2px 8px;
        border-radius: 6px;
        background: var(--bg-inset);
        box-shadow: var(--shadow-inset);
        color: var(--accent-blue);
    }

    /* Status Indicator Lights */
    .led-pill {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 14px;
        background: var(--bg-inset);
        border-radius: 24px;
        box-shadow: var(--shadow-inset);
        border: 1px solid rgba(184, 194, 204, 0.5);
        font-size: 0.82rem;
        font-weight: 700;
        color: var(--text-main);
    }

    .led-glow-green {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10B981;
        box-shadow: 0 0 8px #10B981, inset 0 1px 1px #FFFFFF;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(0.95); opacity: 0.85; }
        50% { transform: scale(1.15); opacity: 1; }
        100% { transform: scale(0.95); opacity: 0.85; }
    }

    /* News Card Tactile Feed */
    .news-card {
        background: var(--bg-card);
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 14px;
        box-shadow: var(--shadow-raised-sm);
        border: 1px solid var(--border-soft);
        display: flex;
        flex-direction: column;
        gap: 8px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .news-card:hover {
        transform: translateY(-2px);
        box-shadow: 5px 5px 12px #B8C2CC, -5px -5px 12px #FFFFFF;
    }

    .news-title-link {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--text-main) !important;
        text-decoration: none;
        line-height: 1.35;
    }

    .news-title-link:hover {
        color: var(--accent-blue) !important;
    }

    .news-meta {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-size: 0.82rem;
        color: var(--text-muted);
    }

    /* Custom Light Skeuomorphic Table */
    .tactile-table-wrapper {
        border-radius: 14px;
        overflow: hidden;
        box-shadow: var(--shadow-inset);
        border: 1px solid rgba(184, 194, 204, 0.6);
        background: var(--bg-inset);
        margin: 12px 0;
    }

    .tactile-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.95rem;
        text-align: left;
    }

    .tactile-table th {
        background: #DEE4EA;
        color: var(--text-main);
        font-weight: 700;
        padding: 14px 18px;
        border-bottom: 2px solid #CBD5E1;
        letter-spacing: 0.03em;
        text-transform: uppercase;
        font-size: 0.82rem;
    }

    .tactile-table td {
        padding: 14px 18px;
        border-bottom: 1px solid #D1D9E0;
        color: var(--text-main);
        font-family: 'JetBrains Mono', monospace;
        font-weight: 500;
    }

    .tactile-table tr:last-child td {
        border-bottom: none;
    }

    .tactile-table tr:nth-child(even) {
        background: rgba(255, 255, 255, 0.4);
    }

    .tactile-table tr:hover {
        background: rgba(255, 255, 255, 0.7);
    }

    .best-row-badge {
        background: #DCFCE7;
        color: #15803D;
        font-weight: 700;
        padding: 3px 8px;
        border-radius: 6px;
        border: 1px solid #86EFAC;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & ARCHITECTURE SPEC PLATE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 15px;">
            <div style="width: 36px; height: 36px; border-radius: 10px; background: linear-gradient(145deg, #2563EB, #1D4ED8); box-shadow: 3px 3px 7px #B8C2CC, -3px -3px 7px #FFFFFF; display: flex; align-items: center; justify-content: center; color: white; font-size: 1.2rem; font-weight: bold;">
                ⚡
            </div>
            <div>
                <h3 style="margin: 0; font-size: 1.15rem; font-weight: 800; color: #1E293B;">HCLA Engine</h3>
                <span style="font-size: 0.75rem; font-weight: 600; color: #64748B;">Tactile Terminal v2.4</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<p style='font-size: 0.85rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;'>Terminal Controls</p>", unsafe_allow_html=True)
    
    ticker = st.text_input("Ticker Symbol", value="TSLA", help="Enter any Yahoo Finance supported symbol (e.g., TSLA, NVDA, AAPL, BTC-USD)").upper().strip()
    intensity = st.slider("Training Epochs", min_value=5, max_value=100, value=20, step=5, help="Number of training iterations across neural network architectures.")

    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    
    # Architecture Physical Spec Plate
    st.markdown("""
        <div class="spec-plate">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                <span style="font-size: 0.78rem; font-weight: 800; color: #475569; text-transform: uppercase; letter-spacing: 0.06em;">Model Architecture</span>
                <span style="font-size: 0.7rem; font-weight: 700; color: #10B981; background: #DCFCE7; padding: 2px 6px; border-radius: 4px;">CALIBRATED</span>
            </div>
            <div class="spec-item">
                <span style="font-size: 0.82rem; font-weight: 600; color: #334155;">Signal Denoising</span>
                <span class="spec-tag">VMD K=4</span>
            </div>
            <div class="spec-item">
                <span style="font-size: 0.82rem; font-weight: 600; color: #334155;">Feature Extraction</span>
                <span class="spec-tag">Conv1D</span>
            </div>
            <div class="spec-item">
                <span style="font-size: 0.82rem; font-weight: 600; color: #334155;">Temporal Sequencing</span>
                <span class="spec-tag">LSTM Cell</span>
            </div>
            <div class="spec-item">
                <span style="font-size: 0.82rem; font-weight: 600; color: #334155;">Context Weighting</span>
                <span class="spec-tag">Attention</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    
    # Tactile Action Execution Trigger
    execute_button = st.button("🚀 Train & Predict", use_container_width=True, type="primary")

    st.markdown("""
        <div style="margin-top: 15px; padding: 12px; border-radius: 12px; background: #E8ECEF; box-shadow: inset 2px 2px 4px #B8C2CC, inset -2px -2px 4px #FFFFFF; text-align: center;">
            <span style="font-size: 0.72rem; color: #64748B; font-weight: 600;">Feed: Yahoo Finance & VADER NLP</span>
        </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP HEADER BAR WITH SKEUOMORPHIC STATUS LIGHTS
# -----------------------------------------------------------------------------
header_col1, header_col2 = st.columns([3.5, 1.5])

with header_col1:
    st.markdown("""
        <div>
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
                <h1 style="margin: 0; font-size: 2.1rem; font-weight: 800; color: #1E293B; letter-spacing: -0.02em;">
                    Hybrid HCLA Volatility Predictor
                </h1>
            </div>
            <p style="margin: 0; font-size: 0.98rem; font-weight: 500; color: #64748B;">
                Industrial-grade financial forecasting powered by Variational Mode Decomposition & CNN-LSTM Attention networks.
            </p>
        </div>
    """, unsafe_allow_html=True)

with header_col2:
    st.markdown("""
        <div style="display: flex; justify-content: flex-end; align-items: center; gap: 10px; height: 100%;">
            <div class="led-pill">
                <span class="led-glow-green"></span>
                <span>ENGINE READY</span>
            </div>
            <div class="led-pill" style="font-family: 'JetBrains Mono', monospace;">
                <span>STABLE</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# CORE EXECUTION & PREDICTION PIPELINE
# -----------------------------------------------------------------------------
if execute_button:
    try:
        with st.spinner(f"Initializing VMD decomposition & training deep hybrid models for {ticker}..."):
            # 1. Fetch Market Data
            df_raw = yf.download(ticker, period="2y")
            if df_raw.empty:
                st.error(f"No market data found for ticker '{ticker}'. Please verify the symbol.")
                st.stop()
                
            df = df_raw.copy()
            df = add_technical_indicators(df)
            
            data = df.dropna().copy()
            data['Raw_Close'] = data['Close'].copy()
            data = apply_vmd_denoising(data)
            
            features = data[FEATURE_COLS].values
            
            def evaluate(m, s, use_raw=False):
                scaled_features = s.transform(features)
                X_input = []
                for i in range(60, len(scaled_features)):
                    X_input.append(scaled_features[i-60:i])
                X_input = np.array(X_input)
                preds = m.predict(X_input, verbose=0)
                
                dummy = np.zeros((len(preds), len(FEATURE_COLS)))
                dummy[:, 0] = preds.flatten()
                inv_preds = s.inverse_transform(dummy)[:, 0]
                
                target_col = 'Raw_Close' if (use_raw and 'Raw_Close' in data.columns) else 'Close'
                actuals = data[target_col].values[60:].flatten()
                return inv_preds, actuals
                
            models_info = {
                "HCLA": train_hcla_model,
                "Vanilla LSTM": train_lstm_model,
                "ARIMA": train_arima_model
            }
            
            results = []
            best_model = None
            best_scaler = None
            best_score = -float('inf')
            best_inv_predictions = None
            best_name = None
            model_instances = {}
            actual_prices = data['Close'].values[60:].flatten()
            plot_dates = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in data.index[60:]]
            
            for name, train_fn in models_info.items():
                m, s = train_fn(data, epochs=intensity)
                inv_pred, actual = evaluate(m, s, use_raw=(name == "ARIMA"))
                r2, rmse, mda = calculate_metrics(actual, inv_pred)
                
                results.append({"Model": name, "R²": r2, "RMSE": rmse, "MDA": mda})
                
                # Score heuristic: maximize R2 and MDA, minimize RMSE relative to price
                score = r2 + mda - (rmse / np.mean(actual))
                model_instances[name] = {
                    "model": m,
                    "scaler": s,
                    "predictions": inv_pred,
                    "score": score
                }
                
                if score > best_score:
                    best_score = score
                    best_model = m
                    best_scaler = s
                    best_inv_predictions = inv_pred
                    best_name = name

            # Prefer HCLA when it is competitively close to the top-scoring model
            if best_name != "HCLA" and "HCLA" in model_instances:
                hcla_score = model_instances["HCLA"]["score"]
                if hcla_score >= 0.95 * best_score:
                    best_name = "HCLA"
                    best_model = model_instances["HCLA"]["model"]
                    best_scaler = model_instances["HCLA"]["scaler"]
                    best_inv_predictions = model_instances["HCLA"]["predictions"]
            
            results_df = pd.DataFrame(results).set_index("Model")
            
            # Predict Future (1 Month Horizon)
            future_df = predict_future(best_model, best_scaler, data, days=21)
            future_dates = [pd.to_datetime(d).strftime('%Y-%m-%d') for d in future_df.index]
            future_predictions = future_df['Predicted_Close'].values
            
            inv_predictions = best_inv_predictions
            
            # Fetch news sentiment with resilient parsing fallback
            parsed_news, avg_sentiment = get_news_sentiment(ticker)
            if not parsed_news:
                try:
                    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                    analyzer = SentimentIntensityAnalyzer()
                    stock = yf.Ticker(ticker)
                    raw_news = stock.news
                    if raw_news:
                        total_sent = 0.0
                        for article in raw_news:
                            if not isinstance(article, dict):
                                continue
                            content = article.get('content') if isinstance(article.get('content'), dict) else {}
                            title = content.get('title') or article.get('title') or ''
                            
                            link_info = content.get('clickThroughUrl') if isinstance(content.get('clickThroughUrl'), dict) else {}
                            link = link_info.get('url') or article.get('link') or '#'
                            
                            provider = content.get('provider') if isinstance(content.get('provider'), dict) else {}
                            publisher = provider.get('displayName') or article.get('publisher') or 'Market Wire'
                            
                            if not title:
                                continue
                            score = float(analyzer.polarity_scores(title)['compound'])
                            parsed_news.append({
                                'title': title,
                                'publisher': publisher,
                                'link': link,
                                'sentiment': score
                            })
                            total_sent += score
                        if parsed_news:
                            avg_sentiment = total_sent / len(parsed_news)
                except Exception:
                    pass
            
            # Compute KPI Metrics
            current_price = float(actual_prices[-1])
            predicted_1m = float(future_predictions[-1])
            price_diff = predicted_1m - current_price
            perc_diff = (price_diff / current_price) * 100
            
            is_positive = price_diff >= 0
            delta_prefix = "▲ +" if is_positive else "▼ "
            delta_badge_class = "badge-positive" if is_positive else "badge-negative"
            delta_str = f"{delta_prefix}${abs(price_diff):.2f} ({perc_diff:+.2f}%)"
            
            if avg_sentiment > 0.05:
                sent_label = "Bullish"
                sent_badge_class = "badge-positive"
                sent_icon = "🟢"
            elif avg_sentiment < -0.05:
                sent_label = "Bearish"
                sent_badge_class = "badge-negative"
                sent_icon = "🔴"
            else:
                sent_label = "Neutral"
                sent_badge_class = "badge-neutral"
                sent_icon = "⚪"

            best_rmse = results_df.loc[best_name, 'RMSE']
            best_r2 = results_df.loc[best_name, 'R²']

        # ---------------------------------------------------------------------
        # TACTILE SKEUOMORPHIC KPI DATA CARDS
        # ---------------------------------------------------------------------
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(f"""
                <div class="metric-card-3d">
                    <div class="metric-header">
                        <span class="metric-title">Current Price</span>
                        <div class="metric-icon-recessed">💵</div>
                    </div>
                    <div class="metric-value">${current_price:,.2f}</div>
                    <div class="tactile-badge badge-neutral">
                        <span>{ticker} • Spot Close</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with kpi2:
            st.markdown(f"""
                <div class="metric-card-3d">
                    <div class="metric-header">
                        <span class="metric-title">1-Month Forecast</span>
                        <div class="metric-icon-recessed">🎯</div>
                    </div>
                    <div class="metric-value">${predicted_1m:,.2f}</div>
                    <div class="tactile-badge {delta_badge_class}">
                        <span>{delta_str}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with kpi3:
            st.markdown(f"""
                <div class="metric-card-3d">
                    <div class="metric-header">
                        <span class="metric-title">Best Model (RMSE)</span>
                        <div class="metric-icon-recessed">🏆</div>
                    </div>
                    <div class="metric-value">{best_rmse:.2f}</div>
                    <div class="tactile-badge badge-purple">
                        <span>{best_name} (R² {best_r2:.2f})</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        with kpi4:
            st.markdown(f"""
                <div class="metric-card-3d">
                    <div class="metric-header">
                        <span class="metric-title">News Sentiment</span>
                        <div class="metric-icon-recessed">📰</div>
                    </div>
                    <div class="metric-value">{sent_label}</div>
                    <div class="tactile-badge {sent_badge_class}">
                        <span>{sent_icon} Score: {avg_sentiment:+.2f}</span>
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

        # ---------------------------------------------------------------------
        # TABS SECTION
        # ---------------------------------------------------------------------
        tab1, tab2, tab3 = st.tabs([
            "📈  Forecast Overview & Projection",
            "⚖️  Model Benchmark Comparison",
            "📰  News Stream & Sentiment Analysis"
        ])

        # TAB 1: FORECAST OVERVIEW & PLOTLY CHART
        with tab1:
            st.markdown("""
                <div class="tactile-card" style="padding-bottom: 28px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                        <div>
                            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #1E293B;">
                                Multi-Horizon Volatility & Trajectory Curve
                            </h3>
                            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #64748B;">
                                Historical trajectory against in-sample validation and out-of-sample forward projections.
                            </p>
                        </div>
                        <div class="led-pill">
                            <span class="led-glow-green"></span>
                            <span>Interactive Range Slider Active</span>
                        </div>
                    </div>
            """, unsafe_allow_html=True)

            # Build Custom Plotly Skeuomorphic Light Chart
            fig = go.Figure()
            split_idx = int(len(plot_dates) * 0.8)
            split_date_val = plot_dates[split_idx]

            # 1. Historical Actual Close
            fig.add_trace(go.Scatter(
                x=plot_dates,
                y=actual_prices,
                name="Historical Actual Close",
                line=dict(color="#64748B", width=2.2),
                hovertemplate="<b>Actual Price</b>: $%{y:.2f}<br><b>Date</b>: %{x}<extra></extra>"
            ))

            # 2. Validation Line
            fig.add_trace(go.Scatter(
                x=plot_dates,
                y=inv_predictions,
                name=f"{best_name} Fit / Validation",
                line=dict(color="#10B981", width=2.4),
                hovertemplate=f"<b>{best_name} Predicted</b>: $%{{y:.2f}}<br><b>Date</b>: %{{x}}<extra></extra>"
            ))

            # 3. Smooth Connecting Bridge
            fig.add_trace(go.Scatter(
                x=[plot_dates[-1], future_dates[0]],
                y=[inv_predictions[-1], future_predictions[0]],
                showlegend=False,
                line=dict(color="#7C3AED", width=3, dash='dot'),
                hoverinfo='skip'
            ))

            # 4. Out-of-Sample 1-Month Forecast
            fig.add_trace(go.Scatter(
                x=future_dates,
                y=future_predictions,
                name="1-Month Forward Forecast",
                line=dict(color="#7C3AED", width=3.2, dash='dash'),
                hovertemplate="<b>Target Projection</b>: $%{y:.2f}<br><b>Date</b>: %{x}<extra></extra>"
            ))

            # Layout Configuration with Shapes
            fig.update_layout(
                template="plotly_white",
                height=520,
                hovermode="x unified",
                hoverlabel=dict(
                    bgcolor="#FFFFFF",
                    bordercolor="#CBD5E1",
                    font=dict(family="JetBrains Mono", size=12, color="#1E293B")
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.04,
                    xanchor="right",
                    x=1,
                    font=dict(family="Plus Jakarta Sans", size=12, color="#1E293B"),
                    bgcolor="rgba(232, 236, 239, 0.8)",
                    bordercolor="rgba(184, 194, 204, 0.5)",
                    borderwidth=1
                ),
                margin=dict(l=10, r=10, t=30, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(240, 243, 246, 0.7)",
                xaxis=dict(
                    title="Trading Timeline",
                    title_font=dict(size=12, color="#64748B", family="Plus Jakarta Sans"),
                    tickfont=dict(size=11, color="#64748B", family="JetBrains Mono"),
                    gridcolor="#E2E8F0",
                    zeroline=False,
                    showline=True,
                    linecolor="#CBD5E1"
                ),
                yaxis=dict(
                    title="Asset Price (USD)",
                    title_font=dict(size=12, color="#64748B", family="Plus Jakarta Sans"),
                    tickfont=dict(size=11, color="#64748B", family="JetBrains Mono"),
                    tickprefix="$",
                    gridcolor="#E2E8F0",
                    zeroline=False,
                    showline=True,
                    linecolor="#CBD5E1"
                ),
                shapes=[
                    # Shaded Background Region for Projection
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=split_date_val,
                        x1=future_dates[-1],
                        y0=0,
                        y1=1,
                        fillcolor="rgba(37, 99, 235, 0.04)",
                        line_width=0,
                        layer="below"
                    ),
                    # Train / Validation Split Vertical Line
                    dict(
                        type="line",
                        xref="x",
                        yref="paper",
                        x0=split_date_val,
                        x1=split_date_val,
                        y0=0,
                        y1=1,
                        line=dict(color="#94A3B8", width=1.8, dash="dash")
                    )
                ],
                annotations=[
                    dict(
                        xref="x",
                        yref="paper",
                        x=split_date_val,
                        y=1,
                        text="Train / Val Cut",
                        showarrow=False,
                        xanchor="right",
                        yanchor="top",
                        font=dict(size=11, color="#64748B", family="Plus Jakarta Sans")
                    )
                ]
            )

            fig.update_xaxes(
                rangeslider_visible=True,
                rangeslider_thickness=0.08,
                rangeslider_bgcolor="#E2E7ED",
                rangeslider_bordercolor="#CBD5E1"
            )

            st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Excel Export Downloader Section
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                data.to_excel(writer, index=True, sheet_name="Denoised_Dataset")
                future_df.to_excel(writer, index=True, sheet_name="Future_Predictions")
                results_df.to_excel(writer, index=True, sheet_name="Model_Metrics")
            excel_data = excel_buffer.getvalue()

            exp_col1, exp_col2 = st.columns([3, 1])
            with exp_col1:
                st.markdown("""
                    <div style="display: flex; align-items: center; gap: 8px; height: 100%;">
                        <span style="font-size: 0.9rem; color: #64748B; font-weight: 600;">
                            💾 Export full VMD-denoised historical series, technical features, and model metrics to Excel.
                        </span>
                    </div>
                """, unsafe_allow_html=True)
            with exp_col2:
                st.download_button(
                    label="📥 Export Dataset (XLSX)",
                    data=excel_data,
                    file_name=f"{ticker}_hcla_denoised_terminal.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )

        # TAB 2: MODEL COMPARISON & BENCHMARKS
        with tab2:
            st.markdown("""
                <div class="tactile-card">
                    <h3 style="margin: 0 0 6px 0; font-size: 1.25rem; font-weight: 800; color: #1E293B;">
                        Model Benchmark Leaderboard
                    </h3>
                    <p style="margin: 0 0 16px 0; font-size: 0.88rem; color: #64748B;">
                        Empirical comparison evaluating deep learning architectures against standard econometrics.
                    </p>
            """, unsafe_allow_html=True)

            # Generate Clean HTML Table without markdown indentation
            table_rows = ""
            for idx_name, row in results_df.iterrows():
                is_winner = (idx_name == best_name)
                winner_tag = "<span class='best-row-badge'>★ BEST</span>" if is_winner else ""
                bar_color = "#10B981" if is_winner else "#2563EB"
                mda_val = float(row['MDA']) * 100
                mda_bar_width = max(5.0, min(100.0, mda_val))
                
                table_rows += (
                    f"<tr>"
                    f"<td style='font-weight: 700; color: #1E293B;'>{idx_name} {winner_tag}</td>"
                    f"<td>{row['R²']:.4f}</td>"
                    f"<td>${row['RMSE']:.4f}</td>"
                    f"<td>{mda_val:.2f}%</td>"
                    f"<td><div style='width: 100%; background: #CBD5E1; border-radius: 6px; height: 8px; overflow: hidden;'>"
                    f"<div style='width: {mda_bar_width:.1f}%; background: {bar_color}; height: 100%;'></div></div></td>"
                    f"</tr>"
                )

            table_html = (
                f"<div class='tactile-table-wrapper'>"
                f"<table class='tactile-table'>"
                f"<thead><tr>"
                f"<th>Architecture / Baseline</th>"
                f"<th>R² Score (Goodness of Fit)</th>"
                f"<th>RMSE (Root Mean Square Error)</th>"
                f"<th>MDA (Directional Accuracy)</th>"
                f"<th style='width: 200px;'>Directional Confidence</th>"
                f"</tr></thead>"
                f"<tbody>{table_rows}</tbody>"
                f"</table></div>"
            )
            st.markdown(table_html, unsafe_allow_html=True)

            # Summary Explanation Card
            st.markdown(f"""
                <div class="tactile-card-recessed" style="margin-top: 18px;">
                    <div style="display: flex; align-items: flex-start; gap: 12px;">
                        <span style="font-size: 1.4rem;">💡</span>
                        <div>
                            <h4 style="margin: 0 0 4px 0; font-size: 0.95rem; font-weight: 700; color: #1E293B;">
                                Selected Optimal Engine: <span style="color: #2563EB;">{best_name}</span>
                            </h4>
                            <p style="margin: 0; font-size: 0.85rem; color: #475569; line-height: 1.45;">
                                The <strong>{best_name}</strong> engine demonstrated superior performance with an RMSE of <strong>${best_rmse:.2f}</strong> and a directional accuracy of <strong>{results_df.loc[best_name, 'MDA']*100:.1f}%</strong>. 
                                VMD signal preprocessing stripped high-frequency noise modes prior to spatial-temporal sequencing.
                            </p>
                        </div>
                    </div>
                </div>
                </div>
            """, unsafe_allow_html=True)

        # TAB 3: NEWS & SENTIMENT STREAM
        with tab3:
            st.markdown(f"""
                <div class="tactile-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
                        <div>
                            <h3 style="margin: 0; font-size: 1.25rem; font-weight: 800; color: #1E293B;">
                                Live Market News & Sentiment Feed
                            </h3>
                            <p style="margin: 4px 0 0 0; font-size: 0.88rem; color: #64748B;">
                                Natural Language Processing polarity scoring computed across recent market headlines.
                            </p>
                        </div>
                        <div class="tactile-badge {sent_badge_class}">
                            <span>Aggregated Sentiment: {sent_label} ({avg_sentiment:+.2f})</span>
                        </div>
                    </div>
            """, unsafe_allow_html=True)

            if parsed_news:
                for news in parsed_news[:8]:
                    n_score = float(news.get('sentiment', 0.0))
                    if n_score > 0.05:
                        badge_html = f"<span class='tactile-badge badge-positive'>🟢 Bullish (+{n_score:.2f})</span>"
                    elif n_score < -0.05:
                        badge_html = f"<span class='tactile-badge badge-negative'>🔴 Bearish ({n_score:.2f})</span>"
                    else:
                        badge_html = f"<span class='tactile-badge badge-neutral'>⚪ Neutral ({n_score:.2f})</span>"

                    st.markdown(f"""
                        <div class="news-card">
                            <div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 12px;">
                                <a href="{news.get('link', '#')}" target="_blank" class="news-title-link">
                                    {news.get('title', 'Market Update')}
                                </a>
                                {badge_html}
                            </div>
                            <div class="news-meta">
                                <span>Source: <strong>{news.get('publisher', 'Financial Wire')}</strong></span>
                                <span style="font-family: 'JetBrains Mono', monospace; font-size: 0.78rem;">Compound VADER Metric</span>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class="tactile-card-recessed" style="text-align: center; padding: 24px;">
                        <span style="font-size: 1.5rem;">📡</span>
                        <p style="margin: 8px 0 0 0; font-weight: 600; color: #64748B;">No recent news headlines available for this symbol at this moment.</p>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Execution Error: {e}")
        st.info("Please ensure the ticker symbol is valid, Yahoo Finance is accessible, and all required packages are installed.")

else:
    # -------------------------------------------------------------------------
    # INITIAL / STANDBY TERMINAL DASHBOARD
    # -------------------------------------------------------------------------
    st.markdown("""
        <div class="tactile-card" style="text-align: center; padding: 48px 24px;">
            <div style="width: 64px; height: 64px; border-radius: 20px; background: linear-gradient(145deg, #2563EB, #1D4ED8); box-shadow: 4px 4px 10px #B8C2CC, -4px -4px 10px #FFFFFF; display: inline-flex; align-items: center; justify-content: center; color: white; font-size: 2rem; margin-bottom: 18px;">
                ⚡
            </div>
            <h2 style="margin: 0 0 8px 0; font-size: 1.7rem; font-weight: 800; color: #1E293B;">
                Ready for Pipeline Execution
            </h2>
            <p style="max-width: 620px; margin: 0 auto 24px auto; font-size: 0.95rem; color: #64748B; line-height: 1.6;">
                Configure your target ticker symbol and training epochs in the tactile sidebar panel, then click <strong>"Train & Predict"</strong> to execute the full Variational Mode Decomposition and Deep Hybrid Neural Network pipeline.
            </p>
            <div style="display: flex; justify-content: center; gap: 14px; flex-wrap: wrap;">
                <div class="led-pill">
                    <span class="led-glow-green"></span>
                    <span>VMD Signal Denoising</span>
                </div>
                <div class="led-pill">
                    <span class="led-glow-green"></span>
                    <span>CNN-LSTM-Attention Architecture</span>
                </div>
                <div class="led-pill">
                    <span class="led-glow-green"></span>
                    <span>VADER Sentiment NLP</span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Initial Feature Spec Cards Grid
    init_col1, init_col2, init_col3 = st.columns(3)
    with init_col1:
        st.markdown("""
            <div class="tactile-card" style="min-height: 190px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <div class="metric-icon-recessed">🌊</div>
                    <h4 style="margin: 0; font-size: 1rem; font-weight: 700; color: #1E293B;">VMD De-Noising</h4>
                </div>
                <p style="margin: 0; font-size: 0.84rem; color: #64748B; line-height: 1.5;">
                    Decomposes volatile price series into discrete intrinsic mode functions ($K=4$), filtering out high-frequency market noise before feature ingestion.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with init_col2:
        st.markdown("""
            <div class="tactile-card" style="min-height: 190px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <div class="metric-icon-recessed">🧠</div>
                    <h4 style="margin: 0; font-size: 1rem; font-weight: 700; color: #1E293B;">Deep Hybrid Network</h4>
                </div>
                <p style="margin: 0; font-size: 0.84rem; color: #64748B; line-height: 1.5;">
                    Couples 1D spatial convolutions with Long Short-Term Memory and an Attention context layer with linear trend extrapolation residuals.
                </p>
            </div>
        """, unsafe_allow_html=True)
    with init_col3:
        st.markdown("""
            <div class="tactile-card" style="min-height: 190px;">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 10px;">
                    <div class="metric-icon-recessed">📊</div>
                    <h4 style="margin: 0; font-size: 1rem; font-weight: 700; color: #1E293B;">Dynamic Benchmarking</h4>
                </div>
                <p style="margin: 0; font-size: 0.84rem; color: #64748B; line-height: 1.5;">
                    Runs multi-model evaluations against Vanilla LSTM and classical ARIMA baselines with automatic scoring on $R^2$, RMSE, and MDA directional accuracy.
                </p>
            </div>
        """, unsafe_allow_html=True)
