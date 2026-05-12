"""
Flask API for Stock Forecasting Dashboard
Endpoints:
  GET  /api/forecast/<ticker>?horizon=10
  GET  /api/tickers
  GET  /api/stats/<ticker>
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
from forecaster import run_forecast, generate_price_series, engineer_features
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
CORS(app)

SUPPORTED_TICKERS = [
    {"symbol": "RY.TO",   "name": "Royal Bank of Canada",   "sector": "Financials"},
    {"symbol": "SHOP.TO", "name": "Shopify Inc.",            "sector": "Technology"},
    {"symbol": "TD.TO",   "name": "Toronto-Dominion Bank",  "sector": "Financials"},
    {"symbol": "CNR.TO",  "name": "Canadian National Railway", "sector": "Industrials"},
    {"symbol": "ENB.TO",  "name": "Enbridge Inc.",          "sector": "Energy"},
    {"symbol": "BCE.TO",  "name": "BCE Inc.",                "sector": "Telecom"},
]


@app.route("/api/tickers")
def get_tickers():
    return jsonify({"tickers": SUPPORTED_TICKERS})


@app.route("/api/forecast/<ticker>")
def forecast(ticker: str):
    horizon = int(request.args.get("horizon", 10))
    horizon = max(1, min(horizon, 30))  # clamp 1–30
    
    ticker = ticker.upper()
    valid_symbols = [t["symbol"] for t in SUPPORTED_TICKERS]
    if ticker not in valid_symbols:
        return jsonify({"error": f"Ticker {ticker} not supported. Choose from: {valid_symbols}"}), 400
    
    result = run_forecast(ticker, horizon=horizon)
    return jsonify(result)


@app.route("/api/history/<ticker>")
def history(ticker: str):
    ticker = ticker.upper()
    days   = int(request.args.get("days", 252))
    
    df = generate_price_series(ticker, days=days + 100)
    df = engineer_features(df).tail(days)
    
    payload = {
        "ticker": ticker,
        "dates":  pd.to_datetime(df.index).strftime('%Y-%m-%d').tolist(),
        "close":  df['Close'].round(2).tolist(),
        "ma20":   df['MA_20'].round(2).tolist(),
        "ma50":   df['MA_50'].round(2).tolist(),
        "volume": df['Volume'].tolist(),
        "rsi":    df['RSI_14'].round(2).tolist(),
        "bb_upper": df['BB_upper'].round(2).tolist(),
        "bb_lower": df['BB_lower'].round(2).tolist(),
    }
    return jsonify(payload)


@app.route("/api/stats/<ticker>")
def stats(ticker: str):
    ticker = ticker.upper()
    df_raw = generate_price_series(ticker, days=500)
    df     = engineer_features(df_raw)
    
    recent = df.tail(252)
    ann_return = (recent['Close'].iloc[-1] / recent['Close'].iloc[0]) ** (252/len(recent)) - 1
    ann_vol    = recent['Return_1d'].std() * np.sqrt(252)
    sharpe     = ann_return / ann_vol if ann_vol != 0 else 0
    max_dd     = ((recent['Close'] / recent['Close'].cummax()) - 1).min()
    
    return jsonify({
        "ticker":           ticker,
        "last_close":       round(float(df['Close'].iloc[-1]), 2),
        "52w_high":         round(float(df['Close'].tail(252).max()), 2),
        "52w_low":          round(float(df['Close'].tail(252).min()), 2),
        "annualised_return":round(ann_return, 4),
        "annualised_vol":   round(ann_vol, 4),
        "sharpe_ratio":     round(sharpe, 4),
        "max_drawdown":     round(float(max_dd), 4),
        "current_rsi":      round(float(df['RSI_14'].iloc[-1]), 2),
        "current_bb_pct":   round(float(df['BB_pct'].iloc[-1]), 4),
    })


@app.route("/")
def index():
    return jsonify({
        "service": "Stock Forecasting API",
        "version": "1.0.0",
        "endpoints": ["/api/tickers", "/api/forecast/<ticker>", "/api/history/<ticker>", "/api/stats/<ticker>"]
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port)