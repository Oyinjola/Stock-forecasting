"""
Canadian Stock Forecasting Model

This project looks at a few TSX-listed stocks and uses regression +
a simple autoregressive model to estimate short-term price movement.

Note:
- The price data is simulated for now.
- Later version could connect to yfinance or another market data API.
- Macro values are sample Canadian market indicators.
"""

"""
Stock Price Forecasting Model
Uses regression + ARIMA time series to forecast Canadian/US equities
Outputs predictions and statistical diagnostics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ── Data simulation (replace with yfinance for real data) ─────────────────────
def generate_price_series(ticker: str, days: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic OHLCV data for demo purposes."""
    np.random.seed(seed + hash(ticker) % 100)
    dates = pd.date_range(end=datetime.today(), periods=days, freq='B')
    
    # Geometric Brownian Motion
    mu = 0.0003
    sigma = 0.018
    S0 = np.random.uniform(20, 200)
    
    returns = np.random.normal(mu, sigma, days)
    prices = S0 * np.exp(np.cumsum(returns))
    
    df = pd.DataFrame({
        'Date': dates,
        'Open':  prices * np.random.uniform(0.995, 1.005, days),
        'High':  prices * np.random.uniform(1.001, 1.020, days),
        'Low':   prices * np.random.uniform(0.980, 0.999, days),
        'Close': prices,
        'Volume': np.random.randint(500_000, 5_000_000, days)
    }).set_index('Date')
    
    return df


# ── Feature Engineering ───────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build predictive features: moving averages, RSI, volatility, returns."""
    df = df.copy()
    
    # Returns
    df['Return_1d']  = df['Close'].pct_change(1)
    df['Return_5d']  = df['Close'].pct_change(5)
    df['Return_20d'] = df['Close'].pct_change(20)
    
    # Moving averages
    for w in [5, 10, 20, 50]:
        df[f'MA_{w}'] = df['Close'].rolling(w).mean()
        df[f'MA_{w}_ratio'] = df['Close'] / df[f'MA_{w}']
    
    # Volatility (rolling std of returns)
    df['Vol_10d'] = df['Return_1d'].rolling(10).std() * np.sqrt(252)
    df['Vol_20d'] = df['Return_1d'].rolling(20).std() * np.sqrt(252)
    
    # RSI (14-day)
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI_14'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_mid']   = df['Close'].rolling(20).mean()
    bb_std         = df['Close'].rolling(20).std()
    df['BB_upper'] = df['BB_mid'] + 2 * bb_std
    df['BB_lower'] = df['BB_mid'] - 2 * bb_std
    df['BB_pct']   = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Volume ratio
    df['Vol_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
    
    # Target: next-day return
    df['Target'] = df['Return_1d'].shift(-1)
    
    return df.dropna()


# ── Linear Regression Forecast ────────────────────────────────────────────────
def linear_regression_forecast(X_train, y_train, X_test):
    """Manual OLS regression — no sklearn dependency needed."""
    # Add intercept
    ones = np.ones((X_train.shape[0], 1))
    X_b  = np.hstack([ones, X_train])
    
    # Beta = (X'X)^-1 X'y
    try:
        beta = np.linalg.lstsq(X_b, y_train, rcond=None)[0]
    except np.linalg.LinAlgError:
        beta = np.zeros(X_b.shape[1])
    
    ones_test = np.ones((X_test.shape[0], 1))
    X_test_b  = np.hstack([ones_test, X_test])
    return X_test_b @ beta, beta


def compute_r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0


# ── ARIMA (manual AR(p) component) ───────────────────────────────────────────
def ar_forecast(series: np.ndarray, p: int = 5, horizon: int = 10) -> np.ndarray:
    """Autoregressive model AR(p) fit via OLS."""
    n = len(series)
    X, y = [], []
    for i in range(p, n):
        X.append(series[i-p:i][::-1])
        y.append(series[i])
    
    X, y = np.array(X), np.array(y)
    ones = np.ones((X.shape[0], 1))
    X_b  = np.hstack([ones, X])
    beta = np.linalg.lstsq(X_b, y, rcond=None)[0]
    
    forecast = []
    window = list(series[-p:])
    for _ in range(horizon):
        x_next = np.array([1.0] + window[-p:][::-1])
        pred   = float(x_next @ beta)
        forecast.append(pred)
        window.append(pred)
    
    return np.array(forecast)


# ── Confidence Intervals ──────────────────────────────────────────────────────
def prediction_intervals(residuals: np.ndarray, forecast: np.ndarray,
                          alpha: float = 0.05) -> tuple:
    """Bootstrap 95% prediction intervals."""
    std = np.std(residuals)
    z   = 1.96  # 95% CI
    lower = forecast - z * std
    upper = forecast + z * std
    return lower, upper


# ── Main Pipeline ─────────────────────────────────────────────────────────────
def run_forecast(ticker: str = "RY.TO", horizon: int = 10) -> dict:
    print(f"\n{'='*55}")
    print(f"  Stock Forecast: {ticker}  |  Horizon: {horizon} days")
    print(f"{'='*55}")
    
    # 1. Load data
    df_raw  = generate_price_series(ticker)
    df      = engineer_features(df_raw)
    
    # 2. Feature matrix
    feature_cols = ['Return_5d', 'Return_20d', 'MA_5_ratio', 'MA_20_ratio',
                    'Vol_10d', 'RSI_14', 'BB_pct', 'Vol_ratio']
    
    X = df[feature_cols].values
    y = df['Target'].values
    
    # Train/test split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Standardize
    mu_x, std_x = X_train.mean(axis=0), X_train.std(axis=0) + 1e-9
    X_train_s   = (X_train - mu_x) / std_x
    X_test_s    = (X_test  - mu_x) / std_x
    
    # 3. Linear regression
    y_pred_test, beta = linear_regression_forecast(X_train_s, y_train, X_test_s)
    r2       = compute_r_squared(y_test, y_pred_test)
    residuals = np.asarray(y_test) - np.asarray(y_pred_test)
    rmse     = np.sqrt(np.mean(residuals**2))
    mae      = np.mean(np.abs(residuals))
    
    print(f"\n[Regression Model]")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  MAE  : {mae:.6f}")
    
    # 4. AR(5) price-level forecast
    close_prices = np.asarray(df['Close'].values)
    price_returns = np.asarray(df['Return_1d'].values)
    
    ar_returns  = ar_forecast(price_returns, p=5, horizon=horizon)
    last_price  = close_prices[-1]
    
    # Convert return forecast → price forecast
    price_forecast = [last_price]
    for r in ar_returns:
        price_forecast.append(price_forecast[-1] * (1 + r))
    price_forecast = np.array(price_forecast[1:])
    
    # Prediction intervals
    ar_residuals    = price_returns[5:] - ar_forecast(price_returns, p=5, horizon=len(price_returns)-5)[:len(price_returns)-5]
    lower, upper    = prediction_intervals(ar_residuals * last_price, price_forecast)
    
    forecast_dates = pd.date_range(
        start=df.index[-1] + timedelta(days=1),
        periods=horizon, freq='B'
    )
    
    print(f"\n[AR(5) Price Forecast — next {horizon} trading days]")
    print(f"  Last close : ${last_price:.2f}")
    for i, (d, p, lo, hi) in enumerate(zip(forecast_dates, price_forecast, lower, upper)):
        marker = " ◀ today+1" if i == 0 else ""
        print(f"  {d.strftime('%Y-%m-%d')} : ${p:.2f}  [{lo:.2f}, {hi:.2f}]{marker}")
    
    # 5. Directional accuracy (on test set)
    dir_acc = np.mean(np.sign(y_pred_test) == np.sign(y_test))
    print(f"\n[Directional Accuracy] : {dir_acc:.1%}")
    
    # 6. Summary stats
    print(f"\n[Descriptive Stats — last 252 days]")
    recent = df['Close'].tail(252)
    ann_return = (recent.iloc[-1] / recent.iloc[0]) ** (252/len(recent)) - 1
    ann_vol    = df['Return_1d'].tail(252).std() * np.sqrt(252)
    sharpe     = ann_return / ann_vol if ann_vol != 0 else 0
    print(f"  Annualised return : {ann_return:.1%}")
    print(f"  Annualised vol    : {ann_vol:.1%}")
    print(f"  Sharpe ratio      : {sharpe:.2f}")
    
    return {
        "ticker": ticker,
        "last_price": round(last_price, 2),
        "forecast_prices": price_forecast.tolist(),
        "forecast_dates": [d.strftime('%Y-%m-%d') for d in forecast_dates],
        "lower_ci": lower.tolist(),
        "upper_ci": upper.tolist(),
        "r2": round(r2, 4),
        "rmse": round(rmse, 6),
        "directional_accuracy": round(dir_acc, 4),
        "annualised_return": round(ann_return, 4),
        "annualised_vol": round(ann_vol, 4),
        "sharpe": round(sharpe, 4),
    }


if __name__ == "__main__":
    tickers = ["RY.TO", "SHOP.TO", "TD.TO"]
    for t in tickers:
        result = run_forecast(t, horizon=10)