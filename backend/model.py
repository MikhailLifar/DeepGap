import datetime as dt
from datetime import datetime as dtdt

import joblib

import numpy as np
import pandas as pd


def preprocess_for_tabular(df: pd.DataFrame, min_records=20):
    if len(df) < min_records:
        df['Close_Return'] = df['Close'].pct_change() * 100
        return df, None, None
    # assert len(df) >= min_records, 'Too small data size for caluclating all the features'

    df = df.copy()
    
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # important - adding a row for the next day
    current_day = df.index[-1].today()
    next_day = current_day + dt.timedelta(days=1)
    df.loc[next_day, :] = -1.

    # Calculate returns for Close, High, and Low
    df['Close_Return'] = df['Close'].pct_change() * 100
    df['High_Return'] = ((df['High'] / df['Close'].shift(1)) - 1) * 100
    df['Low_Return'] = ((df['Low'] / df['Close'].shift(1)) - 1) * 100
    # Drop the first row with NaN values
    df = df.dropna()

    # Create lagged features (1-5 days)
    for lag in range(1, 6):
        df[f'Close_Return_lag_{lag}'] = df['Close_Return'].shift(lag)
        df[f'High_Return_lag_{lag}'] = df['High_Return'].shift(lag)
        df[f'Low_Return_lag_{lag}'] = df['Low_Return'].shift(lag)

    # Create moving averages (excluding current price)
    df['Close_Return_MA_5'] = df['Close_Return'].shift(1).rolling(window=5).mean()
    df['Close_Return_MA_10'] = df['Close_Return'].shift(1).rolling(window=10).mean()

    # Create more sophisticated rolling statistics
    # Standard deviation
    df['Close_Return_Std_5'] = df['Close_Return'].shift(1).rolling(window=5).std()

    # Polynomial fit coefficients (quadratic fit to last 5 days)
    def get_poly_coefficients(window):
        if len(window) < 5:
            return [np.nan, np.nan, np.nan]
        x = np.arange(5)
        y = window.values
        # Fit quadratic polynomial: ax^2 + bx + c
        coeffs = np.polyfit(x, y, 2)
        return coeffs

    # Apply polynomial fit to rolling windows
    poly_coeffs = df['Close_Return'].shift(1).rolling(window=5).apply(
        lambda w: get_poly_coefficients(w)[0], raw=False
    )
    df['Close_Return_Poly_a'] = poly_coeffs

    poly_coeffs = df['Close_Return'].shift(1).rolling(window=5).apply(
        lambda w: get_poly_coefficients(w)[1], raw=False
    )
    df['Close_Return_Poly_b'] = poly_coeffs

    # Trend strength (R-squared of the polynomial fit)
    def get_r_squared(window):
        if len(window) < 5:
            return np.nan
        x = np.arange(5)
        y = window.values
        # Fit quadratic polynomial
        coeffs = np.polyfit(x, y, 2)
        p = np.poly1d(coeffs)
        # Calculate R-squared
        yhat = p(x)
        ybar = np.sum(y) / len(y)
        ssreg = np.sum((yhat - ybar)**2)
        sstot = np.sum((y - ybar)**2)
        return ssreg / sstot if sstot > 0 else 0

    df['Close_Return_Trend_Strength'] = df['Close_Return'].shift(1).rolling(window=5).apply(
        lambda w: get_r_squared(w), raw=False
    )

    # Drop rows with NaN values
    df = df.dropna()

    lag_features = []
    for lag in range(1, 6):
        lag_features.extend([
            f'Close_Return_lag_{lag}', 
            f'High_Return_lag_{lag}', 
            f'Low_Return_lag_{lag}'
        ])

    rolling_features = [
        'Close_Return_MA_5', 'Close_Return_MA_10', 
        'Close_Return_Std_5',
        'Close_Return_Poly_a', 'Close_Return_Poly_b',
        'Close_Return_Trend_Strength'
    ]

    features = lag_features + rolling_features

    # Define target (next day's close return)
    target = 'Close_Return'

    return df, features, target


def load_model():
    """Load your pretrained model."""
    # Replace with your actual model path
    return joblib.load("models/et_regressor_model.joblib")


def predict(model, X: pd.DataFrame):
    """Generate prediction (dummy implementation)."""
    # Replace with actual prediction logic
    predicted_value, = model.predict(X)
    return {"return": predicted_value, "trend": "up" if predicted_value > 0. else "down"}
