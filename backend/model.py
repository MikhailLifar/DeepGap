import datetime as dt
from datetime import datetime as dtdt
import os
from typing import List, Tuple, Optional

import joblib
import torch
import torch.nn as nn

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


# ============================================================================
# LSTM Model for Multi-Stock Prediction
# ============================================================================

class LSTMModel(nn.Module):
    """
    LSTM model for time series forecasting.
    Modular architecture allows easy extension (e.g., attention, multiple outputs).
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float = 0.2,
        bidirectional: bool = False
    ):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        # Calculate LSTM output size
        lstm_output_size = hidden_size * 2 if bidirectional else hidden_size

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, 1)  # Single output for regression

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            output: Tensor of shape (batch_size,)
        """
        # LSTM forward pass
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Use the last output (many-to-one architecture)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_size)

        # Fully connected layers
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out.squeeze(-1)  # (batch_size,)


# Configuration matching new training script (lstm_b30_s12_l2_news)
LSTM_CONFIG = {
    'SEQUENCE_LENGTH': 12,  # Monthly data, 12 months
    'HIDDEN_SIZE': 128,
    'NUM_LAYERS': 2,
    'DROPOUT': 0.2,
    'BIDIRECTIONAL': False,
    'NUM_BASE_STOCKS': 30,
    'NUM_NEWS_FEATURES': 5,  # news_count, avg_title_len, avg_text_len, text_count, news_mask
    'NUM_FEATURES': 36,  # 30 base stocks + 1 target stock + 5 news features
    'MONTHLY': True,  # Use monthly aggregated data
    'RETURN_CLIP': 20.0,  # Clip returns to [-20, 20]
}

# Base stocks - starting with top 10, extended to 30 based on training data
BASE_STOCKS = [
    'SBER',  # Sberbank - Banking
    'VTBR',  # VTB Bank - Banking
    'GAZP',  # Gazprom - Energy/Oil & Gas
    'LKOH',  # Lukoil - Energy/Oil & Gas
    'MGNT',  # Magnit - Retail
    'X5',    # X5 Group - Retail
    'YNDX',  # Yandex - Technology/Internet
    'MTSS',  # MTS - Telecommunications
    'NLMK',  # NLMK - Steel/Metals
    'PLZL',  # Polymetal - Gold/Mining
    'ROSN',  # Rosneft - Energy/Oil & Gas
    'TATN',  # Tatneft - Energy/Oil & Gas
    'GMKN',  # Nornickel - Mining
    'SNGS',  # Surgutneftegas - Energy/Oil & Gas
    'OZON',  # Ozon - E-commerce
    'FIVE',  # X5 Retail Group alternative
    'IRKT',  # Irkut - Aerospace
    'PIKK',  # PIK - Real Estate
    'AQUA',  # Aqua - Consumer
    'CBOM',  # Credit Bank of Moscow - Banking
    'POSI',  # Positive Technologies - IT
    'RUSI',  # RussNeft - Energy
    'ALRS',  # Alrosa - Mining
    'MOEX',  # Moscow Exchange - Financial
    'AFKS',  # Sistema - Conglomerate
    'RTKM',  # Rostelecom - Telecom
    'CHMF',  # Severstal - Steel
    'MAGN',  # MMK - Steel
    'POLY',  # Polymetal alternative
    'HYDR',  # RusHydro - Energy
]


def load_lstm_model(model_path: str = "models/lstm_b30_s12_l2_news.pth", device: str = "cpu"):
    """
    Load the trained LSTM model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint file
        device: Device to load the model on ('cpu' or 'cuda')
    
    Returns:
        Loaded LSTM model in evaluation mode
    """
    device_obj = torch.device(device)
    
    # Initialize model with same architecture as training
    model = LSTMModel(
        input_size=LSTM_CONFIG['NUM_FEATURES'],
        hidden_size=LSTM_CONFIG['HIDDEN_SIZE'],
        num_layers=LSTM_CONFIG['NUM_LAYERS'],
        dropout=LSTM_CONFIG['DROPOUT'],
        bidirectional=LSTM_CONFIG['BIDIRECTIONAL']
    )
    
    # Load checkpoint - handle both .pth (state_dict) and .pt (full checkpoint) formats
    checkpoint = torch.load(model_path, map_location=device_obj)
    
    # Check if it's a full checkpoint or just state_dict
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Full checkpoint format
        model.load_state_dict(checkpoint['model_state_dict'])
    elif isinstance(checkpoint, dict) and any(k.startswith(('lstm.', 'fc1.', 'fc2.')) for k in checkpoint.keys()):
        # It's a state_dict (keys are layer names)
        model.load_state_dict(checkpoint)
    else:
        # Try loading as state_dict
        try:
            model.load_state_dict(checkpoint)
        except Exception as e:
            raise ValueError(f"Could not load model from {model_path}: {e}")
    
    model.to(device_obj)
    model.eval()
    
    return model


def prepare_lstm_input(
    target_stock: str,
    base_stocks: List[str],
    stock_fetcher,
    news_fetcher=None,
    sequence_length: int = 12,
    monthly: bool = True,
    return_clip_range: Tuple[float, float] = (-20.0, 20.0)
) -> Tuple[torch.Tensor, pd.DataFrame]:
    """
    Prepare input data for LSTM prediction with monthly data and news features.
    
    Fetches historical data for base stocks and target stock, converts to monthly returns,
    adds news features, and prepares the input tensor in the same format as training.
    
    Args:
        target_stock: Ticker symbol of the stock to predict
        base_stocks: List of base stock tickers
        stock_fetcher: StockDataFetcher instance
        news_fetcher: NewsDataFetcher instance (optional)
        sequence_length: Number of months to look back (default: 12)
        monthly: Whether to use monthly aggregated data (default: True)
        return_clip_range: Range to clip returns (default: (-20.0, 20.0))
    
    Returns:
        input_tensor: Tensor of shape (1, sequence_length, num_features) ready for model
        target_df: DataFrame with the target stock monthly data (for getting last close price)
    
    Raises:
        ValueError: If insufficient data is available
    """
    from datetime import datetime, timedelta
    from data_fetcher import calculate_monthly_returns
    
    # Calculate date range (need at least sequence_length months)
    end_date = datetime.now()
    # Add extra buffer: sequence_length months + 6 months for safety
    start_date = end_date - timedelta(days=(sequence_length + 6) * 30)
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')
    
    # Fetch data for target stock
    target_df = stock_fetcher.fetch_stock_prices(
        ticker=target_stock,
        start_date=start_date_str,
        end_date=end_date_str,
        force=False
    )
    
    if len(target_df) == 0:
        raise ValueError(f"No data available for target stock {target_stock}")
    
    # Convert to monthly returns
    if monthly:
        target_df = calculate_monthly_returns(target_df)
    else:
        target_df = target_df.copy()
        target_df['Date'] = pd.to_datetime(target_df['Date'])
        target_df = target_df.sort_values('Date').reset_index(drop=True)
        target_df['Close_Return'] = target_df['Close'].pct_change() * 100
        target_df = target_df.dropna().reset_index(drop=True)
    
    if len(target_df) < sequence_length:
        raise ValueError(
            f"Insufficient data for target stock {target_stock}: "
            f"got {len(target_df)} months, need {sequence_length}"
        )
    
    # Clip returns
    target_df['Close_Return'] = np.clip(
        np.array(target_df['Close_Return'].values, dtype=np.float32),
        return_clip_range[0],
        return_clip_range[1]
    )
    
    # Get target dates for alignment
    target_dates = target_df['Date'].tail(sequence_length).values
    
    # Fetch data for base stocks
    base_stock_returns = []
    base_stocks_data = stock_fetcher.fetch_multiple_stocks(
        tickers=base_stocks,
        start_date=start_date_str,
        end_date=end_date_str,
        force=False
    )
    
    for base_stock in base_stocks:
        if base_stock not in base_stocks_data or len(base_stocks_data[base_stock]) == 0:
            print(f"Warning: No data for base stock {base_stock}, using zeros")
            base_stock_returns.append(np.zeros(sequence_length, dtype=np.float32))
            continue
        
        try:
            base_df = base_stocks_data[base_stock]
            
            # Convert to monthly returns
            if monthly:
                base_df = calculate_monthly_returns(base_df)
            else:
                base_df = base_df.copy()
                base_df['Date'] = pd.to_datetime(base_df['Date'])
                base_df = base_df.sort_values('Date').reset_index(drop=True)
                base_df['Close_Return'] = base_df['Close'].pct_change() * 100
                base_df = base_df.dropna().reset_index(drop=True)
            
            if len(base_df) < sequence_length:
                print(f"Warning: Insufficient data for base stock {base_stock}, using zeros")
                base_stock_returns.append(np.zeros(sequence_length, dtype=np.float32))
                continue
            
            # Clip returns
            base_df['Close_Return'] = np.clip(
                np.array(base_df['Close_Return'].values, dtype=np.float32),
                return_clip_range[0],
                return_clip_range[1]
            )
            
            # Align dates with target stock
            base_df_indexed = base_df.set_index('Date')
            base_aligned = []
            
            for date in target_dates:
                date_pd = pd.to_datetime(date)
                # Try exact match first
                if date_pd in base_df_indexed.index:
                    base_aligned.append(base_df_indexed.loc[date_pd, 'Close_Return'])
                else:
                    # Find closest date if exact match not found
                    date_diff = np.abs((base_df['Date'] - date_pd).dt.total_seconds())
                    closest_idx = date_diff.idxmin()
                    base_aligned.append(base_df.loc[closest_idx, 'Close_Return'])
            
            base_stock_returns.append(np.array(base_aligned, dtype=np.float32))
            
        except Exception as e:
            print(f"Error processing base stock {base_stock}: {e}, using zeros")
            base_stock_returns.append(np.zeros(sequence_length, dtype=np.float32))
    
    # Get target stock returns for the last sequence_length months
    target_returns = target_df['Close_Return'].tail(sequence_length).values.astype(np.float32)
    
    # Prepare news features if available
    news_features = None
    if news_fetcher is not None:
        try:
            news_df = news_fetcher.get_monthly_news_features()
            if len(news_df) > 0:
                # Align news features with target dates
                news_aligned = []
                news_df_indexed = news_df.set_index('month_start')
                
                for date in target_dates:
                    date_pd = pd.to_datetime(date)
                    # Find the month start for this date
                    month_start = date_pd.replace(day=1)
                    
                    if month_start in news_df_indexed.index:
                        row = news_df_indexed.loc[month_start]
                        news_aligned.append([
                            row.get('news_count', 0.0),
                            row.get('avg_title_len', 0.0),
                            row.get('avg_text_len', 0.0),
                            row.get('text_count', 0.0),
                            row.get('news_mask', 0.0),
                        ])
                    else:
                        # No news for this month, use zeros
                        news_aligned.append([0.0, 0.0, 0.0, 0.0, 0.0])
                
                news_features = np.array(news_aligned, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Error processing news features: {e}, using zeros")
            news_features = np.zeros((sequence_length, 5), dtype=np.float32)
    
    # Combine all features
    if news_features is not None:
        # Shape: (sequence_length, num_base_stocks + 1 + 5)
        feature_matrix = np.column_stack(base_stock_returns + [target_returns])
        feature_matrix = np.concatenate([feature_matrix, news_features], axis=1)
    else:
        # Shape: (sequence_length, num_base_stocks + 1)
        feature_matrix = np.column_stack(base_stock_returns + [target_returns])
        # Add zero news features if news not available
        zero_news = np.zeros((sequence_length, 5), dtype=np.float32)
        feature_matrix = np.concatenate([feature_matrix, zero_news], axis=1)
    
    # Convert to tensor: (1, sequence_length, num_features)
    input_tensor = torch.FloatTensor(feature_matrix).unsqueeze(0)
    
    return input_tensor, target_df


def predict_lstm(
    target_stock: str,
    get_moex_history_func,
    model_path: str = "models/lstm_b30_s12_l2_news.pth",
    news_csv_path: Optional[str] = "data/ALL_sentiment.csv",
    device: str = "cpu",
    model=None  # Optional: pass pre-loaded model to avoid reloading
) -> dict:
    """
    Predict next month's close return for a target stock using the trained LSTM model.
    
    This function fetches fresh data for:
    - Target stock prices
    - Base stock prices (30 stocks)
    - News sentiment features (monthly aggregated)
    
    Args:
        target_stock: Ticker symbol of the stock to predict
        get_moex_history_func: Function to fetch MOEX data (from utils.get_moex_history)
        model_path: Path to the trained model checkpoint
        news_csv_path: Path to news sentiment CSV file
        device: Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        Dictionary with prediction results:
        {
            'return': predicted_close_return (float),
            'trend': 'up' or 'down' (str),
            'price': predicted_close_price (float)
        }
    """
    from data_fetcher import StockDataFetcher, NewsDataFetcher
    
    # Initialize data fetchers
    stock_fetcher = StockDataFetcher(
        get_moex_history_func=get_moex_history_func,
        board="TQBR",
        sleep_sec=0.5
    )
    
    news_fetcher = None
    if news_csv_path:
        news_fetcher = NewsDataFetcher(news_csv_path=news_csv_path)
    
    # Load model if not provided
    if model is None:
        model = load_lstm_model(model_path, device)
    
    # Prepare input data
    input_tensor, target_df = prepare_lstm_input(
        target_stock=target_stock,
        base_stocks=BASE_STOCKS[:LSTM_CONFIG['NUM_BASE_STOCKS']],
        stock_fetcher=stock_fetcher,
        news_fetcher=news_fetcher,
        sequence_length=LSTM_CONFIG['SEQUENCE_LENGTH'],
        monthly=LSTM_CONFIG['MONTHLY'],
        return_clip_range=(-LSTM_CONFIG['RETURN_CLIP'], LSTM_CONFIG['RETURN_CLIP'])
    )
    
    # Move tensor to device
    device_obj = torch.device(device)
    input_tensor = input_tensor.to(device_obj)
    
    # Make prediction
    with torch.no_grad():
        predicted_return = model(input_tensor).item()
    
    # Get last close price to calculate predicted price
    last_close_price = float(target_df['Close'].iloc[-1])
    
    # Convert return to price
    # Return is in percentage (e.g., 1.5 means 1.5%), so divide by 100
    predicted_price = last_close_price * (1.0 + predicted_return / 100.0)
    
    return {
        'return': float(predicted_return),
        'trend': 'up' if predicted_return > 0.0 else 'down',
        'price': float(predicted_price)
    }
