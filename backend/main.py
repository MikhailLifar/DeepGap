import datetime
import os
import threading
import time
from datetime import datetime as dtdt
from pathlib import Path
from typing import Dict, Optional

from fastapi import FastAPI, HTTPException
import pandas as pd
from utils import get_moex_history
from model import load_model, predict, preprocess_for_tabular, predict_lstm, load_lstm_model, BASE_STOCKS, LSTM_CONFIG  # LSTM prediction function
from data_fetcher import StockDataFetcher, calculate_monthly_returns

app = FastAPI()

# Get absolute paths based on script location
_BACKEND_DIR = Path(__file__).parent.resolve()
_DATA_DIR = _BACKEND_DIR / "data"
_MODELS_DIR = _BACKEND_DIR / "models"

# Cache the LSTM model globally to avoid reloading on every request
_lstm_model_cache = None
_lstm_model_path = str(_MODELS_DIR / "lstm_b30_s12_l2_news.pth")

# Popular tickers to cache (matching frontend list)
POPULAR_TICKERS = ["SBER", "GAZP", "LKOH", "GMKN", "NLMK", "ROSN", "TATN", "MGNT", "OZON", "PLZL"]

# Cache for stock data and predictions
# Structure: {ticker: {"data": df_dict, "pred": prediction_dict, "last_updated": datetime}}
_stock_cache: Dict[str, Dict] = {}
_cache_lock = threading.Lock()
_cache_update_period_hours = 12  # Default: update every 12 hours

# Cache for base stocks data (used by LSTM predictions - loaded once to avoid reloading 30 times)
_base_stocks_cache: Dict[str, pd.DataFrame] = {}
_base_stocks_cache_lock = threading.Lock()
_base_stocks_cache_loaded = False

def get_cached_lstm_model():
    """Get or load the cached LSTM model."""
    global _lstm_model_cache
    if _lstm_model_cache is None:
        _lstm_model_cache = load_lstm_model(_lstm_model_path, device="cpu")
    return _lstm_model_cache


def _load_base_stocks_cache():
    """Load and cache base stocks data (used by LSTM predictions). Only loads once."""
    global _base_stocks_cache, _base_stocks_cache_loaded
    
    with _base_stocks_cache_lock:
        if _base_stocks_cache_loaded:
            return  # Already loaded
        
        print("Loading base stocks cache for LSTM predictions...")
        stock_fetcher = StockDataFetcher(
            get_moex_history_func=get_moex_history,
            board="TQBR",
            sleep_sec=0.5
        )
        
        # Calculate date range (need at least 12 months + buffer)
        from datetime import timedelta
        end_date = dtdt.now()
        start_date = end_date - timedelta(days=(12 + 6) * 30)  # 18 months buffer
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        base_stocks_to_load = BASE_STOCKS[:LSTM_CONFIG['NUM_BASE_STOCKS']]
        base_stocks_data = stock_fetcher.fetch_multiple_stocks(
            tickers=base_stocks_to_load,
            start_date=start_date_str,
            end_date=end_date_str,
            force=False
        )
        
        # Convert to monthly returns and cache
        for ticker, df in base_stocks_data.items():
            if len(df) > 0:
                try:
                    monthly_df = calculate_monthly_returns(df)
                    _base_stocks_cache[ticker] = monthly_df
                except Exception as e:
                    print(f"Warning: Failed to process base stock {ticker}: {e}")
                    _base_stocks_cache[ticker] = pd.DataFrame()
        
        _base_stocks_cache_loaded = True
        print(f"✓ Loaded base stocks cache for {len(_base_stocks_cache)} stocks")


def _load_ticker_data(ticker: str) -> Optional[Dict]:
    """Load data for a ticker from MOEX and return as dict. Period: (2015, current date)."""
    try:
        end = dtdt.now()
        start = datetime.date(2015, 1, 1)  # Start from 2015-01-01
        start_str = f'{start:%Y-%m-%d}'
        end_str = f'{end:%Y-%m-%d}'
        
        data = get_moex_history(
            ticker=ticker,
            start_date=start_str,
            end_date=end_str,
            board="TQBR",
            sleep_sec=0.5,
            force=False,
        )
        
        # Check if data is empty (ticker doesn't exist or has no data)
        if data.empty or len(data) == 0:
            print(f"No data found for ticker {ticker}")
            return None
        
        return data.to_dict()
    except Exception as e:
        print(f"Error loading data for {ticker}: {e}")
        return None


def _predict_lstm_with_cached_base_stocks(ticker: str) -> Optional[Dict]:
    """Predict using LSTM model with cached base stocks data to avoid reloading."""
    from datetime import timedelta
    import torch
    import numpy as np
    from data_fetcher import NewsDataFetcher
    
    try:
        cached_model = get_cached_lstm_model()
        
        # Get cached base stocks data
        with _base_stocks_cache_lock:
            base_stocks_data = _base_stocks_cache.copy()
        
        if not base_stocks_data:
            raise ValueError("Base stocks cache not loaded yet")
        
        # Create stock fetcher for target stock only
        stock_fetcher = StockDataFetcher(
            get_moex_history_func=get_moex_history,
            board="TQBR",
            sleep_sec=0.5
        )
        
        # Calculate date range for target stock
        end_date = dtdt.now()
        start_date = end_date - timedelta(days=(12 + 6) * 30)
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Fetch target stock data
        target_df = stock_fetcher.fetch_stock_prices(
            ticker=ticker,
            start_date=start_date_str,
            end_date=end_date_str,
            force=False
        )
        
        if len(target_df) == 0:
            raise ValueError(f"No data available for target stock {ticker}")
        
        # Convert target to monthly returns
        target_df = calculate_monthly_returns(target_df)
        
        if len(target_df) < 12:
            raise ValueError(f"Insufficient data for target stock {ticker}: got {len(target_df)} months, need 12")
        
        # Clip returns
        target_df['Close_Return'] = np.clip(
            np.array(target_df['Close_Return'].values, dtype=np.float32),
            -20.0, 20.0
        )
        
        target_dates = target_df['Date'].tail(12).values
        target_returns = target_df['Close_Return'].tail(12).values.astype(np.float32)
        
        # Process cached base stocks
        base_stock_returns = []
        base_stocks_list = BASE_STOCKS[:LSTM_CONFIG['NUM_BASE_STOCKS']]
        
        for base_stock in base_stocks_list:
            if base_stock not in base_stocks_data or len(base_stocks_data[base_stock]) == 0:
                base_stock_returns.append(np.zeros(12, dtype=np.float32))
                continue
            
            try:
                base_df = base_stocks_data[base_stock].copy()
                base_df['Close_Return'] = np.clip(
                    np.array(base_df['Close_Return'].values, dtype=np.float32),
                    -20.0, 20.0
                )
                
                # Align dates with target stock
                base_df_indexed = base_df.set_index('Date')
                base_aligned = []
                
                for date in target_dates:
                    date_pd = pd.to_datetime(date)
                    if date_pd in base_df_indexed.index:
                        base_aligned.append(base_df_indexed.loc[date_pd, 'Close_Return'])
                    else:
                        date_diff = np.abs((base_df['Date'] - date_pd).dt.total_seconds())
                        closest_idx = date_diff.idxmin()
                        base_aligned.append(base_df.loc[closest_idx, 'Close_Return'])
                
                base_stock_returns.append(np.array(base_aligned, dtype=np.float32))
            except Exception as e:
                print(f"Error processing cached base stock {base_stock}: {e}, using zeros")
                base_stock_returns.append(np.zeros(12, dtype=np.float32))
        
        # Get news features
        news_features = None
        news_csv_path = str(_DATA_DIR / "ALL_sentiment.csv")
        try:
            news_fetcher = NewsDataFetcher(news_csv_path=news_csv_path)
            news_df = news_fetcher.get_monthly_news_features()
            if len(news_df) > 0:
                news_aligned = []
                news_df_indexed = news_df.set_index('month_start')
                
                for date in target_dates:
                    date_pd = pd.to_datetime(date)
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
                        news_aligned.append([0.0, 0.0, 0.0, 0.0, 0.0])
                
                news_features = np.array(news_aligned, dtype=np.float32)
        except Exception as e:
            print(f"Warning: Error processing news features: {e}, using zeros")
            news_features = np.zeros((12, 5), dtype=np.float32)
        
        # Combine features
        if news_features is not None:
            feature_matrix = np.column_stack(base_stock_returns + [target_returns])
            feature_matrix = np.concatenate([feature_matrix, news_features], axis=1)
        else:
            feature_matrix = np.column_stack(base_stock_returns + [target_returns])
            zero_news = np.zeros((12, 5), dtype=np.float32)
            feature_matrix = np.concatenate([feature_matrix, zero_news], axis=1)
        
        # Convert to tensor and predict
        input_tensor = torch.FloatTensor(feature_matrix).unsqueeze(0)
        
        with torch.no_grad():
            predicted_return = cached_model(input_tensor).item()
        
        # Get last close price
        last_close_price = float(target_df['Close'].iloc[-1])
        predicted_price = last_close_price * (1.0 + predicted_return / 100.0)
        
        return {
            'return': float(predicted_return),
            'trend': 'up' if predicted_return > 0.0 else 'down',
            'price': float(predicted_price)
        }
    except Exception as e:
        print(f"Error in cached LSTM prediction for {ticker}: {e}")
        return None


def _load_ticker_prediction(ticker: str) -> Optional[Dict]:
    """Load prediction for a ticker."""
    try:
        # Try cached version first (uses cached base stocks)
        prediction = _predict_lstm_with_cached_base_stocks(ticker)
        if prediction is not None:
            return {"prediction": prediction}
    except Exception as e:
        print(f"Cached prediction failed for {ticker}: {e}, trying fallback...")
    
    # Fallback to original method
    try:
        cached_model = get_cached_lstm_model()
        prediction = predict_lstm(
            target_stock=ticker,
            get_moex_history_func=get_moex_history,
            model_path=_lstm_model_path,
            news_csv_path=str(_DATA_DIR / "ALL_sentiment.csv"),
            device="cpu",
            model=cached_model
        )
        return {"prediction": prediction}
    except Exception as e:
        print(f"Error loading prediction for {ticker}: {e}")
        # Try fallback to tabular model
        try:
            model = load_model()
            today_str = f'{dtdt.now():%Y-%m-%d}'
            today = dtdt.strptime(today_str, r'%Y-%m-%d')
            month_ago = today - datetime.timedelta(days=30)
            month_ago_str = f'{month_ago:%Y-%m-%d}'
            
            data = get_moex_history(
                ticker=ticker,
                start_date=month_ago_str,
                end_date=today_str,
                board="TQBR",
                sleep_sec=1.,
                force=False,
            )
            
            data, features, _ = preprocess_for_tabular(data)
            
            if features is not None:
                prediction = predict(model, data.loc[[data.index[-1], ], features])
            else:
                predicted_value = data['Close_Return'].mean()
                prediction = {"return": predicted_value, "trend": "up" if predicted_value > 0. else "down"}
            prediction['price'] = data.loc[data.index[-2], 'Close'] * (1.0 + 0.01 * prediction['return'])
            return {"prediction": prediction}
        except Exception as e2:
            print(f"Fallback prediction also failed for {ticker}: {e2}")
            return None


def _update_cache_for_ticker(ticker: str):
    """Update cache for a single ticker."""
    print(f"Updating cache for {ticker}...")
    data = _load_ticker_data(ticker)
    pred = _load_ticker_prediction(ticker)
    
    if data is not None:
        with _cache_lock:
            _stock_cache[ticker] = {
                "data": data,
                "pred": pred,
                "last_updated": dtdt.now()
            }
        print(f"✓ Updated cache for {ticker}")
    else:
        print(f"✗ Failed to update cache for {ticker}")


def _update_all_cache():
    """Update cache for all popular tickers."""
    # Ensure base stocks cache is loaded first
    if not _base_stocks_cache_loaded:
        _load_base_stocks_cache()
    
    print(f"Starting cache update for {len(POPULAR_TICKERS)} tickers...")
    for ticker in POPULAR_TICKERS:
        _update_cache_for_ticker(ticker)
    print(f"Cache update completed. Cached {len(_stock_cache)} tickers.")


def _periodic_cache_update():
    """Background thread that periodically updates the cache."""
    while True:
        time.sleep(_cache_update_period_hours * 3600)  # Convert hours to seconds
        print(f"Starting periodic cache update (every {_cache_update_period_hours} hours)...")
        # Reload base stocks cache periodically
        global _base_stocks_cache_loaded, _base_stocks_cache
        with _base_stocks_cache_lock:
            _base_stocks_cache_loaded = False
            _base_stocks_cache.clear()
        _update_all_cache()


@app.on_event("startup")
async def startup_event():
    """Load cache at startup (non-blocking)."""
    # Load cache in background thread so server starts immediately
    def startup_cache_load():
        print("Loading cache at startup...")
        # First load base stocks cache (needed for predictions)
        _load_base_stocks_cache()
        # Then load popular tickers data and predictions
        _update_all_cache()
        print("Startup cache loading completed.")
    
    # Start background thread for initial cache load
    startup_thread = threading.Thread(target=startup_cache_load, daemon=True)
    startup_thread.start()
    
    # Start background thread for periodic updates
    update_thread = threading.Thread(target=_periodic_cache_update, daemon=True)
    update_thread.start()
    print(f"Started background cache update thread (period: {_cache_update_period_hours} hours)")

@app.get("/fetch_data/{stock_name}")
def fetch_data(stock_name: str):
    """Fetch historical data from MOEX (served from cache if available)."""
    # Check cache first
    with _cache_lock:
        cached = _stock_cache.get(stock_name)
    
    if cached and cached.get("data") is not None:
        # Return from cache
        return cached["data"]
    
    # If not in cache, load it (this should rarely happen after startup)
    print(f"Cache miss for {stock_name}, loading from MOEX...")
    data_dict = _load_ticker_data(stock_name)
    if data_dict is not None:
        # Update cache for future requests
        with _cache_lock:
            if stock_name not in _stock_cache:
                _stock_cache[stock_name] = {}
            _stock_cache[stock_name]["data"] = data_dict
            _stock_cache[stock_name]["last_updated"] = dtdt.now()
        return data_dict
    else:
        # Try fallback: direct fetch in case _load_ticker_data had an issue
        try:
            end = dtdt.now()
            start = datetime.date(2015, 1, 1)  # Start from 2015-01-01
            start_str = f'{start:%Y-%m-%d}'
            end_str = f'{end:%Y-%m-%d}'
            
            data = get_moex_history(
                ticker=stock_name,
                start_date=start_str,
                end_date=end_str,
                board="TQBR",
                sleep_sec=0.5,
                force=False,
            )
            
            # Check if data is empty
            if data.empty or len(data) == 0:
                raise HTTPException(
                    status_code=404,
                    detail=f"Ticker '{stock_name}' not found or has no data available on MOEX"
                )
            
            return data.to_dict()
        except HTTPException:
            raise
        except Exception as e:
            # If MOEX API returns an error (e.g., ticker doesn't exist)
            raise HTTPException(
                status_code=404,
                detail=f"Ticker '{stock_name}' not found or could not be loaded from MOEX: {str(e)}"
            )

@app.post("/predict/{stock_name}")
def predict_price(stock_name: str, days: int = 30):
    """Predict stock price using pretrained LSTM model (served from cache if available)."""
    # Check cache first
    with _cache_lock:
        cached = _stock_cache.get(stock_name)
    
    if cached and cached.get("pred") is not None:
        # Return from cache
        return cached["pred"]
    
    # If not in cache, load it (this should rarely happen after startup)
    print(f"Cache miss for prediction of {stock_name}, generating prediction...")
    pred_dict = _load_ticker_prediction(stock_name)
    if pred_dict is not None:
        # Update cache for future requests
        with _cache_lock:
            if stock_name not in _stock_cache:
                _stock_cache[stock_name] = {}
            _stock_cache[stock_name]["pred"] = pred_dict
            if "last_updated" not in _stock_cache[stock_name]:
                _stock_cache[stock_name]["last_updated"] = dtdt.now()
        return pred_dict
    else:
        # Fallback to original behavior if loading fails
        try:
            cached_model = get_cached_lstm_model()
            prediction = predict_lstm(
                target_stock=stock_name,
                get_moex_history_func=get_moex_history,
                model_path=_lstm_model_path,
                news_csv_path=str(_DATA_DIR / "ALL_sentiment.csv"),
                device="cpu",
                model=cached_model
            )
            return {"prediction": prediction}
        except Exception as e:
            print(f"LSTM prediction failed for {stock_name}: {e}, falling back to tabular model")
            model = load_model()
            
            today_str = f'{dtdt.now():%Y-%m-%d}'
            today = dtdt.strptime(today_str, r'%Y-%m-%d')
            month_ago = today - datetime.timedelta(days=30)
            month_ago_str = f'{month_ago:%Y-%m-%d}'

            data = get_moex_history(
                ticker=stock_name,
                start_date=month_ago_str,
                end_date=today_str,
                board="TQBR",
                sleep_sec=1.,
                force=False,
            )
            
            data, features, _ = preprocess_for_tabular(data)

            if features is not None:
                prediction = predict(model, data.loc[[data.index[-1], ], features])
            else:
                predicted_value = data['Close_Return'].mean()
                prediction = {"return": predicted_value, "trend": "up" if predicted_value > 0. else "down"}
            prediction['price'] = data.loc[data.index[-2], 'Close'] * (1.0 + 0.01 * prediction['return'])
            return {"prediction": prediction}
