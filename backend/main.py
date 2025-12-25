import datetime
import os
from datetime import datetime as dtdt
from pathlib import Path

from fastapi import FastAPI
from utils import get_moex_history
from model import load_model, predict, preprocess_for_tabular, predict_lstm, load_lstm_model  # LSTM prediction function

app = FastAPI()

# Get absolute paths based on script location
_BACKEND_DIR = Path(__file__).parent.resolve()
_DATA_DIR = _BACKEND_DIR / "data"
_MODELS_DIR = _BACKEND_DIR / "models"

# Cache the LSTM model globally to avoid reloading on every request
_lstm_model_cache = None
_lstm_model_path = str(_MODELS_DIR / "lstm_b30_s12_l2_news.pth")

def get_cached_lstm_model():
    """Get or load the cached LSTM model."""
    global _lstm_model_cache
    if _lstm_model_cache is None:
        _lstm_model_cache = load_lstm_model(_lstm_model_path, device="cpu")
    return _lstm_model_cache

@app.get("/fetch_data/{stock_name}")
def fetch_data(stock_name: str):
    """Fetch historical data from MOEX."""
    end = dtdt.now()
    today = end.today()
    start = datetime.date(today.year - 10, today.month, today.day)
    start = f'{start:%Y-%m-%d}'
    end = f'{end:%Y-%m-%d}'

    data = get_moex_history(
                ticker=stock_name,
                start_date=start,
                end_date=end,
                board="TQBR",
                sleep_sec=0.5,
                force=False,
            )
    return data.to_dict()

@app.post("/predict/{stock_name}")
def predict_price(stock_name: str, days: int = 30):
    """Predict stock price using pretrained LSTM model."""
    try:
        # Use cached LSTM model for faster predictions
        cached_model = get_cached_lstm_model()
        # Use LSTM model for prediction
        prediction = predict_lstm(
            target_stock=stock_name,
            get_moex_history_func=get_moex_history,
            model_path=_lstm_model_path,
            news_csv_path=str(_DATA_DIR / "ALL_sentiment.csv"),
            device="cpu",
            model=cached_model  # Use cached model
        )
        return {"prediction": prediction}
    except Exception as e:
        # Fallback to old model if LSTM prediction fails
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
