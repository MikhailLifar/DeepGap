from fastapi import FastAPI
from utils import get_moex_history
from model import load_model, predict  # We'll create this next

app = FastAPI()

@app.get("/fetch_data/{stock_name}")
def fetch_data(stock_name: str, start: str = "2020-01-01", end: str = "2023-01-01"):
    """Fetch historical data from MOEX."""
    data = get_moex_history(
                ticker=stock_name,
                start_date=start,
                end_date=end,
                board="TQBR",
                sleep_sec=1.,
                force=False,
                min_rows_cache=20,
                max_age_days_cache=1,
            )
    return data.to_dict()

@app.post("/predict/{stock_name}")
def predict_price(stock_name: str, days: int = 30):
    """Predict stock price using pretrained model."""
    model = load_model()
    prediction = predict(model, stock_name, days)
    return {"prediction": prediction}
