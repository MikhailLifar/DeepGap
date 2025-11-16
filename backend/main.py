import datetime
from datetime import datetime as dtdt

from fastapi import FastAPI
from utils import get_moex_history
from model import load_model, predict, preprocess_for_tabular  # We'll create this next

app = FastAPI()

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
    """Predict stock price using pretrained model."""
    model = load_model()
    
    today_str = f'{dtdt.now():%Y-%m-%d}'
    today = dtdt.strptime(today_str, r'%Y-%m-%d')
    # next_day = today + datetime.timedelta(days=1)
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

    # print('Next day: ', next_day)
    # print('Data index tail: ', data.index[-5:-1])

    if features is not None:
        prediction = predict(model, data.loc[[data.index[-1], ], features])
    else:
        predicted_value = data['Close_Return'].mean()
        prediction = {"return": predicted_value, "trend": "up" if predicted_value > 0. else "down"}
    prediction['price'] = data.loc[data.index[-2], 'Close'] * (1.0 + 0.01 * prediction['return'])
    return {"prediction": prediction}
