import joblib
import pandas as pd

def load_model():
    """Load your pretrained model."""
    # Replace with your actual model path
    return joblib.load("models/et_regressor_model.joblib")

def predict(model, stock_name: str, days: int):
    """Generate prediction (dummy implementation)."""
    # Replace with actual prediction logic
    return {"price": 150.0, "trend": "up"}
