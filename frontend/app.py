import dash
from dash import dcc, html, Input, Output, no_update
from dash import dash_table
import plotly.graph_objects as go
import requests
from datetime import datetime

MATRIX_GREEN = "#39FF14"
GOLD = "#d4af37"
DARK_BG = "#0b0f0c"
RED = "#ff4d4d"
CARBON_BG = "#0e1116"
PANEL_BG = "#0a0e12"
GRID_LINE = "#132318"

app = dash.Dash(__name__)

# Company display names and popular tickers
TICKER_TO_NAME = {
    "SBER": "Сбербанк",
    "GAZP": "Газпром",
    "LKOH": "Лукойл",
    "GMKN": "Норникель",
    "NLMK": "НЛМК",
    "ROSN": "Роснефть",
    "TATN": "Татнефть",
    "MGNT": "Магнит",
    "OZON": "Ozon",
    "YNDX": "Яндекс",
    "PLZL": "Полюс",
}
POPULAR_TICKERS = ["SBER", "GAZP", "LKOH", "GMKN", "NLMK", "ROSN", "TATN", "MGNT", "OZON", "PLZL"]

# Simple cache to store preloaded data and predictions at server start
STOCK_CACHE = {}  # {ticker: {"data": df_like_dict, "pred": prediction_json}}

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "backgroundColor": DARK_BG,
        "backgroundImage": (
            "radial-gradient(1200px 600px at 20% -10%, rgba(212,175,55,0.08), transparent 60%),"
            "radial-gradient(1000px 500px at 110% 10%, rgba(57,255,20,0.06), transparent 60%)"
        ),
        "backgroundAttachment": "fixed",
        "color": MATRIX_GREEN,
        "fontFamily": "Verdana, Geneva, Tahoma, sans-serif",
        "padding": "0",
        "margin": "0",
        "display": "flex",
        "flexDirection": "row",
    },
    children=[
        # LEFT: Candlestick chart (fills height)
        html.Div(
            style={
                "flex": "6 1 0%",
                "padding": "14px 12px 14px 14px",
                "borderRight": f"1px solid rgba(212,175,55,0.2)",
                "boxSizing": "border-box",
                "backgroundColor": "transparent",
            },
            children=[
                html.H2(
                    "Time Series (Japanese Candles)",
                    style={
                        "color": GOLD,
                        "margin": "8px 0 12px 0",
                        "letterSpacing": "0.5px",
                        "fontWeight": "600",
                    },
                ),
                dcc.Graph(
                    id="candles-graph",
                    style={
                        "height": "calc(100vh - 88px)",
                        "backgroundColor": PANEL_BG,
                        "borderRadius": "10px",
                        "boxShadow": (
                            "inset 0 0 0 1px rgba(212,175,55,0.16), "
                            "0 0 24px rgba(212,175,55,0.06)"
                        ),
                    },
                    config={"displayModeBar": False},
                ),
            ],
        ),
        # RIGHT: Top (25%) dropdown, Bottom (75%) table
        html.Div(
            style={
                "flex": "4 1 0%",
                "display": "flex",
                "flexDirection": "column",
                "boxSizing": "border-box",
                "padding": "14px 14px 14px 12px",
            },
            children=[
                html.Div(
                    style={
                        "height": "25vh",
                        "padding": "14px",
                        "boxSizing": "border-box",
                        "borderBottom": f"1px solid rgba(212,175,55,0.2)",
                        "display": "flex",
                        "flexDirection": "column",
                        "gap": "12px",
                        "justifyContent": "flex-start",
                        "backgroundColor": PANEL_BG,
                        "borderRadius": "10px",
                        "boxShadow": (
                            "inset 0 0 0 1px rgba(212,175,55,0.16), "
                            "0 0 20px rgba(57,255,20,0.05)"
                        ),
                    },
                    children=[
                        html.H2(
                            "Select Ticker",
                            style={
                                "color": GOLD,
                                "margin": "4px 0 0 0",
                                "letterSpacing": "0.4px",
                                "fontWeight": "600",
                            },
                        ),
                        dcc.Dropdown(
                            id="stock-dropdown",
                            options=[{"label": f"{TICKER_TO_NAME.get(t, t)} ({t})", "value": t} for t in POPULAR_TICKERS],
                            value="SBER",
                            style={
                                "backgroundColor": CARBON_BG,
                                "color": MATRIX_GREEN,
                                "border": f"1px solid rgba(212,175,55,0.3)",
                                "borderRadius": "10px",
                                "height": "44px",
                                "boxShadow": "0 0 0 0 rgba(212,175,55,0)",
                            },
                        ),
                        html.Div(
                            id="prediction-output",
                            style={
                                "marginTop": "12px",
                                "color": MATRIX_GREEN,
                                "fontSize": "16px",
                                "opacity": 0.9,
                            },
                        ),
                    ],
                ),
                html.Div(
                    style={
                        "height": "calc(100vh - 25vh)",
                        "padding": "14px",
                        "boxSizing": "border-box",
                        "backgroundColor": PANEL_BG,
                        "borderRadius": "10px",
                        "boxShadow": (
                            "inset 0 0 0 1px rgba(212,175,55,0.16), "
                            "0 0 20px rgba(57,255,20,0.05)"
                        ),
                    },
                    children=[
                        html.H2(
                            "Today & Forecast",
                            style={
                                "color": GOLD,
                                "margin": "0 0 10px 0",
                                "letterSpacing": "0.4px",
                                "fontWeight": "600",
                            },
                        ),
                        dash_table.DataTable(
                            id="metrics-table",
                            columns=[
                                {"name": "Ticker", "id": "ticker"},
                                {"name": "Today Price", "id": "today_price", "type": "numeric", "format": {"specifier": ".2f"}},
                                {"name": "Today Return", "id": "today_return", "type": "numeric", "format": {"specifier": ".2%"}},
                                {"name": "Forecasted Price", "id": "forecast_price", "type": "numeric", "format": {"specifier": ".2f"}},
                                {"name": "Forecasted Return", "id": "forecast_return", "type": "numeric", "format": {"specifier": ".2%"}},
                            ],
                            data=[],
                            style_table={
                                "overflowX": "auto",
                                "backgroundColor": "transparent",
                                "borderRadius": "8px",
                            },
                            style_header={
                                "backgroundColor": CARBON_BG,
                                "color": GOLD,
                                "border": f"1px solid rgba(212,175,55,0.2)",
                                "borderBottom": f"1px solid rgba(212,175,55,0.3)",
                                "fontWeight": "600",
                            },
                            style_cell={
                                "backgroundColor": CARBON_BG,
                                "color": MATRIX_GREEN,
                                "border": f"1px solid rgba(212,175,55,0.08)",
                                "textAlign": "center",
                                "fontFamily": "Verdana, Geneva, Tahoma, sans-serif",
                                "fontSize": "14px",
                                "padding": "10px 8px",
                                "whiteSpace": "nowrap",
                            },
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "#0d1218",
                                },
                                {
                                    "if": {"state": "active"},
                                    "backgroundColor": "#101826",
                                    "border": f"1px solid rgba(212,175,55,0.2)",
                                },
                                {
                                    "if": {"filter_query": "{today_return} > 0", "column_id": "today_return"},
                                    "color": MATRIX_GREEN,
                                },
                                {
                                    "if": {"filter_query": "{today_return} <= 0", "column_id": "today_return"},
                                    "color": RED,
                                },
                                {
                                    "if": {"filter_query": "{forecast_return} > 0", "column_id": "forecast_return"},
                                    "color": MATRIX_GREEN,
                                },
                                {
                                    "if": {"filter_query": "{forecast_return} <= 0", "column_id": "forecast_return"},
                                    "color": RED,
                                },
                            ],
                            style_as_list_view=True,
                        ),
                    ],
                ),
            ],
        ),
    ],
)


def _build_candlestick_figure(df_like, stock_name: str):
    # Try to extract date axis
    date_key = None
    for dk in ("Date", "DATE", "date"):
        if dk in df_like:
            date_key = dk
            break

    # Try common OHLC keys
    keys_map_options = [
        ("Open", "High", "Low", "Close"),
        ("OPEN", "HIGH", "LOW", "CLOSE"),
        ("open", "high", "low", "close"),
    ]
    x_vals = None
    o_key = h_key = l_key = c_key = None
    for o, h, l, c in keys_map_options:
        if all(k in df_like for k in (o, h, l, c)):
            o_key, h_key, l_key, c_key = o, h, l, c
            break
    # If OHLC missing, fallback to line Close
    if o_key is None:
        # Try to use any 'Close' variant for a line chart
        for c_try in ("Close", "CLOSE", "close"):
            if c_try in df_like:
                # Align to dates if available
                if date_key:
                    order_keys = list(df_like[date_key].keys())
                    x_vals = [ _safe_parse_date(df_like[date_key][k]) for k in order_keys ]
                    y_vals = [ df_like[c_try].get(k) for k in order_keys ]
                else:
                    x_vals = list(df_like[c_try].keys())
                    y_vals = list(df_like[c_try].values())
                fig = go.Figure(
                    data=[go.Scatter(x=x_vals, y=y_vals, mode="lines", line=dict(color=GOLD, width=2))]
                )
                fig.update_layout(
                    title=f"{stock_name} Prices",
                    paper_bgcolor=DARK_BG,
                    plot_bgcolor=DARK_BG,
                    font=dict(color=MATRIX_GREEN),
                    xaxis=dict(gridcolor="#124d12", zerolinecolor="#124d12"),
                    yaxis=dict(gridcolor="#124d12", zerolinecolor="#124d12"),
                )
                return fig
        # No usable data
        fig = go.Figure()
        fig.update_layout(
            title=f"{stock_name} (no price data)",
            paper_bgcolor=DARK_BG,
            plot_bgcolor=DARK_BG,
            font=dict(color=MATRIX_GREEN),
        )
        return fig

    # Build candlestick figure
    if date_key:
        order_keys = list(df_like[date_key].keys())
        x_vals = [ _safe_parse_date(df_like[date_key][k]) for k in order_keys ]
        opens = [ df_like[o_key].get(k) for k in order_keys ]
        highs = [ df_like[h_key].get(k) for k in order_keys ]
        lows = [ df_like[l_key].get(k) for k in order_keys ]
        closes = [ df_like[c_key].get(k) for k in order_keys ]
    else:
        # Fallback to whatever order in OHLC dicts (likely numeric indices)
        x_vals = list(df_like[o_key].keys())
        opens = list(df_like[o_key].values())
        highs = list(df_like[h_key].values())
        lows = list(df_like[l_key].values())
        closes = list(df_like[c_key].values())
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x_vals if date_key else [_safe_parse_date(k) for k in x_vals],
                open=opens,
                high=highs,
                low=lows,
                close=closes,
                increasing_line_color=GOLD,
                increasing_fillcolor=GOLD,
                decreasing_line_color=RED,
                decreasing_fillcolor=RED,
            )
        ]
    )
    fig.update_layout(
        title=f"{stock_name} Candles",
        paper_bgcolor=DARK_BG,
        plot_bgcolor=DARK_BG,
        font=dict(color=MATRIX_GREEN),
        xaxis=dict(gridcolor="#124d12", zerolinecolor="#124d12"),
        yaxis=dict(gridcolor="#124d12", zerolinecolor="#124d12"),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    return fig


def _safe_parse_date(s):
    try:
        return datetime.fromisoformat(s)
    except Exception:
        return s


def _compute_returns(close_series_dict, forecast_price):
    # close_series_dict is an ordered mapping (date -> price)
    if not close_series_dict or len(close_series_dict) < 2:
        return None
    keys = list(close_series_dict.keys())
    last = float(close_series_dict[keys[-1]])
    prev = float(close_series_dict[keys[-2]])
    today_ret = (last / prev - 1.0) if prev != 0 else 0.0
    forecast_ret = (float(forecast_price) / last - 1.0) if last != 0 else 0.0
    return last, today_ret, forecast_ret


def _build_table_rows_from_cache():
    rows = []
    for t in POPULAR_TICKERS:
        cached = STOCK_CACHE.get(t) or {}
        df_like = cached.get("data") or {}
        pred = cached.get("pred") or {}
        # extract forecast price
        pred_price = None
        if isinstance(pred, dict):
            pred_price = (pred.get("prediction") or {}).get("price")
        # extract close series
        close_key = None
        for ck in ("Close", "CLOSE", "close"):
            if ck in df_like:
                close_key = ck
                break
        if close_key and pred_price is not None:
            metrics = _compute_returns(df_like[close_key], pred_price)
            if metrics:
                today_price, today_return, forecast_return = metrics
                rows.append({
                    "ticker": f"{TICKER_TO_NAME.get(t, t)} ({t})",
                    "today_price": float(today_price),
                    "today_return": float(today_return),
                    "forecast_price": float(pred_price),
                    "forecast_return": float(forecast_return),
                })
    return rows


@app.callback(
    [
        Output("candles-graph", "figure"),
        Output("prediction-output", "children"),
        Output("metrics-table", "data"),
    ],
    [Input("stock-dropdown", "value")],
)
def update_views(stock_name):
    # Fetch data and prediction
    table_data = _build_table_rows_from_cache()

    if not stock_name:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Select a ticker to view data",
            paper_bgcolor=PANEL_BG,
            plot_bgcolor=PANEL_BG,
            font=dict(color=MATRIX_GREEN),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
        )
        return empty_fig, "", table_data

    df_like = requests.get(f"http://localhost:8000/fetch_data/{stock_name}").json()
    pred_resp = requests.post(f"http://localhost:8000/predict/{stock_name}").json()

    # Build candlestick or fallback line
    figure = _build_candlestick_figure(df_like, stock_name)

    # Extract close series for metrics
    close_key = None
    for ck in ("Close", "CLOSE", "close"):
        if ck in df_like:
            close_key = ck
            break

    forecast_price = None
    # Expected structure: {'prediction': {'price': ...}}
    if isinstance(pred_resp, dict):
        prediction_obj = pred_resp.get("prediction") or {}
        forecast_price = prediction_obj.get("price")

    pred_text = ""
    if close_key and forecast_price is not None:
        metrics = _compute_returns(df_like[close_key], forecast_price)
        if metrics:
            today_price, today_return, forecast_return = metrics
            pred_text = f"Predicted Price: {float(forecast_price):.2f}"

    return figure, pred_text, table_data


def preload_cache():
    # Preload latest data and predictions for popular tickers
    for t in POPULAR_TICKERS:
        try:
            data = requests.get(f"http://localhost:8000/fetch_data/{t}", timeout=30).json()
            pred = requests.post(f"http://localhost:8000/predict/{t}", timeout=30).json()
            STOCK_CACHE[t] = {"data": data, "pred": pred}
        except Exception:
            # Leave missing tickers out silently
            pass


if __name__ == "__main__":
    # Preload data for table before starting server
    preload_cache()
    app.run(debug=True, host="0.0.0.0", port=8050)
