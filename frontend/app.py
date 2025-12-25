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
                "flex": "2 1 0%",
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
                dcc.Loading(
                    id="graph-loading",
                    type="default",
                    children=dcc.Graph(
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
                    color=GOLD,
                ),
            ],
        ),
        # RIGHT: Top (25%) dropdown, Bottom (75%) table
        html.Div(
            style={
                "flex": "1 1 0%",
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
                            "Search Ticker",
                            style={
                                "color": GOLD,
                                "margin": "4px 0 0 0",
                                "letterSpacing": "0.4px",
                                "fontWeight": "600",
                            },
                        ),
                        html.Div(
                            children=[
                                dcc.Input(
                                    id="stock-search",
                                    type="text",
                                    value="SBER",
                                    placeholder="Enter ticker symbol (e.g., SBER, GAZP, LKOH...)",
                                    debounce=True,
                                    style={
                                        "width": "100%",
                                        "height": "44px",
                                        "backgroundColor": CARBON_BG,
                                        "color": MATRIX_GREEN,
                                        "border": f"1px solid rgba(212,175,55,0.3)",
                                        "borderRadius": "10px",
                                        "padding": "10px 12px",
                                        "fontSize": "14px",
                                        "fontFamily": "Verdana, Geneva, Tahoma, sans-serif",
                                        "boxShadow": "0 0 0 0 rgba(212,175,55,0)",
                                    },
                                ),
                                html.Div(
                                    "Popular: " + ", ".join([f"{TICKER_TO_NAME.get(t, t)} ({t})" for t in POPULAR_TICKERS[:5]]),
                                    style={
                                        "marginTop": "6px",
                                        "fontSize": "11px",
                                        "color": "rgba(212,175,55,0.6)",
                                        "fontStyle": "italic",
                                    },
                                ),
                            ],
                        ),
                        html.Div(
                            id="loading-indicator",
                            style={
                                "marginTop": "8px",
                                "color": GOLD,
                                "fontSize": "14px",
                                "display": "none",
                            },
                            children=[
                                html.Div("⏳ Loading data from MOEX... This may take some time.", style={"color": GOLD}),
                            ],
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
                            "Current & Predicted Monthly",
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
                                {"name": "Current Month Price", "id": "today_price", "type": "numeric", "format": {"specifier": ".3f"}},
                                {"name": "Current Month Return", "id": "today_return", "type": "numeric", "format": {"specifier": ".3%"}},
                                {"name": "Predicted Monthly Price", "id": "forecast_price", "type": "numeric", "format": {"specifier": ".3f"}},
                                {"name": "Predicted Monthly Return", "id": "forecast_return", "type": "numeric", "format": {"specifier": ".3%"}},
                                {"name": "Recommendation", "id": "recommendation"},
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
                                "whiteSpace": "normal",
                                "height": "auto",
                                "lineHeight": "1.2",
                                "textAlign": "center",
                                "padding": "8px 4px",
                            },
                            style_cell={
                                "backgroundColor": CARBON_BG,
                                "color": MATRIX_GREEN,
                                "border": f"1px solid rgba(212,175,55,0.08)",
                                "textAlign": "center",
                                "fontFamily": "Verdana, Geneva, Tahoma, sans-serif",
                                "fontSize": "13px",
                                "padding": "8px 4px",
                                "whiteSpace": "normal",
                            },
                            style_header_conditional=[
                                {
                                    "if": {"column_id": "ticker"},
                                    "width": "15%",
                                },
                                {
                                    "if": {"column_id": "today_price"},
                                    "width": "17%",
                                },
                                {
                                    "if": {"column_id": "today_return"},
                                    "width": "17%",
                                },
                                {
                                    "if": {"column_id": "forecast_price"},
                                    "width": "17%",
                                },
                                {
                                    "if": {"column_id": "forecast_return"},
                                    "width": "17%",
                                },
                                {
                                    "if": {"column_id": "recommendation"},
                                    "width": "17%",
                                },
                            ],
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
                                {
                                    "if": {"filter_query": "{recommendation} = buy", "column_id": "recommendation"},
                                    "color": MATRIX_GREEN,
                                    "fontWeight": "600",
                                },
                                {
                                    "if": {"filter_query": "{recommendation} = sell", "column_id": "recommendation"},
                                    "color": RED,
                                    "fontWeight": "600",
                                },
                                {
                                    "if": {"filter_query": "{recommendation} = keep", "column_id": "recommendation"},
                                    "color": GOLD,
                                    "fontWeight": "600",
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


def _get_recommendation(predicted_return: float) -> str:
    """
    Get recommendation based on predicted monthly return.
    
    Args:
        predicted_return: Predicted monthly return as a decimal (e.g., 0.05 for 5%)
    
    Returns:
        'buy', 'keep', or 'sell'
    """
    # Convert to percentage for threshold comparison
    return_pct = predicted_return * 100
    
    if return_pct > 2.0:
        return "buy"
    elif return_pct < -2.0:
        return "sell"
    else:
        return "keep"


def _build_table_rows_from_cache(force_reload=False):
    """Build table rows from cache, fetching from backend only if needed.
    
    Args:
        force_reload: If True, fetch from backend even if in local cache.
                     If False, only use local cache (backend handles initial loading).
    """
    rows = []
    for t in POPULAR_TICKERS:
        # Check local cache first
        cached = STOCK_CACHE.get(t)
        if not cached or force_reload:
            # If not in local cache, fetch from backend (which uses its own cache)
            try:
                data = requests.get(f"http://localhost:8000/fetch_data/{t}", timeout=10).json()
                pred = requests.post(f"http://localhost:8000/predict/{t}", timeout=30).json()
                # Update local cache for future use
                STOCK_CACHE[t] = {"data": data, "pred": pred}
                cached = STOCK_CACHE.get(t)
            except Exception as e:
                # If backend hasn't finished loading yet, skip for now
                # Table will update once backend is ready
                print(f"Error fetching {t} for table (backend may still be loading): {e}")
                continue
        
        if not cached:
            continue
            
        df_like = cached.get("data") or {}
        pred = cached.get("pred") or {}
        # extract forecast price and return
        pred_price = None
        pred_return = None
        if isinstance(pred, dict):
            prediction_obj = pred.get("prediction") or {}
            pred_price = prediction_obj.get("price")
            pred_return = prediction_obj.get("return")  # Monthly return in percentage
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
                # Use predicted return from API if available, otherwise use calculated forecast_return
                if pred_return is not None:
                    # pred_return is in percentage (e.g., 1.5 for 1.5%), convert to decimal
                    forecast_return = pred_return / 100.0
                recommendation = _get_recommendation(forecast_return)
                rows.append({
                    "ticker": f"{TICKER_TO_NAME.get(t, t)} ({t})",
                    "today_price": float(today_price),
                    "today_return": float(today_return),
                    "forecast_price": float(pred_price),
                    "forecast_return": float(forecast_return),
                    "recommendation": recommendation.upper(),
                })
    return rows


@app.callback(
    [
        Output("candles-graph", "figure"),
        Output("prediction-output", "children"),
        Output("metrics-table", "data"),
        Output("loading-indicator", "style"),
    ],
    [Input("stock-search", "value")],
    # Allow initial call to show data on page load
)
def update_views(stock_name):
    # Only build table from cache (don't force reload - backend handles initial loading)
    # This prevents redundant requests on every input change
    table_data = _build_table_rows_from_cache(force_reload=False)

    if not stock_name:
        empty_fig = go.Figure()
        empty_fig.update_layout(
            title="Enter a ticker symbol to view data",
            paper_bgcolor=PANEL_BG,
            plot_bgcolor=PANEL_BG,
            font=dict(color=MATRIX_GREEN),
            xaxis_showgrid=False,
            yaxis_showgrid=False,
        )
        return empty_fig, "", table_data, {"marginTop": "8px", "display": "none"}

    # Normalize ticker (uppercase, strip whitespace)
    stock_name = stock_name.upper().strip()

    # Use cache if available, otherwise fetch
    cached = STOCK_CACHE.get(stock_name)
    loading_style = {"marginTop": "8px", "display": "none"}
    
    if cached:
        df_like = cached.get("data")
        pred_resp = cached.get("pred")
    else:
        # Fetch if not in cache - show loading indicator
        loading_style = {"marginTop": "8px", "display": "block"}
        try:
            response = requests.get(f"http://localhost:8000/fetch_data/{stock_name}", timeout=60)
            # Check for HTTP errors
            if response.status_code == 404:
                error_detail = response.json().get("detail", f"Ticker '{stock_name}' not found")
                empty_fig = go.Figure()
                empty_fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    text=f"Ticker '{stock_name}' not found<br>Please check the ticker symbol",
                    showarrow=False,
                    font=dict(size=16, color=RED),
                    bgcolor="rgba(0, 0, 0, 0.8)",
                    bordercolor=RED,
                    borderwidth=2,
                    borderpad=15,
                    align="center",
                )
                empty_fig.update_layout(
                    title=f"Ticker '{stock_name}' not found",
                    paper_bgcolor=PANEL_BG,
                    plot_bgcolor=PANEL_BG,
                    font=dict(color=MATRIX_GREEN),
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                )
                loading_style = {"marginTop": "8px", "display": "none"}
                error_msg = html.Div(
                    f"Error: {error_detail}",
                    style={
                        "color": RED,
                        "fontSize": "14px",
                        "marginTop": "8px",
                    }
                )
                return empty_fig, error_msg, table_data, loading_style
            response.raise_for_status()  # Raise exception for other HTTP errors
            df_like = response.json()
            
            # Check if data is empty (shouldn't happen with proper backend handling, but check anyway)
            if not df_like or (isinstance(df_like, dict) and not any(df_like.values())):
                empty_fig = go.Figure()
                empty_fig.add_annotation(
                    xref="paper", yref="paper",
                    x=0.5, y=0.5,
                    text=f"No data available for ticker '{stock_name}'",
                    showarrow=False,
                    font=dict(size=16, color=RED),
                    bgcolor="rgba(0, 0, 0, 0.8)",
                    bordercolor=RED,
                    borderwidth=2,
                    borderpad=15,
                    align="center",
                )
                empty_fig.update_layout(
                    title=f"No data for '{stock_name}'",
                    paper_bgcolor=PANEL_BG,
                    plot_bgcolor=PANEL_BG,
                    font=dict(color=MATRIX_GREEN),
                    xaxis_showgrid=False,
                    yaxis_showgrid=False,
                )
                loading_style = {"marginTop": "8px", "display": "none"}
                error_msg = html.Div(
                    f"No data available for ticker '{stock_name}'. Please check the ticker symbol.",
                    style={
                        "color": RED,
                        "fontSize": "14px",
                        "marginTop": "8px",
                    }
                )
                return empty_fig, error_msg, table_data, loading_style
            
            # Try to get prediction, but don't fail if it doesn't work for non-existent tickers
            try:
                pred_resp = requests.post(f"http://localhost:8000/predict/{stock_name}", timeout=90).json()
            except Exception:
                # If prediction fails, that's okay - we'll just show data without prediction
                pred_resp = None
            
            # Update cache for future use
            STOCK_CACHE[stock_name] = {"data": df_like, "pred": pred_resp}
            loading_style = {"marginTop": "8px", "display": "none"}
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error fetching data for {stock_name}: {e}")
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                text=f"Error loading '{stock_name}'<br>Please check the ticker symbol",
                showarrow=False,
                font=dict(size=16, color=RED),
                bgcolor="rgba(0, 0, 0, 0.8)",
                bordercolor=RED,
                borderwidth=2,
                borderpad=15,
                align="center",
            )
            empty_fig.update_layout(
                title=f"Error loading {stock_name}",
                paper_bgcolor=PANEL_BG,
                plot_bgcolor=PANEL_BG,
                font=dict(color=MATRIX_GREEN),
                xaxis_showgrid=False,
                yaxis_showgrid=False,
            )
            loading_style = {"marginTop": "8px", "display": "none"}
            error_msg = html.Div(
                f"Error: Could not load data for ticker '{stock_name}'. Please verify the ticker symbol is correct.",
                style={
                    "color": RED,
                    "fontSize": "14px",
                    "marginTop": "8px",
                }
            )
            return empty_fig, error_msg, table_data, loading_style
        except Exception as e:
            print(f"Error fetching data for {stock_name}: {e}")
            # Return empty figure on error
            empty_fig = go.Figure()
            empty_fig.add_annotation(
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                text=f"Error loading '{stock_name}'<br>{str(e)[:100]}",
                showarrow=False,
                font=dict(size=16, color=RED),
                bgcolor="rgba(0, 0, 0, 0.8)",
                bordercolor=RED,
                borderwidth=2,
                borderpad=15,
                align="center",
            )
            empty_fig.update_layout(
                title=f"Error loading {stock_name}",
                paper_bgcolor=PANEL_BG,
                plot_bgcolor=PANEL_BG,
                font=dict(color=MATRIX_GREEN),
                xaxis_showgrid=False,
                yaxis_showgrid=False,
            )
            loading_style = {"marginTop": "8px", "display": "none"}
            error_msg = html.Div(
                f"Error loading data: {str(e)}",
                style={
                    "color": RED,
                    "fontSize": "14px",
                    "marginTop": "8px",
                }
            )
            return empty_fig, error_msg, table_data, loading_style

    # Build candlestick or fallback line
    figure = _build_candlestick_figure(df_like, stock_name)

    # Extract close series for metrics
    close_key = None
    for ck in ("Close", "CLOSE", "close"):
        if ck in df_like:
            close_key = ck
            break

    forecast_price = None
    pred_return = None
    # Expected structure: {'prediction': {'price': ..., 'return': ...}}
    if pred_resp and isinstance(pred_resp, dict):
        prediction_obj = pred_resp.get("prediction") or {}
        forecast_price = prediction_obj.get("price")
        pred_return = prediction_obj.get("return")  # Monthly return in percentage

    pred_text = ""
    if close_key and forecast_price is not None:
        metrics = _compute_returns(df_like[close_key], forecast_price)
        if metrics:
            today_price, today_return, forecast_return = metrics
            # Use predicted return from API if available, otherwise use calculated forecast_return
            if pred_return is not None:
                # pred_return is in percentage (e.g., 1.5 for 1.5%), convert to decimal
                forecast_return = pred_return / 100.0
            recommendation = _get_recommendation(forecast_return)
            recommendation_color = MATRIX_GREEN if recommendation == "buy" else (RED if recommendation == "sell" else GOLD)
            pred_text = html.Div([
                html.Div(
                    f"Current Monthly Price: {float(today_price):.3f}",
                    style={
                        "marginBottom": "8px",
                        "color": GOLD,
                        "fontSize": "16px",
                        "fontWeight": "600",
                    }
                ),
                html.Div(
                    f"Predicted Monthly Price: {float(forecast_price):.3f}",
                    style={
                        "marginBottom": "8px",
                        "color": MATRIX_GREEN,
                        "fontSize": "16px",
                    }
                ),
                html.Div(
                    f"Recommendation: {recommendation.upper()}",
                    style={
                        "color": recommendation_color,
                        "fontSize": "16px",
                        "fontWeight": "600",
                    }
                ),
            ])

    return figure, pred_text, table_data, loading_style


if __name__ == "__main__":
    # Backend handles data caching and preloading, so frontend starts immediately
    print("Starting Dash server...")
    print("Note: Backend handles data caching and periodic updates (12 hours)")
    app.run(debug=True, host="0.0.0.0", port=8050)
