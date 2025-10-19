import dash
from dash import dcc, html, Input, Output
import requests

app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Stock Price Prediction"),
    dcc.Dropdown(
        id='stock-dropdown',
        options=[
            {'label': 'SBER', 'value': 'SBER'},
            {'label': 'GAZP', 'value': 'GAZP'},
        ],
        value='SBER'
    ),
    dcc.Graph(id='price-graph'),
    html.Div(id='prediction-output')
])

@app.callback(
    [Output('price-graph', 'figure'),
     Output('prediction-output', 'children')],
    [Input('stock-dropdown', 'value')]
)
def update_graph(stock_name):
    # Fetch data from FastAPI
    data = requests.get(f"http://localhost:8000/fetch_data/{stock_name}").json()
    prediction = requests.post(f"http://localhost:8000/predict/{stock_name}").json()

    data = data['Close']
    
    # Create figure (dummy example)
    figure = {
        'data': [{'x': list(data.keys()), 'y': list(data.values()), 'type': 'line'}],
        'layout': {'title': f'{stock_name} Prices'}
    }

    # # Display prediction
    pred_text = f"Predicted Price: {prediction['prediction']['price']}"
    # pred_text = ''

    return figure, pred_text

if __name__ == '__main__':
    app.run(debug=True)
