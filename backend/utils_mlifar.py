import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta


def get_moex_history(symbol, start_date, end_date, dest_dir, save=True):
    """
    Download historical daily stock data from MOEX API

    Parameters:
    symbol (str): Stock ticker symbol
    start_date (str): Start date in 'YYYY-MM-DD' format
    end_date (str): End date in 'YYYY-MM-DD' format

    Returns:
    pandas.DataFrame: DataFrame with historical stock data
    """
    
    dest_prefix = os.path.join(dest_dir, f'{symbol}_{start_date}')
    df = None

    dest_path = [os.path.join(dest_dir, f) for f in os.listdir(dest_dir) if f.startswith(dest_prefix)]
    if len(dest_path) == 1:
        dest_path, = dest_path

    if not len(dest_path):
        url = f"https://iss.moex.com/iss/engines/stock/markets/shares/securities/{symbol}/candles.json"
        params = {
            'from': start_date,
            'till': end_date,
            'interval': 24,  # Daily candles
            'iss.meta': 'off'
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            # Extract candles data
            candles_data = data['candles']['data']
            columns = data['candles']['columns']

            # Create DataFrame
            df = pd.DataFrame(candles_data, columns=columns)

            # Rename columns for clarity
            df = df.rename(columns={
                'begin': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume',
                'value': 'value'
            })

            # Convert date to datetime and set as index
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)

            if save:
                dest_path = f'{dest_prefix}_{df["date"].max().strftime("%Y-%m-%d")}'
                df.to_csv(dest_path, index=True, index_label='date_index')

        except Exception as e:
            print(f"Error downloading data for {symbol}: {e}")
            df = pd.DataFrame()
        except KeyboardInterrupt:
            print(f'Interrupted data scraping for {symbol}')
            df = pd.DataFrame()

    else:
        df = pd.read_csv(dest_path, index_col='date_index')

    return df


def complete_data(data_path, complete_to_date):
    df = pd.read_csv(data_path, index_col='date_index')
    basename = os.path.splitext(os.path.split(data_path)[1])[0]
    ticker = basename.split('_')[0]

    end_date = df.index.max()
    # print(ticker, end_date)

    while end_date < complete_to_date:
        new_df = get_moex_history(ticker, end_date=complete_to_date, start_date=end_date)




