"""Data fetcher classes for LSTM prediction pipeline."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


def calculate_monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert daily stock data to monthly returns.
    
    Args:
        df: DataFrame with Date and Close columns
    
    Returns:
        DataFrame with monthly aggregated data (Date, Close, Close_Return)
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Group by month and take the last day of each month
    df['Month'] = df['Date'].dt.to_period('M').dt.to_timestamp()
    month_end = df.groupby('Month', as_index=False).tail(1)
    month_end = month_end[['Month', 'Close']].copy()
    month_end = month_end.sort_values('Month')
    
    # Calculate monthly returns
    month_end['Close_Return'] = month_end['Close'].pct_change() * 100
    month_end = month_end.rename(columns={'Month': 'Date'})
    month_end = month_end[['Date', 'Close', 'Close_Return']].copy()
    month_end = month_end.dropna().reset_index(drop=True)
    
    return month_end


class StockDataFetcher:
    """Fetches stock price data from MOEX."""
    
    def __init__(
        self,
        get_moex_history_func,
        board: str = "TQBR",
        sleep_sec: float = 0.5
    ):
        """
        Args:
            get_moex_history_func: Function to fetch MOEX data (from utils.get_moex_history)
            board: MOEX board name (default: TQBR)
            sleep_sec: Sleep time between requests
        """
        self.get_moex_history = get_moex_history_func
        self.board = board
        self.sleep_sec = sleep_sec
    
    def fetch_stock_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        force: bool = False
    ) -> pd.DataFrame:
        """
        Fetch stock prices for a single ticker.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force: Force refresh data
        
        Returns:
            DataFrame with Date, Open, High, Low, Close, Volume columns
        """
        return self.get_moex_history(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            board=self.board,
            sleep_sec=self.sleep_sec,
            force=force
        )
    
    def fetch_multiple_stocks(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        force: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch stock prices for multiple tickers.
        
        Args:
            tickers: List of stock ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            force: Force refresh data
        
        Returns:
            Dictionary mapping ticker to DataFrame
        """
        result = {}
        for ticker in tickers:
            try:
                df = self.fetch_stock_prices(
                    ticker=ticker,
                    start_date=start_date,
                    end_date=end_date,
                    force=force
                )
                if len(df) > 0:
                    result[ticker] = df
            except Exception as e:
                print(f"Error fetching {ticker}: {e}")
                result[ticker] = pd.DataFrame()
        return result


class NewsDataFetcher:
    """Fetches and processes news sentiment data."""
    
    def __init__(self, news_csv_path: str):
        """
        Args:
            news_csv_path: Path to news sentiment CSV file
        """
        self.news_csv_path = news_csv_path
        self._news_df: Optional[pd.DataFrame] = None
    
    def _load_news_data(self) -> pd.DataFrame:
        """Load and process news data from CSV."""
        if self._news_df is not None:
            return self._news_df
        
        try:
            df = pd.read_csv(self.news_csv_path)
            
            # Check if month_start column exists
            if 'month_start' not in df.columns:
                # Try to infer from date column
                if 'Date' in df.columns or 'date' in df.columns:
                    date_col = 'Date' if 'Date' in df.columns else 'date'
                    df['month_start'] = pd.to_datetime(df[date_col]).dt.to_period('M').dt.to_timestamp()
                else:
                    raise ValueError("News CSV must include month_start or Date/date column")
            
            df['month_start'] = pd.to_datetime(df['month_start'])
            
            # Fill missing text/title
            df['title'] = df.get('title', '').fillna('')
            df['text'] = df.get('text', '').fillna('')
            
            # Calculate features
            df['title_len'] = df['title'].str.len()
            df['text_len'] = df['text'].str.len()
            df['has_text'] = (df['text'].str.len() > 0).astype(int)
            
            # Aggregate by month
            agg = df.groupby('month_start').agg(
                news_count=('title', 'size'),
                avg_title_len=('title_len', 'mean'),
                avg_text_len=('text_len', 'mean'),
                text_count=('has_text', 'sum'),
            )
            agg['news_mask'] = (agg['news_count'] > 0).astype(int)
            
            self._news_df = agg.reset_index()
            return self._news_df
            
        except Exception as e:
            print(f"Error loading news data from {self.news_csv_path}: {e}")
            return pd.DataFrame()
    
    def get_monthly_news_features(self) -> pd.DataFrame:
        """
        Get monthly aggregated news features.
        
        Returns:
            DataFrame with columns: month_start, news_count, avg_title_len, 
            avg_text_len, text_count, news_mask
        """
        return self._load_news_data()

