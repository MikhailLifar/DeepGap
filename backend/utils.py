# backend/utils.py

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from pathlib import Path

import pandas as pd

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


MOEX_BASE = "https://iss.moex.com/iss"
# Ликвидные акции торгуются на TQBR (режим T+)
DEFAULT_BOARD = "TQBR"

# Get the directory where this script is located
_BACKEND_DIR = Path(__file__).parent.resolve()
_DEFAULT_DATA_DIR = _BACKEND_DIR / "data"



# ---------- ВСПОМОГАТЕЛЬНЫЕ ПРОВЕРКИ ----------
def has_fresh_file(path: str, min_rows: int = 250, max_age_days: int = 7) -> bool:

    """

    Возвращает True, если файл существует, достаточно большой и не старше max_age_days.

    """

    if not os.path.isfile(path):
        return False

    try:
        size_ok = os.path.getsize(path) > 0
        if not size_ok:
            return False

        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        age_days = (datetime.now() - mtime).days

        if age_days > max_age_days:
            return False

        df = pd.read_csv(path)
        if len(df) < min_rows:
            return False

        return True

    except Exception:
        return False



# ---------- HTTP СЕССИЯ С РЕТРАЯМИ ----------

def _requests_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=0.8,           # 0.8, 1.6, 2.4, 3.2, ...
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def _moex_history_page(
    ticker: str,
    start: int,
    date_from: str,
    date_till: str,
    board: str = DEFAULT_BOARD,
    session: requests.Session | None = None,
) -> Dict:
    session = session or _requests_session()
    url = (
        f"{MOEX_BASE}/history/engines/stock/markets/shares/boards/{board}/securities/{ticker}.json"
        f"?from={date_from}&till={date_till}&start={start}"
    )
    resp = session.get(url, timeout=30)   # увеличенный таймаут
    resp.raise_for_status()
    return resp.json()


# ---------- ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ ----------

def _download_moex_range(
    ticker: str,
    start_date: str,
    end_date: str,
    board: str,
    session: requests.Session,
    sleep_sec: float = 0.5,
) -> pd.DataFrame:
    """
    Скачивает данные MOEX для указанного диапазона дат.
    Возвращает очищенный DataFrame с колонками Date,Open,High,Low,Close,Volume.
    """
    # Первая страница
    start = 0
    all_rows: List[List] = []
    columns: List[str] = []

    first = _moex_history_page(ticker, start, start_date, end_date, board, session=session)
    columns = first.get("history", {}).get("columns", [])
    rows = first.get("history", {}).get("data", [])
    cursor_cols = first.get("history.cursor", {}).get("columns", [])
    cursor_data = first.get("history.cursor", {}).get("data", [])

    if not rows and not cursor_data:
        return pd.DataFrame()

    # cursor_data: [[TOTAL, PAGESIZE, ...]]
    if cursor_data:
        total = cursor_data[0][cursor_cols.index("TOTAL")] if "TOTAL" in cursor_cols else len(rows)
        page_size = cursor_data[0][cursor_cols.index("PAGESIZE")] if "PAGESIZE" in cursor_cols else len(rows)
    else:
        total, page_size = len(rows), len(rows)

    # Пагинация с ручными ретраями
    from tqdm import tqdm

    with tqdm(total=total or 0, desc=f"MOEX {ticker} [{start_date} to {end_date}]", unit="rows") as pbar:
        while True:
            if rows:
                all_rows.extend(rows)
                pbar.update(len(rows))

            start += page_size

            if total is not None and start >= total:
                break

            time.sleep(sleep_sec)

            for attempt in range(5):
                try:
                    page = _moex_history_page(ticker, start, start_date, end_date, board, session=session)
                    rows = page.get("history", {}).get("data", [])
                    break
                except requests.exceptions.RequestException as e:
                    wait = 1.5 * (attempt + 1)
                    print(f"[{ticker}] start={start} failed: {e.__class__.__name__}. retry in {wait:.1f}s...")
                    time.sleep(wait)
                    rows = []

            if not rows:
                print(f"[{ticker}] no rows at start={start}, stop paging.")
                break

    if not all_rows:
        return pd.DataFrame()

    raw = pd.DataFrame(all_rows, columns=columns)
    # print(f'Check 0: {len(raw)}')

    # Переименуем в привычные имена
    rename_map = {
        "TRADEDATE": "Date",
        "OPEN": "Open",
        "HIGH": "High",
        "LOW": "Low",
        "CLOSE": "Close",
        "VOLUME": "Volume",
        "VALUE": "Value",
        "NUMTRADES": "NumTrades",
    }

    for src, dst in rename_map.items():
        if src in raw.columns:
            raw[dst] = raw[src]

    keep = [c for c in ["Date", "Open", "High", "Low", "Close", "Volume"] if c in raw.columns]
    df = raw[keep].copy()
    # print(f'Check 1: {len(df)}')

    # Чистка
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    # print(f'Check 2: {len(df)}')

    return df


def get_moex_history(
    ticker: str,
    start_date: str,
    end_date: str,
    out_dir: Optional[str] = None,
    board: str = DEFAULT_BOARD,
    sleep_sec: float = 0.5,
    force: bool = False,
) -> pd.DataFrame:
    """
    Скачивает историю торгов MOEX для тикера и сохраняет CSV: Date,Open,High,Low,Close,Volume.
    Умная логика: загружает только недостающие диапазоны дат и объединяет с существующими данными.
    Если файл уже есть и «свежий» — по умолчанию пропустит (кроме force=True).
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        out_dir: Output directory for CSV files (default: backend/data, absolute path)
        board: MOEX board name (default: TQBR)
        sleep_sec: Sleep time between requests
        force: Force refresh even if file exists
    """
    # Use absolute path by default to avoid working directory issues
    if out_dir is None:
        out_dir = str(_DEFAULT_DATA_DIR)
    else:
        # Convert relative paths to absolute paths
        out_dir_path = Path(out_dir)
        if not out_dir_path.is_absolute():
            # If relative, make it relative to the backend directory
            out_dir = str(_BACKEND_DIR / out_dir)
        else:
            out_dir = str(out_dir_path)
    
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{ticker}.csv")

    # Преобразуем даты в datetime для сравнения
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    existing_df = pd.DataFrame()
    existing_path = out_path

    # Загружаем существующие данные, если файл есть
    if os.path.isfile(existing_path) and not force:
        try:
            existing_df = pd.read_csv(existing_path, parse_dates=["Date"])
            if "Date" in existing_df.columns and len(existing_df) > 0:
                existing_df["Date"] = pd.to_datetime(existing_df["Date"])
                existing_df = existing_df.sort_values("Date").reset_index(drop=True)
        except Exception:
            # Если файл повреждён, игнорируем его
            existing_df = pd.DataFrame()

    # Определяем, какие диапазоны нужно скачать
    ranges_to_download: List[tuple[str, str]] = []

    if len(existing_df) == 0:
        # Нет существующих данных - скачиваем весь диапазон
        ranges_to_download.append((start_date, end_date))
    else:
        existing_start = existing_df["Date"].min()
        existing_end = existing_df["Date"].max()

        # Проверяем, нужно ли скачивать данные до существующего диапазона
        if start_dt < existing_start:
            # Скачиваем от start_date до дня перед existing_start
            prev_day = (existing_start - timedelta(days=1)).strftime("%Y-%m-%d")
            ranges_to_download.append((start_date, prev_day))

        # Проверяем, нужно ли скачивать данные после существующего диапазона
        if end_dt > existing_end:
            # Скачиваем от дня после existing_end до end_date
            next_day = (existing_end + timedelta(days=1)).strftime("%Y-%m-%d")
            ranges_to_download.append((next_day, end_date))

    # Если нечего скачивать, возвращаем существующие данные (уже отфильтрованные)
    if not ranges_to_download:
        # Фильтруем существующие данные по запрошенному диапазону
        mask = (existing_df["Date"] >= start_dt) & (existing_df["Date"] <= end_dt)
        existing_df = existing_df[mask].copy()
        return existing_df

    # Скачиваем недостающие диапазоны
    session = _requests_session()
    downloaded_dfs: List[pd.DataFrame] = []

    for range_start, range_end in ranges_to_download:
        print(f"[{ticker}] Downloading missing range: {range_start} to {range_end}")
        df_range = _download_moex_range(ticker, range_start, range_end, board, session, sleep_sec)
        if len(df_range) > 0:
            downloaded_dfs.append(df_range)
        print(f'Downloaded {len(df_range)} records')

    # Объединяем все данные
    all_dfs = [existing_df] + downloaded_dfs
    all_dfs = [df for df in all_dfs if len(df) > 0]
    # print(f'Check 0: {list(len(df) for df in all_dfs)}')

    if not all_dfs:
        # Если ничего не получилось, возвращаем пустой DataFrame
        return pd.DataFrame()

    # Объединяем и дедуплицируем
    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_df = merged_df.sort_values("Date").drop_duplicates(subset=["Date"]).reset_index(drop=True)
    # Сохраняем объединённые данные
    # print(f'Check 1: {len(merged_df)}')
    merged_df.to_csv(out_path, index=False)

    # Фильтруем по запрошенному диапазону (на случай, если скачали больше)
    mask = (merged_df["Date"] >= start_dt) & (merged_df["Date"] <= end_dt)
    merged_df = merged_df[mask].copy().sort_values("Date").reset_index(drop=True)

    return merged_df


def complete_data(csv_path: str, out_path: Optional[str] = None) -> pd.DataFrame:

    """

    Приводит произвольный CSV к стандартизированным колонкам и чистит данные.

    Сохраняет на место (или в out_path). Возвращает DataFrame.

    """

    if not os.path.isfile(csv_path):

        print(f"[complete_data] файл не найден: {csv_path}")

        return pd.DataFrame()



    df = pd.read_csv(csv_path)

    lower = {c.lower(): c for c in df.columns}



    def pick(*names):

        for n in names:

            if n in lower:

                return lower[n]

        return None



    c_date = pick("date", "tradedate", "time", "timestamp")

    c_close = pick("close", "price", "close_price", "adj_close")

    c_open  = pick("open")

    c_high  = pick("high")

    c_low   = pick("low")

    c_vol   = pick("volume", "vol")



    cols = {}

    if c_date:  cols["Date"]   = pd.to_datetime(df[c_date], errors="coerce")

    if c_open:  cols["Open"]   = pd.to_numeric(df[c_open], errors="coerce")

    if c_high:  cols["High"]   = pd.to_numeric(df[c_high], errors="coerce")

    if c_low:   cols["Low"]    = pd.to_numeric(df[c_low], errors="coerce")

    if c_close: cols["Close"]  = pd.to_numeric(df[c_close], errors="coerce")

    if c_vol:   cols["Volume"] = pd.to_numeric(df[c_vol], errors="coerce")



    out = pd.DataFrame(cols)

    if "Date" not in out or "Close" not in out:

        print("[complete_data] нет обязательных колонок Date/Close")

        return pd.DataFrame()



    out = (

        out.sort_values("Date")

           .drop_duplicates(subset=["Date"])

           .reset_index(drop=True)

    )

    for col in ["Open", "High", "Low", "Close", "Volume"]:

        if col in out.columns:

            out[col] = out[col].ffill().bfill()



    save_to = out_path or csv_path

    out.to_csv(save_to, index=False)

    return out

