from utils import *


def main():
    # Define blue chips and growing companies
    blue_chips = [
        'SBER',  # Sberbank
        'GAZP',  # Gazprom
        'LKOH',  # Lukoil
        'ROSN',  # Rosneft
        'MGNT',  # Magnit
        'NLMK',  # NLMK
        'GMKN',  # Nornickel
        'SNGS',  # Surgutneftegas
        'PLZL',  # Polymetal
        'TATN'   # Tatneft
    ]

    growing_companies = [
        'YNDX',  # Yandex
        'TCS',   # TCS Group
        'OZON',  # Ozon
        'FIVE',  # Five
        'IRKT',  # Irkut
        'PIKK',  # PIK
        'AQUA',  # Aqua
        'CBOM',  # Cbom
        'POSI',  # Positiv
        'RUSI'   # RussNeft
    ]

    # Combine all tickers
    all_tickers = blue_chips + growing_companies

    # Set date range (10 years of data)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')

    # # Download data for each ticker
    # for ticker in all_tickers:
    #     print(f"Downloading data for {ticker}...")
    #     df = get_moex_history(ticker, start_date, end_date, '../data')

    #     if not df.empty:
    #         print(f"Successfully downloaded {len(df)} records for {ticker}")
    #     else:
    #         print(f"No data downloaded for {ticker}")

    #     # Sleep to avoid overwhelming the API
    #     time.sleep(1)

    directory = os.path.join('.', 'data')
    for f in os.listdir(directory):
        complete_data(os.path.join(directory, f), None)


if __name__ == '__main__':
    main()
