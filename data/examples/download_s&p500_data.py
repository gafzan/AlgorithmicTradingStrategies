"""download_s&p500_data.py
Downloads adj. close and volume data of all the stocks in the S&P 500 from Yahoo Finance using
yahoo_fin (https://theautomatic.net/yahoo_fin-documentation/)
"""

from datetime import datetime as dt
from pathlib import Path

from yahoo_fin.stock_info import tickers_sp500
from data.yahoo_finance_api import get_ohlc_volume

SAVE_FOLDER_PATH = Path("")


def main():
    tickers = tickers_sp500()  # list of tickers as strings
    data_dict = get_ohlc_volume(ticker=tickers)  # returns a dictionary with data labels (e.g. 'open') as keys and DataFrames as values

    for data_label in ['adjclose', 'volume']:
        data_dict[data_label].to_csv(SAVE_FOLDER_PATH / f"sp500_{data_label}_{dt.now().strftime('%Y%m%d')}.csv")


if __name__ == '__main__':
    main()



