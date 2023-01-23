"""correlation_analysis.py"""

import matplotlib.pyplot as plt
from matplotlib import cm

from analysis.financial_stats_signals_indicators import rolling_pairwise_correlation
from data.yahoo_finance_api import get_adj_close_price_df

COLOR_MAP = cm.get_cmap('Dark2')  # background color
plt.rcParams['axes.facecolor'] = 'ivory'  # coloring scheme


def main():
    # parameters
    price_return_lag = 5
    corr_lag = 50
    tickers = [e.strip() for e in input('Enter ticker separated by a comma (,): ').split(',')]

    # load the data, clean it and reformat it based on the frequency
    adj_close_df = get_adj_close_price_df(ticker=tickers)
    adj_close_df.fillna(method='ffill', inplace=True)
    adj_close_df.dropna(axis=0, inplace=True)
    adj_close_df = adj_close_df.iloc[::price_return_lag, :]

    # calculate the correlation and plot it
    corr_df = rolling_pairwise_correlation(df=adj_close_df, lag=corr_lag)
    corr_df.plot(grid=True, title='Correlation', colormap=COLOR_MAP)
    plt.show()


if __name__ == '__main__':
    main()


