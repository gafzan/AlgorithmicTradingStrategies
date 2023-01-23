"""factor_analysis.py"""

import pandas_datareader as pdr
import pandas as pd
from sklearn import linear_model


def main():
    print('load data')
    stock_price_df = pd.read_csv('<path>.csv', index_col=0)
    # ticker = 'SAND.ST'
    tickers = list(stock_price_df)[:1]
    stock_price_df = stock_price_df[tickers]

    # clean data
    stock_price_df.index = pd.to_datetime(stock_price_df.index)  # convert the index to a DateTimeIndex

    # calculate monthly returns
    stock_price_df = stock_price_df.asfreq('B')  # fill in the missing day in the index
    stock_price_df.fillna(method='ffill', inplace=True)  # asfreq('B') can generate additional nan
    monthly_ret_df = stock_price_df.asfreq('BM').pct_change()
    monthly_ret_df.dropna(axis=0, inplace=True)

    # dictionary with 0: monthly factor returns, 1: yearly factor returns
    monthly_factor_returns = pdr.DataReader(name='F-F_Research_Data_Factors', data_source='famafrench',
                                            start=monthly_ret_df.index[0], end=monthly_ret_df.index[-1])[0] / 100
    monthly_factor_returns.index = monthly_ret_df.index  # FF factors uses a PeriodIndex
    merged_df = monthly_factor_returns.join(monthly_ret_df)  # merge the two DataFrames

    # calculate the excess returns
    for ticker in tickers:
        merged_df[f'{ticker}-RF'] = merged_df[ticker] - merged_df.RF
        merged_df.drop(ticker, axis=1, inplace=True)

    # perform OLS regression
    reg = linear_model.LinearRegression(fit_intercept=True)
    y = 'SAND.ST-RF'
    x = ['Mkt-RF', 'SMB', 'HML']
    reg.fit(merged_df[x], merged_df[y])
    print(reg.intercept_)
    print(reg.coef_)


if __name__ == '__main__':
    main()
