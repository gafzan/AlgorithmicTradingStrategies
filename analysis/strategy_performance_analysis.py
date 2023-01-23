"""strategy_performance_analysis.py"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.financial_stats_signals_indicators import rolling_holding_period_returns
from analysis.financial_stats_signals_indicators import realized_volatility
from analysis.financial_stats_signals_indicators import maximum_drawdown


MONTH_NAME_MAP = {
    1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov',
    12: 'Dec'
}


def monthly_return_df(price_df: pd.DataFrame)->pd.DataFrame:
    """
    Returns a DataFrame with a year and month multi index, instrument names as columns and monthly returns as values
    :param price_df: DataFrame
    :return: DataFrame
    """
    # clean the results
    result_df = price_df.copy()
    result_df.index = pd.to_datetime(result_df.index)  # convert the index to a DateTimeIndex
    result_df = result_df.asfreq('B')  # fill in the missing day in the index
    result_df.fillna(method='ffill', inplace=True)  # fill forward nan

    # create two new columns with the year and month
    result_df['year'] = result_df.index.year  # create a column with the year of the date
    result_df['month'] = result_df.index.month  # create a column with the month of the date

    # calculate the return between each end of month dates
    monthly_returns = result_df.asfreq('BM').set_index(['year', 'month']).pct_change()
    return monthly_returns


def monthly_return_table(price_df: pd.DataFrame)->{pd.DataFrame, dict}:
    """
    Creates a DataFrame with monthly returns with year as index and month as column. If the specified DataFrame contains
    more than one price column a dict is returned with the col name as key and monthly return DataFrame as value
    :param price_df: DataFrame
    :return: DataFrame or dict
    """
    # get the monthly return DataFrame
    month_ret_all_df = monthly_return_df(price_df=price_df)

    # loop through each instrument and pivot the DataFrame such that the index is the year and the columns are months
    result = {}
    for col in month_ret_all_df:
        month_ret_df = month_ret_all_df[[col]].reset_index()
        month_ret_df = month_ret_df.pivot(index='year', columns='month', values=col)
        month_ret_df.rename(columns=MONTH_NAME_MAP, inplace=True)
        result[col] = month_ret_df

    # if there is only one instrument, return a DataFrame else return a dict with the instrument name as key and
    # monthly return DataFrame as value
    if len(result.keys()) == 1:
        return result[list(result.keys())[0]]
    else:
        return result


def plot_pairwise_correlation(df: pd.DataFrame)->None:
    """
    Plots a heat-map of correlations in a table

    Source: https://seaborn.pydata.org/examples/many_pairwise_correlations.html
    :param df: DataFrame
    :return: None
    """
    sns.set(style="white")
    corr = df.corr()  # compute the correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=np.bool))  # generate a mask for the upper triangle
    cmap = sns.diverging_palette(220, 10, as_cmap=True)  # generate a custom diverging colormap
    # draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})
    plt.show()


def return_risk_metrics_standard(price_df: pd.DataFrame):
    """
    Return a DataFrame with average rolling 1 year return and volatility, Sharpe ratio (defined as the ratio between
    the return and vol) as well as Maximum Drawdown observed over the netire period
    :param price_df: DataFrame
    :return: DataFrame
    """
    avg_1y_ret = rolling_holding_period_returns(price_df=price_df, lag=252).mean(axis=0)
    avg_1y_vol = realized_volatility(price_df=price_df, lag=252).mean(axis=0)
    sharpe_ratio = avg_1y_ret / avg_1y_vol
    max_dd = maximum_drawdown(price_df=price_df)
    result = {
        'Avg. 1Y return (%)': avg_1y_ret,
        'Avg. 1Y volatility (%)': avg_1y_vol,
        'Sharpe ratio': sharpe_ratio,
        'Max Drawdown (%)': max_dd
    }
    result_df = pd.DataFrame(result).T
    result_df.iloc[[0, 1, 3]] = result_df.iloc[[0, 1, 3]] * 100
    return result_df



