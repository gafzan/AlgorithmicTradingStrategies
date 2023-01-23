"""financial_stats_signals_indicators.py"""

import pandas as pd
import numpy as np


def simple_moving_average(df: pd.DataFrame, lag: int, relative: bool = False)->pd.DataFrame:
    """
    Calculates a simple moving average ('SMA') or rolling average over the specified lag parameter.
    Can be normalized wrt to the original values.
    :param df: DataFrame
    :param lag: int
    :param relative: bool
    :return: DataFrame
    """
    sma_df = df.rolling(window=lag).mean()
    if relative:
        return sma_df / df.fillna(method='ffill').values
    else:
        return sma_df


def exponentially_weighted_return(price_df: pd.DataFrame, lambda_: float) -> pd.DataFrame:
    """
    Calculates the exponentially weighted returns
    :param price_df: DataFrame containing the closing levels
    :param lambda_: float the weight
    :return: pd.DataFrame
    """
    price_return_df = price_df.pct_change()
    price_return_df = price_return_df.iloc[1:, :]
    number_days = price_return_df.shape[0]
    exp_weighting_var = [np.exp(-lambda_ / number_days * (1 + t_day)) for t_day in range(number_days - 1, -1, -1)]
    exp_weighting_s = pd.Series(index=price_return_df.index, data=exp_weighting_var)
    return price_return_df.mul(exp_weighting_s, axis=0)


def realized_volatility(price_df: pd.DataFrame, lag: int, log_normal_returns: bool = False,
                        annualising_factor: int = 252, with_mean: bool = True)->pd.DataFrame:
    """
    Returns the realized volatility i.e. the standard deviation of returns.
    :param price_df: DataFrame (assumed to be a price DataFrame
    :param lag: int
    :param log_normal_returns: bool
    :param annualising_factor: int (default 252)
    :param with_mean: bool (some formulas for volatility does not adjust the return before squaring)
    :return: DataFrame
    """
    # either use log normal returns or arithmetic returns
    if log_normal_returns:
        return_df = np.log(price_df) - np.log(price_df.shift(1))
    else:
        # by default the fill method is forward fill of the price before the return calculation so NaN are set to 0
        return_df = price_df.pct_change(fill_method=None)

    return _realized_volatility(returns_df=return_df, lag=lag, annualising_factor=annualising_factor,
                                with_mean=with_mean)


def downside_volatility(price_df: pd.DataFrame, lag: int, minimum_accepted_return: float = 0.0,
                        log_normal_returns: bool = False,  annualising_factor: int = 252,
                        with_mean: bool = False)->pd.DataFrame:
    """
    Returns the standard deviation of returns below the minimum accepted return (MAR). MAR = 0 means the volatiltiy of
    negative returns
    :param price_df: DataFrame (assumed to be a price DataFrame
    :param lag: int
    :param minimum_accepted_return: float
    :param log_normal_returns: bool
    :param annualising_factor: int (default 252)
    :param with_mean: bool (some formulas for volatility does not adjust the return before squaring)
    :return: DataFrame
    """
    # either use log normal returns or arithmetic returns
    if log_normal_returns:
        return_df = np.log(price_df) - np.log(price_df.shift(1))
    else:
        # by default the fill method is forward fill of the price before the return calculation so NaN are set to 0
        return_df = price_df.pct_change(fill_method=None)

    # set all returns above the MAR to nan
    return_df -= minimum_accepted_return
    return_df[return_df > 0] = np.nan

    # calculate the realized volatility on each column after dropping the nan i.e. the returns above the MAR
    down_vol = return_df.apply(lambda ts: _realized_volatility(returns_df=ts.dropna(), lag=lag,
                                                               annualising_factor=annualising_factor, with_mean=with_mean))
    return down_vol.fillna(method='ffill')


def _realized_volatility(returns_df: pd.DataFrame, lag: int, annualising_factor: int, with_mean: bool):
    """
    Returns the rolling volatility i.e. standard deviations of given returns
    :param returns_df: DataFrame of returns
    :param lag: int
    :param annualising_factor: int
    :param with_mean: bool
    :return: DataFrame
    """
    if with_mean:
        return np.sqrt(annualising_factor) * returns_df.rolling(window=lag).std()
    else:
        # first calculate the rolling mean of the squared returns
        squared_mean_return_df = simple_moving_average(df=returns_df.apply(np.square), lag=lag)
        # squared_mean_return_df = return_df.apply(np.square).rolling(window=lag).mean()
        return np.sqrt(annualising_factor) * (lag / (lag - 1) * squared_mean_return_df).apply(np.sqrt)


def rolling_holding_period_returns(price_df: pd.DataFrame, lag: int, lag_ending: int = 0):
    """
    Calculate the rolling holding period return for each column defined as:
        Price(t - ending_lag) / Price(t - lag) - 1
    :param price_df: DataFrame
    :param lag: int
    :param lag_ending: int
    :return:
    """
    return price_df.shift(lag_ending).pct_change(lag - lag_ending, fill_method=None)


def rolling_sharpe_ratio(price_df: pd.DataFrame)->pd.DataFrame:
    """
    Returns a DataFrame with the rolling Sharpe Ratio defined as the ratio between rolling 252 days holding period
    return and rolling 252 days realized volatility.

    Theoretically you should adjust the returns with the "risk-free"
    rate but I have never seen anyone working in finance who actually does that. Why? Because if rates are not negative
    the Sharpe ratio will be worse...

    :param price_df: DataFrame
    :return: DataFrame
    """
    annual_ret_df = rolling_holding_period_returns(price_df=price_df, lag=252)
    annual_vol_df = realized_volatility(price_df=price_df, lag=252)
    return annual_ret_df / annual_vol_df


def rolling_drawdown(price_df: pd.DataFrame, look_back_period: int = None) -> pd.DataFrame:
    """
    Assumes that price_df is a DataFrame and look_back_period is an int. If look_back_period is not assigned, the
    'peak/maximum' will be observed continuously. Returns a DataFrame containing the drawdown for each underlying i.e.
    'price' / 'maximum priced over look back period' - 1.
    :param price_df: DataFrame
    :param look_back_period: int (if None assumes that the observation period is the entire history)
    :return: DataFrame
    """
    if look_back_period is None:
        look_back_period = len(price_df.index)
    price_df = price_df.fillna(method='ffill').copy()
    rolling_max_df = price_df.rolling(window=look_back_period, min_periods=1).max()
    drawdown_df = price_df / rolling_max_df - 1.0
    return drawdown_df


def maximum_drawdown(price_df: pd.DataFrame, look_back_period: int = None) -> pd.Series:
    """
    Assumes that price_df is a DataFrame and look_back_period is an int. If look_back_period is not assigned, the
    'peak/maximum' will be observed continuously. Returns a Series containing the maximum drawdown for each underlying
    i.e. the lowest 'price' / 'maximum priced over look back period' - 1 observed.
    :param price_df: DataFrame
    :param look_back_period: int (if None assumes that the observation period is the entire history)
    :return: Series
    """
    drawdown_df = rolling_drawdown(price_df, look_back_period)
    return drawdown_df.min()


def rolling_pairwise_correlation(df: pd.DataFrame, lag: int)->pd.DataFrame:
    """
    Returns a DataFrame with the rolling correlations for each pair excluding duplicates and when correlations of the
    same columns.
    This function is quite slow for DataFrames with many columns...
    :param df: DataFrame
    :param lag: int
    :return: DataFrame
    """
    cols = df.columns
    corr_df = pd.DataFrame()
    for i in range(len(cols)):
        for j in range(len(cols)):
            if j > i:
                corr_df[f"{cols[i]}_vs_{cols[j]}"] = df[cols[i]].rolling(window=lag).corr(df[cols[j]])
    return corr_df


def relative_strength_index(price_df: pd.DataFrame, lag: int = 14, ewma: bool = True):
    """
    Returns a DataFrame with the rolling Relative Strength Index (RSI). Technical indicator for price strength or
    weakness.
    100 - 100 / (1 + (avg. gains / avg. loss))
    :param price_df: DataFrame
    :param lag: window of the average (14 by default for some reason...)
    :param ewma: use exponential smoothing (same method as www.tradingview.com)
    :return:
    """
    gain_df = np.maximum(price_df.diff(), 0)
    loss_df = np.maximum(-price_df.diff(), 0)

    if ewma:
        ma_gain_df = gain_df.ewm(alpha=1 / lag, adjust=True).mean()
        ma_loss_df = loss_df.ewm(alpha=1 / lag, adjust=True).mean()
    else:
        ma_gain_df = gain_df.rolling(window=lag).mean()
        ma_loss_df = loss_df.rolling(window=lag).mean()

    return 100 - (100 / (1 + ma_gain_df / ma_loss_df))


def beta(price_df: pd.DataFrame, benchmark_price: {pd.DataFrame, pd.Series}, lag: int,
         log_normal_returns: bool = False)->pd.DataFrame:
    """
    Calculates the rolling beta as the ratio of the covariance and variance. Covariance is calculated on the returns of
    the given price DataFrame and the benchmark. Variance is calculate on the returns of the benchmark. Note that there
    is no lag here for the return calculation. In case one needs to look at the beta for weekly returns one needs to
    first convert the frequency of the price DataFrame of to weekly and then use it as an input into this function.
    :param price_df: DataFrame
    :param benchmark_price: DataFrame or Series
    :param lag: int observation window for the rolling beta calculation
    :param log_normal_returns: bool
    :return: DataFrame
    """

    if isinstance(price_df, pd.Series):
        price_df = price_df.to_frame()

    if 'benchmark' in price_df.columns:
        raise ValueError("can't have a column named 'benchmark' in price_df")

    # convert the benchmark price to a DataFrame if necessary and rename the column
    benchmark_price = benchmark_price.copy()
    if isinstance(benchmark_price, pd.Series):
        benchmark_price = benchmark_price.to_frame()
    benchmark_price.columns = ['benchmark']

    # use log normal returns or arithmetic returns on the merged DataFrame (exact match using the index of price_df)
    return_df = price_return(price_df=price_df.join(benchmark_price), log_normal_returns=log_normal_returns)

    # calculate realized beta as the ratio between the covariance and the valriance
    covariance_df = return_df.rolling(window=lag).cov(return_df['benchmark'])
    variance_df = return_df['benchmark'].rolling(window=lag).var()
    beta_df = covariance_df.divide(variance_df, axis='index')
    beta_df.drop('benchmark', axis=1, inplace=True)
    return beta_df


def price_return(price_df: pd.DataFrame, log_normal_returns: bool, lag: int = 1)->pd.DataFrame:
    """
    Returns a DataFrame with arithmetic or logarithmic return
    :param price_df: DataFrame
    :param log_normal_returns: bool
    :param lag: int
    :return: DataFrame
    """
    # either use log normal returns or arithmetic returns
    if log_normal_returns:
        return_df = np.log(price_df) - np.log(price_df.shift(lag))
    else:
        # by default the fill method is forward fill of the price before the return calculation so NaN are set to 0
        return_df = price_df.pct_change(periods=lag, fill_method=None)
    return return_df


