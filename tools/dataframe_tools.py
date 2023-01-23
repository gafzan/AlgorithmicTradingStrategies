"""dataframe_tools.py"""

import pandas as pd
import numpy as np


def winsorize_dataframe(df: pd.DataFrame, lower_pct: float = 0, upper_pct: float = 1):
    """
    Returns a new DataFrame where each data is set to nan if it is outside the percentiles. The percentiles are
    calculated per row
    :param df: DataFrame
    :param lower_pct: float
    :param upper_pct: float
    :return: DataFrame
    """
    if lower_pct >= upper_pct:
        raise ValueError("'upper_pct' ({}) needs to be strictly larger than 'lower_pct' ({})".format(upper_pct, lower_pct))
    df_q = df.quantile(q=[lower_pct, upper_pct], axis=1).T
    df_q.columns = ['low', 'high']
    # if data is outside the percentiles, set to nan else 1
    win_array = np.where(df.lt(df_q['low'], axis='index') | df.gt(df_q['high'], axis='index'), np.nan, 1)
    return df * win_array


def comparative_filter_df(df: pd.DataFrame, filter_type: str, inclusive: bool, threshold: float,
                          or_equal: bool = True):
    """
    Returns a DataFrame with value 1 in case the number has passed the comparison, else nan.
    :param df: DataFrame
    :param filter_type: str {'smaller' or 'larger}
    :param inclusive: bool if True the passing value will be 1 else, the passing value will be nan
    :param threshold: float
    :param or_equal: bool
    :return: DataFrame
    """
    filter_type = filter_type.lower()
    if filter_type not in ['smaller', 'larger']:
        raise ValueError("'filter_type' needs to be equal to 'smaller' or 'larger'")

    # perform comparison with threshold
    if filter_type == 'smaller':
        if or_equal:
            comp_df = df.le(threshold, axis=0)  # less than or equal
        else:
            comp_df = df.lt(threshold, axis=0)  # strictly less than
    else:
        if or_equal:
            comp_df = df.ge(threshold, axis=0)  # greater than or equal
        else:
            comp_df = df.gt(threshold, axis=0)  # strictly greater than
    filter_bool = np.where(comp_df, inclusive, not inclusive)
    filter_df = pd.DataFrame(index=df.index, columns=df.columns, data=filter_bool)
    filter_df *= 1  # convert True and False to 1 and 0
    filter_df *= np.where(~df.isnull(), 1, np.nan)  # nan in case original data is nan
    filter_df.replace(0, np.nan, inplace=True)
    return filter_df


def rank_filter_df(df: pd.DataFrame, filter_type: str, inclusive: bool, rank_threshold: {int, float},
                   or_equal: bool = True):
    """
    Returns a DataFrame with 1 if the element ha passed the specified ranking filter, else nan. Ranking threshold can be
    in percentage terms or a number.
    :param df: DataFrame
    :param filter_type: str
    :param inclusive: bool if True the passing value will be 1 else, the passing value will be nan
    :param rank_threshold: float or int
    :param or_equal: bool
    :return: DataFrame
    """
    filter_type = filter_type.lower()
    if filter_type not in ['top', 'bottom']:
        raise ValueError("'filter_type' needs to be equal to 'top' or 'bottom'")
    ascending = filter_type == 'bottom'

    # check the inputs
    if rank_threshold <= 0:
        raise ValueError("'rank_threshold' needs to be an int or float strictly larger than 0")

    # rank the elements in the DataFrame
    ranked_df = df.rank(axis='columns', method='first', ascending=ascending, numeric_only=True)

    # set DataFrame to 1 if the ranking filter is passed, else nan
    if rank_threshold < 1:
        num_numeric_per_row = ranked_df.count(axis=1)
        rank_threshold = round(num_numeric_per_row * rank_threshold)

    return comparative_filter_df(df=ranked_df, filter_type='smaller', inclusive=inclusive, threshold=rank_threshold,
                                 or_equal=or_equal)


def aggregate_df(df_list: list, agg_method: str):
    """
    Returns a DataFrame that is an aggregate of all the DataFrames in the given list
    :param df_list: list of DataFrame
    :param agg_method: str
    :return:
    """
    # convert to numpy array since it is waaaaay faster
    array_list = [df.values for df in df_list]

    # adjust the name of the aggregate function
    agg_np = getattr(np, agg_method)(a=array_list, axis=0)
    if isinstance(df_list[0], pd.DataFrame):
        return pd.DataFrame(
            columns=df_list[0].columns,
            index=df_list[0].index,
            data=agg_np
        )
    else:
        return pd.Series(
            index=df_list[0].index,
            data=agg_np
        )


def cap_floor_df(df, floor: float = None, cap: float = None):
    """
    Returns a DataFrame where all the elements are capped and floored at the specified values
    :param df: DataFrame
    :param floor: float
    :param cap: float
    :return: DataFrame
    """
    return pd.DataFrame(
        data=np.clip(df.values, floor, cap),
        index=df.index,
        columns=df.columns
    )
