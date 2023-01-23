"""rsi_strategy.py

Strategy picking stocks that look attractive according to their Relative Strength Index (RSI)

Rebalance frequency: monthly
Filter based on:
    * liquidity
    * RSI < 30
    * lowest N RSI of those below 30
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from file_paths_HIDDEN import SPX_STOCK_PRICE_FILE_PATH
from file_paths_HIDDEN import SPX_STOCK_VOLUME_FILE_PATH

from strategy_building_blocks.backtest_calendar import BacktestCalendar
from strategy_building_blocks.filter_handler import FilterHandler
from strategy_building_blocks.filters import LiquidityFilter
from strategy_building_blocks.filters import RelativeStrengthIndexFilter
from strategy_building_blocks.filters import PerformanceFilter
from strategy_building_blocks.weight import EqualWeight
from strategy_building_blocks.strategy import Strategy

from analysis.strategy_performance_analysis import return_risk_metrics_standard

COLOR_MAP = cm.get_cmap('Dark2')  # background color
plt.rcParams['axes.facecolor'] = 'ivory'  # coloring scheme

pd.set_option('display.max_columns', None)


def get_data(file_path: str, ffill_na: bool = True)->pd.DataFrame:
    """
    Returns a DataFrame by reading a CSV file, converting index to DateTimeIndex, forward fill nan and dropping nan
    :param file_path: str path to a csv file
    :param ffill_na: bool if True forward fill na, else replace them with zero (good for volume data)
    :return: DataFrame
    """
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    if ffill_na:
        df.fillna(method='ffill', inplace=True)
    else:
        df.fillna(0, inplace=True)
    return df


def main():
    # __________________________________________________________________________________________________________________
    # parameters
    liq_avg_lag = 60
    min_liq = 100_000_000
    plot_results = True
    low_rsi_threshold = 30  # long stocks that has RSI below this level
    tc = 0.000

    # __________________________________________________________________________________________________________________
    # get the data
    daily_price_df = get_data(file_path=SPX_STOCK_PRICE_FILE_PATH)
    daily_volume_df = get_data(file_path=SPX_STOCK_VOLUME_FILE_PATH, ffill_na=False)

    # __________________________________________________________________________________________________________________
    # calendar
    reb_rules = {
        'start_date': '1/1/2000',
        'end_date': '30/12/2022',
        'days': -1,
        'frequency': 'quarterly'
    }
    calendar = BacktestCalendar(rebalance_rules=reb_rules, reweight_lag=1)

    # __________________________________________________________________________________________________________________
    # define filters
    liq_filter = LiquidityFilter(avg_lag=liq_avg_lag, liquidity_threshold=min_liq, price_df=daily_price_df,
                                 volume_df=daily_volume_df)

    bullish_rsi_filter = RelativeStrengthIndexFilter(value=low_rsi_threshold, filter_type='smaller', price_df=daily_price_df)
    perf_filter = PerformanceFilter(observation_lag=[60, 120], value=20, price_df=daily_price_df, filter_type='top')
    combo_filter = FilterHandler()

    # __________________________________________________________________________________________________________________
    # weighting scheme
    eqw = EqualWeight()

    # __________________________________________________________________________________________________________________
    # setup the strategy and run the back test
    strat = Strategy(instrument_price_df=daily_price_df, calendar=calendar, strategy_filter=combo_filter, weight=eqw)
    strat.transaction_costs = tc
    strat.strategy_filter.filter_collection = [liq_filter, bullish_rsi_filter]
    rsi_strat_bt = strat.run_backtest()
    rsi_strat_bt.columns = ['rsi_strategy']

    # with extra momentum filter
    strat.strategy_filter.filter_collection = [liq_filter, [bullish_rsi_filter, perf_filter]]
    rsi_mom_strat_bt = strat.run_backtest()
    rsi_mom_strat_bt.columns = ['rsi_mom_strategy']

    # compare the strategy to just an equal weighted portfolio
    strat.strategy_filter = liq_filter  # no RSI filter
    eqw_strat_bt = strat.run_backtest()
    eqw_strat_bt.columns = ['eqw_strategy']

    # combine the DataFrames
    bt_df = rsi_strat_bt.join(rsi_mom_strat_bt)
    bt_df = bt_df.join(eqw_strat_bt)

    # __________________________________________________________________________________________________________________
    # analyse the results
    result = return_risk_metrics_standard(price_df=bt_df)
    print(result)

    if plot_results:
        bt_df.plot(grid=True, title='Performance', colormap=COLOR_MAP)  # standard line plot
        plt.show()


if __name__ == '__main__':
    main()

