"""low_volatility.py"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm

from file_paths_HIDDEN import SPX_STOCK_PRICE_FILE_PATH
from file_paths_HIDDEN import SPX_STOCK_VOLUME_FILE_PATH

# import stuff used for index construction
from strategy_building_blocks.strategy import Strategy
from strategy_building_blocks.backtest_calendar import BacktestCalendar
from strategy_building_blocks.filters import VolatilityFilter
from strategy_building_blocks.filters import LiquidityFilter
from strategy_building_blocks.filter_handler import FilterHandler
from strategy_building_blocks.weight import EqualWeight

# import stuff for analysis
from analysis.strategy_performance_analysis import return_risk_metrics_standard
from reporting.excel import save_and_format_excel
from file_paths_HIDDEN import SAVE_FOLDER_PATH_BACKTEST

COLOR_MAP = cm.get_cmap('Dark2')  # background color
plt.rcParams['axes.facecolor'] = 'ivory'  # coloring scheme


def get_data(file_path: str, ffill_na: bool = True)->pd.DataFrame:
    """
    Returns a DataFrame by reading a CSV file, converting index to DateTimeIndex, forward fill nan and dropping nan
    :param file_path: str path to a csv file
    :param ffill_na: bool if True forward fill the nan, else replace them as 0
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
    portfolio_size = 25
    vol_lag = [20, 60]  # maximum of 20 and 60 days volatility
    liquidity_threshold = 10_000_000
    liq_avg_lag = 60  # the observation window for calculating average liquidity
    transaction_costs = 0.0015

    save_results = True
    plot_results = False

    # __________________________________________________________________________________________________________________
    # get the data
    daily_adj_close_df = get_data(file_path=SPX_STOCK_PRICE_FILE_PATH)
    daily_volume_df = get_data(file_path=SPX_STOCK_VOLUME_FILE_PATH, ffill_na=False)

    # __________________________________________________________________________________________________________________
    # set up the components of the strategy
    # step 1: define the calendar
    calendar = BacktestCalendar(
        rebalance_rules={
                'start_date': '1/1/2000',
                'days': -1,
                'frequency': 'monthly'
            },
        reweight_lag=1
    )

    # step 2: setup the filters
    # first setup a liquidity filter
    liquidity_filter = LiquidityFilter(avg_lag=liq_avg_lag, liquidity_threshold=liquidity_threshold,
                                       price_df=daily_adj_close_df, volume_df=daily_volume_df)
    # low volatility filter
    low_vol_filter = VolatilityFilter(vol_lag=vol_lag, value=portfolio_size, price_df=daily_adj_close_df,
                                      winsorize_lower_pct=0.1)
    liq_low_vol_filter = FilterHandler(filter_collection=[liquidity_filter, low_vol_filter])

    # step 3: set up the weighting scheme
    eqw = EqualWeight()

    # __________________________________________________________________________________________________________________
    # setup the strategy and run the back test
    strat = Strategy(instrument_price_df=daily_adj_close_df,
                     calendar=calendar,
                     weight=eqw,
                     strategy_filter=liq_low_vol_filter,
                     transaction_costs=transaction_costs)

    # run the back test of the low volatility strategy
    low_vol_bt_df = strat.run_backtest()
    low_vol_bt_df.columns = ['LOW_VOL_EQW']

    # run the back test of the benchmark
    strat.strategy_filter = liquidity_filter
    eqw_bt_df = strat.run_backtest()
    eqw_bt_df.columns = ['EQW']

    bt_df = low_vol_bt_df.join(eqw_bt_df)  # combine into one DataFrame

    # __________________________________________________________________________________________________________________
    # analysis
    result_df = return_risk_metrics_standard(price_df=bt_df)
    print(result_df)

    if save_results:
        # save result in excel
        strategy_name = 's&p_500_low_volatility'
        save_file_path = SAVE_FOLDER_PATH_BACKTEST / f'{strategy_name}_{dt.date.today().strftime("%Y%m%d")}.xlsx'
        save_and_format_excel(data={'performance': bt_df, 'risk & return': result_df}, save_file_path=save_file_path)

    if plot_results:
        bt_df.plot(grid=True, title='Performance', colormap=COLOR_MAP)  # standard line plot
        plt.show()


if __name__ == '__main__':
    main()




