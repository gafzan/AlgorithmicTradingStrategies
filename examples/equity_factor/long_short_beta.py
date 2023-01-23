"""long_short_beta.py
Long stocks that has a low beta w.r.t. S&P 500 and short high beta stocks
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from file_paths_HIDDEN import SPX_STOCK_PRICE_FILE_PATH
from file_paths_HIDDEN import SPX_STOCK_VOLUME_FILE_PATH
from file_paths_HIDDEN import SPX_INDEX_PRICE_FILE_PATH

from strategy_building_blocks.backtest_calendar import BacktestCalendar
from strategy_building_blocks.filters import BetaFilter
from strategy_building_blocks.filters import LiquidityFilter
from strategy_building_blocks.filter_handler import FilterHandler
from strategy_building_blocks.weight import EqualWeight
from strategy_building_blocks.weight import CustomFixedWeight
from strategy_building_blocks.strategy import Strategy

from analysis.strategy_performance_analysis import return_risk_metrics_standard


COLOR_MAP = cm.get_cmap('Dark2')  # background color
plt.rcParams['axes.facecolor'] = 'ivory'  # coloring scheme

pd.set_option('display.max_columns', None)


def get_data(file_path: str, ffill_na: bool=True)->pd.DataFrame:
    """
    Returns a DataFrame by reading a CSV file, converting index to DateTimeIndex, forward fill nan and dropping nan
    :param file_path: str path to a csv file
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
    liquidity_threshold = 1_000_000
    liq_lag = 60  # the observation window for calculating average liquidity
    beta_calc_window = [60, 252]  # the observation window for calculating the rolling beta with S&P 500
    num_low_beta_stocks = 50  # the number of low beta stocks included
    num_high_beta_stocks = 50  # the number of high beta stocks included
    tc = 0.0015  # transaction costs
    plot_results = True

    # __________________________________________________________________________________________________________________
    # get the data
    daily_price_df = get_data(file_path=SPX_STOCK_PRICE_FILE_PATH)
    daily_volume_df = get_data(file_path=SPX_STOCK_VOLUME_FILE_PATH, ffill_na=False)
    daily_spx_price_df = get_data(file_path=SPX_INDEX_PRICE_FILE_PATH)

    # __________________________________________________________________________________________________________________
    # set up the components of the strategy
    # step 1: define the calendar
    # rebalance on the 4th business day of each month and calculate the weights 2 days before
    reb_rules = {
        'start_date': '1/1/2005',
        'end_date': '30/12/2022',
        'days': 4,
        'frequency': 'quarterly'
    }
    calendar = BacktestCalendar(rebalance_rules=reb_rules, reweight_lag=2)

    # step 2: setup the filters
    # first setup a liquidity filter
    liquidity_filter = LiquidityFilter(avg_lag=liq_lag, liquidity_threshold=liquidity_threshold,
                                       price_df=daily_price_df, volume_df=daily_volume_df)

    # go long the stocks with the lowest beta to S&P 500 and short the highest beta stocks
    # winsorize the results to avoid outliers
    pos_beta_filter = BetaFilter(beta_calc_lag=beta_calc_window, filter_type='larger', value=0, or_equal=False,
                                 price_df=daily_price_df, benchmark_price=daily_spx_price_df)
    low_beta_filter = BetaFilter(beta_calc_lag=beta_calc_window, filter_type='bottom', value=num_low_beta_stocks,
                                 winsorize_lower_pct=0.05, price_df=daily_price_df, benchmark_price=daily_spx_price_df)
    high_beta_filter = BetaFilter(beta_calc_lag=beta_calc_window, filter_type='top', value=num_high_beta_stocks,
                                  winsorize_upper_pct=0.95, price_df=daily_price_df, benchmark_price=daily_spx_price_df)

    # the above filters will be combined first such that we will perform a liquidity filter and from those stocks we
    # will pick the highest/lowest beta stocks
    combo_filter = FilterHandler()
    # [] are treated as an 'intersection' of two sets and () is equivalent to a 'union' of sets
    combo_filter.filter_collection = [liquidity_filter, ([low_beta_filter, pos_beta_filter], {'filter': high_beta_filter, 'position': 'short'})]
    # combo_filter.filter_collection = [liquidity_filter, {'filter': high_beta_filter, 'position': 'short'}]

    # step 3: set up the weighting scheme
    eqw = EqualWeight(net_zero=True)

    # __________________________________________________________________________________________________________________
    # setup the strategy and run the back test
    strat_name = 'long_short_beta'
    strat = Strategy(instrument_price_df=daily_price_df)
    strat.calendar = calendar
    strat.strategy_filter = combo_filter
    strat.weight = eqw
    strat.transaction_costs = tc
    strat.columns = [strat_name]

    strat_bt = strat.run_backtest()
    bt_df = strat_bt.join(daily_spx_price_df)  # add S&P 500 for the sake of comparison

    # strategy combining the beta long/short with S&P 500
    for spx_weight in [0.5, 0.6, 0.7, 0.8, 0.9]:
        weight_custom = CustomFixedWeight(weight_map={'^GSPC': spx_weight, strat_name: 1 - spx_weight})
        combo_strat = Strategy(instrument_price_df=bt_df,
                               calendar=calendar,
                               weight=weight_custom)

        combo_strat_bt = combo_strat.run_backtest()
        combo_strat_bt.columns = [f'{round(spx_weight*100, 0)}% SPX + {round((1 - spx_weight) * 100, 0)}% {strat_name}']
        bt_df = bt_df.join(combo_strat_bt)

    bt_df /= bt_df.iloc[0, :]

    # __________________________________________________________________________________________________________________
    # analyse the result
    risk_return_df = return_risk_metrics_standard(price_df=bt_df)
    print(risk_return_df)

    if plot_results:
        bt_df.plot(grid=True, title='Performance', colormap=COLOR_MAP)  # standard line plot
        plt.show()


if __name__ == '__main__':
    main()

