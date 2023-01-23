"""s&p_500_momentum.py

Compare "Golden Cross Strategy" on S&P 500 to just buy and hold
Long when leading SMA > lagging SMA

compare return and risk
* with and without transaction costs
* use various rebalance frequencies (monthly and quarterly)

Conclusion:

"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm

from file_paths_HIDDEN import SPX_INDEX_PRICE_FILE_PATH

# import stuff used for index construction
from strategy_building_blocks.strategy import Strategy
from strategy_building_blocks.backtest_calendar import BacktestCalendar
from strategy_building_blocks.filters import SimpleMovingAverageCrossFilter
from strategy_building_blocks.weight import EqualWeight

# import stuff for analysis
from analysis.strategy_performance_analysis import return_risk_metrics_standard
from reporting.excel import save_and_format_excel
from file_paths_HIDDEN import SAVE_FOLDER_PATH_BACKTEST

COLOR_MAP = cm.get_cmap('Dark2')  # background color
plt.rcParams['axes.facecolor'] = 'ivory'  # coloring scheme

pd.set_option('display.max_columns', 100)


def main():

    # parameters
    # sma_lead = 50
    # sma_lag = 200
    sma_variations = [(20, 50), (20, 100), (50, 100), (50, 200), (100, 200)]
    save_results = True

    # __________________________________________________________________________________________________________________
    # download data from excel
    spy_price_df = pd.read_csv(SPX_INDEX_PRICE_FILE_PATH, index_col=0)
    spy_price_df.index = pd.to_datetime(spy_price_df.index)
    spy_price_df.sort_index(inplace=True)

    # __________________________________________________________________________________________________________________
    # define the components of the index
    calendar = BacktestCalendar(
        {
            'start_date': '1/1/2007',
            'days': -1,
            'frequency': 'weekly'
        }
    )
    eqw = EqualWeight(weight_floor=0)  # no shorts

    # SMA filter
    sma_filter = SimpleMovingAverageCrossFilter(price_df=spy_price_df)
    sma_filter.filter_type = 'larger'
    sma_filter.or_equal = True
    sma_filter.value = 0

    # __________________________________________________________________________________________________________________
    # setup and run strategy
    strat = Strategy(instrument_price_df=spy_price_df,
                     calendar=calendar,
                     weight=eqw,
                     transaction_costs=0.008)

    # run the back test of the benchmark
    eqw_bt_df = strat.run_backtest()
    eqw_bt_df.columns = ['Buy and Hold']

    # run the back test of the momentum strategy
    bt_df = eqw_bt_df.copy()
    for sma_param in sma_variations:
        sma_filter.leading_window = sma_param[0]
        sma_filter.lagging_window = sma_param[1]
        strat.strategy_filter = sma_filter

        sma_bt_df = strat.run_backtest()
        sma_bt_df.columns = [f'SMA_CROSS({sma_param[0]}, {sma_param[1]})']
        bt_df = bt_df.join(sma_bt_df) # combine into one DataFrame

    # __________________________________________________________________________________________________________________
    # analysis
    result_df = return_risk_metrics_standard(price_df=bt_df)
    print(result_df)

    if save_results:
        # save result in excel
        strategy_name = 'spy_sma_cross'
        save_file_path = SAVE_FOLDER_PATH_BACKTEST / f'{strategy_name}_{dt.date.today().strftime("%Y%m%d")}.xlsx'
        save_and_format_excel(data={'performance': bt_df, 'risk & return': result_df}, save_file_path=save_file_path)

    bt_df.plot(grid=True, title='Performance', colormap=COLOR_MAP)  # standard line plot
    plt.show()


if __name__ == '__main__':
    main()




