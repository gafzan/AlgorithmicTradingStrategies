"""dow_jones_momentum.py

Idea is to compare two equally weighted strategies
1) simple buy and hold of each stock
2) filter out N stock based on their price momentum

compare return and risk
* with and without transaction costs
* compare various momentum filters
* use various rebalance frequencies (monthly and quarterly)

Conclusion (Dow Jones):
The strategy does not work on Dow Jones
Even when ignoring transaction costs, the momentum filter does not add value vs simple EQW
Before transaction costs, monthly rebalancing performs better than quarterly but the relationship switches when
introducing 8bps transaction cost for the rebalancing.

"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib import cm

from file_paths_HIDDEN import DOW_JONES_PRICE_FILE_PATH

# import stuff used for index construction
from strategy_building_blocks.strategy import Strategy
from strategy_building_blocks.backtest_calendar import BacktestCalendar
from strategy_building_blocks.filters import PerformanceFilter
from strategy_building_blocks.weight import EqualWeight

# import stuff for analysis
from analysis.strategy_performance_analysis import return_risk_metrics_standard
from reporting.excel import save_and_format_excel
from file_paths_HIDDEN import SAVE_FOLDER_PATH_BACKTEST

COLOR_MAP = cm.get_cmap('Dark2')  # background color
plt.rcParams['axes.facecolor'] = 'ivory'  # coloring scheme


def main():

    # download data from excel
    daily_adj_close_df = pd.read_csv(DOW_JONES_PRICE_FILE_PATH, index_col=0)
    daily_adj_close_df.fillna(method='ffill', inplace=True)
    daily_adj_close_df.index = pd.to_datetime(daily_adj_close_df.index)
    daily_adj_close_df.sort_index(inplace=True)

    # define the components of the index
    calendar = BacktestCalendar(
        {
            'start_date': '1/1/2000',
            'days': -1,
            'frequency': 'quarterly'
        }
    )
    eqw = EqualWeight()

    # momentum filter
    top_perfs = 10
    momentum_filter = PerformanceFilter(observation_lag=[20, 60, 180], filter_type='top', value=top_perfs,
                                        price_df=daily_adj_close_df)

    # setup strategy
    strat = Strategy(instrument_price_df=daily_adj_close_df,
                     calendar=calendar,
                     weight=eqw,
                     transaction_costs=0.0015)

    # run the back test of the benchmark
    eqw_bt_df = strat.run_backtest()
    eqw_bt_df.columns = ['EQW']

    # run the back test of the momentum strategy
    strat.strategy_filter = momentum_filter
    mom_bt_df = strat.run_backtest()
    mom_bt_df.columns = ['MOM_EQW']

    bt_df = mom_bt_df.join(eqw_bt_df)  # combine into one DataFrame

    # analysis
    result_df = return_risk_metrics_standard(price_df=bt_df)
    print(result_df)

    # save result in excel
    strategy_name = 'dow_jones_momentum'
    save_file_path = SAVE_FOLDER_PATH_BACKTEST / f'{strategy_name}_{dt.date.today().strftime("%Y%m%d")}.xlsx'

    save_and_format_excel(data={'performance': bt_df, 'risk & return': result_df}, save_file_path=save_file_path)


if __name__ == '__main__':
    main()




