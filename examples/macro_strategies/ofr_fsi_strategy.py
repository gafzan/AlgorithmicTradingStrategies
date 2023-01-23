"""ofr_fsi_strategy.py
Calculate a strategy that goes long equities when the OFR Financial Stress Alpha Index is bullish, else a mix between
bonds and equities
"""

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import cm

from examples.indicators.ofr_financial_stress_alpha_index import get_ofr_fsi_alpha_signal

from file_paths_HIDDEN import OFR_FSI_STRATEGY_DATA_PATH

from strategy_building_blocks.backtest_calendar import BacktestCalendar
from strategy_building_blocks.weight import SignalBasedWeight
from strategy_building_blocks.weight import EqualWeight
from strategy_building_blocks.weight import CustomFixedWeight
from strategy_building_blocks.strategy import Strategy

from analysis.strategy_performance_analysis import return_risk_metrics_standard

COLOR_MAP = cm.get_cmap('Dark2')  # background color
plt.rcParams['axes.facecolor'] = 'ivory'  # coloring scheme

pd.set_option('display.max_columns', None)


def get_data():
    """
    Returns a DataFrame with prices of equity, gold and US bond etf
    :return: DataFrame
    """
    df = pd.read_csv(OFR_FSI_STRATEGY_DATA_PATH, index_col=0)

    # clean data forward fill na and then drop remaining na and normalize results
    df.index = pd.to_datetime(df.index)  # make sure the index is a DateTimeIndex
    df.fillna(method='ffill', inplace=True)
    df.dropna(axis=0, inplace=True)
    df /= df.iloc[0, :]
    df *= 100.0
    return df


def get_signal_time_series():
    """
    Returns a Series that is either 'bearish' or 'bullish' depending on the value of the OFR Financial Stress Alpha
    :return: Series
    """
    ofr_fsi_alpha_signal = get_ofr_fsi_alpha_signal()
    signal_time_series = ofr_fsi_alpha_signal.copy()
    signal_time_series[ofr_fsi_alpha_signal > 0] = 'bearish'
    signal_time_series[ofr_fsi_alpha_signal <= 0] = 'bullish'
    return signal_time_series


def main():

    create_plot = True

    # data
    daily_prices = get_data()

    # setup the signal
    signal_time_series = get_signal_time_series()
    signal_weight_map = {
        'bullish':
            {
                'SPY': 1.0
            },
        'bearish':
            {
                'SPY': 0.2,
                'GLD': 0.4,
                'IEF': 0.4
            }
    }

    # ____________________________________________________________________________
    # setup the strategy
    # calendar
    calendar = BacktestCalendar(
        {
            'start_date': '1/1/2005',
            'end_date': '30/12/2022',
            'days': -1,
            'frequency': 'monthly'
        }
    )

    # weight component
    signal_weight = SignalBasedWeight(signal_time_series=signal_time_series,
                                      signal_weight_map=signal_weight_map)

    strat = Strategy()
    strat.instrument_price_df = daily_prices
    strat.calendar = calendar
    strat.weight = signal_weight

    strat.annual_fee = 0.005
    strat.transaction_costs = {
        'GLD': 0.01,
        'IEF': 0.0015,
        'SPY': 0.0015
    }

    # ____________________________________________________________________________
    # run the back test and analyse the result
    strat_bt = strat.run_backtest()

    # calculate EQW strategy for comparison
    strat.weight = EqualWeight()
    strat_eqw_bt = strat.run_backtest()
    strat_eqw_bt.columns = ['strategy_eqw']

    # adjusted fixed weights
    strat.weight = CustomFixedWeight(weight_map={'GLD': 0.2, 'IEF': 0.2, 'SPY': 0.6})
    strat_fixed_w_df = strat.run_backtest()
    strat_fixed_w_df.columns = ['strategy_fixed_w']

    # combine into one DataFrame
    bt_df = strat_bt.join(daily_prices)
    bt_df = bt_df.join(strat_eqw_bt)
    bt_df = bt_df.join(strat_fixed_w_df)
    risk_return_df = return_risk_metrics_standard(price_df=bt_df)
    print(risk_return_df)

    if create_plot:
        bt_df.plot(grid=True, title='Performance', colormap=COLOR_MAP)  # standard line plot
        plt.show()


if __name__ == '__main__':
    main()

