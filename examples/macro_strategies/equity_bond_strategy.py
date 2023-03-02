"""cross_asset_basket.py"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm


from data.yahoo_finance_api import get_adj_close_price_df
from analysis.financial_stats_signals_indicators import rolling_pairwise_correlation
from analysis.strategy_performance_analysis import return_risk_metrics_standard

from strategy_building_blocks.strategy import Strategy
from strategy_building_blocks.weight import CustomFixedWeight
from strategy_building_blocks.weight import VolatilityWeight
from strategy_building_blocks.backtest_calendar import BacktestCalendar

COLOR_MAP = cm.get_cmap('Dark2')  # background color
plt.rcParams['axes.facecolor'] = 'ivory'  # coloring scheme

pd.set_option('display.max_columns', None)


def clean_daily_price(df: pd.DataFrame, normalize: bool = True):
    df.fillna(inplace=True, method='ffill')
    df.dropna(inplace=True)
    if normalize:
        df /= df.iloc[0, :]
    return df


def corr_analysis_main():
    """
    Plot rolling correlations for equity and fixed income tickers
    SPY: SPDR S&P 500 ETF Trust
    AGG: iShares Core US Aggregate Bond ETF
    VSIGX: Vanguard Intermediate-Term Treasury Index Fund Admiral Shares
    """
    # download the data and clean it
    tickers = ["SPY", "AGG", "VSIGX"]
    daily_price_df = get_adj_close_price_df(ticker=tickers)
    daily_price_df = clean_daily_price(df=daily_price_df)

    # for all correlation lags calculate the correlation and plot the results
    corr_lags = [250]
    total_corr_df = None
    for corr_l in corr_lags:
        # calculate correlation
        corr_df = rolling_pairwise_correlation(df=daily_price_df, lag=corr_l)
        corr_df.columns = [f"{col}_{corr_l}D" for col in corr_df.columns]  # change column name
        if total_corr_df is None:
            total_corr_df = corr_df
        else:
            total_corr_df = total_corr_df.join(
                corr_df
            )
    total_corr_df.plot()
    plt.show()


def strategy_main():
    # parameters
    equity_weights = [0.9, 0.8, 0.7, 0.6]

    # data
    tickers = ["SPY", "AGG"]
    daily_price_df = get_adj_close_price_df(ticker=tickers)
    daily_price_df = clean_daily_price(df=daily_price_df)

    # calendar
    calendar = BacktestCalendar(
        rebalance_rules={
                'start_date': '1/1/2004',
                'days': -1,
                'frequency': 'monthly'
            },
        reweight_lag=1
    )

    # strategy
    strat = Strategy(instrument_price_df=daily_price_df,
                     calendar=calendar, transaction_costs=0.0002)

    bt_strat_total_df = None
    for equity_w in equity_weights:
        ticker_weight_map = {
            "SPY": equity_w,
            "AGG": 1.0 - equity_w
        }

        # weights
        w = CustomFixedWeight(weight_map=ticker_weight_map)

        strat.weight = w

        # run back test
        bt_strat = strat.run_backtest()
        bt_strat.columns = [f'equity weight={equity_w}']

        if bt_strat_total_df is None:
            bt_strat_total_df = bt_strat
        else:
            bt_strat_total_df = bt_strat_total_df.join(bt_strat)

    inv_vol_w = VolatilityWeight(vol_lag=[60, 20], price_df=daily_price_df)
    strat.weight = inv_vol_w
    bt_inv_vol_strat = strat.run_backtest()
    bt_inv_vol_strat.columns = ['inv_vol']
    bt_strat_total_df = bt_strat_total_df.join(bt_inv_vol_strat)

    # add the underlying components
    bt_strat_total_df = bt_strat_total_df.join(daily_price_df)
    bt_strat_total_df = clean_daily_price(df=bt_strat_total_df)

    # analysis
    bt_result = return_risk_metrics_standard(price_df=bt_strat_total_df)
    print(bt_result)

    # plot results
    axis_1 = bt_strat_total_df.plot(grid=True, colormap=COLOR_MAP)
    axis_1.set_ylabel('Performance')

    # add rolling correlation to the plot
    corr_df = rolling_pairwise_correlation(df=daily_price_df, lag=252)
    axis_2 = corr_df.plot(secondary_y=True, ax=axis_1)
    axis_2.set_ylabel('Correlation')

    plt.show()


if __name__ == '__main__':
    strategy_main()
