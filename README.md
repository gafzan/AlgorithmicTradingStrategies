# Algorithmic Trading Strategies
***Version 1.0.0***

This project aims to be a toolbox for designing, back testing and analysing algorithmic trading strategies. At the moment, it is not meant
to be an automatic trading tool but rather be an environment where you can efficiently back test systematic trading strategies. 
The filters, signals and weighting schemes as well as the analysis tools available will expand over time.

A strategy is assumed to have the following components: **calendar**, **filter**, **weights**. A fourth component will 
be **overlay** that will include such things as *stop-loss features*, *volatility targeting*, *beta hedges* etc.

#### 1. Calendar
The timing of deciding the allocation to the components in the strategy is dictated by specified calendars. There are three 
available calendars that one can specify for the strategy: 1) Review calendar, 2) Reweight calendar and 3) Rebalance 
calendar.

- **Review calendar**: the dates when your *investment universe is reviewed*. This is usually a less frequent filter e.g. 
once a year. For example one can have a strategy that every 1st business day of the year performs an eligibility filter like liquidity or market capitalization. 

- **Reweight calendar**: the dates when your strategy *calculates* the weights for the strategy. 

- **Rebalance calendar**: finally the *allocation is implemented* based on the weights calculated as of the previous reweight
calendar date.

At least a rebalance calendar needs to be specified.

The back test calendar can be as simple as just monthly or more convoluted. Say for example you want to select the 5th 
business day, 1st Thursday and 2nd Friday of every month of March and September (I mean why wouldn't you?)

The way you would define this rule is as below:

```
calendar_rule = {
    'start_date': '1/1/2000',
    'frequency': 'monthly',
    'frequency_values': [3, 9],
    'days': [5, 'thu', ('fri', 2)]
}
```

Setting up the back test calendar is done using the `BackestCalendar` class:

```
from strategy_building_blocks.backtest_calendar import BacktestCalendar  # import the calendar generator

calendar = BacktestCalendar()  # initialize an instance of a BacktestCalendar object
calendar.rebalance_rules = calendar_rule  # specify the rule
calendar_df = calendar.get_calendars(as_dataframe=True)  # run the calendar and store the result as a DataFrame
```

Notice that since we have not specified the rules for the *review* or *reweight* calendars they are both set to the 
rebalance calendar by default.


#### 2. Filter
Common filters are liquidity, market capitalization or for example volatility. A sequence of filters can be implemented 
either as a *union* of filters (picking low volatility stocks and stocks with high price momentum), *intersection* 
(low volatility stocks that also has high price momentum) or a combination of the two. A filter can also be specified to 
find components to short.

#### 3. Weights
A weighting scheme can be selected from a range of allocation methods like *equal weighting - "1/N"*, inverse 
proportional to realized volatility or custom weights depending on a separate indicator or signal. 

To be added: optimized weights like *minimum variance*

---
#### Example - S&P 500 Momentum 
As an example of a simple (and frankly dumb) strategy we will pick the 25 stocks in the S&P 500 with the highest price 
momentum. On the last business day of every month we will allocate 4% to each 25 stocks based on a momentum signal defined 
as the average performance over 20, 60 and 120-days. As an eligibility filter, the 
average 60 day liquidity needs to be at least USD 10 million.

**Strategy**:
- Average 60-day liquidity > USD 10 million
- Pick top 25 stocks with the highest momentum
    * Momentum signal: average of 20, 50 and 120-day performance
- Allocate 4% (1/25) to each stock
- Rebalance the portfolio on the last business day of each month

One can download historical price and volume data from Yahoo Finance using [yahoo_fin](https://theautomatic.net/yahoo_fin-documentation/). 
In the project folder data > examples there is a script called `download_s&p500_data.py` that uses yahoo_fin to download 
historical price and volume data for all <ins>current</ins> tickers in the S&P 500 (i.e. de-listed tickers are not included: 
*survivorship bias*) and saves the results in csv files.

Once the data is downloaded and stored (by running `download_s&p500_data.py`) we can start to setup the back test.

Store the price in a DataFrame (do the same for volume data): 
```
import pandas as pd

# get the data
price_file_path = "<full path to csv file>"
daily_adj_close_df = pd.read_csv(price_file_path, index_col=0)  # set the first column as an index
daily_adj_close_df.index = pd.to_datetime(daily_adj_close_df.index)
daily_adj_close_df.fillna(method='ffill', inplace=True)  
```

Setup the monthly calendar using the `BacktestCalendar` class. 

```
from strategy_building_blocks.backtest_calendar import BacktestCalendar

calendar = BacktestCalendar(
        rebalance_rules={
                'start_date': '1/1/2000',
                'days': -1,
                'frequency': 'monthly'
            },
        reweight_lag=1
    )
```

For the filters we will use the two filter classes `LiquidityFilter` and `PerformanceFilter`.

```
from strategy_building_blocks.filters import PerformanceFilter
from strategy_building_blocks.filters import LiquidityFilter

# filter by the average 20, 60 and 120-day performance and pick the top 25
momentum_filter = PerformanceFilter(observation_lag=[20, 60, 120], 
                                    filter_type='top', 
                                    value=25,
                                    aggregator_method='mean',
                                    price_df=daily_adj_close_df)

# filter the stocks that has an average 60-day liquidity (price x volume) larger than 1,000,000. 
# by default the filter_type attribute is set to 'larger'
liquidity_filter = LiquidityFilter(avg_lag=60, 
                                   liquidity_threshold=10_000_000,
                                   price_df=daily_adj_close_df, 
                                   volume_df=daily_volume_df)
                               
```

We want to take the *intersection* of these two filters (liquid stocks with high momentum). This is done by utilizing 
`FilterHandler` class and input a list of filters. A union of filters is represented by a tuple.

```
from strategy_building_blocks.filter_handler import FilterHandler

liq_mom_filter = FilterHandler(filter_collection=[liquidity_filter, momentum_filter])
```

For the equal weighting scheme we will use the `EqualWeight` class:

```
from strategy_building_blocks.weight import EqualWeight

eqw = EqualWeight()
```

Finally we can setup our strategy and run the back test using the `Strategy` class

```
from strategy_building_blocks.strategy import Strategy

strat = Strategy(instrument_price_df=daily_adj_close_df,
                 calendar=calendar,
                 weight=eqw,
                 strategy_filter=liq_mom_filter)
 
mom_bt_df = strat.run_backtest()            
```

As a starting point for our analysis we can look at the strategy's average annual return, realized volatility, 
risk-adjusted return aka "Sharpe ratio" and Maximum Drawdown (the worst point-to-point return you could have experienced). 
This table can be generated by running `return_risk_metrics_standard`.

```
from analysis.strategy_performance_analysis import return_risk_metrics_standard

result_df = return_risk_metrics_standard(price_df=mom_bt_df)
print(result_df)
```

Using data as of 6 February 2023, below is the result of the back test and S&P 500 index for the same period:
```
                         strategy    S&P 500
Avg. 1Y return (%)      23.580310   6.454363
Avg. 1Y volatility (%)  23.872361  17.829992
Sharpe ratio             0.987766   0.361995
Max Drawdown (%)       -46.412229 -56.775388
```

However, using the S&P 500 as a benchmark of our performance filter is not fair. First, the S&P 500 time series includes
companies that have been for, various reasons, de-listed. Since our data only includes stocks that are trading today, we are
only including the "winners", creating an upward bias in our back test. This is called *survivorship bias*.

A more fairer comparison would be to run our back test, all else equal except that we don't use the performance filter.
This can be done by re-running the back test with `strat.strategy_filter = liquidity_filter`.

````
                          MOM_EQW        EQW
Avg. 1Y return (%)      23.580310  16.482435
Avg. 1Y volatility (%)  23.872361  18.915180
Sharpe ratio             0.987766   0.871387
Max Drawdown (%)       -46.412229 -51.054810
````

So our momentum filter achieves higher annualized return with just slightly higher volatility (leading to a higher Sharpe ratio) 
with improved downside risk.

Ignoring explicit costs like commissions for now, we also need to take into account implicit costs like a bid/ask spread. Since 
we are using historical closing prices we can't assume that we are able to execute at the close but instead execute
at close adjusted by half of the bid/ask spread. Not only does this spread differ between stocks (Apple has lower bid/ask spread than a 
penny stock with low liquidity) but also across time with higher bid/ask spreads in times of stress.
 
For simplicity we assume that it is 0.15% for all stocks in our universe. Re-running the back test with `transaction_costs=0.0015` leads to the below result:

````
                          MOM_EQW        EQW
Avg. 1Y return (%)      21.312626  16.439120
Avg. 1Y volatility (%)  23.874989  18.914900
Sharpe ratio             0.892676   0.869110
Max Drawdown (%)       -49.514885 -51.079176
````

We still out perform our benchmark but the result is not as striking as just comparing to S&P 500.

The full script (with additional plotting and functions that saves and formats the results in excel) for the S&P 500 Momentum
strategy as well as other strategies can be found in the *examples* folder 

Licensed under [Apache License 2.0](LICENSE)







