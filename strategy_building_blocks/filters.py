"""filters_old.py"""

from tools.dataframe_tools import rank_filter_df
from tools.dataframe_tools import comparative_filter_df
from tools.dataframe_tools import winsorize_dataframe
from tools.dataframe_tools import aggregate_df
from analysis.financial_stats_signals_indicators import simple_moving_average
from analysis.financial_stats_signals_indicators import beta
from analysis.financial_stats_signals_indicators import realized_volatility
from analysis.financial_stats_signals_indicators import relative_strength_index


import pandas as pd
import numpy as np
from abc import ABC, abstractmethod


class BacktestFilter(ABC):
    """
    This is an abstract class i.e. a blueprint for other sub-classes. This is to provide a common interface for
    different implementations. An abstract method are methods that must be overridden inside the sub-class definition.
    """

    def __init__(self, previous_filter_df: pd.DataFrame = None):
        self.previous_filter_df = previous_filter_df

    def adjust_input_df_before_filtering(self, input_df: pd.DataFrame = None):
        if self.previous_filter_df is not None:
            # forward fill the values from the previous filter
            # does not need to assume the same calendar and column names
            prev_filter_df = self.previous_filter_df.reindex_like(input_df[self.previous_filter_df.columns],
                                                                  method="ffill").reindex_like(input_df)
            # unless the value in the previous DataFrame is null, multiply the input DataFrame value by 1
            input_df *= np.where(prev_filter_df.isnull(), np.nan, 1)
        return input_df

    @abstractmethod
    def run(self):
        raise NotImplementedError("This method needs to be overridden by a sub-class")

    @staticmethod
    def get_filter_desc():
        return ''

    def __repr__(self):
        return "<{}({})>".format(type(self).__name__, self.get_filter_desc())


class PriceAvailabilityFilter(BacktestFilter):
    """Class definition of PriceAvailabilityFilter
    For a given window N, the filter returns 1 if the previous N values are numbers, else nan
    """

    def __init__(self, price_df: pd.DataFrame = None, window: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.price_df = price_df
        self.window = window

    def run(self)->pd.DataFrame:

        if self.price_df is not None:
            price_df = self.adjust_input_df_before_filtering(input_df=self.price_df)
        else:
            raise ValueError("'price_df' not assigned")
        # calculate a moving average over the given observation window. if there is a value then the prices are
        # available i.e. set value to 1, else nan
        filter_result_df = pd.DataFrame(np.where(price_df.rolling(window=self.window).mean().isnull(), np.nan, 1),
                                        columns=price_df.columns, index=price_df.index)
        return filter_result_df


class NumericRankingComparisonFilter(BacktestFilter):
    """Class definition of NumericRankingComparisonFilter"""

    def __init__(self, numeric_df: pd.DataFrame = None, filter_type: str=None, value: {int, float}=None, inclusive: bool = True, or_equal: bool = True,
                 winsorize_upper_pct: float = None, winsorize_lower_pct: float = None, *args, **kwargs):
        """
        Calls run() to perform a numeric ranking filter
        :param numeric_df: DataFrame
        :param filter_type: the type of filter to apply
            'top': the 'value'th highest elements
            'bottom': the 'value'th bottom elements
            'larger': elements larger than 'value'
            'smaller': elements smaller than 'value'
        :param value: int or float defines the threshold rule like top 10, bottom 5% or larger than 100
        :param inclusive: filter in or out?
        :param or_equal: bool '>' vs. '=>' e.g. should you include or exclude the 10th largest in the top 10?
        :param winsorize_upper_pct: float upper percentile when winsorizing the data before ranking and filtering
        :param winsorize_lower_pct: float lower percentile when winsorizing the data before ranking and filtering
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.numeric_df = numeric_df
        self.filter_type = filter_type
        self.value = value
        self.inclusive = inclusive
        self.or_equal = or_equal
        self.winsorize_upper_pct = winsorize_upper_pct
        self.winsorize_lower_pct = winsorize_lower_pct

    def winsorize_data(self, df: pd.DataFrame):
        lower_pct = self.winsorize_lower_pct if self.winsorize_lower_pct else 0
        upper_pct = self.winsorize_upper_pct if self.winsorize_upper_pct else 1
        return winsorize_dataframe(df=df, lower_pct=lower_pct, upper_pct=upper_pct)

    def run(self):

        if self.value is None:
            raise ValueError("'value' needs to be specified. If you are using a new class that inherits from "
                             "NumericRankingComparisonFilter make sure that you define a new variable and use a "
                             "get setter property to map to 'value'")
        if self.filter_type is None:
            raise ValueError("'filter_type' not specified.\n'top': the 'value'th highest elements\n'bottom': the "
                             "'value'th bottom elements\n'larger': elements larger than 'value'\n'smaller': elements "
                             "smaller than 'value'")

        df = self.get_data()
        df = self.adjust_input_df_before_filtering(input_df=df)

        if self.winsorize_lower_pct or self.winsorize_upper_pct:
            df = self.winsorize_data(df=df)

        if self.filter_type in ['top', 'bottom']:
            filter_result_df = rank_filter_df(df=df, filter_type=self.filter_type, inclusive=self.inclusive,
                                              rank_threshold=self.value, or_equal=self.or_equal)
        elif self.filter_type in ['larger', 'smaller']:
            filter_result_df = comparative_filter_df(df=df, filter_type=self.filter_type,
                                                     inclusive=self.inclusive,
                                                     threshold=self.value, or_equal=self.or_equal)
        else:
            raise ValueError("'filter_type' needs to be equal to 'top', 'bottom', 'larger', 'smaller'")

        return filter_result_df

    def get_data(self):
        if self.numeric_df is not None:
            return self.numeric_df
        else:
            raise ValueError("'numeric_df' is not specified")

    @property
    def filter_type(self):
        return self._filter_type

    @filter_type.setter
    def filter_type(self, filter_type: str):
        eligible_filter_types = ['top', 'bottom', 'larger', 'smaller']
        if filter_type is None or filter_type.lower() in eligible_filter_types:
            self._filter_type = filter_type
        else:
            raise ValueError("'filter_type' needs to be specified as '%s'" % "', '".join(eligible_filter_types))


class FinancialSignalFilter(NumericRankingComparisonFilter):

    def __init__(self, fill_na_method: str = None, columnwise_drop_na: bool = False, *args, **kwargs):
        super().__init__(numeric_df=None, *args, **kwargs)
        self.fill_na_method = fill_na_method
        self.columnwise_drop_na = columnwise_drop_na

    def get_data(self):
        """
        Returns a DataFrame that will be used for ranking/comparison. First an input DataFrame is retrieved using
        get_financial_signal_input_df and a signal is computed (stored in a DataFrame) and results are cleaned for nan
        if applicable.
        :return: DataFrame
        """
        input_df = self.get_financial_signal_input_df()
        if input_df is None:
            raise ValueError("the input used for the financial signal is None. Make sure all required data attributes "
                             "are specified")
        if self.columnwise_drop_na:
            raise ValueError("'columnwise_drop_na' set to True leads to error (work in progress...)")
            df = input_df.apply(lambda ts: self.compute_signal(ts.dropna()).reindex(input_df.index))
        else:
            df = self.compute_signal(df=input_df)
        if self.fill_na_method:
            df.fillna(method=self.fill_na_method, inplace=True)
        return df

    @abstractmethod
    def get_financial_signal_input_df(self)->pd.DataFrame:
        raise NotImplementedError("This method needs to be overridden by a sub-class")

    @abstractmethod
    def compute_signal(self, df: pd.DataFrame)->pd.DataFrame:
        raise NotImplementedError("This method needs to be overridden by a sub-class")


class LiquidityFilter(FinancialSignalFilter):
    """Class definition of LiquidityFilter"""

    def __init__(self, avg_lag: {int, list, tuple}, liquidity_threshold: float, liquidity_df: pd.DataFrame = None, price_df: pd.DataFrame = None,
                 volume_df: pd.DataFrame = None, aggregator_method: str = 'min', filter_type: str = 'larger', *args, **kwargs):
        """
        Performs a liquidity filter i.e. filter outs instruments with too low liquidity
        :param avg_lag: int window for the rolling average
        :param liquidity_df: DataFrame (if not specified, both 'price_df' and 'volume_df' needs to be specified
        :param price_df: DataFrame
        :param volume_df: DataFrame
        :param aggregator_method: str method to aggregate the results when a list or tuples of avg_lag has been specified
        :param filter_type: the type of filter to apply
            'top': the 'value'th highest elements
            'bottom': the 'value'th bottom elements
            'larger': elements larger than 'value'
            'smaller': elements smaller than 'value'
        :param args:
        :param kwargs:
        """
        super().__init__(filter_type=filter_type, *args, **kwargs)
        self.avg_lag = avg_lag
        self.liquidity_threshold = liquidity_threshold
        self.liquidity_df = liquidity_df
        self.price_df = price_df
        self.volume_df = volume_df
        self.aggregator_method = aggregator_method

    def compute_signal(self, df: pd.DataFrame)->pd.DataFrame:
        """
        Calculates a rolling average of liquidity
        :return: DataFrame
        """

        if isinstance(self.avg_lag, int):
            return simple_moving_average(df=df, lag=self.avg_lag)
        else:
            # assumes 'avg_lag' is a tuple or list of int so loop through each avg_lag and aggregate the results into
            # one DataFrame using an aggregator method e.g. 'max'
            signal_df_list = []
            for lag in self.avg_lag:
                signal_df_list.append(
                    simple_moving_average(df=df, lag=lag)
                )
            return aggregate_df(df_list=signal_df_list, agg_method=self.aggregator_method)

    def get_financial_signal_input_df(self)->pd.DataFrame:
        """
        Returns the liquidity if specified or calculate the product of volume and price as a DataFrame
        :return: DataFrame
        """
        if self.liquidity_df is None:
            # make the volume have the same calendar as prices
            vm_df = self.volume_df.reindex(index=self.price_df.index, columns=self.price_df.columns, fill_value=np.nan)
            liq_df = self.price_df * vm_df.values
        else:
            liq_df = self.liquidity_df
        return liq_df

    @property
    def liquidity_threshold(self):
        return self.value

    @liquidity_threshold.setter
    def liquidity_threshold(self, liquidity_threshold: float):
        self.value = liquidity_threshold


class PerformanceFilter(FinancialSignalFilter):
    """Definition of PerformanceFilter"""

    def __init__(self, observation_lag: {int, tuple, list}, filter_type: str, value: float, price_df: pd.DataFrame = None,
                 aggregator_method: str = 'mean', *args, **kwargs):
        """
        Runs a filter based on the performance i.e. Underlying[t2] / Underlying[t1]
        :param observation_lag: int, tuple or list
        :param filter_type: the type of filter to apply
            'top': the 'value'th highest elements
            'bottom': the 'value'th bottom elements
            'larger': elements larger than 'value'
            'smaller': elements smaller than 'value'
        :param value: int or float defines the threshold rule like top 10, bottom 5% or larger than 100
        :param price_df: DataFrame
        :param aggregator_method: str needs to be specified when 'observation_lag' is an iterable
        :param args:
        :param kwargs:
        """
        super().__init__(filter_type=filter_type, value=value, *args, **kwargs)
        self.observation_lag = observation_lag
        self.price_df = price_df
        self.aggregator_method = aggregator_method

    def compute_signal(self, df: pd.DataFrame):
        """
        Computes the rolling performance
        :param df: DataFrame assumes that it is a price DataFrame
        :return: DataFrame
        """
        if isinstance(self.observation_lag, int):
            return self.calculate_perf(df=df, lag=self.observation_lag)
        else:
            # assumes 'observation_lag' is a tuple or list of int so loop through each avg_lag and aggregate the results
            # into one DataFrame using an aggregator method e.g. 'mean'
            signal_df_list = []
            for lag in self.observation_lag:
                signal_df_list.append(
                    self.calculate_perf(df=df, lag=lag)
                )
            return aggregate_df(df_list=signal_df_list, agg_method=self.aggregator_method)

    def get_financial_signal_input_df(self)->pd.DataFrame:
        return self.price_df

    @staticmethod
    def calculate_perf(df: pd.DataFrame, lag: int)->pd.DataFrame:
        return 1 + df.pct_change(periods=lag, fill_method=None)


class SimpleMovingAverageCrossFilter(FinancialSignalFilter):
    """Definition of SimpleMovingAverageCrossFilter"""

    def __init__(self, price_df: pd.DataFrame, lagging_window: int = None, leading_window: int = 1,
                 normalize_sma: bool = True, use_sma_ratio: bool = False, *args, **kwargs):
        """
        Performs a ranking/comparison filter of the difference
        :param price_df:
        :param lagging_window: int
        :param leading_window: int
        :param normalize_sma: bool if True, the SMA is normalized by the latest spot
        :param use_sma_ratio: bool if True, the indicator is SMA_leading / SMA_lagging instead of SMA_leading - SMA_lagging
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.price_df = price_df
        self.lagging_window = lagging_window
        self.leading_window = leading_window
        self.relative = normalize_sma
        self.use_sma_ratio = use_sma_ratio

    def get_financial_signal_input_df(self)->pd.DataFrame:
        return self.price_df

    def compute_signal(self, df: pd.DataFrame):
        """
        Calculates the signal as the distance between the leading and lagging SMA where each SMA is relative to the spot
        :param df: DataFrame
        :return: DataFrame
        """
        if self.lagging_window is None:
            raise ValueError("'lagging_window' is not specified")
        lead_sma_df = simple_moving_average(df=df, lag=self.leading_window, relative=self.relative)
        lag_sma_df = simple_moving_average(df=df, lag=self.lagging_window, relative=self.relative)
        if self.use_sma_ratio:
            return lead_sma_df / lag_sma_df
        else:
            return lead_sma_df - lag_sma_df


class RelativeStrengthIndexFilter(FinancialSignalFilter):
    """Definition of RelativeStrengthIndexFilter"""

    def __init__(self, value: float = 0.3, filter_type: str = 'smaller', lag: int = 14, ewma: bool = True,
                 price_df: pd.DataFrame = None, *args, **kwargs):
        """
        Filter based on Relative Strength Index
        :param value: float
        :param filter_type: the type of filter to apply
            'top': the 'value'th highest elements
            'bottom': the 'value'th bottom elements
            'larger': elements larger than 'value'
            'smaller': elements smaller than 'value'
        :param lag: int (default 14)
        :param ewma: bool if True use exponential smoothing (same method as www.tradingview.com)
        :param price_df: DataFrame
        :param args:
        :param kwargs:
        """
        super().__init__(value=value, filter_type=filter_type, *args, **kwargs)
        self.lag = lag
        self.ewma = ewma
        self.price_df = price_df

    def compute_signal(self, df: pd.DataFrame):
        return relative_strength_index(price_df=df, lag=self.lag, ewma=self.ewma)

    def get_financial_signal_input_df(self):
        return self.price_df


class VolatilityFilter(FinancialSignalFilter):
    """Definition of VolatilityFilter"""

    def __init__(self, vol_lag: {int, tuple, list}, value: float, filter_type: str = 'bottom',
                 log_normal_returns: bool = False, annualising_factor: int = 252, with_mean: bool = False,
                 price_df: pd.DataFrame = None, aggregator_method: str = 'max', *args, **kwargs):
        """
        Filter based on realized volatility
        :param vol_lag: int, list or tuple
        :param value: float
        :param filter_type: the type of filter to apply
            'top': the 'value'th highest elements
            'bottom': the 'value'th bottom elements
            'larger': elements larger than 'value'
            'smaller': elements smaller than 'value'
        :param log_normal_returns: bool
        :param annualising_factor: int
        :param with_mean: bool
        :param price_df: DataFrame
        :param aggregator_method: str
        :param args:
        :param kwargs:
        """
        super().__init__(value=value, filter_type=filter_type, *args, **kwargs)
        self.vol_lag = vol_lag
        self.log_normal_returns = log_normal_returns
        self.price_df = price_df
        self.aggregator_method = aggregator_method
        self.annualising_factor = annualising_factor
        self.with_mean = with_mean

    def get_financial_signal_input_df(self)->pd.DataFrame:
        return self.price_df

    def compute_signal(self, df: pd.DataFrame):
        if isinstance(self.vol_lag, int):
            return realized_volatility(price_df=df, lag=self.vol_lag, log_normal_returns=self.log_normal_returns,
                                       annualising_factor=self.annualising_factor, with_mean=self.with_mean)
        else:
            # assumes 'vol_lag' is a tuple or list of int so loop through each avg_lag and aggregate the results
            # into one DataFrame using an aggregator method e.g. 'mean'
            signal_df_list = []
            for lag in self.vol_lag:
                signal_df_list.append(
                    realized_volatility(price_df=df, lag=lag, log_normal_returns=self.log_normal_returns,
                                        annualising_factor=self.annualising_factor, with_mean=self.with_mean)
                )
            return aggregate_df(df_list=signal_df_list, agg_method=self.aggregator_method)


class BetaFilter(FinancialSignalFilter):
    """Definition of BetaFilter"""

    def __init__(self, beta_calc_lag: {int, tuple, list}, filter_type: str, value: float,
                 log_normal_returns: bool = False, price_df: pd.DataFrame = None,
                 benchmark_price: {pd.Series, pd.DataFrame}=None, aggregator_method: str = 'mean', *args, **kwargs):
        """
        Filter based on beta between the instruments and a specified benchmark time series
        :param beta_calc_lag: int observation window for the beta calculation
        :param filter_type: str
        :param value: float
        :param log_normal_returns: bool
        :param price_df: DataFrame
        :param benchmark_price: DataFrame or Series
        :param aggregator_method: str
        :param args:
        :param kwargs:
        """
        super().__init__(filter_type=filter_type, value=value, *args, **kwargs)
        self.beta_calc_lag = beta_calc_lag
        self.log_normal_returns = log_normal_returns
        self.price_df = price_df
        self.benchmark_price = benchmark_price
        self.aggregator_method = aggregator_method

    def get_financial_signal_input_df(self)->pd.DataFrame:
        return self.price_df

    def compute_signal(self, df: pd.DataFrame):
        """
        Calculates the rolling beta as the ratio of the covariance and variance. Covariance is calculated on the returns of
        the given price DataFrame and the benchmark. Variance is calculate on the returns of the benchmark. Note that there
        is no lag here for the return calculation. In case one needs to look at the beta for weekly returns one needs to
        first convert the frequency of the price DataFrame of to weekly and then use it as an input into this function.
        :param df: DataFrame
        :return: DataFrame
        """
        benchmark_price = self.benchmark_price.copy()
        if self.columnwise_drop_na:
            raise ValueError("'columnwise_drop_na' set to True leads to error (work in progress...)")
            # make it so that only calculate the betas when both the benchmark and the instruments have prices
            # adjust benchmark
            benchmark_price = benchmark_price.reindex(df.index)
            benchmark_price.dropna(inplace=True)

            # adjust price of instruments
            df = df.reindex(benchmark_price.index)

        if isinstance(self.beta_calc_lag, int):
            beta_df = beta(price_df=df, benchmark_price=benchmark_price, lag=self.beta_calc_lag, log_normal_returns=self.log_normal_returns)
            return beta_df
        else:
            # assumes 'beta_calc_lag' is a tuple or list of int so loop through each avg_lag and aggregate the results
            # into one DataFrame using an aggregator method e.g. 'mean'
            signal_df_list = []
            for lag in self.beta_calc_lag:
                signal_df_list.append(
                    beta(price_df=df, benchmark_price=benchmark_price, lag=lag,
                         log_normal_returns=self.log_normal_returns)
                )
            return aggregate_df(df_list=signal_df_list, agg_method=self.aggregator_method)

