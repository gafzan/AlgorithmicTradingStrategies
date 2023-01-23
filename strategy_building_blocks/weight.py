"""weight_old.py"""

from tools.dataframe_tools import cap_floor_df, aggregate_df
from analysis.financial_stats_signals_indicators import realized_volatility

import pandas as pd
import numpy as np

from abc import ABC, abstractmethod


class Weight(ABC):

    def __init__(self, eligibility_df: pd.DataFrame = None):
        self.eligibility_df = eligibility_df

    @abstractmethod
    def run(self):
        raise NotImplementedError("This method needs to be overridden by a sub-class")

    @staticmethod
    def get_weight_desc():
        return ''

    @property
    def eligibility_df(self):
        return self._filter_df

    @eligibility_df.setter
    def eligibility_df(self, eligibility_df: pd.DataFrame):
        # makes sure that eligibility_df is None or a DataFrame containing only 1, -1, 0 and nan
        if isinstance(eligibility_df, pd.DataFrame):
            allowed_filter_val = [1, -1, 0, np.nan]
            if all(e in [1, -1, 0] or np.isnan(e) for e in np.unique(eligibility_df)):
                self._filter_df = eligibility_df.fillna(0)
            else:
                raise ValueError("the values in 'eligibility_df' can only be %s" % ", ".join([str(e) for e in allowed_filter_val]))
        elif eligibility_df is None:
            self._filter_df = eligibility_df
        else:
            raise ValueError("'eligibility_df' can only be specified as DataFrame")

    def __repr__(self):
        return "<{}({})>".format(type(self).__name__, self.get_weight_desc())


class CustomFixedWeight(Weight):
    """Definition of CustomFixedWeight"""

    def __init__(self, weight_map: dict = None, *args, **kwargs):
        """
        Setup weights based on a custom map of weights
        :param weight_map: dict with instrument names as keys and weights as values
        Example: assuming you want to have 60% / 40% SPY and VGLT the map can be written as
            {
                'SPY': 0.6,
                'VGLT': 0.4
            }
        :param args:
        :param kwargs:
        """
        super().__init__(*args, *kwargs)
        self.weight_map = weight_map

    def run(self):
        if self.eligibility_df is None:
            raise ValueError("'eligibility_df' is not specified")

        weight_df = pd.DataFrame(self.weight_map, index=self.eligibility_df.index)
        weight_df = weight_df.reindex_like(self.eligibility_df)
        weight_df *= self.eligibility_df
        return weight_df


class SignalBasedWeight(Weight):
    """Definition of SignalWeight"""

    def __init__(self, signal_time_series: {pd.Series, pd.DataFrame}=None, signal_weight_map: dict = None, *args, **kwargs):
        """
        Weights are chosen based on a signal and a corresponding signal->weight mapper (dict)
        :param signal_time_series: DataFrame or Series
        :param signal_weight_map: nested dict with the signal values as first level of keys and instrument name - weight
        dict as values. The instrument name - weight dict has instrument names as keys and weights as values.

        Example: Assume you have a signal that has 'bull' and 'bear' as the only signals and you want to go long 100%
        'SPY' if the signal is 'bull', else go 50%/50% 'SPY' and 'VGLT' if 'bear'.
        The mapping would then be:
            {
                'bull': {
                            'SPY': 1.0
                        },
                'bear': {
                            'SPY': 0.5,
                            'VGLT': 0.5
                        }
            }
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.signal_time_series = signal_time_series
        self.signal_weight_map = signal_weight_map

    def run(self):
        """
        Returns a DataFrame with weights that are based on a signal time series and a mapping (dict) between the signal
        values and the weights per instrument
        :return: DataFrame
        """
        weight_df = self._get_weights_from_signal()
        if self.eligibility_df is not None:
            weight_df = weight_df.reindex_like(self.eligibility_df[weight_df.columns], method='ffill').reindex_like(self.eligibility_df)
            weight_df *= self.eligibility_df.values
        return weight_df

    def _get_weights_from_signal(self):
        """
        Returns a DataFrame with dates as index, instrument names as column headers and weights as values
        :return: DataFrame
        """

        self._check_signal()

        # first map the dictionary per signal value to each row then split these dictionaries using json_normalize()
        # splits the dictionary into columns with weights
        mapped_weights_df = pd.json_normalize(
            self.signal_time_series.map(self.signal_weight_map)
        )
        mapped_weights_df.index = self.signal_time_series.index
        return mapped_weights_df

    def _check_signal(self):
        if self.signal_time_series is None:
            raise ValueError("'signal_time_series' needs to be specified")
        elif isinstance(self.signal_time_series, pd.DataFrame):
            if self.signal_time_series.shape[1] > 1:
                raise ValueError(f"'signal_time_series' is a DataFrame with {self.signal_time_series[1]} columns "
                                 f"(can only have 1 column).")


class _ProportionalWeight(Weight):
    """Definition of _ProportionalWeight"""

    def __init__(self, weight_sum: float = 1, net_zero: bool = False, weight_cap: float = None, weight_floor: float = None,
                 inverse: bool = False, *args, **kwargs):
        """
        Weighting class for calculating proportional weights based on data. A subclass needs to overwrite the abstract
        method get_data().
        :param weight_sum: float the sum of all the component weights
        :param net_zero: bool only applicable when there are short positions. If net_zero is true, then the sum of all
        long and short positions is zero. Else the weights will not depend on if it is a short or long position
        :param weight_cap: float the highest allowed weight per instrument
        :param weight_floor: float the lowest allowed weight per instrument
        :param inverse: bool inverse proportionally if True
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.weight_sum = weight_sum
        self.net_zero = net_zero
        self.weight_cap = weight_cap
        self.weight_floor = weight_floor
        self.inverse = inverse

    def run(self)->pd.DataFrame:
        """
        Retrieves the data, takes the inverse if applicable, calculates the weights applies cap and floor
        :return: DataFrame
        """
        # get the data used for proportional weighting
        df = self.get_data()
        if self.eligibility_df is not None:
            df = df.reindex_like(self.eligibility_df[df.columns], method='ffill').reindex_like(self.eligibility_df)
            df *= self.eligibility_df.values

        if self.inverse:
            df = df.apply(lambda x: 1 / x)  # inverse of all values
            df.replace([np.nan, np.Inf, -np.Inf], 0, inplace=True)

        # calculate the weights
        weight_df = self._calculate_proportional_weights(data_df=df)

        # check cap and floor
        # TODO allocate the excess below floor and above cap to the other weights (short positions this might not make
        #  sense)
        if self.weight_floor or self.weight_cap:
            weight_df = cap_floor_df(df=weight_df, floor=self.weight_floor, cap=self.weight_cap)
        return weight_df

    @abstractmethod
    def get_data(self):
        raise NotImplementedError("This method needs to be overridden by a sub-class")

    def _calculate_proportional_weights(self, data_df: pd.DataFrame):
        if self.net_zero:
            # the sum of all the long and short positions (if any) are zero
            pos_col_sum = data_df[data_df > 0.0].sum(axis=1)
            neg_col_sum = data_df[data_df < 0.0].sum(axis=1).abs()
            weight_df = data_df.copy().replace(np.nan, 0)
            weight_df[weight_df > 0.0] = weight_df[weight_df > 0.0].divide(pos_col_sum, axis=0)
            weight_df[weight_df < 0.0] = weight_df[weight_df < 0.0].divide(neg_col_sum, axis=0)
        else:
            # weight row i, col j = value i, j divided by the sum of row i
            weight_df = data_df.div(data_df.abs().sum(axis=1), axis=0)
        weight_df *= self.weight_sum
        return weight_df


class ProportionalWeight(_ProportionalWeight):
    """ProportionalWeight"""

    def __init__(self, data_df: pd.DataFrame = None, *args, **kwargs):
        """
        Proportional weight based on specified DataFrame
        :param data_df: DataFrame
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.data_df = data_df

    def get_data(self):
        if self.data_df is None:
            raise ValueError("'data_df' is not specified")
        return self.data_df


class EqualWeight(_ProportionalWeight):
    """Definition of EqualWeight"""

    def get_data(self):
        return self.eligibility_df.abs()

    def get_weight_desc(self):
        return f'sum={self.weight_sum}, net zero={self.net_zero}'


class FinancialSignalWeight(_ProportionalWeight):
    """Definition of FinancialSignalWeight"""

    def __init__(self, fill_na_method: str = None, columnwise_drop_na: bool = False, *args, **kwargs):
        """
        Additional attributes and methods for weights based on financial signals
        :param fill_na_method: str method to fill na for the financial signal before calculating weights
        :param columnwise_drop_na: bool if True, drop nan for each column of input to the financial signal
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.fill_na_method = fill_na_method
        self.columnwise_drop_na = columnwise_drop_na

    def get_data(self):
        """
        Get the financial signal input and compute the signal DataFrame
        :return: DataFrame
        """
        input_df = self.get_financial_signal_input_df()
        if self.columnwise_drop_na:
            raise ValueError("'columnwise_drop_na' set to True leads to error (work in progress...)")
            df = input_df.apply(lambda ts: self.compute_signal(ts.dropna()))
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


class VolatilityWeight(FinancialSignalWeight):

    def __init__(self, vol_lag: {int, list}, price_df: pd.DataFrame = None, log_normal_returns: bool = False,
                 annualising_factor: int = 252, with_mean: bool = True, aggregator_method: str = 'max',
                 inverse: bool = True, *args, **kwargs):
        """
        Calculates weights that are proportional to volatility
        :param vol_lag: int or list of int
        :param price_df: DataFrame
        :param aggregator_method: str needs to be specified when 'observation_lag' is an iterable
        :param log_normal_returns: bool if True use log normal returns else arithmetic returns
        :param annualising_factor: int (by default 252)
        :param with_mean: bool (some formulas for volatility does not adjust the return before squaring)
        :param inverse: bool (default True)
        :param args:
        :param kwargs:
        """
        super().__init__(inverse=inverse, *args, **kwargs)
        self.vol_lag = vol_lag
        self.price_df = price_df
        self.aggregator_method = aggregator_method
        self.log_normal_returns = log_normal_returns
        self.annualising_factor = annualising_factor
        self.with_mean = with_mean

    def compute_signal(self, df: pd.DataFrame)->pd.DataFrame:
        """
        Calculates the rolling realized volatility of the given price DataFrame
        :return: DataFrame
        """

        if isinstance(self.vol_lag, int):
            return realized_volatility(price_df=df,
                                       lag=self.vol_lag,
                                       log_normal_returns=self.log_normal_returns,
                                       annualising_factor=self.annualising_factor,
                                       with_mean=self.with_mean)
        else:
            # assumes 'avg_lag' is a tuple or list of int so loop through each avg_lag and aggregate the results into
            # one DataFrame using an aggregator method e.g. 'max'
            signal_df_list = []
            for lag in self.vol_lag:
                signal_df_list.append(
                    realized_volatility(price_df=df,
                                        lag=lag,
                                        log_normal_returns=self.log_normal_returns,
                                        annualising_factor=self.annualising_factor,
                                        with_mean=self.with_mean)
                )
            return aggregate_df(df_list=signal_df_list, agg_method=self.aggregator_method)

    def get_financial_signal_input_df(self)->pd.DataFrame:
        """
        Returns the price DataFrame to be used to calculate the realized volatility
        :return: DataFrame
        """
        return self.price_df

    def get_weight_desc(self):
        return f'sum={self.weight_sum}, net zero={self.net_zero}'

