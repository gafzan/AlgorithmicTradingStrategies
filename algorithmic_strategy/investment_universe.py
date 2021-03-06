"""
investment_universe.py
"""
import pandas as pd
from pandas.tseries.offsets import BDay
import pandas.core.common as com
import numpy as np

# my modules
from database.financial_database import FinancialDatabase
from database.config_database import __MY_DATABASE_NAME__
from financial_analysis.finance_tools import rolling_average
from dataframe_tools import merge_two_dataframes_as_of


class InvestmentUniverse:
    """Class definition for InvestmentUniverse"""

    def __init__(self, tickers: {str, list, tuple}, start=None, end=None, periods=None, freq=None,
                 observation_calendar: pd.DatetimeIndex = None):
        if com.count_not_none(start, end, periods, freq) != 0:
            self._observation_calendar = pd.date_range(start, end, periods, freq)
        else:
            if observation_calendar is None:
                raise ValueError('Need to specify observation_calendar or the parameters of an observation calendar i.e'
                                 'start, end, periods, freq')
            elif observation_calendar.is_monotonic_increasing:
                self._observation_calendar = observation_calendar
            else:
                raise ValueError('observation_calendar needs to be an instance of a DatatimeIndex object that is '
                                 'monotonic increasing')

        self.tickers = tickers
        self._financial_database_handler = FinancialDatabase(__MY_DATABASE_NAME__)
        self._filter_has_been_applied = False
        self._filter_desc_list = []

    def get_start_end_dates(self):
        return min(self._observation_calendar), max(self._observation_calendar)

    def apply_custom_filter(self, custom_eligibility_df: pd.DataFrame, filter_desc: str = 'custom filter'):
        if list(custom_eligibility_df) != self.tickers:
            raise ValueError('Column headers (i.e. tickers) are not the same.'
                             '\nTickers in current investment universe: %s' % ', '.join(self.tickers)
                             + '\nTickers in custom filter: %s' % ', '.join(list(custom_eligibility_df)))
        elif not (custom_eligibility_df.index.is_monotonic_increasing and isinstance(custom_eligibility_df.index, pd.DatetimeIndex)):
            raise ValueError('Index needs to be a monotonically increasing DatetimeIndex.')
        self._apply_dataframe(custom_eligibility_df, filter_desc)

    def _apply_dataframe(self, df: pd.DataFrame, filter_desc: str):
        self._filter_desc_list.append(filter_desc)
        # merge (as of) the new filter to the current observation calendar
        new_filter = merge_two_dataframes_as_of(pd.DataFrame(index=self.observation_calendar), df)
        if self._filter_has_been_applied:
            self._eligibility_df = self._eligibility_df * new_filter.values
        else:
            self._eligibility_df = new_filter
        self._filter_has_been_applied = True

    def get_eligible_tickers(self) -> list:
        """
        Return a list with all tickers that has at least one 1 in their eligibility column i.e. the stocks that has
        passed the filters at least once.
        :return: list
        """
        stock_is_eligible_df = pd.DataFrame(data=self._eligibility_df.sum().gt(0), index=list(self._eligibility_df),
                                            columns=['eligibility'])
        return list(stock_is_eligible_df[stock_is_eligible_df['eligibility']].index)

    # ------------------------------------------------------------------------------------------------------------------
    # filter methods
    def apply_liquidity_filter(self, avg_lag: int, liquidity_threshold: float, currency: str = None):
        if avg_lag < 1:
            raise ValueError('avg_lag needs to be an int larger or equal to 1.')
        start_date, end_date = self.get_start_end_dates()
        liquidity_data = self._financial_database_handler.get_liquidity_df(self.tickers, start_date - BDay(avg_lag + 10), end_date, currency)
        avg_liquidity = rolling_average(liquidity_data, avg_lag)
        liquidity_eligibility = pd.DataFrame(data=np.where(avg_liquidity > liquidity_threshold, 1, 0),
                                             index=avg_liquidity.index,
                                             columns=avg_liquidity.columns)
        if currency is None:
            currency = ''
        self._apply_dataframe(liquidity_eligibility, '{} day avg. liquidity > {} {}'.format(avg_lag, currency.upper(), liquidity_threshold))

    def apply_close_price_history_filter(self, minimum_number_consecutive_published_prices: int, tolerance: float = 0.95):
        closing_price_data = self._get_closing_price_data(lag=minimum_number_consecutive_published_prices)

        # is NaN only when there is less than minimum_number_consecutive_published_prices x tolerance available prices
        rolling_avg_df = closing_price_data.rolling(window=minimum_number_consecutive_published_prices,
                                                    min_periods=int(tolerance * minimum_number_consecutive_published_prices)).mean()
        price_history_eligibility = pd.DataFrame(np.where(rolling_avg_df.isna(), 0, 1), index=rolling_avg_df.index,
                                                 columns=rolling_avg_df.columns)
        self._apply_dataframe(price_history_eligibility, '{}% of prices has been published for the past {} days'
                              .format(tolerance * 100, minimum_number_consecutive_published_prices))

    def apply_published_close_price_filter(self, max_number_days_since_publishing: int):
        closing_price_data = self._get_closing_price_data(lag=max_number_days_since_publishing)
        # first avg is calculated to check the availability at the start of the data (in case you observe at the start
        # of the available data)
        strict_rolling_avg_df = closing_price_data.rolling(window=max_number_days_since_publishing).mean()
        strict_rolling_avg_df.fillna(method='ffill', inplace=True)
        rolling_avg_df = closing_price_data.rolling(window=max_number_days_since_publishing, min_periods=1).mean()  # is NaN only when there is not a single value within the given period
        rolling_avg_df *= strict_rolling_avg_df.values
        price_availability_eligibility = pd.DataFrame(np.where(rolling_avg_df.isna(), 0, 1), index=rolling_avg_df.index,
                                                      columns=rolling_avg_df.columns)
        self._apply_dataframe(price_availability_eligibility, 'price published for the past {} days.'.format(max_number_days_since_publishing))

    # ------------------------------------------------------------------------------------------------------------------
    # get setter methods
    def get_eligibility_df(self, only_eligibile_tickers: bool = False):
        if self._filter_has_been_applied:
            if only_eligibile_tickers:
                eligible_tickers = self.get_eligible_tickers()
                if not len(eligible_tickers):
                    raise ValueError('No tickers passed the filter: %s' % ', '.join(self._filter_desc_list))
                return self._eligibility_df[eligible_tickers].replace(0, np.nan)
            else:
                return self._eligibility_df.replace(0, np.nan)
        else:
            raise ValueError('No filter has been applied yet.')

    def _get_closing_price_data(self, lag: int) -> pd.DataFrame:
        if lag < 1:
            raise ValueError('lag when loading prices needs to be an int larger or equal to 1.')
        start_date, end_date = self.get_start_end_dates()
        closing_price_data = self._financial_database_handler.get_close_price_df(self.tickers, start_date - BDay(lag + 10), end_date)
        return closing_price_data

    def _get_liquidity_data(self, lag: int, currency: {str, None}) -> pd.DataFrame:
        if lag < 1:
            raise ValueError('lag when loading liquidity needs to be an int larger or equal to 1.')
        start_date, end_date = self.get_start_end_dates()
        liquidity_data = self._financial_database_handler.get_liquidity_df(self.tickers, start_date - BDay(lag + 10), end_date, currency)
        return liquidity_data

    @property
    def observation_calendar(self):
        return self._observation_calendar

    @observation_calendar.setter
    def observation_calendar(self, observation_calendar: pd.DatetimeIndex):
        """
        Check if the observation calendar is monotonically increasing. Reset the eligibility DataFrame.
        :param observation_calendar:DatetimeIndex
        :return: None
        """
        if observation_calendar.is_monotonic_increasing and isinstance(observation_calendar, pd.DatetimeIndex):
            self._observation_calendar = observation_calendar
            self._eligibility_df = pd.DataFrame(columns=self._tickers, index=observation_calendar)
            self._filter_desc_list = []
        else:
            ValueError('observation_calendar needs to be a DatetimeIndex that is monotonic increasing.')

    @property
    def tickers(self):
        return self._tickers

    @tickers.setter
    def tickers(self, tickers: {str, list, tuple}):
        """
        Convert to list if ticker is str. Reset the eligibility DataFrame.
        :param tickers: str, list, tuple
        :return:
        """
        if isinstance(tickers, str):
            tickers = [tickers]
        elif type(tickers) not in [list, tuple]:
            raise ValueError('tickers needs to be a string, list and tuple.')
        self._tickers = [ticker.upper() for ticker in tickers]
        self._eligibility_df = pd.DataFrame(columns=self._tickers, index=self.observation_calendar)
        self._filter_desc_list = []

    def get_desc(self):
        if len(self._filter_desc_list):
            return '%s' % ', '.join(self._filter_desc_list)
        else:
            return 'no filter'

    def __repr__(self):
        return '<InvestmentUniverse(filter={})>'.format(self.get_desc())

